#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrivGraph 实验脚本（崩溃恢复 + 进度打印）

Job 1 主对比：    PrivGraph vs Ours，3 数据集 × 7 ε × 10 reps = 420 reps
Job 2 超参敏感性：Chameleon × ε=2.0，5 个 inter_ratio × 30 reps = 150 reps

用法：
  python run_experiments.py             # 跑全部
  python run_experiments.py --job=1     # 只跑主对比
  python run_experiments.py --job=2     # 只跑超参
  python run_experiments.py --summary-only  # 只看现有 CSV 的均值汇总
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import laplace
from sklearn import metrics
import community

import comm
from utils import (
    get_mat, community_init,
    get_uptri_arr, get_upmat, FO_pp,
    cal_diam, cal_overlap, cal_kl, cal_rel, cal_MAE,
    step6_original, step6_v6_cl_compensated,
)

# ============================ 配置区 ============================
DATA_DIR    = './data'
RESULTS_DIR = './results'
LOG_DIR     = './logs'

# Job 1
DATASETS    = ['Chamelon', 'Facebook', 'CA-HepPh' , 'Enron']  # 注意 Chamelon 与文件名一致
METHODS     = ['privgraph', 'ours']
EPSILONS    = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
N_REPS_MAIN = 10

# Job 2
HP_DATASET      = 'Chamelon'
HP_INTER_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30]
HP_EPSILON      = 2.0
N_REPS_HP       = 30

# 默认超参（与 main.py 一致）
DEFAULT_INTER_RATIO = 0.10
DEFAULT_E1_R = 1.0 / 3.0
DEFAULT_E2_R = 1.0 / 3.0
DEFAULT_N    = 20
DEFAULT_T    = 1.0
# ===============================================================

MAIN_CSV = os.path.join(RESULTS_DIR, 'main_comparison.csv')
HP_CSV   = os.path.join(RESULTS_DIR, 'hp_inter_ratio.csv')


# ----------------------------- 工具 ------------------------------
class Tee:
    """同时写到 stdout 和日志文件"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def setup_dirs():
    for d in (RESULTS_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def fmt_eta(seconds):
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}min'
    return f'{seconds/3600:.2f}h'


def load_done_keys(csv_path, key_cols):
    """读取 CSV 中已完成的 key 集合（用于跳过）"""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()
        keys = set()
        for _, row in df[key_cols].iterrows():
            # 把 numpy 类型转回 Python 原生类型，保证 hash 一致
            key = tuple(
                v.item() if hasattr(v, 'item') else v
                for v in row.values
            )
            keys.add(key)
        return keys
    except Exception as e:
        print(f'[警告] 读取 {csv_path} 失败: {e}（将重新跑全部）')
        return set()


def append_row(csv_path, row, cols):
    """追加一行到 CSV，文件不存在则写表头"""
    df = pd.DataFrame([row], columns=cols)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


# --------------------------- 核心流程 ----------------------------
def precompute_baseline(mat0, name):
    """对原图算一次评测基准"""
    print(f'  [基准] 构建图 ...', flush=True)
    g = nx.from_numpy_array(mat0, create_using=nx.Graph)
    n_node = g.number_of_nodes()
    n_edge = g.number_of_edges()
    print(f'  [基准] 节点={n_node}, 边={n_edge}')

    print(f'  [基准] Louvain 社区划分 ...', flush=True)
    par = community.best_partition(g)
    mod = community.modularity(par, g)

    print(f'  [基准] 度分布 / 聚类系数 / 直径 ...', flush=True)
    deg = np.sum(mat0, 0)
    deg_dist = np.bincount(np.int64(deg))
    cc = nx.transitivity(g)
    diam = cal_diam(mat0)

    print(f'  [基准] 特征向量中心性（可能较慢）...', flush=True)
    try:
        evc = nx.eigenvector_centrality(g, max_iter=10000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        evc = nx.eigenvector_centrality_numpy(g)
    evc_sorted = dict(sorted(evc.items(), key=lambda x: x[1], reverse=True))
    evc_ak = list(evc_sorted.keys())
    evc_val = np.array(list(evc_sorted.values()))

    print(f'  [基准] 完成。mod={mod:.4f}, cc={cc:.4f}, diam={diam}', flush=True)

    return {
        'graph':    g,
        'n_node':   n_node,
        'par':      par,
        'mod':      mod,
        'cc':       cc,
        'diam':     diam,
        'deg_dist': deg_dist,
        'evc_ak':   evc_ak,
        'evc_val':  evc_val,
    }


def run_one_rep(mat0, baseline, method, epsilon, inter_ratio,
                e1_r=DEFAULT_E1_R, e2_r=DEFAULT_E2_R,
                N=DEFAULT_N, t=DEFAULT_T):
    """跑一次完整 PrivGraph 流程，返回评测 dict"""
    e1 = e1_r * epsilon
    e2 = e2_r * epsilon
    e3 = (1.0 - e1_r - e2_r) * epsilon
    ev_lambda = 1.0 / e3
    dd_lam = 2.0 / e3

    g0 = baseline['graph']
    mat0_node = baseline['n_node']

    # ---- Step 1-2: 社区划分 ----
    pvarr_init = community_init(mat0, g0, epsilon=e1, nr=N, t=t)
    part_init = {i: int(pvarr_init[i]) for i in range(len(pvarr_init))}
    par_final = comm.best_partition(g0, part_init, epsilon_EM=e2)
    pvarr = np.array(list(par_final.values()))
    comm_n = int(pvarr.max()) + 1
    pvs = [list(np.where(pvarr == i)[0]) for i in range(comm_n)]

    # ---- Step 3-4: 边向量 ----
    ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)
    for i in range(comm_n):
        pi = pvs[i]
        ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, comm_n):
            pj = pvs[j]
            ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
            ev_mat[j, i] = ev_mat[i, j]

    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

    # ---- Step 5: 度序列 ----
    dd_s = []
    for i in range(comm_n):
        dd1 = mat0[np.ix_(pvs[i], pvs[i])]
        dd1 = np.sum(dd1, 1)
        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1
        dd_s.append(list(dd1))

    # ---- Step 6: 图重建 ----
    if method == 'privgraph':
        mat2 = step6_original(mat0_node, comm_n, pvs, dd_s, ev_mat)
    elif method == 'ours':
        mat2 = step6_v6_cl_compensated(
            mat0_node, comm_n, pvs, dd_s, ev_mat,
            inter_ratio=inter_ratio,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    # 对称化 + 0/1 化
    mat2 = mat2 + np.transpose(mat2)
    mat2 = np.triu(mat2, 1)
    mat2 = mat2 + np.transpose(mat2)
    mat2[mat2 > 0] = 1

    # ---- 评测 ----
    g2 = nx.from_numpy_array(mat2, create_using=nx.Graph)
    par2 = community.best_partition(g2)
    mod2 = community.modularity(par2, g2)
    cc2 = nx.transitivity(g2)

    deg2 = np.sum(mat2, 0)
    deg_dist2 = np.bincount(np.int64(deg2))

    try:
        evc2 = nx.eigenvector_centrality(g2, max_iter=10000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        evc2 = nx.eigenvector_centrality_numpy(g2)
    evc2_sorted = dict(sorted(evc2.items(), key=lambda x: x[1], reverse=True))
    evc2_ak = list(evc2_sorted.keys())
    evc2_val = np.array(list(evc2_sorted.values()))

    diam2 = cal_diam(mat2)
    evc_kn = int(0.01 * mat0_node)

    return {
        'nmi':        float(metrics.normalized_mutual_info_score(
                          list(baseline['par'].values()),
                          list(par2.values()))),
        'evc_overlap':float(cal_overlap(baseline['evc_ak'], evc2_ak, evc_kn)),
        'evc_mae':    float(cal_MAE(baseline['evc_val'], evc2_val, k=evc_kn)),
        'deg_kl':     float(cal_kl(baseline['deg_dist'], deg_dist2)),
        'diam_rel':   float(cal_rel(baseline['diam'], diam2)),
        'cc_rel':     float(cal_rel(baseline['cc'], cc2)),
        'mod_rel':    float(cal_rel(baseline['mod'], mod2)),
        'edges':      int(g2.number_of_edges()),
    }


# --------------------------- Job 1 ----------------------------
def run_main_comparison():
    print('\n' + '=' * 78)
    print('  Job 1: 主对比实验（PrivGraph vs Ours）')
    print('=' * 78)

    cols = [
        'dataset', 'method', 'epsilon', 'rep',
        'nmi', 'evc_overlap', 'evc_mae', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel',
        'edges', 'time_sec', 'timestamp',
    ]
    key_cols = ['dataset', 'method', 'epsilon', 'rep']
    done = load_done_keys(MAIN_CSV, key_cols)

    tasks = []
    for dataset in DATASETS:
        for method in METHODS:
            for eps in EPSILONS:
                for rep in range(N_REPS_MAIN):
                    key = (str(dataset), str(method), float(eps), int(rep))
                    if key not in done:
                        tasks.append(key)

    total = len(tasks) + len(done)
    print(f'已完成: {len(done)}/{total}')
    print(f'剩余:   {len(tasks)}')
    if not tasks:
        print('  [跳过] 全部已完成')
        return

    cache = {}
    t_start = time.time()

    for i, (dataset, method, eps, rep) in enumerate(tasks):
        # 数据集懒加载
        if dataset not in cache:
            data_path = os.path.join(DATA_DIR, f'{dataset}.txt')
            if not os.path.exists(data_path):
                print(f'[错误] 数据集文件不存在: {data_path}, 跳过该数据集所有任务')
                cache[dataset] = None
                continue
            print(f'\n[加载数据集 {dataset}]')
            mat0, _ = get_mat(data_path)
            baseline = precompute_baseline(mat0, dataset)
            cache[dataset] = (mat0, baseline)

        if cache[dataset] is None:
            continue
        mat0, baseline = cache[dataset]

        t0 = time.time()
        try:
            res = run_one_rep(mat0, baseline, method, eps,
                              inter_ratio=DEFAULT_INTER_RATIO)
            row = {
                'dataset':   dataset,
                'method':    method,
                'epsilon':   float(eps),
                'rep':       int(rep),
                **res,
                'time_sec':  round(time.time() - t0, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            append_row(MAIN_CSV, row, cols)

            elapsed = time.time() - t_start
            done_n = i + 1
            avg = elapsed / done_n
            eta = (len(tasks) - done_n) * avg
            print(f'[{done_n:>4}/{len(tasks)}] '
                  f'{dataset:<10} {method:<9} ε={eps:<3} rep={rep:<2} | '
                  f'NMI={res["nmi"]:.4f} mod_rel={res["mod_rel"]:.4f} '
                  f'deg_kl={res["deg_kl"]:.3f} | '
                  f't={time.time()-t0:5.1f}s ETA={fmt_eta(eta)}',
                  flush=True)

        except KeyboardInterrupt:
            print('\n[中断] 已保存到上一 rep。下次启动会自动续跑。')
            raise
        except Exception as e:
            print(f'[错误] {dataset}/{method}/ε={eps}/rep={rep}: {e}')
            traceback.print_exc()
            continue

    print(f'\nJob 1 完成。总耗时: {fmt_eta(time.time() - t_start)}')


# --------------------------- Job 2 ----------------------------
def run_hp_sensitivity():
    print('\n' + '=' * 78)
    print('  Job 2: 超参敏感性扫描（inter_ratio）')
    print('=' * 78)

    cols = [
        'dataset', 'inter_ratio', 'epsilon', 'rep',
        'nmi', 'evc_overlap', 'evc_mae', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel',
        'edges', 'time_sec', 'timestamp',
    ]
    key_cols = ['dataset', 'inter_ratio', 'epsilon', 'rep']
    done = load_done_keys(HP_CSV, key_cols)

    tasks = []
    for ir in HP_INTER_RATIOS:
        for rep in range(N_REPS_HP):
            key = (str(HP_DATASET), float(ir), float(HP_EPSILON), int(rep))
            if key not in done:
                tasks.append(key)

    total = len(tasks) + len(done)
    print(f'已完成: {len(done)}/{total}')
    print(f'剩余:   {len(tasks)}')
    if not tasks:
        print('  [跳过] 全部已完成')
        return

    data_path = os.path.join(DATA_DIR, f'{HP_DATASET}.txt')
    if not os.path.exists(data_path):
        print(f'[错误] 数据集文件不存在: {data_path}')
        return
    print(f'\n[加载数据集 {HP_DATASET}]')
    mat0, _ = get_mat(data_path)
    baseline = precompute_baseline(mat0, HP_DATASET)

    t_start = time.time()
    for i, (dataset, ir, eps, rep) in enumerate(tasks):
        t0 = time.time()
        try:
            res = run_one_rep(mat0, baseline, 'ours', eps, inter_ratio=ir)
            row = {
                'dataset':     dataset,
                'inter_ratio': float(ir),
                'epsilon':     float(eps),
                'rep':         int(rep),
                **res,
                'time_sec':    round(time.time() - t0, 2),
                'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            append_row(HP_CSV, row, cols)

            elapsed = time.time() - t_start
            done_n = i + 1
            avg = elapsed / done_n
            eta = (len(tasks) - done_n) * avg
            print(f'[{done_n:>4}/{len(tasks)}] '
                  f'ir={ir:<5} ε={eps} rep={rep:<2} | '
                  f'NMI={res["nmi"]:.4f} mod_rel={res["mod_rel"]:.4f} '
                  f'deg_kl={res["deg_kl"]:.3f} | '
                  f't={time.time()-t0:5.1f}s ETA={fmt_eta(eta)}',
                  flush=True)

        except KeyboardInterrupt:
            print('\n[中断] 已保存到上一 rep。下次启动会自动续跑。')
            raise
        except Exception as e:
            print(f'[错误] ir={ir}/ε={eps}/rep={rep}: {e}')
            traceback.print_exc()
            continue

    print(f'\nJob 2 完成。总耗时: {fmt_eta(time.time() - t_start)}')


# --------------------------- Summary --------------------------
def print_summary():
    print('\n' + '=' * 78)
    print('  实验结果汇总（按均值聚合）')
    print('=' * 78)

    if os.path.exists(MAIN_CSV):
        df = pd.read_csv(MAIN_CSV)
        print(f'\nJob 1 主对比 - {len(df)} 行：')
        if not df.empty:
            agg = df.groupby(['dataset', 'method', 'epsilon']).agg(
                nmi=('nmi', 'mean'),
                mod_rel=('mod_rel', 'mean'),
                deg_kl=('deg_kl', 'mean'),
                cc_rel=('cc_rel', 'mean'),
                diam_rel=('diam_rel', 'mean'),
                n=('rep', 'count'),
            ).round(4)
            print(agg.to_string())
    else:
        print('\nJob 1: 无数据')

    if os.path.exists(HP_CSV):
        df = pd.read_csv(HP_CSV)
        print(f'\nJob 2 inter_ratio 敏感性 - {len(df)} 行：')
        if not df.empty:
            agg = df.groupby(['inter_ratio']).agg(
                nmi=('nmi', 'mean'),
                nmi_std=('nmi', 'std'),
                mod_rel=('mod_rel', 'mean'),
                deg_kl=('deg_kl', 'mean'),
                cc_rel=('cc_rel', 'mean'),
                n=('rep', 'count'),
            ).round(4)
            print(agg.to_string())
    else:
        print('\nJob 2: 无数据')


# ---------------------------- main ----------------------------
def main():
    parser = argparse.ArgumentParser(description='PrivGraph 实验脚本（崩溃恢复）')
    parser.add_argument('--job', type=int, choices=[0, 1, 2], default=0,
                        help='0=全部, 1=主对比, 2=超参敏感性')
    parser.add_argument('--summary-only', action='store_true',
                        help='只打印已有 CSV 的均值汇总，不跑实验')
    args = parser.parse_args()

    setup_dirs()

    if args.summary_only:
        print_summary()
        return

    log_path = os.path.join(LOG_DIR,
                            f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_f)

    print(f'实验启动 @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'日志:  {log_path}')
    print(f'结果:  {RESULTS_DIR}/')
    t0 = time.time()

    try:
        if args.job in (0, 1):
            run_main_comparison()
        if args.job in (0, 2):
            run_hp_sensitivity()
    except KeyboardInterrupt:
        print('\n用户中断。下次启动可自动从断点恢复。')
        sys.exit(130)

    print(f'\n全部任务完成。总耗时: {fmt_eta(time.time() - t0)}')
    print_summary()


if __name__ == '__main__':
    main()