#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enron 预算分配扫描脚本

目标：验证假设——Enron 上 CC Rel 爆炸的真因是 ε1/ε2 不足导致社区划分质量过低，
而非 step6 度序列放大的固有副作用。

策略：
  - 仅在 Enron 上跑
  - 仅使用 ours（不是 ours_v2，因为 v2 已验证对 Enron 无效）
  - 扫多种 (e1_r, e2_r, e3_r) 组合
  - 关键观察指标：CC Rel、NMI、Mod Rel

用法：
  python enron_eps_sweep.py                    # 默认 ε=2.0，每组 10 reps
  python enron_eps_sweep.py --eps 1.0 3.0      # 多个 ε
  python enron_eps_sweep.py --reps 5           # 减少 reps 加速

输出: ./results/enron_eps_sweep.csv
"""

import os
import sys
import time
import argparse
import traceback
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import laplace
from scipy import sparse
from sklearn import metrics
import community
from concurrent.futures import ProcessPoolExecutor, as_completed

import comm
from utils import (
    get_mat, community_init,
    get_uptri_arr, get_upmat, FO_pp,
    cal_diam, cal_overlap, cal_kl, cal_rel, cal_MAE,
    step6_v6_cl_compensated,
)

# ============================ 配置 ============================
DATA_DIR    = './data'
RESULTS_DIR = './results'
LOG_DIR     = './logs'
CACHE_DIR   = './cache'

CSV_PATH = os.path.join(RESULTS_DIR, 'enron_eps_sweep.csv')

# 预算分配方案（e1_r + e2_r + e3_r = 1.0）
BUDGET_SCHEMES = {
    'A_default':  (1.0/3, 1.0/3, 1.0/3),     # 论文默认
    'B_aggressive': (0.20, 0.60, 0.20),       # 你的提议：激进加强 ε2
    'C_balanced': (0.25, 0.50, 0.25),         # 中等加强 ε2
    'D_save_e3':  (0.15, 0.55, 0.30),         # 加强 ε2，保 ε3
    'E_low_e1':   (0.10, 0.45, 0.45),         # 牺牲 ε1，e2/e3 平衡
    'F_e2_heavy': (0.20, 0.50, 0.30),         # ε2 重，但 ε3 不太弱
}

DEFAULT_INTER_RATIO = 0.10
DEFAULT_N = 20
DEFAULT_T = 1.0

# 直径近似
DIAM_APPROX_THRESHOLD = 10000
DIAM_APPROX_SAMPLES = 30


# ----------------------------- 工具 ------------------------------
def setup_dirs():
    for d in (RESULTS_DIR, LOG_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def fmt_eta(seconds):
    if seconds < 60:   return f'{seconds:.0f}s'
    if seconds < 3600: return f'{seconds/60:.1f}min'
    return f'{seconds/3600:.2f}h'


def load_done_keys(csv_path, key_cols):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()
        keys = set()
        for _, row in df[key_cols].iterrows():
            key = tuple(v.item() if hasattr(v, 'item') else v
                        for v in row.values)
            keys.add(key)
        return keys
    except Exception as e:
        print(f'[警告] 读取 {csv_path} 失败: {e}')
        return set()


def append_row(csv_path, row, cols):
    df = pd.DataFrame([row], columns=cols)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


# ------------------------ 直径近似 --------------------------
def cal_diam_approx(mat_sp, n_samples=DIAM_APPROX_SAMPLES, seed=None):
    import random
    rng = random.Random(seed)
    g = nx.from_scipy_sparse_array(mat_sp, create_using=nx.Graph)
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return 0
    ccs = list(nx.connected_components(g))
    if not ccs:
        return 0
    cc = max(ccs, key=len)
    sub = g.subgraph(cc)
    best = 0
    nodes = list(sub.nodes())
    seeds = rng.sample(nodes, min(n_samples, len(nodes)))
    for s in seeds:
        lengths = nx.single_source_shortest_path_length(sub, s)
        u = max(lengths, key=lengths.get)
        lengths2 = nx.single_source_shortest_path_length(sub, u)
        d = max(lengths2.values())
        if d > best:
            best = d
    return best


def cal_diam_smart(mat_sp, n_node):
    if n_node >= DIAM_APPROX_THRESHOLD:
        return cal_diam_approx(mat_sp)
    return cal_diam(mat_sp.toarray())


# --------------------------- baseline ----------------------------
def precompute_baseline(mat0_sp, name):
    print(f'  [基准:{name}] 构建图 ...', flush=True)
    g = nx.from_scipy_sparse_array(mat0_sp, create_using=nx.Graph)
    n_node = g.number_of_nodes()
    n_edge = g.number_of_edges()
    print(f'  [基准:{name}] 节点={n_node}, 边={n_edge}', flush=True)

    par = community.best_partition(g)
    mod = community.modularity(par, g)

    deg = np.asarray(mat0_sp.sum(0, dtype=np.int64)).flatten()
    deg_dist = np.bincount(np.int64(deg))
    cc = nx.transitivity(g)
    diam = cal_diam_smart(mat0_sp, n_node)

    try:
        evc = nx.eigenvector_centrality(g, max_iter=10000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        evc = nx.eigenvector_centrality_numpy(g)
    evc_sorted = dict(sorted(evc.items(), key=lambda x: x[1], reverse=True))
    evc_ak = list(evc_sorted.keys())
    evc_val = np.array(list(evc_sorted.values()))

    print(f'  [基准:{name}] mod={mod:.4f}, cc={cc:.4f}, diam={diam}', flush=True)
    return {
        'n_node': n_node, 'par': par, 'mod': mod, 'cc': cc, 'diam': diam,
        'deg_dist': deg_dist, 'evc_ak': evc_ak, 'evc_val': evc_val,
    }


def load_or_build_baseline(dataset):
    cache_path = os.path.join(CACHE_DIR, f'{dataset}.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                blob = pickle.load(f)
            mat0_sp = blob['mat0_sp']
            if mat0_sp.dtype != np.int8:
                mat0_sp = mat0_sp.astype(np.int8)
            return mat0_sp, blob['baseline']
        except Exception as e:
            print(f'[警告] 缓存损坏: {e}')

    data_path = os.path.join(DATA_DIR, f'{dataset}.txt')
    print(f'\n[加载 {dataset}]', flush=True)
    mat0, _ = get_mat(data_path)
    mat0_sp = sparse.csr_matrix(mat0).astype(np.int8)
    del mat0
    baseline = precompute_baseline(mat0_sp, dataset)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({'mat0_sp': mat0_sp, 'baseline': baseline}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f'[警告] 缓存写入失败: {e}', flush=True)
    return mat0_sp, baseline


# --------------------------- 核心：跑一次 ----------------------------
def run_one_rep(mat0_sp, baseline, epsilon, e1_r, e2_r,
                inter_ratio=DEFAULT_INTER_RATIO,
                N=DEFAULT_N, t=DEFAULT_T):
    e1 = e1_r * epsilon
    e2 = e2_r * epsilon
    e3 = (1.0 - e1_r - e2_r) * epsilon
    ev_lambda = 1.0 / e3
    dd_lam = 2.0 / e3

    mat0_node = mat0_sp.shape[0]
    g0 = nx.from_scipy_sparse_array(mat0_sp, create_using=nx.Graph)

    # Step 1-2: 社区划分
    mat0_dense_for_init = mat0_sp.toarray()
    pvarr_init = community_init(mat0_dense_for_init, g0, epsilon=e1, nr=N, t=t)
    del mat0_dense_for_init
    part_init = {i: int(pvarr_init[i]) for i in range(len(pvarr_init))}
    par_final = comm.best_partition(g0, part_init, epsilon_EM=e2)
    pvarr = np.array(list(par_final.values()))
    comm_n = int(pvarr.max()) + 1
    pvs = [list(np.where(pvarr == i)[0]) for i in range(comm_n)]

    # Step 3-4
    comm_vec = np.empty(mat0_node, dtype=np.int32)
    for ci, members in enumerate(pvs):
        comm_vec[members] = ci
    M = sparse.csr_matrix(
        (np.ones(mat0_node, dtype=np.int64),
         (np.arange(mat0_node), comm_vec)),
        shape=(mat0_node, comm_n)
    )
    mat0_int64 = mat0_sp.astype(np.int64)
    ev_mat = np.asarray((M.T @ mat0_int64 @ M).todense()).astype(np.int64)
    del mat0_int64

    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

    # Step 5
    dd_s = []
    for i in range(comm_n):
        members = pvs[i]
        sub = mat0_sp[members][:, members]
        dd1 = np.asarray(sub.sum(1, dtype=np.int64)).flatten()
        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1
        dd_s.append(list(dd1))

    # Step 6: ours
    mat2 = step6_v6_cl_compensated(
        mat0_node, comm_n, pvs, dd_s, ev_mat,
        inter_ratio=inter_ratio,
    )

    # 对称化
    if mat2.dtype != np.int8:
        try:
            mat2 = mat2.astype(np.int8, copy=False)
        except Exception:
            pass
    np.fill_diagonal(mat2, 0)
    iu = np.triu_indices_from(mat2, k=1)
    upper = (mat2[iu] > 0) | (mat2.T[iu] > 0)
    mat2[:] = 0
    mat2[iu] = upper.astype(mat2.dtype)
    mat2 += mat2.T

    # 评测
    mat2_sp = sparse.csr_matrix(mat2)
    del mat2
    g2 = nx.from_scipy_sparse_array(mat2_sp, create_using=nx.Graph)
    par2 = community.best_partition(g2)
    mod2 = community.modularity(par2, g2)
    cc2 = nx.transitivity(g2)

    deg2 = np.asarray(mat2_sp.sum(0, dtype=np.int64)).flatten()
    deg_dist2 = np.bincount(np.int64(deg2))

    try:
        evc2 = nx.eigenvector_centrality(g2, max_iter=10000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        evc2 = nx.eigenvector_centrality_numpy(g2)
    evc2_sorted = dict(sorted(evc2.items(), key=lambda x: x[1], reverse=True))
    evc2_ak = list(evc2_sorted.keys())
    evc2_val = np.array(list(evc2_sorted.values()))

    diam2 = cal_diam_smart(mat2_sp, mat0_node)
    evc_kn = max(1, int(0.01 * mat0_node))
    n_edges = g2.number_of_edges()

    return {
        'nmi':         float(metrics.normalized_mutual_info_score(
                           list(baseline['par'].values()),
                           list(par2.values()))),
        'evc_overlap': float(cal_overlap(baseline['evc_ak'], evc2_ak, evc_kn)),
        'evc_mae':     float(cal_MAE(baseline['evc_val'], evc2_val, k=evc_kn)),
        'deg_kl':      float(cal_kl(baseline['deg_dist'], deg_dist2)),
        'diam_rel':    float(cal_rel(baseline['diam'], diam2)),
        'cc_rel':      float(cal_rel(baseline['cc'], cc2)),
        'mod_rel':     float(cal_rel(baseline['mod'], mod2)),
        'edges':       int(n_edges),
        'comm_n':      int(comm_n),
    }


# -------------------- worker --------------------
def _worker_main(args):
    dataset, scheme_name, e1_r, e2_r, eps, rep = args
    t0 = time.time()
    try:
        mat0_sp, baseline = load_or_build_baseline(dataset)
        seed = (hash((dataset, scheme_name, float(eps), int(rep))) & 0x7FFFFFFF)
        np.random.seed(seed)
        res = run_one_rep(mat0_sp, baseline, eps, e1_r, e2_r)
        return True, args, res, time.time() - t0
    except Exception as e:
        return False, args, f'{e}\n{traceback.format_exc()}', time.time() - t0


# -------------------- 任务管理 --------------------
def run_sweep(eps_list, reps, workers, schemes_to_run):
    cols = [
        'dataset', 'scheme', 'e1_r', 'e2_r', 'e3_r', 'epsilon', 'rep',
        'nmi', 'evc_overlap', 'evc_mae', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel',
        'edges', 'comm_n', 'time_sec', 'timestamp',
    ]
    key_cols = ['dataset', 'scheme', 'epsilon', 'rep']
    done = load_done_keys(CSV_PATH, key_cols)

    tasks = []
    for scheme_name in schemes_to_run:
        e1_r, e2_r, e3_r = BUDGET_SCHEMES[scheme_name]
        for eps in eps_list:
            for rep in range(reps):
                key = ('Enron', scheme_name, float(eps), int(rep))
                if key not in done:
                    tasks.append(('Enron', scheme_name, e1_r, e2_r,
                                  float(eps), int(rep)))

    print(f'\n方案: {schemes_to_run}')
    print(f'ε: {eps_list}, reps: {reps}')
    print(f'已完成: {len(done)} | 剩余: {len(tasks)}\n')
    if not tasks:
        print('全部已完成')
        return

    # 预热缓存
    load_or_build_baseline('Enron')

    t_start = time.time()
    n_total = len(tasks)
    done_n = 0

    def _handle(ok, args, res_or_err, dt):
        nonlocal done_n
        done_n += 1
        dataset, scheme_name, e1_r, e2_r, eps, rep = args
        if not ok:
            print(f'[错误] {scheme_name}/ε={eps}/rep={rep}: {res_or_err[:300]}',
                  flush=True)
            return
        res = res_or_err
        e3_r = 1.0 - e1_r - e2_r
        row = {
            'dataset': dataset, 'scheme': scheme_name,
            'e1_r': e1_r, 'e2_r': e2_r, 'e3_r': e3_r,
            'epsilon': eps, 'rep': rep,
            **res,
            'time_sec': round(dt, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        append_row(CSV_PATH, row, cols)
        elapsed = time.time() - t_start
        eta = (n_total - done_n) * (elapsed / done_n)
        print(f'[{done_n:>3}/{n_total}] {scheme_name:<14} ε={eps} rep={rep} '
              f'(e1={e1_r:.2f},e2={e2_r:.2f},e3={e3_r:.2f}) '
              f'comm_n={res["comm_n"]:<4} | '
              f'NMI={res["nmi"]:.3f} mod_rel={res["mod_rel"]:.3f} '
              f'cc_rel={res["cc_rel"]:.3f} deg_kl={res["deg_kl"]:.2f} | '
              f't={dt:.0f}s ETA={fmt_eta(eta)}', flush=True)

    if workers <= 1:
        for args in tasks:
            ok, _, res_or_err, dt = _worker_main(args)
            _handle(ok, args, res_or_err, dt)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker_main, a): a for a in tasks}
            for fut in as_completed(futs):
                ok, args, res_or_err, dt = fut.result()
                _handle(ok, args, res_or_err, dt)

    print(f'\n完成。总耗时: {fmt_eta(time.time() - t_start)}')
    print_summary()


def print_summary():
    if not os.path.exists(CSV_PATH):
        return
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return
    print('\n' + '=' * 78)
    print('  Enron 预算分配扫描结果（按 scheme × ε 聚合）')
    print('=' * 78)
    agg = df.groupby(['scheme', 'epsilon']).agg(
        nmi=('nmi', 'mean'),
        mod_rel=('mod_rel', 'mean'),
        cc_rel=('cc_rel', 'mean'),
        deg_kl=('deg_kl', 'mean'),
        comm_n=('comm_n', 'mean'),
        n=('rep', 'count'),
    ).round(4)
    print(agg.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, nargs='+',
                        default=[2.0],
                        help='要扫的 ε 列表（默认仅 2.0，快速诊断）')
    parser.add_argument('--reps', type=int, default=10,
                        help='每组重复次数')
    parser.add_argument('--workers', type=int, default=3,
                        help='并行 worker 数（Enron 建议 ≤ 3）')
    parser.add_argument('--schemes', type=str, nargs='+',
                        default=list(BUDGET_SCHEMES.keys()),
                        help='要跑的方案名')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    setup_dirs()
    if args.summary_only:
        print_summary()
        return

    print('Enron 预算分配扫描')
    print(f'方案表: {list(BUDGET_SCHEMES.items())}')
    run_sweep(args.eps, args.reps, args.workers, args.schemes)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
