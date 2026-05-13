#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrivGraph 实验脚本（崩溃恢复 + 进度打印 + 稀疏加速 + 多进程并行）

修复版改动（v4）:
  1. mat0 稀疏矩阵统一用 int8 dtype（邻接矩阵只有 0/1，省 8 倍内存）
  2. 按数据集分组分发任务：大图（如 Enron）自动降级 worker 数
  3. mat2 后处理改为 in-place，减少 33696² 大矩阵的副本

Job 1 主对比：    PrivGraph vs Ours，4 数据集 × 7 ε × 10 reps
Job 2 超参敏感性：Chameleon × ε=2.0，5 个 inter_ratio × 30 reps = 150 reps

用法：
  python run_experiments.py                  # 跑全部（自动并行 + 自动降级）
  python run_experiments.py --workers=4      # 小图最多 4 进程；大图自动降到更少
  python run_experiments.py --workers=1      # 强制串行（调试用）
  python run_experiments.py --summary-only   # 只看现有 CSV 的均值汇总
"""

import os
import sys
import time
import argparse
import traceback
import pickle
from datetime import datetime
from collections import defaultdict

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
    step6_original, step6_v6_cl_compensated,
)

# ============================ 配置区 ============================
DATA_DIR    = './data'
RESULTS_DIR = './results'
LOG_DIR     = './logs'
CACHE_DIR   = './cache'

# Job 1
DATASETS    = ['Chamelon', 'Facebook', 'CA-HepPh', 'Enron']
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

# 大图启用近似直径的阈值
DIAM_APPROX_THRESHOLD = 10000
DIAM_APPROX_SAMPLES   = 30

# ===== 按数据集大小动态限制 worker 数（节点数 → 最大 worker）=====
# 估算：dense int8 邻接矩阵 ~= N² bytes
#       Enron 33696² ≈ 1.06 GB；step6 输出的 mat2 也是 N² ≈ 1 GB（按 int8 估）
#       所以单 worker 峰值约 2-3 GB
# 假设可用内存 ~10 GB，则 Enron 最多 3-4 个 worker
WORKER_CAP_BY_SIZE = [
    (5000,    None),   # < 5k 节点：不限
    (15000,   6),      # 5k - 15k：最多 6 个
    (40000,   3),      # 15k - 40k（含 Enron）：最多 3 个
    (100000,  2),      # 40k - 100k：最多 2 个
    (10**9,   1),      # 更大：串行
]
# ===============================================================

MAIN_CSV = os.path.join(RESULTS_DIR, 'main_comparison.csv')
HP_CSV   = os.path.join(RESULTS_DIR, 'hp_inter_ratio.csv')


# ----------------------------- 工具 ------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


def setup_dirs():
    for d in (RESULTS_DIR, LOG_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def fmt_eta(seconds):
    if seconds < 60:   return f'{seconds:.0f}s'
    if seconds < 3600: return f'{seconds/60:.1f}min'
    return f'{seconds/3600:.2f}h'


def cap_workers_by_size(n_node, requested):
    """根据图规模降级 worker 数"""
    for limit, cap in WORKER_CAP_BY_SIZE:
        if n_node < limit:
            return requested if cap is None else min(requested, cap)
    return 1


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


# ------------------------ 直径近似算法 --------------------------
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


def cal_diam_smart(mat_sp_or_dense, n_node):
    if sparse.issparse(mat_sp_or_dense):
        mat_sp = mat_sp_or_dense
    else:
        mat_sp = sparse.csr_matrix(mat_sp_or_dense)
    if n_node >= DIAM_APPROX_THRESHOLD:
        return cal_diam_approx(mat_sp)
    else:
        if sparse.issparse(mat_sp_or_dense):
            return cal_diam(mat_sp_or_dense.toarray())
        return cal_diam(mat_sp_or_dense)


# --------------------------- 核心流程 ----------------------------
def precompute_baseline(mat0_sp, name):
    """对原图算一次评测基准（mat0_sp 为 int8 稀疏矩阵）"""
    print(f'  [基准:{name}] 构建图 ...', flush=True)
    g = nx.from_scipy_sparse_array(mat0_sp, create_using=nx.Graph)
    n_node = g.number_of_nodes()
    n_edge = g.number_of_edges()
    print(f'  [基准:{name}] 节点={n_node}, 边={n_edge}', flush=True)

    print(f'  [基准:{name}] Louvain 社区划分 ...', flush=True)
    par = community.best_partition(g)
    mod = community.modularity(par, g)

    print(f'  [基准:{name}] 度分布 / 聚类系数 / 直径 ...', flush=True)
    # int8 sum 必须升类型，否则可能溢出
    deg = np.asarray(mat0_sp.sum(0, dtype=np.int64)).flatten()
    deg_dist = np.bincount(np.int64(deg))
    cc = nx.transitivity(g)
    diam = cal_diam_smart(mat0_sp, n_node)

    print(f'  [基准:{name}] 特征向量中心性 ...', flush=True)
    try:
        evc = nx.eigenvector_centrality(g, max_iter=10000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        evc = nx.eigenvector_centrality_numpy(g)
    evc_sorted = dict(sorted(evc.items(), key=lambda x: x[1], reverse=True))
    evc_ak = list(evc_sorted.keys())
    evc_val = np.array(list(evc_sorted.values()))

    print(f'  [基准:{name}] 完成。mod={mod:.4f}, cc={cc:.4f}, diam={diam}', flush=True)
    return {
        'n_node':   n_node,
        'par':      par,
        'mod':      mod,
        'cc':       cc,
        'diam':     diam,
        'deg_dist': deg_dist,
        'evc_ak':   evc_ak,
        'evc_val':  evc_val,
    }


def load_or_build_baseline(dataset):
    cache_path = os.path.join(CACHE_DIR, f'{dataset}.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                blob = pickle.load(f)
            mat0_sp = blob['mat0_sp']
            # 兼容旧缓存：如果不是 int8 就转一下
            if mat0_sp.dtype != np.int8:
                mat0_sp = mat0_sp.astype(np.int8)
            return mat0_sp, blob['baseline']
        except Exception as e:
            print(f'[警告] 缓存 {cache_path} 损坏: {e}')

    data_path = os.path.join(DATA_DIR, f'{dataset}.txt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'数据集文件不存在: {data_path}')

    print(f'\n[加载数据集 {dataset}]', flush=True)
    mat0, _ = get_mat(data_path)
    # === 关键：邻接矩阵只有 0/1，int8 足够，省 8 倍内存 ===
    mat0_sp = sparse.csr_matrix(mat0).astype(np.int8)
    del mat0   # 立即释放原始 dense
    baseline = precompute_baseline(mat0_sp, dataset)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({'mat0_sp': mat0_sp, 'baseline': baseline}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  [基准:{dataset}] 已缓存到 {cache_path}', flush=True)
    except Exception as e:
        print(f'[警告] 缓存写入失败: {e}', flush=True)

    return mat0_sp, baseline


def run_one_rep(mat0_sp, baseline, method, epsilon, inter_ratio,
                e1_r=DEFAULT_E1_R, e2_r=DEFAULT_E2_R,
                N=DEFAULT_N, t=DEFAULT_T):
    """
    跑一次完整 PrivGraph 流程，返回评测 dict。
    内存优化：mat0 全程 int8；mat2 后处理 in-place。
    """
    e1 = e1_r * epsilon
    e2 = e2_r * epsilon
    e3 = (1.0 - e1_r - e2_r) * epsilon
    ev_lambda = 1.0 / e3
    dd_lam = 2.0 / e3

    mat0_node = mat0_sp.shape[0]
    g0 = nx.from_scipy_sparse_array(mat0_sp, create_using=nx.Graph)

    # ---- Step 1-2: 社区划分 ----
    # community_init 需要稠密矩阵；int8 形式只占 1 GB（Enron），可接受
    mat0_dense_for_init = mat0_sp.toarray()
    pvarr_init = community_init(mat0_dense_for_init, g0, epsilon=e1, nr=N, t=t)
    del mat0_dense_for_init   # 立即释放
    part_init = {i: int(pvarr_init[i]) for i in range(len(pvarr_init))}
    par_final = comm.best_partition(g0, part_init, epsilon_EM=e2)
    pvarr = np.array(list(par_final.values()))
    comm_n = int(pvarr.max()) + 1
    pvs = [list(np.where(pvarr == i)[0]) for i in range(comm_n)]

    # ---- Step 3-4: 边向量（稀疏 M^T @ A @ M）----
    comm_vec = np.empty(mat0_node, dtype=np.int32)
    for ci, members in enumerate(pvs):
        comm_vec[members] = ci
    M = sparse.csr_matrix(
        (np.ones(mat0_node, dtype=np.int64),
         (np.arange(mat0_node), comm_vec)),
        shape=(mat0_node, comm_n)
    )
    # 用 int64 累加避免 int8 溢出
    mat0_int64 = mat0_sp.astype(np.int64)
    ev_mat = np.asarray((M.T @ mat0_int64 @ M).todense()).astype(np.int64)
    del mat0_int64

    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

    # ---- Step 5: 度序列（稀疏子矩阵）----
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

    # ---- 对称化 + 0/1 化（in-place，减少副本）----
    # 原版要 3 次创建 33696² 副本；这里压缩
    if mat2.dtype != np.int8:
        try:
            mat2 = mat2.astype(np.int8, copy=False)
        except Exception:
            pass
    # 直接拿上三角，再与转置或运算，避免多次大矩阵加法
    np.fill_diagonal(mat2, 0)
    iu = np.triu_indices_from(mat2, k=1)
    upper = (mat2[iu] > 0) | (mat2.T[iu] > 0)
    mat2[:] = 0
    mat2[iu] = upper.astype(mat2.dtype)
    # 对称镜像
    mat2 += mat2.T

    # ---- 评测 ----
    mat2_sp = sparse.csr_matrix(mat2)
    del mat2   # 立即释放稠密 mat2
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

    del mat2_sp, g2   # 主动释放

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
        'edges':      int(n_edges),
    }


# -------------------- 子进程入口（必须顶层定义）--------------------
def _worker_main(args):
    dataset, method, eps, rep, inter_ratio, job_tag = args
    t0 = time.time()
    try:
        mat0_sp, baseline = load_or_build_baseline(dataset)
        seed = (hash((dataset, method, float(eps), int(rep), float(inter_ratio)))
                & 0x7FFFFFFF)
        np.random.seed(seed)
        res = run_one_rep(mat0_sp, baseline, method, eps, inter_ratio=inter_ratio)
        return True, args, res, time.time() - t0
    except Exception as e:
        tb = traceback.format_exc()
        return False, args, f'{e}\n{tb}', time.time() - t0


# --------------------------- Job 1 ----------------------------
def run_main_comparison(workers):
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

    all_tasks = []
    for dataset in DATASETS:
        for method in METHODS:
            for eps in EPSILONS:
                for rep in range(N_REPS_MAIN):
                    key = (str(dataset), str(method), float(eps), int(rep))
                    if key not in done:
                        all_tasks.append((dataset, method, float(eps), int(rep),
                                          DEFAULT_INTER_RATIO, 'main'))

    total = len(all_tasks) + len(done)
    print(f'已完成: {len(done)}/{total}')
    print(f'剩余:   {len(all_tasks)}')
    print(f'请求 worker 数: {workers}（实际会按数据集规模动态降级）')
    if not all_tasks:
        print('  [跳过] 全部已完成')
        return

    # 按数据集分组，每组用不同 worker 数
    tasks_by_ds = defaultdict(list)
    for t in all_tasks:
        tasks_by_ds[t[0]].append(t)

    for ds in DATASETS:
        if ds not in tasks_by_ds:
            continue
        ds_tasks = tasks_by_ds[ds]

        data_path = os.path.join(DATA_DIR, f'{ds}.txt')
        if not os.path.exists(data_path):
            print(f'[错误] {data_path} 不存在，跳过 {ds}')
            continue

        # 主进程预热缓存
        try:
            mat0_sp, _ = load_or_build_baseline(ds)
            n_node = mat0_sp.shape[0]
        except Exception as e:
            print(f'[错误] 预构建 {ds} baseline 失败: {e}')
            continue

        ds_workers = cap_workers_by_size(n_node, workers)
        print(f'\n>>> 数据集 {ds}: 节点={n_node}, 任务={len(ds_tasks)}, '
              f'worker={ds_workers}', flush=True)

        _run_tasks(ds_tasks, ds_workers, MAIN_CSV, cols, kind='main')


# --------------------------- Job 2 ----------------------------
def run_hp_sensitivity(workers):
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
                tasks.append((HP_DATASET, 'ours', float(HP_EPSILON),
                              int(rep), float(ir), 'hp'))

    total = len(tasks) + len(done)
    print(f'已完成: {len(done)}/{total}')
    print(f'剩余:   {len(tasks)}')
    if not tasks:
        print('  [跳过] 全部已完成')
        return

    try:
        mat0_sp, _ = load_or_build_baseline(HP_DATASET)
        n_node = mat0_sp.shape[0]
    except Exception as e:
        print(f'[错误] 预构建 {HP_DATASET} baseline 失败: {e}')
        return

    ds_workers = cap_workers_by_size(n_node, workers)
    print(f'\n>>> 数据集 {HP_DATASET}: 节点={n_node}, worker={ds_workers}', flush=True)

    _run_tasks(tasks, ds_workers, HP_CSV, cols, kind='hp')


# --------------------------- 任务分发 ---------------------------
def _run_tasks(tasks, workers, csv_path, cols, kind):
    t_start = time.time()
    n_total = len(tasks)
    done_n = 0

    if workers <= 1:
        for args in tasks:
            ok, _, res_or_err, dt = _worker_main(args)
            done_n += 1
            _handle_result(ok, args, res_or_err, dt, csv_path, cols,
                           done_n, n_total, t_start, kind)
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker_main, a): a for a in tasks}
                try:
                    for fut in as_completed(futures):
                        ok, args, res_or_err, dt = fut.result()
                        done_n += 1
                        _handle_result(ok, args, res_or_err, dt, csv_path, cols,
                                       done_n, n_total, t_start, kind)
                except KeyboardInterrupt:
                    print('\n[中断] 正在取消未完成任务 ...')
                    for f in futures:
                        f.cancel()
                    raise
        except KeyboardInterrupt:
            print('[中断] 已保存到上一 rep。下次启动会自动续跑。')
            raise

    print(f'\n[{kind}] 完成。总耗时: {fmt_eta(time.time() - t_start)}', flush=True)


def _handle_result(ok, args, res_or_err, dt, csv_path, cols,
                   done_n, n_total, t_start, kind):
    dataset, method, eps, rep, inter_ratio, _ = args
    if not ok:
        print(f'[错误] {dataset}/{method}/ε={eps}/rep={rep} ir={inter_ratio}:\n{res_or_err}',
              flush=True)
        return

    res = res_or_err
    if kind == 'main':
        row = {
            'dataset':   dataset,
            'method':    method,
            'epsilon':   float(eps),
            'rep':       int(rep),
            **res,
            'time_sec':  round(dt, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    else:
        row = {
            'dataset':     dataset,
            'inter_ratio': float(inter_ratio),
            'epsilon':     float(eps),
            'rep':         int(rep),
            **res,
            'time_sec':    round(dt, 2),
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    append_row(csv_path, row, cols)

    elapsed = time.time() - t_start
    avg = elapsed / done_n
    eta = (n_total - done_n) * avg
    tag = (f'{dataset:<10} {method:<9} ε={eps:<3} rep={rep:<2}'
           if kind == 'main'
           else f'ir={inter_ratio:<5} ε={eps} rep={rep:<2}')
    print(f'[{done_n:>4}/{n_total}] {tag} | '
          f'NMI={res["nmi"]:.4f} mod_rel={res["mod_rel"]:.4f} '
          f'deg_kl={res["deg_kl"]:.3f} | '
          f't={dt:5.1f}s ETA={fmt_eta(eta)}',
          flush=True)


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
    parser = argparse.ArgumentParser(description='PrivGraph 实验脚本')
    parser.add_argument('--job', type=int, choices=[0, 1, 2], default=0,
                        help='0=全部, 1=主对比, 2=超参')
    parser.add_argument('--workers', type=int, default=0,
                        help='请求 worker 数（0=自动；大图会按节点数自动降级）')
    parser.add_argument('--summary-only', action='store_true',
                        help='只打印汇总，不跑实验')
    parser.add_argument('--no-cache', action='store_true',
                        help='忽略并重新生成 baseline 缓存')
    args = parser.parse_args()

    setup_dirs()

    if args.no_cache:
        for f in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass
        print('[缓存] 已清空')

    if args.summary_only:
        print_summary()
        return

    if args.workers <= 0:
        import multiprocessing as mp
        workers = max(1, mp.cpu_count() - 1)
    else:
        workers = args.workers

    log_path = os.path.join(LOG_DIR,
                            f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_f)

    print(f'实验启动 @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'日志:   {log_path}')
    print(f'结果:   {RESULTS_DIR}/')
    print(f'缓存:   {CACHE_DIR}/')
    print(f'请求并行数: {workers}（小图保持，大图按 WORKER_CAP_BY_SIZE 降级）')
    t0 = time.time()

    try:
        if args.job in (0, 1):
            run_main_comparison(workers)
        if args.job in (0, 2):
            run_hp_sensitivity(workers)
    except KeyboardInterrupt:
        print('\n用户中断。下次启动可自动从断点恢复。')
        sys.exit(130)

    print(f'\n全部任务完成。总耗时: {fmt_eta(time.time() - t0)}')
    print_summary()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()