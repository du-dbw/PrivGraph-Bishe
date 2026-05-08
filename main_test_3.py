"""
run_diagnose_no_swap.py — 诊断脚本
==================================

目的
----
只跑 Enron 和 CA-HepPh 两个数据集，且**禁用 post_process_edge_swap**，
用来验证 mod_rel 波动是否主要来自后处理那一步。

对比方法
--------
- PrivGraph        : 原始 baseline（不变）
- Ours-Full-NoSwap : 完整流程，但跳过 post_process_edge_swap

结果文件
--------
./result/diag_no_swap_v2_<dataset>.csv

用法
----
python run_diagnose_no_swap.py                  # 两个数据集都跑，每组 10 次
python run_diagnose_no_swap.py --reps 5         # 每组只跑 5 次
python run_diagnose_no_swap.py --datasets Enron # 只跑 Enron

跑完后建议把同一个 (method, eps) 下 reps 次的结果做 std/mean，
和原 multi_*.csv 里 Ours-Full 的 std/mean 比一下，就能看出 swap 的影响。

★ 本版本相比上一版新增：
  - 记录每次 trial 的 comm_n（社区数）
  - 记录 mod_raw（重建图 modularity 绝对值）和 mod_orig（原图 modularity）
  用于诊断 mod_rel 驼峰的真正来源。
"""

import os, time, argparse, traceback
import numpy as np
import pandas as pd
import networkx as nx
import community
from numpy.random import laplace
from sklearn import metrics

from utils import *  # comm, community_init, generate_intra_edge, FO_pp, cal_diam,
                     # cal_rel, cal_kl, cal_overlap, cal_MAE, get_mat,
                     # get_uptri_arr, get_upmat, step6_v3_full_fixed
                     # (post_process_edge_swap 此脚本不调用)


# ===================== 全局配置 =====================
RESULT_DIR = './result'
DATA_DIR   = './data'

EPS_LIST  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
N_REPS    = 10
N_INIT    = 20
T_RES     = 1.0
E1_R      = 1/3
E2_R      = 1/3

# Ours-Full 默认超参（与 run_overnight.py 一致）
INTRA_RATIO = 0.05
INTER_RATIO = 0.10


# ===================== 原始 PrivGraph 重建 =====================
def step6_original(N, comm_n, pvs, dd_s, ev_mat):
    """PrivGraph baseline: CL intra + uniform inter."""
    mat2 = np.zeros([N, N], dtype=np.int8)
    for i in range(comm_n):
        nodes = pvs[i]
        if len(nodes) == 0:
            continue
        mat2[np.ix_(nodes, nodes)] = generate_intra_edge(dd_s[i])
        pi = np.array(pvs[i])
        for j in range(i + 1, comm_n):
            ev1 = ev_mat[i, j]
            if ev1 <= 0:
                continue
            pj = np.array(pvs[j])
            c1 = np.random.choice(pi, ev1)
            c2 = np.random.choice(pj, ev1)
            for k in range(ev1):
                mat2[c1[k], c2[k]] = 1
                mat2[c2[k], c1[k]] = 1
    return mat2


def symmetrize(mat2):
    mat2 = mat2 + np.transpose(mat2)
    mat2 = np.triu(mat2, 1)
    mat2 = mat2 + np.transpose(mat2)
    mat2[mat2 > 0] = 1
    return mat2


# ===================== 原图参考量预计算 =====================
def precompute_reference(mat0):
    G   = nx.from_numpy_array(mat0, create_using=nx.Graph)
    par = community.best_partition(G)
    deg = np.sum(mat0, 0)
    deg_dist = np.bincount(np.int64(deg))
    evc = nx.eigenvector_centrality(G, max_iter=10000)
    evc_a = dict(sorted(evc.items(), key=lambda x: x[1], reverse=True))
    return {
        'G': G, 'par': par, 'deg_dist': deg_dist,
        'evc_ak': list(evc_a.keys()),
        'evc_val': np.array(list(evc_a.values())),
        'diam': cal_diam(mat0),
        'cc': nx.transitivity(G),
        'mod': community.modularity(par, G),
    }


# ===================== 单个 trial（关键：禁用 swap） =====================
def run_trial_no_swap(mat0, n, ref, epsilon, method,
                      intra_ratio=INTRA_RATIO, inter_ratio=INTER_RATIO,
                      e1_r=E1_R, e2_r=E2_R, N=N_INIT, t=T_RES):
    """
    method ∈ {'PrivGraph', 'Ours-Full-NoSwap'}
    与原 run_trial 唯一区别：**完全不调用 post_process_edge_swap**
    """
    e1 = e1_r * epsilon
    e2 = e2_r * epsilon
    e3 = (1 - e1_r - e2_r) * epsilon
    ev_lambda = 1 / e3
    dd_lam    = 2 / e3
    G = ref['G']

    # --- 社区初始化 ---
    mat1_pvarr1 = community_init(mat0, G, epsilon=e1, nr=N, t=t)
    part1 = {i: mat1_pvarr1[i] for i in range(len(mat1_pvarr1))}

    # --- 社区调整 ---
    mat1_par1 = comm.best_partition(G, part1, epsilon_EM=e2)
    mat1_pvarr = np.array(list(mat1_par1.values()))
    comm_n = max(mat1_pvarr) + 1
    mat1_pvs = [list(np.where(mat1_pvarr == i)[0]) for i in range(comm_n)]

    # --- 边向量 + 拉普拉斯噪声 + NormSub ---
    ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)
    for i in range(comm_n):
        pi = mat1_pvs[i]
        ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, comm_n):
            pj = mat1_pvs[j]
            ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
            ev_mat[j, i] = ev_mat[i, j]
    ga = get_uptri_arr(ev_mat, ind=1)
    ga_noise = ga + laplace(0, ev_lambda, len(ga))
    ev_mat = get_upmat(FO_pp(ga_noise), comm_n, ind=1)

    # --- 度序列 + 拉普拉斯噪声 + NormSub ---
    dd_s = []
    for i in range(comm_n):
        dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
        dd1 = np.sum(dd1, 1)
        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1
        dd_s.append(list(dd1))

    # --- 图重建 ---
    if method == 'PrivGraph':
        mat2 = step6_original(n, comm_n, mat1_pvs, dd_s, ev_mat)
    elif method == 'Ours-Full-NoSwap':
        mat2 = step6_v3_full_fixed(n, comm_n, mat1_pvs, dd_s, ev_mat,
                                   intra_ratio=intra_ratio, inter_ratio=inter_ratio)
    else:
        raise ValueError(f"unknown method: {method}")

    mat2 = symmetrize(mat2)

    # --- ★ 关键：post_process_edge_swap 已注释掉 ★ ---
    # if method in ('Ours-P', 'Ours-Full'):
    #     mat2 = post_process_edge_swap(mat2, mat1_pvs, comm_n, n_iter_ratio=swap_ratio)

    # --- 评估 ---
    G2 = nx.from_numpy_array(mat2, create_using=nx.Graph)
    par2 = community.best_partition(G2)
    deg2 = np.sum(mat2, 0)
    deg_dist2 = np.bincount(np.int64(deg2))
    evc2 = nx.eigenvector_centrality(G2, max_iter=10000)
    evc2_a = dict(sorted(evc2.items(), key=lambda x: x[1], reverse=True))

    evc_kn = np.int64(0.01 * n)

    # ★ 新增：单独计算重建图 modularity，用于诊断
    mod_reconstructed = community.modularity(par2, G2)

    return {
        'nmi':         metrics.normalized_mutual_info_score(
                            list(ref['par'].values()), list(par2.values())),
        'evc_overlap': cal_overlap(ref['evc_ak'], list(evc2_a.keys()), evc_kn),
        'evc_MAE':     cal_MAE(ref['evc_val'],
                               np.array(list(evc2_a.values())), k=evc_kn),
        'deg_kl':      cal_kl(ref['deg_dist'], deg_dist2),
        'diam_rel':    cal_rel(ref['diam'], cal_diam(mat2)),
        'cc_rel':      cal_rel(ref['cc'],   nx.transitivity(G2)),
        'mod_rel':     cal_rel(ref['mod'],  mod_reconstructed),
        'mat2_edges':  G2.number_of_edges(),
        'comm_n':      int(comm_n),
        'mod_raw':     round(mod_reconstructed, 6),   # ★ 新增：重建图 modularity 绝对值
        'mod_orig':    round(ref['mod'], 6),           # ★ 新增：原图 modularity
    }


# ===================== Checkpoint 工具 =====================
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            print(f"  ! warn: {path} 损坏，将重建")
    return pd.DataFrame()


def is_done(df, key):
    if df.empty: return False
    mask = np.ones(len(df), dtype=bool)
    for k, v in key.items():
        col = df[k]
        if isinstance(v, float):
            mask &= np.isclose(col.astype(float), v)
        else:
            mask &= (col == v)
    return mask.any()


def append_row(path, row):
    df_row = pd.DataFrame([row])
    df_row.to_csv(path,
                  mode='a' if os.path.exists(path) else 'w',
                  header=not os.path.exists(path),
                  index=False)


# ===================== 进度条 =====================
class Progress:
    def __init__(self, total, label):
        self.total, self.label = total, label
        self.done = self.skipped = 0
        self.t0 = time.time()

    def step(self, dt, skipped=False, extra=''):
        self.done += 1
        if skipped: self.skipped += 1
        ran = self.done - self.skipped
        if ran > 0:
            elapsed = time.time() - self.t0
            rate = elapsed / ran
            eta_min = (self.total - self.done) * rate / 60
            print(f"  [{self.label}] {self.done}/{self.total}  "
                  f"this={dt:5.1f}s  ETA≈{eta_min:6.1f}min  "
                  f"(skip={self.skipped}) {extra}", flush=True)
        else:
            print(f"  [{self.label}] {self.done}/{self.total}  (skipped) {extra}", flush=True)


# ===================== 数据加载 =====================
def load_dataset(name):
    print(f">> Loading {name}.txt ...", flush=True)
    mat0, _ = get_mat(os.path.join(DATA_DIR, name + '.txt'))
    n = mat0.shape[0]
    e = int(np.sum(mat0) / 2)
    print(f"   nodes={n}, edges={e}", flush=True)
    print(f">> Pre-computing reference metrics on {name} ...", flush=True)
    ref = precompute_reference(mat0)
    print(f"   mod_orig={ref['mod']:.6f}", flush=True)
    return mat0, n, ref


# ===================== 主循环 =====================
def run_dataset(name, reps):
    csv_path = os.path.join(RESULT_DIR, f'diag_no_swap_v2_{name}.csv')
    methods = ['Ours-Full-NoSwap']

    try:
        mat0, n, ref = load_dataset(name)
    except Exception as ex:
        print(f"!! 跳过 {name}: {ex}")
        traceback.print_exc()
        return

    df = load_csv(csv_path)
    prog = Progress(len(methods) * len(EPS_LIST) * reps, f'diag-{name}')

    for method in methods:
        for eps in EPS_LIST:
            for exper in range(reps):
                key = {'method': method, 'eps': eps, 'exper': exper}
                t0 = time.time()
                if is_done(df, key):
                    prog.step(0.0, skipped=True); continue
                try:
                    m = run_trial_no_swap(mat0, n, ref, eps, method)
                    row = {**key, **m}
                    append_row(csv_path, row)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    extra = (f"nmi={m['nmi']:.3f} mod_rel={m['mod_rel']:.3f} "
                             f"comm_n={m['comm_n']} "
                             f"mod_raw={m['mod_raw']:.4f} mod_orig={m['mod_orig']:.4f}")
                except Exception as ex:
                    print(f"  !! 失败 {key} -> {ex}"); traceback.print_exc()
                    extra = 'FAILED'
                prog.step(time.time() - t0, extra=extra)

    del mat0, ref


# ===================== 结果摘要（顺手算 std） =====================
def print_summary():
    print("\n" + "=" * 70)
    print("结果摘要：每个 (method, eps) 的 mod_rel / mod_raw / comm_n mean ± std")
    print("=" * 70)
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.startswith('diag_no_swap_v2_') or not f.endswith('.csv'):
            continue
        path = os.path.join(RESULT_DIR, f)
        df = pd.read_csv(path)
        if df.empty:
            print(f"  {f}: empty"); continue
        print(f"\n--- {f} ---")
        agg_dict = dict(
            mod_rel_mean =('mod_rel', 'mean'),
            mod_rel_std  =('mod_rel', 'std'),
            cc_mean      =('cc_rel',  'mean'),
            cc_std       =('cc_rel',  'std'),
            nmi_mean     =('nmi',     'mean'),
            n            =('exper',   'count'),
        )
        if 'comm_n' in df.columns:
            agg_dict['comm_n_mean'] = ('comm_n', 'mean')
            agg_dict['comm_n_std']  = ('comm_n', 'std')
        if 'mod_raw' in df.columns:
            agg_dict['mod_raw_mean'] = ('mod_raw', 'mean')
            agg_dict['mod_raw_std']  = ('mod_raw', 'std')
        if 'mod_orig' in df.columns:
            agg_dict['mod_orig_mean'] = ('mod_orig', 'mean')
        agg = df.groupby(['method', 'eps']).agg(**agg_dict).round(4)
        print(agg.to_string())


# ===================== 入口 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=N_REPS)
    parser.add_argument('--datasets', nargs='+',
                        default=['Enron', 'CA-HepPh'])
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    t_begin = time.time()
    print("=" * 60)
    print(f"Diagnose (NO swap) started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"reps={args.reps}, datasets={args.datasets}")
    print(f"post_process_edge_swap: DISABLED")
    print(f"output prefix: diag_no_swap_v2_*.csv")
    print("=" * 60, flush=True)

    for ds in args.datasets:
        print(f"\n##### Dataset: {ds} #####")
        run_dataset(ds, reps=args.reps)

    print_summary()
    print(f"\n>>> All done in {(time.time()-t_begin)/60:.1f} min")


if __name__ == '__main__':
    main()