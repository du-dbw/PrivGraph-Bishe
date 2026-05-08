"""
run_overnight.py — 一键跑完毕业论文所有实验。

特性
----
- 每个 (config, exper) 完成后立刻 append 到 CSV，崩溃后重跑会自动跳过已完成项。
- 三组实验各自独立 CSV，可单独跑：--only main / multi / hp
- 每个 trial 都在 try/except 里，单点失败不会拖垮整批。
- 实时打印 ETA，便于估计何时跑完。

用法
----
python run_overnight.py                       # 全部跑（推荐）
python run_overnight.py --only main           # 只跑 Chameleon 主对比
python run_overnight.py --only multi          # 只跑多数据集验证
python run_overnight.py --only hp             # 只跑超参扫描
python run_overnight.py --only multi --datasets Facebook CA-HepPh
                                              # 多数据集跳过 Enron
python run_overnight.py --reps 5              # 把每组的重复次数从 10 降到 5

跑完后：所有结果在 ./result/*.csv，直接读 csv 填表。
"""

import os, time, argparse, traceback
import numpy as np
import pandas as pd
import networkx as nx
import community
from numpy.random import laplace
from sklearn import metrics

from utils import *  # comm, community_init, generate_intra_edge, FO_pp, cal_diam,
                     # cal_rel, cal_kl, cal_overlap, cal_MAE, get_mat, get_uptri_arr,
                     # get_upmat, step6_v3_full_fixed, post_process_edge_swap


# ===================== 全局配置 =====================
RESULT_DIR = './result'
DATA_DIR   = './data'

EPS_LIST  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
N_REPS    = 10
N_INIT    = 20
T_RES     = 1.0
E1_R      = 1/3
E2_R      = 1/3

# Ours-Full 默认超参
INTRA_RATIO = 0.05
INTER_RATIO = 0.10
SWAP_RATIO  = 0.30


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


# ===================== 单个 trial =====================
def run_trial(mat0, n, ref, epsilon, method,
              intra_ratio=INTRA_RATIO, inter_ratio=INTER_RATIO, swap_ratio=SWAP_RATIO,
              e1_r=E1_R, e2_r=E2_R, N=N_INIT, t=T_RES):
    """method ∈ {'PrivGraph','Ours-R','Ours-P','Ours-Full'}"""
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
    if method in ('PrivGraph', 'Ours-P'):
        mat2 = step6_original(n, comm_n, mat1_pvs, dd_s, ev_mat)
    elif method in ('Ours-R', 'Ours-Full'):
        mat2 = step6_v3_full_fixed(n, comm_n, mat1_pvs, dd_s, ev_mat,
                                   intra_ratio=intra_ratio, inter_ratio=inter_ratio)
    else:
        raise ValueError(f"unknown method: {method}")

    mat2 = symmetrize(mat2)

    # --- 后处理（可选） ---
    if method in ('Ours-P', 'Ours-Full'):
        mat2 = post_process_edge_swap(mat2, mat1_pvs, comm_n, n_iter_ratio=swap_ratio)

    # --- 评估 ---
    G2 = nx.from_numpy_array(mat2, create_using=nx.Graph)
    par2 = community.best_partition(G2)
    deg2 = np.sum(mat2, 0)
    deg_dist2 = np.bincount(np.int64(deg2))
    evc2 = nx.eigenvector_centrality(G2, max_iter=10000)
    evc2_a = dict(sorted(evc2.items(), key=lambda x: x[1], reverse=True))

    evc_kn = np.int64(0.01 * n)
    return {
        'nmi':         metrics.normalized_mutual_info_score(
                            list(ref['par'].values()), list(par2.values())),
        'evc_overlap': cal_overlap(ref['evc_ak'], list(evc2_a.keys()), evc_kn),
        'evc_MAE':     cal_MAE(ref['evc_val'],
                               np.array(list(evc2_a.values())), k=evc_kn),
        'deg_kl':      cal_kl(ref['deg_dist'], deg_dist2),
        'diam_rel':    cal_rel(ref['diam'], cal_diam(mat2)),
        'cc_rel':      cal_rel(ref['cc'],   nx.transitivity(G2)),
        'mod_rel':     cal_rel(ref['mod'],  community.modularity(par2, G2)),
        'mat2_edges':  G2.number_of_edges(),
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
    return mat0, n, ref


# ===================== Group 1: Chameleon 主对比 =====================
def run_main_comparison(reps):
    csv_path = os.path.join(RESULT_DIR, 'main_chameleon.csv')
    methods = ['PrivGraph', 'Ours-R', 'Ours-P', 'Ours-Full']
    mat0, n, ref = load_dataset('Chamelon')
    df = load_csv(csv_path)
    prog = Progress(len(methods) * len(EPS_LIST) * reps, 'main-Chameleon')

    for method in methods:
        for eps in EPS_LIST:
            for exper in range(reps):
                key = {'method': method, 'eps': eps, 'exper': exper}
                t0 = time.time()
                if is_done(df, key):
                    prog.step(0.0, skipped=True); continue
                try:
                    m = run_trial(mat0, n, ref, eps, method)
                    row = {**key, **m}
                    append_row(csv_path, row)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    extra = f"nmi={m['nmi']:.3f}"
                except Exception as ex:
                    print(f"  !! 失败 {key} -> {ex}"); traceback.print_exc()
                    extra = 'FAILED'
                prog.step(time.time() - t0, extra=extra)


# ===================== Group 2: 多数据集验证 =====================
def run_multi_dataset(datasets, reps):
    methods = ['PrivGraph', 'Ours-Full']
    for ds in datasets:
        csv_path = os.path.join(RESULT_DIR, f'multi_{ds}.csv')
        try:
            mat0, n, ref = load_dataset(ds)
        except Exception as ex:
            print(f"!! 跳过 {ds}: {ex}"); continue
        df = load_csv(csv_path)
        prog = Progress(len(methods) * len(EPS_LIST) * reps, f'multi-{ds}')
        for method in methods:
            for eps in EPS_LIST:
                for exper in range(reps):
                    key = {'method': method, 'eps': eps, 'exper': exper}
                    t0 = time.time()
                    if is_done(df, key):
                        prog.step(0.0, skipped=True); continue
                    try:
                        m = run_trial(mat0, n, ref, eps, method)
                        row = {**key, **m}
                        append_row(csv_path, row)
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        extra = f"nmi={m['nmi']:.3f}"
                    except Exception as ex:
                        print(f"  !! 失败 {key} -> {ex}"); traceback.print_exc()
                        extra = 'FAILED'
                    prog.step(time.time() - t0, extra=extra)
        # 释放此数据集占用的大矩阵内存
        del mat0, ref


# ===================== Group 3: 超参扫描（Chameleon, eps=2.0） =====================
def _hp_sweep(name, sweep_values, hp_key, eps, mat0, n, ref, reps):
    csv_path = os.path.join(RESULT_DIR, f'hp_{name}.csv')
    df = load_csv(csv_path)
    prog = Progress(len(sweep_values) * reps, f'hp-{name}')
    for v in sweep_values:
        for exper in range(reps):
            key = {hp_key: v, 'exper': exper}
            t0 = time.time()
            if is_done(df, key):
                prog.step(0.0, skipped=True); continue
            kw = {'intra_ratio': INTRA_RATIO,
                  'inter_ratio': INTER_RATIO,
                  'swap_ratio':  SWAP_RATIO}
            kw[hp_key] = v
            try:
                m = run_trial(mat0, n, ref, eps, 'Ours-Full', **kw)
                row = {**key, **m}
                append_row(csv_path, row)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                extra = f"nmi={m['nmi']:.3f}"
            except Exception as ex:
                print(f"  !! 失败 {key} -> {ex}"); traceback.print_exc()
                extra = 'FAILED'
            prog.step(time.time() - t0, extra=extra)


def run_hyperparam(reps, eps=2.0):
    mat0, n, ref = load_dataset('Chamelon')
    _hp_sweep('inter', [0.05, 0.10, 0.15, 0.20, 0.30], 'inter_ratio', eps, mat0, n, ref, reps)
    _hp_sweep('intra', [0.00, 0.05, 0.10, 0.15],       'intra_ratio', eps, mat0, n, ref, reps)
    _hp_sweep('swap',  [0.0, 0.1, 0.3, 0.5, 0.7],      'swap_ratio',  eps, mat0, n, ref, reps)


# ===================== 跑完后的小结 =====================
def print_summary():
    print("\n" + "=" * 60)
    print("结果文件汇总（行数 = 已完成 trial 数）")
    print("=" * 60)
    for f in sorted(os.listdir(RESULT_DIR)):
        if f.endswith('.csv'):
            n = len(pd.read_csv(os.path.join(RESULT_DIR, f)))
            print(f"  {f:40s} {n} rows")


# ===================== 入口 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['main', 'multi', 'hp'], default=None)
    parser.add_argument('--reps', type=int, default=N_REPS)
    parser.add_argument('--datasets', nargs='+',
                        default=['Facebook', 'CA-HepPh', 'Enron'])
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    t_begin = time.time()
    print("=" * 60)
    print(f"Overnight runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"reps={args.reps}, only={args.only}, datasets={args.datasets}")
    print("=" * 60, flush=True)

    if args.only in (None, 'main'):
        print("\n##### Group 1: Chameleon 主对比 (4 methods × 7 eps × reps) #####")
        run_main_comparison(reps=args.reps)
    if args.only in (None, 'hp'):
        print("\n##### Group 2: 超参扫描 (Chameleon, eps=2.0) #####")
        run_hyperparam(reps=args.reps)
    if args.only in (None, 'multi'):
        print("\n##### Group 3: 多数据集验证 #####")
        run_multi_dataset(datasets=args.datasets, reps=args.reps)

    print_summary()
    print(f"\n>>> All done in {(time.time()-t_begin)/60:.1f} min")


if __name__ == '__main__':
    main()