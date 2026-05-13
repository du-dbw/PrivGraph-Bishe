"""
run_sweep_inter_ratio.py — 快速遍历 inter_ratio
=================================================
固定 eps=2.0，遍历 inter_ratio，每组 5 次，找最优值。

用法：
  python run_sweep_inter_ratio.py --datasets CA-HepPh
  python run_sweep_inter_ratio.py --datasets CA-HepPh --eps 1.5
"""

import os, time, argparse, traceback
import numpy as np
import pandas as pd
import networkx as nx
import community
from numpy.random import laplace
from sklearn import metrics

from utils import (comm, community_init, generate_intra_edge, FO_pp,
                   cal_diam, cal_rel, cal_kl, cal_overlap, cal_MAE,
                   get_mat, get_uptri_arr, get_upmat)

from step6_v6_cl_compensated import step6_v6_cl_compensated

RESULT_DIR = './result'
DATA_DIR   = './data'

IR_LIST = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
N_REPS  = 5
N_INIT  = 20
T_RES   = 1.0
E1_R    = 1/3
E2_R    = 1/3


def symmetrize(mat2):
    mat2 = mat2 + np.transpose(mat2)
    mat2 = np.triu(mat2, 1)
    mat2 = mat2 + np.transpose(mat2)
    mat2[mat2 > 0] = 1
    return mat2


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
        'edges': G.number_of_edges(),
    }


def run_trial(mat0, n, ref, epsilon, inter_ratio):
    e1 = E1_R * epsilon
    e2 = E2_R * epsilon
    e3 = (1 - E1_R - E2_R) * epsilon
    ev_lambda = 1 / e3
    dd_lam    = 2 / e3
    G = ref['G']

    mat1_pvarr1 = community_init(mat0, G, epsilon=e1, nr=N_INIT, t=T_RES)
    part1 = {i: mat1_pvarr1[i] for i in range(len(mat1_pvarr1))}
    mat1_par1 = comm.best_partition(G, part1, epsilon_EM=e2)
    mat1_pvarr = np.array(list(mat1_par1.values()))
    comm_n = max(mat1_pvarr) + 1
    mat1_pvs = [list(np.where(mat1_pvarr == i)[0]) for i in range(comm_n)]

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

    dd_s = []
    for i in range(comm_n):
        dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
        dd1 = np.sum(dd1, 1)
        dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
        dd1 = FO_pp(dd1)
        dd1[dd1 < 0] = 0
        dd1[dd1 >= len(dd1)] = len(dd1) - 1
        dd_s.append(list(dd1))

    mat2 = step6_v6_cl_compensated(n, comm_n, mat1_pvs, dd_s, ev_mat,
                                    inter_ratio=inter_ratio)
    mat2 = symmetrize(mat2)

    G2 = nx.from_numpy_array(mat2, create_using=nx.Graph)
    par2 = community.best_partition(G2)
    deg2 = np.sum(mat2, 0)
    deg_dist2 = np.bincount(np.int64(deg2))
    evc2 = nx.eigenvector_centrality(G2, max_iter=10000)
    evc2_a = dict(sorted(evc2.items(), key=lambda x: x[1], reverse=True))
    evc_kn = np.int64(0.01 * n)
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
        'edge_ratio':  round(G2.number_of_edges() / max(ref['edges'], 1), 4),
        'comm_n':      int(comm_n),
        'mod_raw':     round(mod_reconstructed, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['CA-HepPh'])
    parser.add_argument('--eps', type=float, default=2.0)
    parser.add_argument('--reps', type=int, default=N_REPS)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    epsilon = args.eps

    for ds_name in args.datasets:
        csv_path = os.path.join(RESULT_DIR, f'sweep_ir_{ds_name}_eps{epsilon}.csv')
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}, eps={epsilon}, reps={args.reps}")
        print(f"inter_ratio list: {IR_LIST}")
        print(f"output: {csv_path}")
        print(f"{'='*60}\n")

        print(f">> Loading {ds_name}.txt ...")
        mat0, _ = get_mat(os.path.join(DATA_DIR, ds_name + '.txt'))
        n = mat0.shape[0]
        print(f"   nodes={n}, edges={int(np.sum(mat0)/2)}")
        print(f">> Pre-computing reference ...")
        ref = precompute_reference(mat0)
        print(f"   mod_orig={ref['mod']:.6f}\n")

        total = len(IR_LIST) * args.reps
        done = 0
        t_start = time.time()

        for ir in IR_LIST:
            results = []
            for exper in range(args.reps):
                done += 1
                t0 = time.time()
                try:
                    m = run_trial(mat0, n, ref, epsilon, ir)
                    results.append(m)
                    row = {'inter_ratio': ir, 'exper': exper, 'eps': epsilon, **m}
                    pd.DataFrame([row]).to_csv(
                        csv_path,
                        mode='a' if os.path.exists(csv_path) else 'w',
                        header=not os.path.exists(csv_path),
                        index=False)
                    dt = time.time() - t0
                    elapsed = time.time() - t_start
                    eta = (total - done) * elapsed / done / 60
                    print(f"  [{done}/{total}] ir={ir:.2f} rep={exper} "
                          f"mod_rel={m['mod_rel']:.3f} nmi={m['nmi']:.3f} "
                          f"edges={m['edge_ratio']:.0%} "
                          f"({dt:.0f}s, ETA≈{eta:.1f}min)")
                except Exception as ex:
                    print(f"  !! ir={ir} rep={exper} FAILED: {ex}")
                    traceback.print_exc()

            if results:
                avg = {k: np.mean([r[k] for r in results])
                       for k in ['mod_rel', 'nmi', 'deg_kl', 'cc_rel',
                                  'edge_ratio', 'mod_raw']}
                print(f"  >>> ir={ir:.2f} AVG: "
                      f"mod_rel={avg['mod_rel']:.3f} "
                      f"nmi={avg['nmi']:.3f} "
                      f"deg_kl={avg['deg_kl']:.2f} "
                      f"cc_rel={avg['cc_rel']:.3f} "
                      f"edges={avg['edge_ratio']:.0%}\n")

        # 最终汇总表
        print(f"\n{'='*60}")
        print(f"汇总: {ds_name} eps={epsilon}")
        print(f"{'='*60}")
        df = pd.read_csv(csv_path)
        agg = df.groupby('inter_ratio').agg(
            mod_rel_mean=('mod_rel', 'mean'),
            mod_rel_std =('mod_rel', 'std'),
            nmi_mean    =('nmi',     'mean'),
            deg_kl_mean =('deg_kl',  'mean'),
            cc_rel_mean =('cc_rel',  'mean'),
            edge_ratio  =('edge_ratio', 'mean'),
            n           =('exper',   'count'),
        ).round(4)
        print(agg.to_string())

        # 找最优
        best_idx = agg['mod_rel_mean'].idxmin()
        print(f"\n★ mod_rel 最低: inter_ratio={best_idx} "
              f"(mod_rel={agg.loc[best_idx, 'mod_rel_mean']:.4f})")

        best_nmi = agg['nmi_mean'].idxmax()
        print(f"★ nmi 最高: inter_ratio={best_nmi} "
              f"(nmi={agg.loc[best_nmi, 'nmi_mean']:.4f})")

        del mat0, ref

    print(f"\n>>> All done in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()