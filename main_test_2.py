import community
import networkx as nx
import time
import numpy as np
import pandas as pd
import os

from numpy.random import laplace
from sklearn import metrics

from utils import *


# ============================================================
# 三个独立的后处理变体（拆自 post_process_edge_swap）
# ============================================================

def post_process_stage1_only(mat2, pvs, comm_n, n_iter_ratio=0.3):
    """只做阶段 1：社区内孤立边的 triangle-closing 交换"""
    mat2 = mat2.copy().astype(np.int8)
    for ci in range(comm_n):
        nodes = np.array(pvs[ci])
        if len(nodes) < 4:
            continue
        sub = mat2[np.ix_(nodes, nodes)]
        rows, cols_idx = np.where(np.triu(sub, 1) > 0)
        intra_edges = list(zip(rows, cols_idx))
        if not intra_edges:
            continue
        n_iter = max(1, int(len(intra_edges) * n_iter_ratio))
        for _ in range(n_iter):
            edge_idx = np.random.randint(len(intra_edges))
            u_local, v_local = intra_edges[edge_idx]
            u_neigh = set(np.where(sub[u_local] > 0)[0])
            v_neigh = set(np.where(sub[v_local] > 0)[0])
            if u_neigh & v_neigh:
                continue
            u_non = set(range(len(nodes))) - u_neigh - {u_local}
            best_w, best_score = None, 0
            for w in u_non:
                w_neigh = set(np.where(sub[w] > 0)[0])
                score = len(u_neigh & w_neigh)
                if score > best_score:
                    best_score, best_w = score, w
            if best_w is None or best_score == 0:
                continue
            ug, vg, wg = nodes[u_local], nodes[v_local], nodes[best_w]
            sub[u_local, v_local] = sub[v_local, u_local] = 0
            sub[u_local, best_w] = sub[best_w, u_local] = 1
            mat2[ug, vg] = mat2[vg, ug] = 0
            mat2[ug, wg] = mat2[wg, ug] = 1
            intra_edges[edge_idx] = (u_local, best_w)
    return mat2


def post_process_stage2_only(mat2, pvs, comm_n):
    """只做阶段 2：跨社区低期望强度边清理"""
    mat2 = mat2.copy().astype(np.int8)
    m_total = np.sum(mat2) / 2
    if m_total == 0:
        return mat2
    degree = np.sum(mat2, axis=1)
    for ci in range(comm_n):
        for cj in range(ci + 1, comm_n):
            pi, pj = pvs[ci], pvs[cj]
            cross = [(u, v) for u in pi for v in pj if mat2[u, v] == 1]
            remove_list = [(u, v) for (u, v) in cross
                           if (degree[u] * degree[v]) / (2 * m_total) <= 1.0]
            if not remove_list:
                continue
            n_remove = max(0, len(cross) - max(1, len(cross) // 2))
            n_remove = min(n_remove, len(remove_list))
            if n_remove > 0:
                chosen = np.random.choice(len(remove_list), n_remove, replace=False)
                for idx in chosen:
                    u, v = remove_list[idx]
                    mat2[u, v] = mat2[v, u] = 0
    return mat2


def post_process_both(mat2, pvs, comm_n, n_iter_ratio=0.3):
    """完整两阶段（与原 post_process_edge_swap 等价）"""
    mat2 = post_process_stage1_only(mat2, pvs, comm_n, n_iter_ratio)
    mat2 = post_process_stage2_only(mat2, pvs, comm_n)
    return mat2


# ============================================================
# 评估单个合成图的所有指标（从原 main_func 抽出来）
# ============================================================

def evaluate_one(mat2, mat0_par, mat0_evc_ak, mat0_evc_val, evc_kn,
                 mat0_deg_dist, mat0_cc, mat0_mod, mat0_diam, mat0_node):
    mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)
    mat2_edge = mat2_graph.number_of_edges()
    mat2_node = mat2_graph.number_of_nodes()

    mat2_par = community.best_partition(mat2_graph)
    mat2_mod = community.modularity(mat2_par, mat2_graph)
    mat2_cc = nx.transitivity(mat2_graph)

    mat2_degree = np.sum(mat2, 0)
    mat2_deg_dist = np.bincount(np.int64(mat2_degree))

    mat2_evc = nx.eigenvector_centrality(mat2_graph, max_iter=10000)
    mat2_evc_a = dict(sorted(mat2_evc.items(), key=lambda x: x[1], reverse=True))
    mat2_evc_ak = list(mat2_evc_a.keys())
    mat2_evc_val = np.array(list(mat2_evc_a.values()))

    mat2_diam = cal_diam(mat2)

    cc_rel = cal_rel(mat0_cc, mat2_cc)
    deg_kl = cal_kl(mat0_deg_dist, mat2_deg_dist)
    mod_rel = cal_rel(mat0_mod, mat2_mod)

    labels_true = list(mat0_par.values())
    labels_pred = list(mat2_par.values())
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)

    evc_overlap = cal_overlap(mat0_evc_ak, mat2_evc_ak, np.int64(0.01 * mat0_node))
    evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)
    diam_rel = cal_rel(mat0_diam, mat2_diam)

    return {
        'nodes': mat2_node, 'edges': mat2_edge,
        'nmi': nmi, 'cc_rel': cc_rel, 'deg_kl': deg_kl, 'mod_rel': mod_rel,
        'evc_overlap': evc_overlap, 'evc_MAE': evc_MAE, 'diam_rel': diam_rel,
    }


# ============================================================
# 改造后的主函数：四种后处理变体并行评估
# ============================================================

def main_func_ablation(dataset_name='Chamelon',
                       eps=[0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                       e1_r=1/3, e2_r=1/3, N=20, t=1.0, exp_num=10,
                       n_iter_ratio=0.3, save_csv=False):

    t_begin = time.time()

    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    # 四种后处理变体
    variants = {
        'none':   lambda m, pvs, cn: m.copy(),
        'stage1': lambda m, pvs, cn: post_process_stage1_only(m, pvs, cn, n_iter_ratio),
        'stage2': lambda m, pvs, cn: post_process_stage2_only(m, pvs, cn),
        'both':   lambda m, pvs, cn: post_process_both(m, pvs, cn, n_iter_ratio),
    }

    cols = ['variant', 'eps', 'exper', 'edges',
            'nmi', 'evc_overlap', 'evc_MAE', 'deg_kl',
            'diam_rel', 'cc_rel', 'mod_rel']
    all_data = pd.DataFrame(None, columns=cols)

    # 原图基准
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s, Nodes:%d, Edges:%d' % (dataset_name, mat0_node, mat0_edge))

    mat0_par = community.best_partition(mat0_graph)
    mat0_degree = np.sum(mat0, 0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))

    mat0_evc = nx.eigenvector_centrality(mat0_graph, max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(), key=lambda x: x[1], reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01 * mat0_node)
    mat0_diam = cal_diam(mat0)
    mat0_cc = nx.transitivity(mat0_graph)
    mat0_mod = community.modularity(mat0_par, mat0_graph)

    # ============================================================
    # 主循环
    # ============================================================
    for ei in range(len(eps)):
        epsilon = eps[ei]
        ti = time.time()

        e1 = e1_r * epsilon
        e2 = e2_r * epsilon
        e3_r = 1 - e1_r - e2_r
        e3 = e3_r * epsilon
        ed, ev = e3, e3
        ev_lambda = 1 / ed
        dd_lam = 2 / ev

        for exper in range(exp_num):
            print('-----------epsilon=%.1f, exper=%d/%d-------------'
                  % (epsilon, exper + 1, exp_num))

            # ----- DP 流程：直到生成 mat2_base（所有变体共用） -----
            mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)
            part1 = {i: mat1_pvarr1[i] for i in range(len(mat1_pvarr1))}
            mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
            mat1_pvarr = np.array(list(mat1_par1.values()))

            mat1_pvs = []
            for i in range(max(mat1_pvarr) + 1):
                pv1 = np.where(mat1_pvarr == i)[0]
                mat1_pvs.append(list(pv1))
            comm_n = max(mat1_pvarr) + 1

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
            ga_noise_pp = FO_pp(ga_noise)
            ev_mat = get_upmat(ga_noise_pp, comm_n, ind=1)

            dd_s = []
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
                dd1 = np.sum(dd1, 1)
                dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1 < 0] = 0
                dd1[dd1 >= len(dd1)] = len(dd1) - 1
                dd_s.append(list(dd1))

            # 阶段 0 + 1 + 2：CL 重建 + 跨社区缩减
            mat2_base = step6_v6_cl_compensated(
                mat0_node, comm_n, mat1_pvs, dd_s, ev_mat, inter_ratio=0.1)

            # 对称化
            mat2_base = mat2_base + np.transpose(mat2_base)
            mat2_base = np.triu(mat2_base, 1)
            mat2_base = mat2_base + np.transpose(mat2_base)
            mat2_base[mat2_base > 0] = 1

            # ----- 对每个变体单独评估 -----
            for vname, vfn in variants.items():
                mat2 = vfn(mat2_base, mat1_pvs, comm_n)
                m = evaluate_one(mat2, mat0_par, mat0_evc_ak, mat0_evc_val,
                                 evc_kn, mat0_deg_dist, mat0_cc, mat0_mod,
                                 mat0_diam, mat0_node)

                row = [vname, epsilon, exper, m['edges'],
                       m['nmi'], m['evc_overlap'], m['evc_MAE'], m['deg_kl'],
                       m['diam_rel'], m['cc_rel'], m['mod_rel']]
                all_data = pd.concat(
                    [all_data, pd.DataFrame([row], columns=cols)],
                    ignore_index=True)

                print('  [%-6s] edges=%d, nmi=%.4f, mod_rel=%.4f, '
                      'cc_rel=%.4f, deg_kl=%.4f'
                      % (vname, m['edges'], m['nmi'], m['mod_rel'],
                         m['cc_rel'], m['deg_kl']))

        print('all_index=%d/%d Done. %.2fs\n' % (ei + 1, len(eps), time.time() - ti))

    # ============================================================
    # 汇总打印
    # ============================================================
    print('\n' + '=' * 90)
    print('Ablation Summary: %s' % dataset_name)
    print('=' * 90)

    metrics_cols = ['nmi', 'mod_rel', 'cc_rel', 'deg_kl',
                    'evc_overlap', 'evc_MAE', 'diam_rel']

    for ep in eps:
        print('\n>>> epsilon = %.1f' % ep)
        print('-' * 90)
        header = '%-8s' % 'variant'
        for mc in metrics_cols:
            header += '%14s' % mc
        print(header)
        print('-' * 90)

        for vname in variants:
            sub = all_data[(all_data['variant'] == vname) &
                           (np.isclose(all_data['eps'], ep))]
            line = '%-8s' % vname
            for mc in metrics_cols:
                vals = sub[mc].astype(float).values
                line += '%14s' % ('%.3f±%.3f' % (vals.mean(), vals.std()))
            print(line)

        # 相对 none 的 Δ
        print('  (Δ vs none, 越好的方向: NMI↑ Mod_Rel↑ CC_Rel→1 Deg_KL↓)')
        none_means = {mc: all_data[(all_data['variant'] == 'none') &
                                   (np.isclose(all_data['eps'], ep))][mc]
                                   .astype(float).mean()
                      for mc in metrics_cols}
        for vname in ['stage1', 'stage2', 'both']:
            sub = all_data[(all_data['variant'] == vname) &
                           (np.isclose(all_data['eps'], ep))]
            line = '  Δ%-6s' % vname
            for mc in metrics_cols:
                delta = sub[mc].astype(float).mean() - none_means[mc]
                line += '%14s' % ('%+.4f' % delta)
            print(line)

    # ----- 保存 CSV -----
    res_path = './result'
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    if save_csv:
        save_name = (res_path + '/' + 'ablation_%s_%d_%.1f_%.2f_%.2f_%d.csv'
                     % (dataset_name, N, t, e1_r, e2_r, exp_num))
        all_data.to_csv(save_name, index=False, sep=',')
        print('\nSaved to: %s' % save_name)

    print('\nAll time: %.2fs' % (time.time() - t_begin))
    return all_data


if __name__ == '__main__':
    dataset_name = 'Facebook'
    eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    e1_r = 1/3
    e2_r = 1/3
    exp_num = 10
    n1 = 20
    t = 1.0

    all_data = main_func_ablation(
        dataset_name=dataset_name, eps=eps,
        e1_r=e1_r, e2_r=e2_r, N=n1, t=t,
        exp_num=exp_num, save_csv=True)