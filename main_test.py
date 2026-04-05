"""
main_test.py — 测试平衡公式分配
================================
直接运行: python main_test.py

对比三种分配策略:
  1. normal:       e1=e2=e3=1/3
  2. aggressive:   e3=0.10 固定 (之前跑的)
  3. balanced:     e3=max(0.10, 0.15/eps) (新公式)
"""

import community
import community as comm
import networkx as nx
import time
import numpy as np
import pandas as pd
from numpy.random import laplace
from sklearn import metrics
from utils import *
import os


def main_test(dataset_name='Chamelon',
              eps=[0.5, 1, 1.5, 2, 2.5, 3, 3.5],
              N=20, t=1.0, exp_num=10,
              mode='balanced',        # 'normal' / 'aggressive' / 'balanced' / 'sweep'
              e3_sweep_vals=None,     # sweep模式下的e3列表
              save_csv=True):
    """
    mode:
      'normal'     → e1=e2=e3=1/3 (论文原始设置)
      'aggressive' → e3=0.10 固定, e1/e2用公式A
      'balanced'   → e3=max(0.10, 0.15/eps), e1用公式A, e2补齐
      'sweep'      → 遍历 e3_sweep_vals 中的每个e3值
    """

    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    # ===== 原始图 =====
    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    n = mat0_node

    print('=' * 60)
    print(f'Dataset: {dataset_name} | Nodes: {mat0_node} | Edges: {mat0_edge}')
    print(f'Mode: {mode}')
    print('=' * 60)

    # ===== 原始图统计 =====
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

    # ===== 确定要跑的 e3 列表 =====
    if mode == 'sweep':
        if e3_sweep_vals is None:
            e3_sweep_vals = [0.05, 0.10, 0.15, 0.20, 0.30]
    else:
        e3_sweep_vals = [None]  # 占位, 只跑一次

    # ===== 结果收集 =====
    cols = ['mode', 'e3_target', 'eps', 'e1_r', 'e2_r', 'e3_r',
            'exper', 'nmi', 'evc_overlap', 'evc_MAE', 'deg_kl',
            'diam_rel', 'cc_rel', 'mod_rel']
    all_data = pd.DataFrame(columns=cols)

    for e3_target in e3_sweep_vals:
        for ei in range(len(eps)):
            epsilon = eps[ei]
            ti = time.time()

            # ===== 计算分配比例 =====
            if mode == 'normal':
                _e1_r, _e2_r, _e3_r = 1/3, 1/3, 1/3
                mode_label = 'normal'

            elif mode == 'aggressive':
                _e1_r = 0.368 + 0.363 / epsilon - 0.033 * np.log(n)
                _e3_r = 0.10
                _e2_r = 1.0 - _e1_r - _e3_r
                mode_label = 'aggressive'

            elif mode == 'balanced':
                _e1_r = 0.368 + 0.363 / epsilon - 0.033 * np.log(n)
                _e3_r = max(0.10, 0.15 / epsilon)
                _e2_r = 1.0 - _e1_r - _e3_r
                mode_label = 'balanced'

            elif mode == 'sweep':
                _e1_r = 0.368 + 0.363 / epsilon - 0.033 * np.log(n)
                _e3_r = e3_target
                _e2_r = 1.0 - _e1_r - _e3_r
                mode_label = f'sweep_e3={e3_target:.2f}'

            # clip + 归一化
            _e1_r = np.clip(_e1_r, 0.05, 0.85)
            _e2_r = np.clip(_e2_r, 0.05, 0.85)
            _e3_r = np.clip(_e3_r, 0.05, 0.85)
            _total = _e1_r + _e2_r + _e3_r
            _e1_r, _e2_r, _e3_r = _e1_r/_total, _e2_r/_total, _e3_r/_total

            e1 = _e1_r * epsilon
            e2 = _e2_r * epsilon
            e3 = _e3_r * epsilon

            ed = e3
            ev = e3
            ev_lambda = 1 / ed
            dd_lam = 2 / ev

            print(f'\n[{mode_label}] eps={epsilon}, '
                  f'e1={_e1_r:.3f}({e1:.3f}), '
                  f'e2={_e2_r:.3f}({e2:.3f}), '
                  f'e3={_e3_r:.3f}({e3:.3f})')

            # ===== 多次实验 =====
            for exper in range(exp_num):
                print(f'  exper {exper+1}/{exp_num}', end=' ')

                # Step1: 社区初始化
                mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)

                # Step2: 社区调整
                part1 = {}
                for i in range(len(mat1_pvarr1)):
                    part1[i] = mat1_pvarr1[i]
                mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
                mat1_pvarr = np.array(list(mat1_par1.values()))

                # Step3: 社区节点提取
                mat1_pvs = []
                for i in range(max(mat1_pvarr) + 1):
                    pv1 = np.where(mat1_pvarr == i)[0]
                    mat1_pvs.append(list(pv1))

                comm_n = max(mat1_pvarr) + 1

                # Step4: 社区间边统计
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

                # Step5: 度序列加噪
                dd_s = []
                for i in range(comm_n):
                    dd1 = mat0[np.ix_(mat1_pvs[i], mat1_pvs[i])]
                    dd1 = np.sum(dd1, 1)
                    dd1 = (dd1 + laplace(0, dd_lam, len(dd1))).astype(int)
                    dd1 = FO_pp(dd1)
                    dd1[dd1 < 0] = 0
                    dd1[dd1 >= len(dd1)] = len(dd1) - 1
                    dd_s.append(list(dd1))

                # Step6: 图重建
                mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
                for i in range(comm_n):
                    pi = mat1_pvs[i]
                    dd_ind = mat1_pvs[i]
                    dd1 = dd_s[i]
                    mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

                    for j in range(i + 1, comm_n):
                        ev1 = ev_mat[i, j]
                        pj = mat1_pvs[j]
                        if ev1 > 0:
                            c1 = np.random.choice(pi, ev1)
                            c2 = np.random.choice(pj, ev1)
                            for ind in range(ev1):
                                mat2[c1[ind], c2[ind]] = 1
                                mat2[c2[ind], c1[ind]] = 1

                mat2 = mat2 + np.transpose(mat2)
                mat2 = np.triu(mat2, 1)
                mat2 = mat2 + np.transpose(mat2)
                mat2[mat2 > 0] = 1

                mat2_graph = nx.from_numpy_array(mat2, create_using=nx.Graph)

                # Step7: 计算指标
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
                evc_overlap = cal_overlap(mat0_evc_ak, mat2_evc_ak, evc_kn)
                evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)
                diam_rel = cal_rel(mat0_diam, mat2_diam)

                print(f'nmi={nmi:.4f} deg_kl={deg_kl:.2f} mod={mod_rel:.4f} '
                      f'evc_o={evc_overlap:.3f} cc={cc_rel:.4f}')

                # 保存
                row = pd.DataFrame([{
                    'mode': mode_label,
                    'e3_target': e3_target if e3_target else _e3_r,
                    'eps': epsilon,
                    'e1_r': round(_e1_r, 4),
                    'e2_r': round(_e2_r, 4),
                    'e3_r': round(_e3_r, 4),
                    'exper': exper,
                    'nmi': nmi,
                    'evc_overlap': evc_overlap,
                    'evc_MAE': evc_MAE,
                    'deg_kl': deg_kl,
                    'diam_rel': diam_rel,
                    'cc_rel': cc_rel,
                    'mod_rel': mod_rel,
                }])
                all_data = pd.concat([all_data, row], ignore_index=True)

            print(f'  eps={epsilon} done. {time.time()-ti:.1f}s')

    # ===== 保存结果 =====
    res_path = './result'
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if save_csv:
        if mode == 'sweep':
            save_name = f'{res_path}/{dataset_name}_sweep_e3.csv'
        else:
            save_name = f'{res_path}/{dataset_name}_{mode}.csv'
        all_data.to_csv(save_name, index=False)
        print(f'\n结果已保存: {save_name}')

    # ===== 打印汇总 =====
    print('\n' + '=' * 60)
    print('汇总 (各eps下的均值)')
    print('=' * 60)
    summary = all_data.groupby(['mode', 'eps'])[
        ['nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']
    ].mean()
    print(summary.round(4).to_string())

    print(f'\n总耗时: {time.time()-t_begin:.1f}s')
    return all_data


if __name__ == '__main__':

    dataset_name = 'Chamelon'
    eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    exp_num = 10

    # ============================================================
    # 选择要跑的实验 (取消注释对应行)
    # ============================================================

    # --- 1. 先跑平衡版, 和你之前的结果对比 ---
    # main_test(dataset_name=dataset_name, eps=eps, exp_num=exp_num,
    #           mode='balanced')

    # --- 2. 跑 normal 基线 (如果之前没存的话) ---
    # main_test(dataset_name=dataset_name, eps=eps, exp_num=exp_num,
    #           mode='normal')

    # --- 3. 跑精简版 e3 扫描 ---
    main_test(dataset_name=dataset_name,
              eps=[1, 2, 3],          # 只跑3个关键eps
              exp_num=exp_num,
              mode='sweep',
              e3_sweep_vals=[0.03, 0.05, 0.10, 0.15, 0.20, 0.30])

    # --- 4. 跑全量 e3 扫描 ---
    # main_test(dataset_name=dataset_name,
    #           eps=eps,
    #           exp_num=exp_num,
    #           mode='sweep',
    #           e3_sweep_vals=[0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30])