"""
修正版 Step 6 对比实验

修正点:
  1. v2_full_fixed: inter_budget 只分配 10% 的社区间边 (对应 cr=0.9)
     其余 90% 社区间边转为社区内边 (通过增大 intra_budget)
  2. v2_simple_soft: 度修复改为软修复, 只修偏差超过阈值的节点
  3. 保留 original 作为 baseline

用法: python run_edge_rebuild_v3.py
"""

import community
import networkx as nx
import time
import numpy as np
import pandas as pd

from numpy.random import laplace
from sklearn import metrics

from utils import *

import os


def step6_original(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat):
    """原始方法 (cr=0.9, os=1.2)"""
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    convert_ratio = 0.9
    oversample = 1.2

    for i in range(comm_n):
        dd_ind = mat1_pvs[i]
        dd1 = dd_s[i]
        mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

        dd_i = np.maximum(np.array(dd_s[i], dtype=np.float64), 1.0)
        prob_i = dd_i / dd_i.sum()

        for j in range(i+1, comm_n):
            ev1 = ev_mat[i, j]
            if ev1 <= 0:
                continue
            pi = mat1_pvs[i]
            pj = mat1_pvs[j]
            dd_j = np.maximum(np.array(dd_s[j], dtype=np.float64), 1.0)
            prob_j = dd_j / dd_j.sum()

            n_convert = int(ev1 * convert_ratio)
            n_inter = ev1 - n_convert
            n_ri = n_convert // 2
            n_rj = n_convert - n_ri

            if n_ri > 0:
                max_ei = len(pi) * (len(pi)-1) // 2
                sample_ri = min(int(n_ri * oversample), max_ei)
                c1 = np.random.choice(pi, sample_ri, p=prob_i)
                c2 = np.random.choice(pi, sample_ri, p=prob_i)
                for ind in range(sample_ri):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1

            if n_rj > 0:
                max_ej = len(pj) * (len(pj)-1) // 2
                sample_rj = min(int(n_rj * oversample), max_ej)
                c1 = np.random.choice(pj, sample_rj, p=prob_j)
                c2 = np.random.choice(pj, sample_rj, p=prob_j)
                for ind in range(sample_rj):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1

            if n_inter > 0:
                c1 = np.random.choice(pi, n_inter, p=prob_i)
                c2 = np.random.choice(pj, n_inter, p=prob_j)
                for ind in range(n_inter):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1

    return mat2

def step6_v2_full_fixed(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat):
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    convert_ratio = 0.85

    node_community = np.zeros(mat0_node, dtype=int)
    for i in range(comm_n):
        for li, ni in enumerate(mat1_pvs[i]):
            node_community[ni] = i

    # ---- 预计算每个社区的 convert_alloc 和 inter_alloc ----
    comm_convert_alloc = {}   # comm_id -> array
    comm_inter_alloc = {}     # comm_id -> array
    node_target_degree = np.zeros(mat0_node, dtype=int)

    for i in range(comm_n):
        nodes = mat1_pvs[i]
        n_nodes = len(nodes)
        if n_nodes == 0:
            comm_convert_alloc[i] = np.zeros(0, dtype=int)
            comm_inter_alloc[i] = np.zeros(0, dtype=int)
            continue

        dd_i = np.array(dd_s[i], dtype=np.float64)
        dd_i_safe = np.maximum(dd_i, 1.0)
        prob = dd_i_safe / dd_i_safe.sum()

        # 社区间边总数 (单侧)
        total_inter_i = sum(ev_mat[i, j] for j in range(comm_n) if j != i)

        # 修复1: 只拿一半的转化量 (和 original 的 n_ri = n_convert//2 一致)
        convert_total = int(total_inter_i * convert_ratio) // 2

        if convert_total > 0 and np.sum(dd_i) > 0:
            convert_alloc = np.round(prob * convert_total).astype(int)
            diff = convert_total - np.sum(convert_alloc)
            sorted_idx = np.argsort(-dd_i)
            for k in range(abs(int(diff))):
                convert_alloc[sorted_idx[k % n_nodes]] += int(np.sign(diff))
        else:
            convert_alloc = np.zeros(n_nodes, dtype=int)
        comm_convert_alloc[i] = convert_alloc

        # 社区间真正保留的 10%
        total_inter_10 = sum(ev_mat[i, j] * (1 - convert_ratio) for j in range(comm_n) if j != i)

        if total_inter_10 > 0 and np.sum(dd_i) > 0:
            inter_alloc = np.round(prob * total_inter_10).astype(int)
            inter_alloc = np.minimum(inter_alloc, dd_i.astype(int))
            # 修复4: 兜底至少 1, 去掉硬上限 0.3
            # (低度节点不至于被压到 0)
        else:
            inter_alloc = np.zeros(n_nodes, dtype=int)
        comm_inter_alloc[i] = inter_alloc

        # 修复2: target = 社区内度 + 转化量 + 社区间量
        for li, ni in enumerate(nodes):
            node_target_degree[ni] = int(dd_i[li]) + int(convert_alloc[li]) + int(inter_alloc[li])

    # ---- 6b: 社区内边 ----
    for i in range(comm_n):
        nodes = mat1_pvs[i]
        if len(nodes) == 0:
            continue
        dd_i = np.array(dd_s[i], dtype=np.float64)
        convert_alloc = comm_convert_alloc[i]
        n_nodes = len(nodes)

        intra_degs = [max(0, min(int(dd_i[li]) + int(convert_alloc[li]), n_nodes - 1))
                      for li in range(n_nodes)]

        mat2[np.ix_(nodes, nodes)] = generate_intra_edge(intra_degs)

    # ---- 6c: 社区间边 (10%, 无碰撞) ----
    node_inter_budget = np.zeros(mat0_node, dtype=int)
    for i in range(comm_n):
        for li, ni in enumerate(mat1_pvs[i]):
            node_inter_budget[ni] = comm_inter_alloc[i][li]

    for i in range(comm_n):
        for j in range(i + 1, comm_n):
            target = int(ev_mat[i, j] * (1 - convert_ratio))
            if target <= 0:
                continue

            pi = np.array(mat1_pvs[i])
            pj = np.array(mat1_pvs[j])
            if len(pi) == 0 or len(pj) == 0:
                continue

            avail_i = np.array([max(0.1, node_inter_budget[n]) for n in pi])
            avail_j = np.array([max(0.1, node_inter_budget[n]) for n in pj])
            prob_i = avail_i / avail_i.sum()
            prob_j = avail_j / avail_j.sum()

            n_sample = min(int(target * 1.3) + 5, len(pi) * len(pj))
            c1 = np.random.choice(len(pi), n_sample, p=prob_i)
            c2 = np.random.choice(len(pj), n_sample, p=prob_j)

            added = 0
            seen = set()
            for k in range(n_sample):
                if added >= target:
                    break
                edge = (c1[k], c2[k])
                if edge not in seen:
                    seen.add(edge)
                    ni, nj = pi[c1[k]], pj[c2[k]]
                    if mat2[ni, nj] == 0:
                        mat2[ni, nj] = 1
                        mat2[nj, ni] = 1
                        node_inter_budget[ni] -= 1
                        node_inter_budget[nj] -= 1
                        added += 1

    # ---- 6d: 软度修复 (修复3: 阈值收紧到 20%) ----
    actual_degree = np.sum(mat2, axis=0).astype(int)

    # 削超度
    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        actual = actual_degree[n]
        excess = actual - target
        threshold = max(int(target * 0.2), 1)   # 收紧
        if excess <= threshold:
            continue

        to_trim = excess - threshold
        neighbors = np.where(mat2[n] > 0)[0]
        inter_nb = [nb for nb in neighbors if node_community[nb] != node_community[n]]
        intra_nb = [nb for nb in neighbors if node_community[nb] == node_community[n]]
        to_remove = []
        if len(inter_nb) > 0:
            np.random.shuffle(inter_nb)
            to_remove.extend(inter_nb[:min(to_trim, len(inter_nb))])
        remaining = to_trim - len(to_remove)
        if remaining > 0 and len(intra_nb) > 0:
            np.random.shuffle(intra_nb)
            to_remove.extend(intra_nb[:min(remaining, len(intra_nb))])
        for nb in to_remove:
            mat2[n, nb] = 0
            mat2[nb, n] = 0
            actual_degree[n] -= 1
            actual_degree[nb] -= 1

    # 补欠度
    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        actual = actual_degree[n]
        deficit = target - actual
        threshold = max(int(target * 0.2), 1)
        if deficit <= threshold:
            continue

        to_add = deficit - threshold
        comm_id = node_community[n]
        candidates = [c for c in mat1_pvs[comm_id] if c != n and mat2[n, c] == 0]
        if len(candidates) == 0:
            continue
        weights = np.array([max(0.1, node_target_degree[c] - actual_degree[c]) for c in candidates])
        weights = weights / weights.sum()
        n_add = min(to_add, len(candidates))
        chosen = np.random.choice(len(candidates), n_add, replace=False, p=weights)
        for idx in chosen:
            c = candidates[idx]
            mat2[n, c] = 1
            mat2[c, n] = 1
            actual_degree[n] += 1
            actual_degree[c] += 1

    return mat2

def step6_v3_full_fixed(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat):
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    convert_ratio = 0.4

    node_community = np.zeros(mat0_node, dtype=int)
    node_target_degree = np.zeros(mat0_node, dtype=int)
    for i in range(comm_n):
        for li, ni in enumerate(mat1_pvs[i]):
            node_community[ni] = i
            node_target_degree[ni] = dd_s[i][li]

    # ---- 阶段1: generate_intra_edge 只用 dd_s (不加 convert) ----
    for i in range(comm_n):
        nodes = mat1_pvs[i]
        if len(nodes) == 0:
            continue
        dd1 = dd_s[i]
        mat2[np.ix_(nodes, nodes)] = generate_intra_edge(dd1)

    # ---- 阶段2: convert 边 —— 无碰撞地往社区内加边 ----
    # 和 Original 一样按 (i,j) 对处理, 但去重
    for i in range(comm_n):
        dd_i = np.maximum(np.array(dd_s[i], dtype=np.float64), 1.0)
        prob_i = dd_i / dd_i.sum()


        for j in range(i + 1, comm_n):
            ev1 = ev_mat[i, j]
            if ev1 <= 0:
                continue

            pi = np.array(mat1_pvs[i])
            pj = np.array(mat1_pvs[j])
            dd_j = np.maximum(np.array(dd_s[j], dtype=np.float64), 1.0)
            prob_j = dd_j / dd_j.sum()

            n_convert = int(ev1 * convert_ratio)
            n_inter = ev1 - n_convert
            n_ri = n_convert // 2
            n_rj = n_convert - n_ri

            # 社区 i 内部加 n_ri 条边 (无碰撞)
            if n_ri > 0 and len(pi) > 1:
                max_ei = len(pi) * (len(pi) - 1) // 2
                actual_target = min(n_ri, max_ei)
                n_sample = min(int(actual_target * 1.5) + 10, max_ei * 2)
                c1_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                c2_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                added = 0
                for k in range(n_sample):
                    if added >= actual_target:
                        break
                    a, b = pi[c1_idx[k]], pi[c2_idx[k]]
                    if a != b and mat2[a, b] == 0:
                        mat2[a, b] = 1
                        mat2[b, a] = 1
                        added += 1

            # 社区 j 内部加 n_rj 条边 (无碰撞)
            if n_rj > 0 and len(pj) > 1:
                max_ej = len(pj) * (len(pj) - 1) // 2
                actual_target = min(n_rj, max_ej)
                n_sample = min(int(actual_target * 1.5) + 10, max_ej * 2)
                c1_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                c2_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                added = 0
                for k in range(n_sample):
                    if added >= actual_target:
                        break
                    a, b = pj[c1_idx[k]], pj[c2_idx[k]]
                    if a != b and mat2[a, b] == 0:
                        mat2[a, b] = 1
                        mat2[b, a] = 1
                        added += 1

            # 社区间边 (无碰撞)
            if n_inter > 0:
                n_sample = min(int(n_inter * 1.3) + 5, len(pi) * len(pj))
                c1_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                c2_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                added = 0
                seen = set()
                for k in range(n_sample):
                    if added >= n_inter:
                        break
                    edge = (c1_idx[k], c2_idx[k])
                    if edge not in seen:
                        seen.add(edge)
                        ni, nj = pi[c1_idx[k]], pj[c2_idx[k]]
                        if mat2[ni, nj] == 0:
                            mat2[ni, nj] = 1
                            mat2[nj, ni] = 1
                            added += 1

    # ---- 阶段3: 软度修复 (阈值 30%) ----
    actual_degree = np.sum(mat2, axis=0).astype(int)

    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        excess = actual_degree[n] - target
        threshold = max(int(target * 0.3), 1)
        if excess <= threshold:
            continue

        to_trim = excess - threshold
        neighbors = np.where(mat2[n] > 0)[0]
        inter_nb = [nb for nb in neighbors if node_community[nb] != node_community[n]]
        intra_nb = [nb for nb in neighbors if node_community[nb] == node_community[n]]
        to_remove = []
        if len(inter_nb) > 0:
            np.random.shuffle(inter_nb)
            to_remove.extend(inter_nb[:min(to_trim, len(inter_nb))])
        remaining = to_trim - len(to_remove)
        if remaining > 0 and len(intra_nb) > 0:
            np.random.shuffle(intra_nb)
            to_remove.extend(intra_nb[:min(remaining, len(intra_nb))])
        for nb in to_remove:
            mat2[n, nb] = 0
            mat2[nb, n] = 0
            actual_degree[n] -= 1
            actual_degree[nb] -= 1

    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        deficit = target - actual_degree[n]
        threshold = max(int(target * 0.3), 1)
        if deficit <= threshold:
            continue

        to_add = deficit - threshold
        comm_id = node_community[n]
        candidates = [c for c in mat1_pvs[comm_id] if c != n and mat2[n, c] == 0]
        if len(candidates) == 0:
            continue
        weights = np.array([max(0.1, node_target_degree[c] - actual_degree[c]) for c in candidates])
        weights = weights / weights.sum()
        n_add = min(to_add, len(candidates))
        chosen = np.random.choice(len(candidates), n_add, replace=False, p=weights)
        for idx in chosen:
            c = candidates[idx]
            mat2[n, c] = 1
            mat2[c, n] = 1
            actual_degree[n] += 1
            actual_degree[c] += 1

    return mat2

def step6_v4_full_fixed(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat):
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    convert_ratio = 0.9

    node_community = np.zeros(mat0_node, dtype=int)
    node_target_degree = np.zeros(mat0_node, dtype=int)
    for i in range(comm_n):
        for li, ni in enumerate(mat1_pvs[i]):
            node_community[ni] = i
            node_target_degree[ni] = dd_s[i][li]

    # ---- 阶段1: generate_intra_edge 只用 dd_s (不加 convert) ----
    for i in range(comm_n):
        nodes = mat1_pvs[i]
        if len(nodes) == 0:
            continue
        dd1 = dd_s[i]
        mat2[np.ix_(nodes, nodes)] = generate_intra_edge(dd1)

    # ---- 阶段2: convert 边 —— 无碰撞地往社区内加边 ----
    # 和 Original 一样按 (i,j) 对处理, 但去重
    for i in range(comm_n):
        # dd_i = np.maximum(np.array(dd_s[i], dtype=np.float64), 1.0)
        # prob_i = dd_i / dd_i.sum()

        # 改成残差权重 (只改这里):
        actual_deg_i = np.array([np.sum(mat2[n]) for n in pi], dtype=np.float64)
        target_deg_i = np.array([dd_s[i][li] for li in range(len(pi))], dtype=np.float64)
        residual_i = np.maximum(target_deg_i - actual_deg_i, 0.1)
        prob_i = residual_i / residual_i.sum()


        for j in range(i + 1, comm_n):
            ev1 = ev_mat[i, j]
            if ev1 <= 0:
                continue

            pi = np.array(mat1_pvs[i])
            pj = np.array(mat1_pvs[j])
            dd_j = np.maximum(np.array(dd_s[j], dtype=np.float64), 1.0)
            prob_j = dd_j / dd_j.sum()

            n_convert = int(ev1 * convert_ratio)
            n_inter = ev1 - n_convert
            n_ri = n_convert // 2
            n_rj = n_convert - n_ri

            # 社区 i 内部加 n_ri 条边 (无碰撞)
            if n_ri > 0 and len(pi) > 1:
                max_ei = len(pi) * (len(pi) - 1) // 2
                actual_target = min(n_ri, max_ei)
                n_sample = min(int(actual_target * 1.5) + 10, max_ei * 2)
                c1_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                c2_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                added = 0
                for k in range(n_sample):
                    if added >= actual_target:
                        break
                    a, b = pi[c1_idx[k]], pi[c2_idx[k]]
                    if a != b and mat2[a, b] == 0:
                        mat2[a, b] = 1
                        mat2[b, a] = 1
                        added += 1

            # 社区 j 内部加 n_rj 条边 (无碰撞)
            if n_rj > 0 and len(pj) > 1:
                max_ej = len(pj) * (len(pj) - 1) // 2
                actual_target = min(n_rj, max_ej)
                n_sample = min(int(actual_target * 1.5) + 10, max_ej * 2)
                c1_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                c2_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                added = 0
                for k in range(n_sample):
                    if added >= actual_target:
                        break
                    a, b = pj[c1_idx[k]], pj[c2_idx[k]]
                    if a != b and mat2[a, b] == 0:
                        mat2[a, b] = 1
                        mat2[b, a] = 1
                        added += 1

            # 社区间边 (无碰撞)
            if n_inter > 0:
                n_sample = min(int(n_inter * 1.3) + 5, len(pi) * len(pj))
                c1_idx = np.random.choice(len(pi), n_sample, p=prob_i)
                c2_idx = np.random.choice(len(pj), n_sample, p=prob_j)
                added = 0
                seen = set()
                for k in range(n_sample):
                    if added >= n_inter:
                        break
                    edge = (c1_idx[k], c2_idx[k])
                    if edge not in seen:
                        seen.add(edge)
                        ni, nj = pi[c1_idx[k]], pj[c2_idx[k]]
                        if mat2[ni, nj] == 0:
                            mat2[ni, nj] = 1
                            mat2[nj, ni] = 1
                            added += 1

    # ---- 阶段3: 软度修复 (阈值 30%) ----
    actual_degree = np.sum(mat2, axis=0).astype(int)

    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        excess = actual_degree[n] - target
        threshold = max(int(target * 0.3), 1)
        if excess <= threshold:
            continue

        to_trim = excess - threshold
        neighbors = np.where(mat2[n] > 0)[0]
        inter_nb = [nb for nb in neighbors if node_community[nb] != node_community[n]]
        intra_nb = [nb for nb in neighbors if node_community[nb] == node_community[n]]
        to_remove = []
        if len(inter_nb) > 0:
            np.random.shuffle(inter_nb)
            to_remove.extend(inter_nb[:min(to_trim, len(inter_nb))])
        remaining = to_trim - len(to_remove)
        if remaining > 0 and len(intra_nb) > 0:
            np.random.shuffle(intra_nb)
            to_remove.extend(intra_nb[:min(remaining, len(intra_nb))])
        for nb in to_remove:
            mat2[n, nb] = 0
            mat2[nb, n] = 0
            actual_degree[n] -= 1
            actual_degree[nb] -= 1

    for n in range(mat0_node):
        target = max(node_target_degree[n], 1)
        deficit = target - actual_degree[n]
        threshold = max(int(target * 0.3), 1)
        if deficit <= threshold:
            continue

        to_add = deficit - threshold
        comm_id = node_community[n]
        candidates = [c for c in mat1_pvs[comm_id] if c != n and mat2[n, c] == 0]
        if len(candidates) == 0:
            continue
        weights = np.array([max(0.1, node_target_degree[c] - actual_degree[c]) for c in candidates])
        weights = weights / weights.sum()
        n_add = min(to_add, len(candidates))
        chosen = np.random.choice(len(candidates), n_add, replace=False, p=weights)
        for idx in chosen:
            c = candidates[idx]
            mat2[n, c] = 1
            mat2[c, n] = 1
            actual_degree[n] += 1
            actual_degree[c] += 1

    return mat2



# ===== 方法注册 =====
METHODS = {
    # 'original':       step6_original,
    'v2_full_fixed':  step6_v3_full_fixed,
}


def main_func(dataset_name='Chamelon', eps=[0.5,1,1.5,2,2.5,3,3.5],
              e1_r=1/3, e2_r=1/3, N=20, t=1.0, exp_num=10,
              save_csv=False, method='original'):

    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps','exper','nmi','evc_overlap','evc_MAE','deg_kl',
            'diam_rel','cc_rel','mod_rel']
    all_data = pd.DataFrame(None, columns=cols)

    mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)
    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s  method=%s' % (dataset_name, method))
    print('Node number:%d' % mat0_node)
    print('Edge number:%d' % mat0_edge)

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

    all_deg_kl, all_mod_rel, all_nmi_arr = [], [], []
    all_evc_overlap, all_evc_MAE, all_cc_rel, all_diam_rel = [], [], [], []

    _e1_r, _e2_r, _e3_r = e1_r, e2_r, 1-e1_r-e2_r

    for ei in range(len(eps)):
        epsilon = eps[ei]
        ti = time.time()

        e1 = _e1_r * epsilon
        e2 = _e2_r * epsilon
        e3 = _e3_r * epsilon
        ev_lambda = 1/e3
        dd_lam = 2/e3

        nmi_arr = np.zeros(exp_num)
        deg_kl_arr = np.zeros(exp_num)
        mod_rel_arr = np.zeros(exp_num)
        cc_rel_arr = np.zeros(exp_num)
        diam_rel_arr = np.zeros(exp_num)
        evc_overlap_arr = np.zeros(exp_num)
        evc_MAE_arr = np.zeros(exp_num)

        for exper in range(exp_num):
            print('-----------[%s][%s] eps=%.1f, exper=%d/%d-------------'
                  % (dataset_name, method, epsilon, exper+1, exp_num))

            mat1_pvarr1 = community_init(mat0, mat0_graph, epsilon=e1, nr=N, t=t)
            part1 = {i: mat1_pvarr1[i] for i in range(len(mat1_pvarr1))}
            mat1_par1 = comm.best_partition(mat0_graph, part1, epsilon_EM=e2)
            mat1_pvarr = np.array(list(mat1_par1.values()))

            mat1_pvs = []
            for i in range(max(mat1_pvarr)+1):
                mat1_pvs.append(list(np.where(mat1_pvarr == i)[0]))
            comm_n = max(mat1_pvarr) + 1

            ev_mat = np.zeros([comm_n, comm_n], dtype=np.int64)
            for i in range(comm_n):
                pi = mat1_pvs[i]
                ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
                for j in range(i+1, comm_n):
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

            # Step6
            step6_func = METHODS[method]
            mat2 = step6_func(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat)

            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2, 1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2 > 0] = 1

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
            nmi = metrics.normalized_mutual_info_score(list(mat0_par.values()), list(mat2_par.values()))
            evc_overlap = cal_overlap(mat0_evc_ak, mat2_evc_ak, np.int64(0.01*mat0_node))
            evc_MAE = cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)
            diam_rel = cal_rel(mat0_diam, mat2_diam)

            nmi_arr[exper] = nmi
            cc_rel_arr[exper] = cc_rel
            deg_kl_arr[exper] = deg_kl
            mod_rel_arr[exper] = mod_rel
            evc_overlap_arr[exper] = evc_overlap
            evc_MAE_arr[exper] = evc_MAE
            diam_rel_arr[exper] = diam_rel

            print('N=%d,E=%d,nmi=%.4f,cc=%.4f,deg=%.4f,mod=%.4f,evc_o=%.4f,evc_m=%.4f,diam=%.4f'
                  % (mat2_node, mat2_edge, nmi, cc_rel, deg_kl, mod_rel, evc_overlap, evc_MAE, diam_rel))

            data_col = np.array([epsilon, exper, nmi, evc_overlap, evc_MAE, deg_kl,
                                 diam_rel, cc_rel, mod_rel]).reshape(1, -1)
            all_data = pd.concat([all_data, pd.DataFrame(data_col, columns=cols)], ignore_index=True)

        all_nmi_arr.append(np.mean(nmi_arr))
        all_cc_rel.append(np.mean(cc_rel_arr))
        all_deg_kl.append(np.mean(deg_kl_arr))
        all_mod_rel.append(np.mean(mod_rel_arr))
        all_evc_overlap.append(np.mean(evc_overlap_arr))
        all_evc_MAE.append(np.mean(evc_MAE_arr))
        all_diam_rel.append(np.mean(diam_rel_arr))

        print('[%s][%s] eps=%d/%d Done.%.2fs\n' % (dataset_name, method, ei+1, len(eps), time.time()-ti))

    res_path = './result'
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if save_csv:
        save_name = res_path + '/' + '%s_%s_%d.csv' % (dataset_name, method, exp_num)
        all_data.to_csv(save_name, index=False, sep=',')

    print('========================================')
    print('dataset:', dataset_name, '  method:', method)
    print('All time:%.2fs' % (time.time()-t_begin))

    return {
        'dataset': dataset_name, 'method': method, 'eps': eps,
        'nmi': all_nmi_arr, 'evc_overlap': all_evc_overlap, 'evc_MAE': all_evc_MAE,
        'deg_kl': all_deg_kl, 'diam_rel': all_diam_rel,
        'cc_rel': all_cc_rel, 'mod_rel': all_mod_rel,
    }


if __name__ == '__main__':

    # ==================== 配置区 ====================
    datasets = ['Chamelon']
    methods  = ['original','v2_full_fixed']
    eps      = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    exp_num  = 5
    # ================================================

    all_results = []
    total = len(datasets) * len(methods)
    idx = 0

    for ds in datasets:
        for m in methods:
            idx += 1
            print('\n' + '#'*70)
            print(f'#  [{idx}/{total}]  dataset={ds}  method={m}')
            print('#'*70 + '\n')
            try:
                r = main_func(dataset_name=ds, eps=eps, exp_num=exp_num,
                              save_csv=True, method=m)
                all_results.append(r)
            except Exception as e:
                print(f'[ERROR] {ds}/{m}: {e}')
                import traceback; traceback.print_exc()

    # 汇总
    print('\n\n' + '='*90)
    print('方法对比汇总 (各指标为所有 epsilon 的均值)')
    print('='*90)

    rows = []
    for r in all_results:
        rows.append({
            'dataset': r['dataset'], 'method': r['method'],
            'nmi': np.mean(r['nmi']), 'evc_overlap': np.mean(r['evc_overlap']),
            'evc_MAE': np.mean(r['evc_MAE']), 'deg_kl': np.mean(r['deg_kl']),
            'diam_rel': np.mean(r['diam_rel']), 'cc_rel': np.mean(r['cc_rel']),
            'mod_rel': np.mean(r['mod_rel']),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    res_path = './result'
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    df.to_csv(res_path + '/method_compare_v3_summary.csv', index=False)
    print('\n保存到 ./result/method_compare_v3_summary.csv')