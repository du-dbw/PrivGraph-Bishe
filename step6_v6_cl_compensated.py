"""
step6_v6_cl_compensated.py
==========================
核心思路：把"省下来"的跨社区边预算，通过放大 dd_s 让 CL 模型自己生成，
而不是随机撒边。CL 按 d_i×d_j 概率放边，天然保持度分布结构。

与 v3 的区别：
  1. 阶段 0（新增）：计算每个社区因 inter_ratio<1 而省下的边数，
     按比例放大 dd_s，让 CL 多生成对应数量的边
  2. 阶段 1：CL 用放大后的 dd_s 生成社区内边
  3. 阶段 2：只放 inter_ratio 比例的跨社区边，intra_ratio 设为 0
     （因为补偿已经在阶段 0 做了）
  4. 阶段 3：删除（无软度修复）
"""

import numpy as np
from utils import generate_intra_edge


def step6_v6_cl_compensated(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat,
                             inter_ratio=0.10):
    """
    参数：
      inter_ratio : 跨社区边保留比例（默认 0.10）
      （不再需要 intra_ratio，因为补偿由 dd_s 放大完成）
    """
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)

    # ================================================================
    # ★ 阶段 0（新增）：计算补偿量，放大 dd_s
    # ================================================================
    # 对每个社区 i，计算它从所有 (i,j) 对中"省下来"的边数
    # 省下的 = Σ_j ev_mat[i,j] × (1 - inter_ratio)
    # 其中一半归 i，一半归 j → i 得到 Σ_j ev_mat[i,j] × (1 - inter_ratio) / 2

    dd_s_scaled = []
    for i in range(comm_n):
        extra_i = 0
        for j in range(comm_n):
            if j == i:
                continue
            ev_ij = ev_mat[i, j] if i < j else ev_mat[j, i]
            if ev_ij > 0:
                extra_i += ev_ij * (1.0 - inter_ratio) / 2.0

        dd_i = list(dd_s[i])  # 拷贝
        dd_sum = sum(dd_i)

        if dd_sum > 0 and extra_i > 0:
            # CL 生成的边数 ≈ sum(dd)/2
            # 要多生成 extra_i 条边 → sum(dd) 需要增加 2*extra_i
            # 放大系数
            scale = 1.0 + (2.0 * extra_i) / dd_sum

            # 按比例放大每个度，保持度分布形状
            dd_i_scaled = []
            for d in dd_i:
                new_d = int(round(d * scale))
                # 社区内度不能超过社区大小-1
                new_d = min(new_d, len(dd_i) - 1)
                new_d = max(new_d, 0)
                dd_i_scaled.append(new_d)
            dd_s_scaled.append(dd_i_scaled)
        else:
            dd_s_scaled.append(dd_i)

    # ---- 阶段 1: CL 用放大后的 dd_s 生成社区内边 ----
    for i in range(comm_n):
        nodes = mat1_pvs[i]
        if len(nodes) == 0:
            continue
        mat2[np.ix_(nodes, nodes)] = generate_intra_edge(dd_s_scaled[i])

    # ---- 阶段 2: 只放 inter_ratio 比例的跨社区边 ----
    # （不再有 intra_ratio 随机补边，补偿已在阶段 0 完成）
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

            n_inter = int(ev1 * inter_ratio)

            if n_inter > 0:
                n_sample = min(int(n_inter * 1.5) + 10, len(pi) * len(pj))
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

    # ---- 阶段 3: ★ 已删除（无软度修复） ★ ----

    return mat2
