import community
import networkx as nx
import time
import numpy as np
import pandas as pd

from numpy.random import laplace
from sklearn import metrics

from utils import *

import os
import itertools


def main_func(dataset_name='Chamelon',eps=[0.5,1,1.5,2,2.5,3,3.5],e1_r=1/3,e2_r=1/3,N=20,t=1.0,exp_num=10,save_csv=False, auto_alloc=False):

    # 记录程序开始时间
    t_begin = time.time()
    # 数据路径
    data_path = './data/' + dataset_name + '.txt'
    # 读取邻接矩阵和节点ID
    mat0,mid = get_mat(data_path)
    
    # ===== 要遍历的超参数组合 =====
    convert_ratio_list = [1.0]
    oversample_list = [1.0,1.1,1.2,1.3]
    param_combos = list(itertools.product(convert_ratio_list, oversample_list))

    # 结果表列名（增加 convert_ratio 和 oversample）
    cols = ['eps','convert_ratio','oversample','exper','nmi','evc_overlap','evc_MAE','deg_kl', \
    'diam_rel','cc_rel','mod_rel']
    
    # 初始化总结果DataFrame
    all_data = pd.DataFrame(None,columns=cols)

    # ===== 原始图构建 =====
    mat0_graph = nx.from_numpy_array(mat0,create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s'%(dataset_name))
    print('Node number:%d'%(mat0_graph.number_of_nodes()))
    print('Edge number:%d'%(mat0_graph.number_of_edges()))

    # ===== 原始图的各种统计信息 =====
    mat0_par = community.best_partition(mat0_graph)
    mat0_degree = np.sum(mat0,0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))
    mat0_evc = nx.eigenvector_centrality(mat0_graph,max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(),key = lambda x:x[1],reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01*mat0_node)
    mat0_diam = cal_diam(mat0)
    mat0_cc = nx.transitivity(mat0_graph)
    mat0_mod = community.modularity(mat0_par,mat0_graph)

    # ===== 存储所有 (eps, convert_ratio, oversample) 组合的平均结果 =====
    summary_rows = []

    # ===== 遍历不同隐私预算 epsilon =====
    for ei in range(len(eps)):
        epsilon = eps[ei]

        if auto_alloc:
            n = mat0_node
            d = 2 * mat0_edge / mat0_node
            _e3_r = np.clip(4.0 / (d * epsilon), 0.05, 0.20)
            _e1_r = 0.368 + 0.363 / epsilon - 0.033 * np.log(n)
            _e2_r = 1.0 - _e1_r - _e3_r
            _e1_r = np.clip(_e1_r, 0.05, 0.50)
            _e2_r = np.clip(_e2_r, 0.15, 0.85)
            _e3_r = np.clip(_e3_r, 0.05, 0.20)
            total = _e1_r + _e2_r + _e3_r
            _e1_r, _e2_r, _e3_r = _e1_r/total, _e2_r/total, _e3_r/total
            total = _e1_r + _e2_r + _e3_r
            _e1_r, _e2_r, _e3_r = _e1_r/total, _e2_r/total, _e3_r/total
            print(f'[AutoAlloc] eps={epsilon}, e1_r={_e1_r:.3f}, e2_r={_e2_r:.3f}, e3_r={_e3_r:.3f}')
        else:
            _e1_r = e1_r
            _e2_r = e2_r
            _e3_r = 1 - e1_r - e2_r

        e1 = _e1_r * epsilon
        e2 = _e2_r * epsilon
        e3 = _e3_r * epsilon
        ed = e3
        ev = e3
        ev_lambda = 1/ed
        dd_lam = 2/ev

        # ===== 遍历 convert_ratio × oversample 组合 =====
        for (convert_ratio, oversample) in param_combos:
            ti = time.time()
            print(f'\n======= eps={epsilon}, convert_ratio={convert_ratio}, oversample={oversample} =======')

            nmi_arr = np.zeros([exp_num])
            deg_kl_arr = np.zeros([exp_num])
            mod_rel_arr = np.zeros([exp_num])
            cc_rel_arr =  np.zeros([exp_num])
            diam_rel_arr = np.zeros([exp_num])
            evc_overlap_arr = np.zeros([exp_num])
            evc_MAE_arr = np.zeros([exp_num])

            # ===== 多次实验 =====
            for exper in range(exp_num):
                print('-----------epsilon=%.1f, cr=%.1f, os=%.1f, exper=%d/%d-------------'
                      %(epsilon, convert_ratio, oversample, exper+1, exp_num))

                t1 = time.time()

                # ===== Step1: 社区初始化 =====
                mat1_pvarr1 = community_init(mat0,mat0_graph,epsilon=e1,nr=N,t=t)

                # ===== 插入修正 =====
                _e2_local = e2
                _e3_local = e3
                _ev_lambda_local = ev_lambda
                _dd_lam_local = dd_lam

                if auto_alloc:
                    K = max(mat1_pvarr1) + 1
                    comm_sizes = [np.sum(mat1_pvarr1 == ci) for ci in range(K)]
                    avg_comm_size = np.mean(comm_sizes)
                    d = 2 * mat0_edge / mat0_node
                    estimated_intra_deg = d * (avg_comm_size / mat0_node)
                    SNR = estimated_intra_deg / (2.0 / e3) if e3 > 0 else 0

                    e2_f = np.clip(1.891 - 0.528/epsilon - 0.087*np.log(n) - 0.001*K - 0.234*SNR, 0.05, 0.85)
                    e3_f = np.clip(0.10, 0.05, 0.85)
                    remaining = 1.0 - _e1_r
                    _e2_r_new = remaining * e2_f / (e2_f + e3_f)
                    _e3_r_new = remaining - _e2_r_new

                    _e2_local = _e2_r_new * epsilon
                    _e3_local = _e3_r_new * epsilon
                    _ev_lambda_local = 1/_e3_local
                    _dd_lam_local = 2/_e3_local

                # 转为字典格式
                part1 = {}
                for i in range(len(mat1_pvarr1)):
                    part1[i] = mat1_pvarr1[i]

                # ===== Step2: 社区调整 =====
                mat1_par1 = comm.best_partition(mat0_graph,part1,epsilon_EM=_e2_local)
                mat1_pvarr = np.array(list(mat1_par1.values()))

                # ===== Step3: 社区节点提取 =====
                mat1_pvs = []
                for i in range(max(mat1_pvarr)+1):
                    pv1 = np.where(mat1_pvarr==i)[0]
                    pvs = list(pv1)
                    mat1_pvs.append(pvs)

                comm_n = max(mat1_pvarr) + 1

                ev_mat = np.zeros([comm_n,comm_n],dtype=np.int64)

                # ===== Step4: 社区间边统计 =====
                for i in range(comm_n):
                    pi = mat1_pvs[i]
                    ev_mat[i,i] = np.sum(mat0[np.ix_(pi,pi)])
                    for j in range(i+1,comm_n):
                        pj = mat1_pvs[j]
                        ev_mat[i,j] = int(np.sum(mat0[np.ix_(pi,pj)]))
                        ev_mat[j,i] = ev_mat[i,j]

                # 加拉普拉斯噪声
                ga = get_uptri_arr(ev_mat,ind=1)
                ga_noise = ga + laplace(0,_ev_lambda_local,len(ga))
                ga_noise_pp = FO_pp(ga_noise)
                ev_mat = get_upmat(ga_noise_pp,comm_n,ind=1)

                # ===== Step5: 度序列加噪 =====
                dd_s = []
                for i in range(comm_n):
                    dd1 = mat0[np.ix_(mat1_pvs[i],mat1_pvs[i])]
                    dd1 = np.sum(dd1,1)
                    dd1 = (dd1 + laplace(0,_dd_lam_local,len(dd1))).astype(int)
                    dd1 = FO_pp(dd1)
                    dd1[dd1<0] = 0
                    dd1[dd1>=len(dd1)] = len(dd1)-1
                    dd1 = list(dd1)
                    dd_s.append(dd1)

                # ===== Step6: 图重建（使用当前 convert_ratio 和 oversample） =====
                mat2 = np.zeros([mat0_node,mat0_node],dtype=np.int8)

                for i in range(comm_n):
                    dd_ind = mat1_pvs[i]
                    dd1 = dd_s[i]
                    mat2[np.ix_(dd_ind,dd_ind)] = generate_intra_edge(dd1)
                    
                    for j in range(i+1, comm_n):
                        ev1 = ev_mat[i,j]
                        if ev1 <= 0:
                            continue
                        
                        pi = mat1_pvs[i]
                        pj = mat1_pvs[j]
                        
                        n_convert = int(ev1 * convert_ratio)
                        n_inter = ev1 - n_convert
                        
                        n_ri = n_convert // 2
                        n_rj = n_convert - n_ri
                        
                        if n_ri > 0:
                            max_ei = len(pi) * (len(pi)-1) // 2
                            sample_ri = min(int(n_ri * oversample), max_ei)
                            c1 = np.random.choice(pi, sample_ri)
                            c2 = np.random.choice(pi, sample_ri)
                            for ind in range(sample_ri):
                                mat2[c1[ind],c2[ind]] = 1
                                mat2[c2[ind],c1[ind]] = 1
                        
                        if n_rj > 0:
                            max_ej = len(pj) * (len(pj)-1) // 2
                            sample_rj = min(int(n_rj * oversample), max_ej)
                            c1 = np.random.choice(pj, sample_rj)
                            c2 = np.random.choice(pj, sample_rj)
                            for ind in range(sample_rj):
                                mat2[c1[ind],c2[ind]] = 1
                                mat2[c2[ind],c1[ind]] = 1
                        
                        if n_inter > 0:
                            c1 = np.random.choice(pi, n_inter)
                            c2 = np.random.choice(pj, n_inter)
                            for ind in range(n_inter):
                                mat2[c1[ind],c2[ind]] = 1
                                mat2[c2[ind],c1[ind]] = 1

                # 对称化邻接矩阵                
                mat2 = mat2 + np.transpose(mat2)
                mat2 = np.triu(mat2,1)
                mat2 = mat2 + np.transpose(mat2)
                mat2[mat2>0] = 1

                mat2_graph = nx.from_numpy_array(mat2,create_using=nx.Graph)

                # ===== Step7: 计算指标 =====
                mat2_edge = mat2_graph.number_of_edges()
                mat2_node = mat2_graph.number_of_nodes()

                mat2_par = community.best_partition(mat2_graph)
                mat2_mod = community.modularity(mat2_par,mat2_graph)
                mat2_cc = nx.transitivity(mat2_graph)
                mat2_degree = np.sum(mat2,0)
                mat2_deg_dist = np.bincount(np.int64(mat2_degree))
                mat2_evc = nx.eigenvector_centrality(mat2_graph,max_iter=10000)
                mat2_evc_a = dict(sorted(mat2_evc.items(),key = lambda x:x[1],reverse=True))
                mat2_evc_ak = list(mat2_evc_a.keys())
                mat2_evc_val = np.array(list(mat2_evc_a.values()))
                mat2_diam = cal_diam(mat2)

                cc_rel = cal_rel(mat0_cc,mat2_cc)
                deg_kl = cal_kl(mat0_deg_dist,mat2_deg_dist)
                mod_rel = cal_rel(mat0_mod,mat2_mod)
                
                labels_true = list(mat0_par.values())
                labels_pred = list(mat2_par.values())
                nmi = metrics.normalized_mutual_info_score(labels_true,labels_pred)

                evc_overlap = cal_overlap(mat0_evc_ak,mat2_evc_ak,np.int64(0.01*mat0_node))
                evc_MAE = cal_MAE(mat0_evc_val,mat2_evc_val,k=evc_kn)
                diam_rel = cal_rel(mat0_diam,mat2_diam)

                nmi_arr[exper] = nmi
                cc_rel_arr[exper] = cc_rel
                deg_kl_arr[exper] = deg_kl
                mod_rel_arr[exper] = mod_rel
                evc_overlap_arr[exper] = evc_overlap
                evc_MAE_arr[exper] = evc_MAE
                diam_rel_arr[exper] = diam_rel

                print('Nodes=%d,Edges=%d,nmi=%.4f,cc_rel=%.4f,deg_kl=%.4f,mod_rel=%.4f,evc_overlap=%.4f,evc_MAE=%.4f,diam_rel=%.4f' \
                    %(mat2_node,mat2_edge,nmi,cc_rel,deg_kl,mod_rel,evc_overlap,evc_MAE,diam_rel))

                data_col = [epsilon, convert_ratio, oversample, exper, nmi, evc_overlap, evc_MAE, deg_kl, \
                    diam_rel, cc_rel, mod_rel]
                col_len = len(data_col)
                data_col = np.array(data_col).reshape(1,col_len)
                data1 = pd.DataFrame(data_col,columns=cols)
                all_data = pd.concat([all_data, data1], ignore_index=True)       

            # ===== 每个 (eps, cr, os) 组合取平均 =====
            summary_rows.append({
                'eps': epsilon,
                'convert_ratio': convert_ratio,
                'oversample': oversample,
                'nmi': np.mean(nmi_arr),
                'evc_overlap': np.mean(evc_overlap_arr),
                'evc_MAE': np.mean(evc_MAE_arr),
                'deg_kl': np.mean(deg_kl_arr),
                'diam_rel': np.mean(diam_rel_arr),
                'cc_rel': np.mean(cc_rel_arr),
                'mod_rel': np.mean(mod_rel_arr),
            })

            print(f'eps={epsilon}, cr={convert_ratio}, os={oversample} Done. %.2fs'%(time.time()-ti))

    # ===== 保存结果 =====
    res_path = './result'
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    # 保存详细结果（每次实验）
    save_name = res_path + '/' + '%s_grid_detail_%d_%.1f_%.2f_%.2f_%d.csv' %(dataset_name,N,t,e1_r,e2_r,exp_num)
    if save_csv:
        all_data.to_csv(save_name, index=False, sep=',')

    # 保存汇总结果（每个组合的平均值）
    summary_df = pd.DataFrame(summary_rows)
    summary_name = res_path + '/' + '%s_grid_summary_%d_%.1f_%.2f_%.2f_%d.csv' %(dataset_name,N,t,e1_r,e2_r,exp_num)
    summary_df.to_csv(summary_name, index=False, sep=',')
    print(f'\n汇总结果已保存到: {summary_name}')

    # ===== 输出汇总表 =====
    print('\n============================')
    print('dataset:', dataset_name)
    print('============================')
    print(summary_df.to_string(index=False))

    # ===== 按 eps 分组找最佳组合 =====
    print('\n===== 每个 eps 下的最佳组合（按 NMI） =====')
    for ep in eps:
        sub = summary_df[summary_df['eps'] == ep]
        best = sub.loc[sub['nmi'].idxmax()]
        print(f"eps={ep}: best cr={best['convert_ratio']}, os={best['oversample']}, "
              f"nmi={best['nmi']:.4f}, deg_kl={best['deg_kl']:.4f}, mod_rel={best['mod_rel']:.4f}")

    print('\nAll time:%.2fs'%(time.time()-t_begin))



if __name__ == '__main__':
    dataset_name = 'Chamelon'
    eps = [0.5,1,1.5,2,2.5,3,3.5]
    e1_r = 1/3
    e2_r = 1/3
    exp_num = 10
    n1 = 20
    t = 1.0

    main_func(dataset_name='Chamelon',eps=eps,e1_r=0.3,e2_r=0.3,N=n1,t=t,exp_num=10,save_csv=True)