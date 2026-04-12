import community
import networkx as nx
import time
import numpy as np
import pandas as pd

from numpy.random import laplace
from sklearn import metrics

from utils import *

import os



def main_func(dataset_name='Chamelon',eps=[0.5,1,1.5,2,2.5,3,3.5],e1_r=1/3,e2_r=1/3,N=20,t=1.0,exp_num=10,save_csv=False, auto_alloc=False):

    # 记录程序开始时间
    t_begin = time.time()
    # 数据路径
    data_path = './data/' + dataset_name + '.txt'
    # 读取邻接矩阵和节点ID
    mat0,mid = get_mat(data_path)
    
    # 结果表列名
    cols = ['eps','exper','nmi','evc_overlap','evc_MAE','deg_kl', \
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
    # 社区划分
    mat0_par = community.best_partition(mat0_graph)
    # 度分布
    mat0_degree = np.sum(mat0,0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree)) # degree distribution
    # 特征向量中心性
    mat0_evc = nx.eigenvector_centrality(mat0_graph,max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(),key = lambda x:x[1],reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01*mat0_node)
    # 直径
    mat0_diam = cal_diam(mat0)
    # 聚类系数
    mat0_cc = nx.transitivity(mat0_graph)
    # 模块度
    mat0_mod = community.modularity(mat0_par,mat0_graph)

    # ===== 存储所有epsilon结果 =====
    all_deg_kl = []
    all_mod_rel = []
    all_nmi_arr = []
    all_evc_overlap = []
    all_evc_MAE = []
    all_cc_rel = []
    all_diam_rel = []

    # ===== 遍历不同隐私预算 epsilon =====
    for ei in range(len(eps)):
        epsilon = eps[ei]
        ti = time.time()

        if auto_alloc:
            # ===== 经验公式: 基于 (eps, n) 自动分配 =====
            n = mat0_node
            d = 2 * mat0_edge / mat0_node
            _e3_r = np.clip(4.0 / (d * epsilon), 0.05, 0.20)
            _e1_r = 0.368 + 0.363 / epsilon - 0.033 * np.log(n)
            _e2_r = 1.0 - _e1_r - _e3_r

            # clip + 归一化
            _e1_r = np.clip(_e1_r, 0.05, 0.50)   # ← 关键: 限制 e1 上限为 0.50
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
        # 噪声参数
        ed = e3
        ev = e3
        ev_lambda = 1/ed
        dd_lam = 2/ev

        # 初始化指标数组
        nmi_arr = np.zeros([exp_num])
        deg_kl_arr = np.zeros([exp_num])
        mod_rel_arr = np.zeros([exp_num])
        cc_rel_arr =  np.zeros([exp_num])
        diam_rel_arr = np.zeros([exp_num])
        evc_overlap_arr = np.zeros([exp_num])
        evc_MAE_arr = np.zeros([exp_num])

        # ===== 多次实验 =====
        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------'%(epsilon,exper+1,exp_num))


            t1 = time.time()

            # ===== Step1: 社区初始化 =====
            mat1_pvarr1 = community_init(mat0,mat0_graph,epsilon=e1,nr=N,t=t)
            # ===== 插入修正 =====
            if auto_alloc:
                K = max(mat1_pvarr1) + 1
                # 用社区大小估算平均社区内度（不访问 mat0）
                comm_sizes = [np.sum(mat1_pvarr1 == ci) for ci in range(K)]
                avg_comm_size = np.mean(comm_sizes)
                # 用全图平均度和社区大小近似社区内度
                d = 2 * mat0_edge / mat0_node   # 这个在函数开头就算过了，是公开信息
                estimated_intra_deg = d * (avg_comm_size / mat0_node)  # 近似
                SNR = estimated_intra_deg / (2.0 / e3) if e3 > 0 else 0

                e2_f = np.clip(1.891 - 0.528/epsilon - 0.087*np.log(n) - 0.001*K - 0.234*SNR, 0.05, 0.85)
                e3_f = np.clip(0.10, 0.05, 0.85)
                remaining = 1.0 - _e1_r
                _e2_r_new = remaining * e2_f / (e2_f + e3_f)
                _e3_r_new = remaining - _e2_r_new

                e2 = _e2_r_new * epsilon
                e3 = _e3_r_new * epsilon
                ev_lambda = 1/e3          
                dd_lam = 2/e3
            # ===== 修正结束 =====

            # mat1_pvarr1 = community_init_dp_neighbor_fixed(
            #     mat0, mat0_graph,
            #     epsilon=e1,
            #     t=t,
            #     alpha=0.3,
            #     beta=0.3,
            #     C=None
            # )

            # 转为字典格式
            part1 = {}
            for i in range(len(mat1_pvarr1)):
                part1[i] = mat1_pvarr1[i]

            # ===== Step2: 社区调整 =====
            mat1_par1 = comm.best_partition(mat0_graph,part1,epsilon_EM=e2)
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
                

            # 加拉普拉斯噪声,且FO_pp截断
            ga = get_uptri_arr(ev_mat,ind=1)
            ga_noise = ga + laplace(0,ev_lambda,len(ga))
        

            ga_noise_pp = FO_pp(ga_noise)

            ev_mat = get_upmat(ga_noise_pp,comm_n,ind=1)

            # ev_total_before = np.sum(ga)           # 加噪前社区间边总数
            # ev_total_after = np.sum(ga_noise_pp)   # 加噪后社区间边总数
            # print(f'[DEBUG Step4] 社区数={comm_n}, 噪声尺度ev_lambda={ev_lambda:.2f}')
            # print(f'[DEBUG Step4] 社区间边: 加噪前={ev_total_before:.0f}, 加噪后={ev_total_after:.0f}, 损失率={1-ev_total_after/(ev_total_before+1e-9):.2%}')


            # ===== Step5: 度序列加噪 =====
            dd_s = []
            intra_degree_sum_before = 0
            intra_degree_sum_after = 0
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i],mat1_pvs[i])]
                dd1 = np.sum(dd1,1)
                intra_degree_sum_before += np.sum(dd1)

                dd1 = (dd1 + laplace(0,dd_lam,len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1<0] = 0
                dd1[dd1>=len(dd1)] = len(dd1)-1
                intra_degree_sum_after += np.sum(dd1)
                dd1 = list(dd1)
                dd_s.append(dd1)


            # ===== Step6: 图重建 =====
            # 原版
            # mat2 = step6_original(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat)

            # 社区内加 50%，社区间保留 100%（总边数放大到 150%）
            mat2 = step6_v3_full_fixed(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat, intra_ratio=0.4, inter_ratio=0.2)

            # 对称化邻接矩阵                
            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2,1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2>0] = 1

            mat2_graph = nx.from_numpy_array(mat2,create_using=nx.Graph)

            # save the graph
            # file_name = './result/' +  'PrivGraph_%s_%.1f_%d.txt' %(dataset_name,epsilon,exper)
            # write_edge_txt(mat2,mid,file_name)

            # ===== [新增] Step 6.5: 后处理剪枝 =====
            # 感觉第二个更好，但是差不太多
            # mat2 = post_process_prune(mat2, mat1_pvs, dd_s, ev_mat, comm_n)
            # mat2 = post_process_edge_swap(mat2, mat1_pvs, comm_n, n_iter_ratio=0.5)

            # ===== Step7: 计算指标 =====
            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()

            mat2_par = community.best_partition(mat2_graph)
            mat2_mod = community.modularity(mat2_par,mat2_graph)

            mat2_cc = nx.transitivity(mat2_graph)

            mat2_degree = np.sum(mat2,0)
            mat2_deg_dist = np.bincount(np.int64(mat2_degree)) # degree distribution
            
            mat2_evc = nx.eigenvector_centrality(mat2_graph,max_iter=10000)
            mat2_evc_a = dict(sorted(mat2_evc.items(),key = lambda x:x[1],reverse=True))
            mat2_evc_ak = list(mat2_evc_a.keys())
            mat2_evc_val = np.array(list(mat2_evc_a.values()))
        

            mat2_diam = cal_diam(mat2)

            # calculate the metrics
            # clustering coefficent
            # ===== 各类指标 =====
            cc_rel = cal_rel(mat0_cc,mat2_cc)

            # degree distribution
            deg_kl = cal_kl(mat0_deg_dist,mat2_deg_dist)

            # modularity
            mod_rel = cal_rel(mat0_mod,mat2_mod)
            
        
            # NMI
            labels_true = list(mat0_par.values())
            labels_pred = list(mat2_par.values())
            nmi = metrics.normalized_mutual_info_score(labels_true,labels_pred)


            # Overlap of eigenvalue nodes 
            evc_overlap = cal_overlap(mat0_evc_ak,mat2_evc_ak,np.int64(0.01*mat0_node))

            # MAE of EVC
            evc_MAE = cal_MAE(mat0_evc_val,mat2_evc_val,k=evc_kn)

            # diameter
            diam_rel = cal_rel(mat0_diam,mat2_diam)

            # 保存结果
            nmi_arr[exper] = nmi
            cc_rel_arr[exper] = cc_rel
            deg_kl_arr[exper] = deg_kl
            mod_rel_arr[exper] = mod_rel
            evc_overlap_arr[exper] = evc_overlap
            evc_MAE_arr[exper] = evc_MAE
            diam_rel_arr[exper] = diam_rel

            print('Nodes=%d,Edges=%d,nmi=%.4f,cc_rel=%.4f,deg_kl=%.4f,mod_rel=%.4f,evc_overlap=%.4f,evc_MAE=%.4f,diam_rel=%.4f' \
                %(mat2_node,mat2_edge,nmi,cc_rel,deg_kl,mod_rel,evc_overlap,evc_MAE,diam_rel))
            # print('原图边数:', mat0_graph.number_of_edges())
            # print('合成图边数:', mat2_graph.number_of_edges())
            # print('原图平均度:', np.mean(np.sum(mat0, 0)))
            # print('合成图平均度:', np.mean(np.sum(mat2, 0)))
                

            data_col = [epsilon,exper,nmi,evc_overlap,evc_MAE,deg_kl, \
                diam_rel,cc_rel,mod_rel]
            col_len = len(data_col)
            data_col = np.array(data_col).reshape(1,col_len)
            data1 = pd.DataFrame(data_col,columns=cols)
            all_data = pd.concat([all_data, data1], ignore_index=True)       

                
        # ===== 每个epsilon取平均 =====
        all_nmi_arr.append(np.mean(nmi_arr))
        all_cc_rel.append(np.mean(cc_rel_arr))
        all_deg_kl.append(np.mean(deg_kl_arr))
        all_mod_rel.append(np.mean(mod_rel_arr))
        all_evc_overlap.append(np.mean(evc_overlap_arr))
        all_evc_MAE.append(np.mean(evc_MAE_arr))
        all_diam_rel.append(np.mean(diam_rel_arr))

        
        print('all_index=%d/%d Done.%.2fs\n'%(ei+1,len(eps),time.time()-ti))

    res_path = './result'
    save_name = res_path + '/' + '%s_%d_%.1f_%.2f_%.2f_%d.csv' %(dataset_name,N,t,e1_r,e2_r,exp_num)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    if save_csv == True:
        all_data.to_csv(save_name,index=False,sep=',')

    print('-----------------------------')
    # ===== 输出最终结果 =====
    print('dataset:',dataset_name)
    
    print('eps=',eps)
    print('all_nmi_arr=',all_nmi_arr)
    print('all_evc_overlap=',all_evc_overlap)
    print('all_evc_MAE=',all_evc_MAE)
    print('all_deg_kl=',all_deg_kl)
    print('all_diam_rel=',all_diam_rel)
    print('all_cc_rel=',all_cc_rel)
    print('all_mod_rel=',all_mod_rel)

    print('All time:%.2fs'%(time.time()-t_begin))



if __name__ == '__main__':
    # set the dataset
    # 'Facebook', 'CA-HepPh', 'Enron'
    dataset_name = 'Chamelon'

    # set the privacy budget, list type
    eps = [0.5,1,1.5,2,2.5,3,3.5]

    # set the ratio of the privacy budget
    e1_r = 1/3
    e2_r = 1/3

    # set the number of experiments
    exp_num = 10

    # set the number of nodes for community initialization
    n1 = 20

    # set the resolution parameter
    t = 1.0

    # run the function
    main_func(dataset_name='Chamelon',eps=eps,e1_r=1/3,e2_r=1/3,N=n1,t=t,exp_num=5)
    # main_func(dataset_name='Chamelon', eps=eps, auto_alloc=True, exp_num=10)
    # main_func(dataset_name=dataset_name,eps=[2],e1_r=e1_r,e2_r=e2_r,N=n1,t=t,exp_num=1)
   

