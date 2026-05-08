import community
import networkx as nx
import time
import numpy as np

from numpy.random import laplace
from sklearn import metrics

from utils import *

import os


# ============================================================
# 模块B: 基于初步社区结构的预算二次再分配
# 在 community_init 之后、community_adjustment 之前调用
# 输入当前 e1/e3 比例，输出新的 e2/e3 比例（e1 不变）
# ============================================================
def realloc_budget_by_community(mat1_pvarr1, e1_r_cur, e3_r_cur,
                                epsilon, mat0_node, mat0_edge):
    n = mat0_node
    K = max(mat1_pvarr1) + 1
    comm_sizes = [np.sum(mat1_pvarr1 == ci) for ci in range(K)]
    avg_comm_size = np.mean(comm_sizes)
    d = 2 * mat0_edge / mat0_node
    estimated_intra_deg = d * (avg_comm_size / mat0_node)
    e3_cur = e3_r_cur * epsilon
    SNR = estimated_intra_deg / (2.0 / e3_cur) if e3_cur > 0 else 0

    e2_f = np.clip(1.891 - 0.528/epsilon - 0.087*np.log(n)
                   - 0.001*K - 0.234*SNR, 0.05, 0.85)
    e3_f = np.clip(0.15, 0.05, 0.85)
    remaining = 1.0 - e1_r_cur
    new_e2_r = remaining * e2_f / (e2_f + e3_f)
    new_e3_r = remaining - new_e2_r

    # 保底: 第二阶段不能把 e3 压得比第一阶段更低
    # new_e3_r = max(new_e3_r, e3_r_cur)
    # new_e2_r = remaining - new_e3_r

    return new_e2_r, new_e3_r


# ============================================================
# 模块C: 图重建（两种策略）
# orig: 原版 Step 6
# new : utils.step6_v3_full_fixed，多两个 ratio 超参
# ============================================================
def reconstruct_graph_orig(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat):
    mat2 = np.zeros([mat0_node, mat0_node], dtype=np.int8)
    for i in range(comm_n):
        # Intra-community
        pi = mat1_pvs[i]
        dd_ind = mat1_pvs[i]
        dd1 = dd_s[i]
        mat2[np.ix_(dd_ind, dd_ind)] = generate_intra_edge(dd1)

        # Inter-community
        for j in range(i+1, comm_n):
            ev1 = ev_mat[i, j]
            pj = mat1_pvs[j]
            if ev1 > 0:
                c1 = np.random.choice(pi, ev1)
                c2 = np.random.choice(pj, ev1)
                for ind in range(ev1):
                    mat2[c1[ind], c2[ind]] = 1
                    mat2[c2[ind], c1[ind]] = 1
    return mat2


def reconstruct_graph_new(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat,
                          intra_ratio=0.05, inter_ratio=0.1):
    return step6_v3_full_fixed(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat,
                               intra_ratio=intra_ratio, inter_ratio=inter_ratio)


def main_func(dataset_name='Chamelon',eps=[0.5,1,1.5,2,2.5,3,3.5],e1_r=1/3,e2_r=1/3,N=20,t=1.0,exp_num=10,save_csv=False,
              use_module_b=False, use_module_c=False,
              intra_ratio=0.05, inter_ratio=0.1):


    t_begin = time.time()

    data_path = './data/' + dataset_name + '.txt'
    mat0,mid = get_mat(data_path)
    

    cols = ['eps','exper','nmi','evc_overlap','evc_MAE','deg_kl', \
    'diam_rel','cc_rel','mod_rel']
    

    all_data = pd.DataFrame(None,columns=cols)

    # original graph
    mat0_graph = nx.from_numpy_array(mat0,create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s'%(dataset_name))
    print('Node number:%d'%(mat0_graph.number_of_nodes()))
    print('Edge number:%d'%(mat0_graph.number_of_edges()))
    print('Modules: B=%s, C=%s' % (use_module_b, use_module_c))


    mat0_par = community.best_partition(mat0_graph)

    mat0_degree = np.sum(mat0,0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree)) # degree distribution

    mat0_evc = nx.eigenvector_centrality(mat0_graph,max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(),key = lambda x:x[1],reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    evc_kn = np.int64(0.01*mat0_node)

    mat0_diam = cal_diam(mat0)

    mat0_cc = nx.transitivity(mat0_graph)

    mat0_mod = community.modularity(mat0_par,mat0_graph)


    all_deg_kl = []
    all_mod_rel = []
    all_nmi_arr = []
    all_evc_overlap = []
    all_evc_MAE = []
    all_cc_rel = []
    all_diam_rel = []


    for ei in range(len(eps)):
        epsilon = eps[ei]
        ti = time.time()
        
        # 当前比例（保留下划线变量名，方便日后接入模块A）
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

    
        nmi_arr = np.zeros([exp_num])
        deg_kl_arr = np.zeros([exp_num])
        mod_rel_arr = np.zeros([exp_num])
        cc_rel_arr =  np.zeros([exp_num])
        diam_rel_arr = np.zeros([exp_num])
        evc_overlap_arr = np.zeros([exp_num])
        evc_MAE_arr = np.zeros([exp_num])


        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------'%(epsilon,exper+1,exp_num))


            t1 = time.time()

            # Community Initialization
            mat1_pvarr1 = community_init(mat0,mat0_graph,epsilon=e1,nr=N,t=t)

            # ===== 模块B: 基于社区结构再分配预算 =====
            if use_module_b:
                _e2_r_new, _e3_r_new = realloc_budget_by_community(
                    mat1_pvarr1, _e1_r, _e3_r, epsilon, mat0_node, mat0_edge)
                e2 = _e2_r_new * epsilon
                e3 = _e3_r_new * epsilon
                ev_lambda = 1/e3
                dd_lam = 2/e3
                if exper == 0:
                    print('[ModuleB] eps=%.1f, e2_r: %.3f -> %.3f, e3_r: %.3f -> %.3f' \
                        %(epsilon, _e2_r, _e2_r_new, _e3_r, _e3_r_new))

            part1 = {}
            for i in range(len(mat1_pvarr1)):
                part1[i] = mat1_pvarr1[i]

            # Community Adjustment
            mat1_par1 = comm.best_partition(mat0_graph,part1,epsilon_EM=e2)
            mat1_pvarr = np.array(list(mat1_par1.values()))

            # Information Extraction
            mat1_pvs = []
            for i in range(max(mat1_pvarr)+1):
                pv1 = np.where(mat1_pvarr==i)[0]
                pvs = list(pv1)
                mat1_pvs.append(pvs)

            comm_n = max(mat1_pvarr) + 1

            ev_mat = np.zeros([comm_n,comm_n],dtype=np.int64)

        
            # edge vector
            for i in range(comm_n):
                pi = mat1_pvs[i]
                ev_mat[i,i] = np.sum(mat0[np.ix_(pi,pi)])
                for j in range(i+1,comm_n):
                    pj = mat1_pvs[j]
                    ev_mat[i,j] = int(np.sum(mat0[np.ix_(pi,pj)]))
                    ev_mat[j,i] = ev_mat[i,j]

            ga = get_uptri_arr(ev_mat,ind=1)
            ga_noise = ga + laplace(0,ev_lambda,len(ga))
        
            ga_noise_pp = FO_pp(ga_noise)
            ev_mat = get_upmat(ga_noise_pp,comm_n,ind=1)

            # degree sequence
            dd_s = []
            for i in range(comm_n):
                dd1 = mat0[np.ix_(mat1_pvs[i],mat1_pvs[i])]
                dd1 = np.sum(dd1,1) 
        
                dd1 = (dd1 + laplace(0,dd_lam,len(dd1))).astype(int)
                dd1 = FO_pp(dd1)
                dd1[dd1<0] = 0
                dd1[dd1>=len(dd1)] = len(dd1)-1

                dd1 = list(dd1)
                dd_s.append(dd1)

            # ===== 模块C: 图重建 =====
            if use_module_c:
                mat2 = reconstruct_graph_new(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat,
                                             intra_ratio=intra_ratio, inter_ratio=inter_ratio)
            else:
                mat2 = reconstruct_graph_orig(mat0_node, comm_n, mat1_pvs, dd_s, ev_mat)

                            
            mat2 = mat2 + np.transpose(mat2)
            mat2 = np.triu(mat2,1)
            mat2 = mat2 + np.transpose(mat2)
            mat2[mat2>0] = 1

            mat2_graph = nx.from_numpy_array(mat2,create_using=nx.Graph)

            # save the graph
            # file_name = './result/' +  'PrivGraph_%s_%.1f_%d.txt' %(dataset_name,epsilon,exper)
            # write_edge_txt(mat2,mid,file_name)

            #evaluate
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


            nmi_arr[exper] = nmi
            cc_rel_arr[exper] = cc_rel
            deg_kl_arr[exper] = deg_kl
            mod_rel_arr[exper] = mod_rel
            evc_overlap_arr[exper] = evc_overlap
            evc_MAE_arr[exper] = evc_MAE
            diam_rel_arr[exper] = diam_rel

            print('Nodes=%d,Edges=%d,nmi=%.4f,cc_rel=%.4f,deg_kl=%.4f,mod_rel=%.4f,evc_overlap=%.4f,evc_MAE=%.4f,diam_rel=%.4f' \
                %(mat2_node,mat2_edge,nmi,cc_rel,deg_kl,mod_rel,evc_overlap,evc_MAE,diam_rel))

     

            data_col = [epsilon,exper,nmi,evc_overlap,evc_MAE,deg_kl, \
                diam_rel,cc_rel,mod_rel]
            col_len = len(data_col)
            data_col = np.array(data_col).reshape(1,col_len)
            data1 = pd.DataFrame(data_col,columns=cols)
            all_data = pd.concat([all_data, data1], ignore_index=True)          

                

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
    e1_r = 0.2
    e2_r = 0.6

    # set the number of experiments
    exp_num = 10

    # set the number of nodes for community initialization
    n1 = 20

    # set the resolution parameter
    t = 1.0

    # ===== 消融测试: 切换开关 =====
    # Baseline (原版)
   #  main_func(dataset_name=dataset_name,eps=eps,e1_r=e1_r,e2_r=e2_r,N=n1,t=t,exp_num=exp_num)

    # +模块B
    # main_func(dataset_name=dataset_name,eps=eps,e1_r=e1_r,e2_r=e2_r,N=n1,t=t,exp_num=exp_num,
    #            use_module_b=True)

    # +模块C
    # main_func(dataset_name=dataset_name,eps=eps,e1_r=e1_r,e2_r=e2_r,N=n1,t=t,exp_num=exp_num,
    #           use_module_c=True, intra_ratio=0.05, inter_ratio=0.1)

    # +模块B+C
    main_func(dataset_name=dataset_name,eps=eps,e1_r=e1_r,e2_r=e2_r,N=n1,t=t,exp_num=exp_num,
               use_module_b=True, use_module_c=True, intra_ratio=0.05, inter_ratio=0.1)