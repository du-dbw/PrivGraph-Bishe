import numpy as np
from numpy.random import laplace
import pandas as pd

import networkx as nx

import community
import comm
import time
import random

import itertools
from heapq import *

from heapq import nlargest


# 从边列表文本文件读取图，构建无向邻接矩阵，并把原始节点 ID 映射成连续索引，方便后续计算。
def get_mat(data_path):
    # data_path = './data/' + dataset_name + '.txt'
    data = np.loadtxt(data_path)

    
    # initial statistics
    dat = (np.append(data[:,0],data[:,1])).astype(int)
    dat_c = np.bincount(dat)

    d = {}
    node = 0
    mid = []
    for i in range(len(dat_c)):
        if dat_c[i] > 0:
            d[i] = node
            mid.append(i)
            node = node + 1
    mid = np.array(mid,dtype=np.int32)

    # initial statistics
    Edge_num = data.shape[0] 
    c = len(d) 


    # genarated adjancent matrix
    mat0 = np.zeros([c,c],dtype=np.uint8)
    for i in range(Edge_num):
        mat0[d[int(data[i,0])],d[int(data[i,1])]] = 1


    # transfer direct to undirect
    mat0 = mat0 + np.transpose(mat0)
    mat0 = np.triu(mat0,1)
    mat0 = mat0 + np.transpose(mat0)
    mat0[mat0>0] = 1
    return mat0,mid
# 带差分隐私的鲁汶社区发现主函数：先随机分社区 → 计算社区间权重 → 加拉普拉斯噪声保护隐私 → 用鲁汶算法合并社区 → 返回最终节点社区标签
def community_init(mat0,mat0_graph,epsilon,nr,t=1.0):

    # t1 = time.time()
    # Divide the nodes randomly
    g1 = list(np.zeros(len(mat0)))
    ind = -1

    for i in range(len(mat0)):
        if i % nr == 0:
            ind = ind + 1
        g1[i] = ind

    random.shuffle(g1)

    mat0_par3 = {}
    for i in range(len(mat0)):
        mat0_par3[i] = g1[i]

    gr1 = max(mat0_par3.values()) + 1

    # mat0_mod3 = community.modularity(mat0_par3,mat0_graph)
    # print('mat0_mod2=%.3f,gr1=%d'%(mat0_mod3,gr1)) 

    #原本 mat0_par3 是 节点 → 社区 的映射（字典），键是节点索引，值是社区编号。

    # 经过这段代码处理后，mat0_par3_pvs 变成了 社区 → 节点列表 的映射（嵌套列表）：

    # mat0_par3_pvs[i] 就是社区 i 包含的所有节点索引。
    mat0_par3_pv = np.array(list(mat0_par3.values()))
    mat0_par3_pvs = []

    # 构建 社区-社区邻接矩阵：
    # 对角线：社区内边数
    # 非对角线：社区间边数
    for i in range(gr1):
        pv = np.where(mat0_par3_pv==i)[0]
        pvs = list(pv)
        mat0_par3_pvs.append(pvs)
    mat_one_level = np.zeros([gr1,gr1])

    for i in range(gr1):
        pi = mat0_par3_pvs[i]
        mat_one_level[i,i] = np.sum(mat0[np.ix_(pi,pi)])
        for j in range(i+1,gr1):
            pj = mat0_par3_pvs[j]
            mat_one_level[i,j] = np.sum(mat0[np.ix_(pi,pj)])
    # print('generate new matrix time:%.2fs'%(time.time()-t1))
    


    lap_noise = laplace(0,1/epsilon,gr1*gr1).astype(np.int32)
    lap_noise = lap_noise.reshape(gr1,gr1)

    # ind=0 → 从对角线开始提取;ind=1 → 从对角线下一列开始提取（即严格上三角，不含对角线）
    ga = get_uptri_arr(mat_one_level,ind=1)
    ga_noise = ga + laplace(0,1/epsilon,len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    mat_one_level_noise = get_upmat(ga_noise_pp,gr1,ind=1)


    noise_diag = np.int32(mat_one_level.diagonal() + laplace(0,2/epsilon,len(mat_one_level)))

    # keep consistency
    # 消除数组中的负数（因为加入拉普拉斯噪声可能会出现负数）,尽量保持总和不变,输出结果是 非负整数数组，适合作为权重矩阵的上三角值
    noise_diag = FO_pp(noise_diag)
  
    mat_one_level_noise = np.triu(mat_one_level_noise,1)
    mat_one_level_noise = mat_one_level_noise + np.transpose(mat_one_level_noise)

    row,col = np.diag_indices_from(mat_one_level_noise) 
    mat_one_level_noise[row,col] = noise_diag
    mat_one_level_noise[mat_one_level_noise<0] = 0

    mat_one_level_graph = nx.from_numpy_array(mat_one_level_noise,create_using=nx.Graph)
    
    # 这块产生的结果就是，对角线是社区自己的权重，矩阵对称，并且加噪
    # Apply the Louvain method

    # Louvain算法
    mat_new_par = community.best_partition(mat_one_level_graph,resolution=t)
    gr2 = max(mat_new_par.values()) + 1 
    mat_new_pv = np.array(list(mat_new_par.values()))
    mat_final_pvs = []
    for i in range(gr2):
        pv = np.where(mat_new_pv==i)[0]
        mat_final_pv = []
        for j in range(len(pv)):
            pvj = pv[j]
            mat_final_pv.extend(mat0_par3_pvs[pvj])
        mat_final_pvs.append(mat_final_pv)

    label1 = np.zeros([len(mat0)],dtype=np.int32)
    for i in range(len(mat_final_pvs)):
        label1[mat_final_pvs[i]] = i

    return label1


def community_init_dp_degree_adaptive(mat0, mat0_graph, epsilon, nr=None, t=1.0, alpha=0.3):
    """
    DP度排序 + 自适应分组版本

    参数：
    epsilon : 总预算（用于初始化阶段）
    nr      : 可选（不再使用固定分组）
    alpha   : 分给“度排序”的预算比例（推荐 0.2~0.4）
    """

    n = len(mat0)

    # ===== Step0: 隐私预算拆分 =====
    e_deg = alpha * epsilon
    e_init = (1 - alpha) * epsilon

    # ===== Step1: DP度序列 =====
    deg = np.sum(mat0, axis=1)

    deg_noise = deg + laplace(0, 1 / e_deg, n)

    # 排序（降序）
    sort_idx = np.argsort(-deg_noise)

    # ===== Step2: 自适应分组（√n）=====
    group_size = int(np.sqrt(n))
    group_size = max(group_size, 1)

    num_groups = int(np.ceil(n / group_size))

    g1 = np.zeros(n, dtype=np.int32)
    for i in range(n):
        g1[sort_idx[i]] = i // group_size

    mat0_par3 = {i: int(g1[i]) for i in range(n)}
    gr1 = max(mat0_par3.values()) + 1

    # ===== Step3: 社区 → 节点集合 =====
    mat0_par3_pv = np.array(list(mat0_par3.values()))
    mat0_par3_pvs = []

    for i in range(gr1):
        pv = np.where(mat0_par3_pv == i)[0]
        mat0_par3_pvs.append(list(pv))

    # ===== Step4: 构建社区级图 =====
    mat_one_level = np.zeros([gr1, gr1])

    for i in range(gr1):
        pi = mat0_par3_pvs[i]
        mat_one_level[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, gr1):
            pj = mat0_par3_pvs[j]
            mat_one_level[i, j] = np.sum(mat0[np.ix_(pi, pj)])

    # ===== Step5: DP扰动 =====
    ga = get_uptri_arr(mat_one_level, ind=1)
    ga_noise = ga + laplace(0, 1 / e_init, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    mat_one_level_noise = get_upmat(ga_noise_pp, gr1, ind=1)

    noise_diag = np.int32(
        mat_one_level.diagonal() + laplace(0, 2 / e_init, len(mat_one_level))
    )
    noise_diag = FO_pp(noise_diag)

    mat_one_level_noise = np.triu(mat_one_level_noise, 1)
    mat_one_level_noise = mat_one_level_noise + mat_one_level_noise.T

    row, col = np.diag_indices_from(mat_one_level_noise)
    mat_one_level_noise[row, col] = noise_diag
    mat_one_level_noise[mat_one_level_noise < 0] = 0

    mat_one_level_graph = nx.from_numpy_array(mat_one_level_noise, create_using=nx.Graph)

    # ===== Step6: Louvain =====
    mat_new_par = community.best_partition(mat_one_level_graph, resolution=t)

    gr2 = max(mat_new_par.values()) + 1
    mat_new_pv = np.array(list(mat_new_par.values()))

    mat_final_pvs = []
    for i in range(gr2):
        pv = np.where(mat_new_pv == i)[0]
        mat_final_pv = []
        for j in range(len(pv)):
            mat_final_pv.extend(mat0_par3_pvs[pv[j]])
        mat_final_pvs.append(mat_final_pv)

    # ===== Step7: 输出标签 =====
    label1 = np.zeros(n, dtype=np.int32)
    for i in range(len(mat_final_pvs)):
        label1[mat_final_pvs[i]] = i

    return label1

def community_init_dp_neighbor_fixed(mat0, mat0_graph, epsilon, nr=None, t=1.0,
                                      alpha=0.3, beta=0.5, C=None):
    n = len(mat0)

    e_score = alpha * epsilon
    e_init  = (1 - alpha) * epsilon
    e_deg   = e_score * 0.5
    e_ns    = e_score * 0.5

    # Phase 1
    deg     = np.sum(mat0, axis=1).astype(float)
    d_noisy = deg + laplace(0, 2.0 / e_deg, n)

    # C 对数增长
    if C is None:
        C = int(np.sqrt(n) * (1.0 + np.log(max(1.0, epsilon)))) + 1
    d_clipped = np.clip(d_noisy, 0.0, float(C))

    # Phase 2
    ns_true  = mat0 @ d_clipped
    ns_noisy = ns_true + laplace(0, 2.0 * C / e_ns, n)

    score = d_noisy + beta * ns_noisy

    # 纯得分排序，不做任何分层
    # group_size = max(1, int(np.sqrt(n)))
    group_size = nr if nr is not None else max(1, int(np.sqrt(n)))


    sort_idx   = np.argsort(-score)

    g1 = np.zeros(n, dtype=np.int32)
    for i in range(n):
        g1[sort_idx[i]] = i // group_size

    mat0_par3 = {i: int(g1[i]) for i in range(n)}
    gr1 = max(mat0_par3.values()) + 1

    mat0_par3_pv  = np.array(list(mat0_par3.values()))
    mat0_par3_pvs = []
    for i in range(gr1):
        pv = np.where(mat0_par3_pv == i)[0]
        mat0_par3_pvs.append(list(pv))

    mat_one_level = np.zeros([gr1, gr1])
    for i in range(gr1):
        pi = mat0_par3_pvs[i]
        mat_one_level[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, gr1):
            pj = mat0_par3_pvs[j]
            mat_one_level[i, j] = np.sum(mat0[np.ix_(pi, pj)])

    ga          = get_uptri_arr(mat_one_level, ind=1)
    ga_noise    = ga + laplace(0, 1.0 / e_init, len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    mat_one_level_noise = get_upmat(ga_noise_pp, gr1, ind=1)

    noise_diag = np.int32(mat_one_level.diagonal() + laplace(0, 2.0 / e_init, gr1))
    noise_diag = FO_pp(noise_diag)

    mat_one_level_noise = np.triu(mat_one_level_noise, 1)
    mat_one_level_noise = mat_one_level_noise + mat_one_level_noise.T
    row, col = np.diag_indices_from(mat_one_level_noise)
    mat_one_level_noise[row, col] = noise_diag
    mat_one_level_noise[mat_one_level_noise < 0] = 0

    mat_one_level_graph = nx.from_numpy_array(mat_one_level_noise, create_using=nx.Graph)
    mat_new_par = community.best_partition(mat_one_level_graph, resolution=t)

    gr2 = max(mat_new_par.values()) + 1
    mat_new_pv    = np.array(list(mat_new_par.values()))
    mat_final_pvs = []
    for i in range(gr2):
        pv = np.where(mat_new_pv == i)[0]
        mat_final_pv = []
        for j in range(len(pv)):
            mat_final_pv.extend(mat0_par3_pvs[pv[j]])
        mat_final_pvs.append(mat_final_pv)

    label1 = np.zeros(n, dtype=np.int32)
    for i in range(len(mat_final_pvs)):
        label1[mat_final_pvs[i]] = i

    return label1









# 把矩阵的上三角部分展平成一维数组，方便加噪声。
def get_uptri_arr(mat_init,ind=0):
    a = len(mat_init)
    res = []
    for i in range(a):
        dat = mat_init[i][i+ind:]
        res.extend(dat)
    arr = np.array(res)
    return arr

# 把一维数组还原成对称矩阵的上三角，恢复社区权重矩阵。
def get_upmat(arr,k,ind=0):
    mat = np.zeros([k,k],dtype=np.int32)
    left = 0
    for i in range(k):
        delta = k - i - ind
        mat[i,i+ind:] = arr[left:left+delta]
        left = left + delta
        
    return mat

# Post processing
# 消除数组中的负数（因为加入拉普拉斯噪声可能会出现负数）,尽量保持总和不变,输出结果是 非负整数数组，适合作为权重矩阵的上三角值
def FO_pp(data_noise,type='norm_sub'):
    if type == 'norm_sub':
        data = norm_sub_deal(data_noise)
        
    if type == 'norm_mul':
        data = norm_mul_deal(data_noise)
    
    return data
# 让噪声数据非负化，同时尽量保持总和不变，用于隐私保护后的数值修正
def norm_sub_deal(data):
    data = np.array(data,dtype=np.int32)
    data_min = np.min(data)
    data_sum = np.sum(data)
    delta_m = 0 - data_min
    
    if delta_m > 0:
        dm = 100000000
        data_seq = np.zeros([len(data)],dtype=np.int32)
        for i in range(0,delta_m):
            data_t = data - i
            data_t[data_t<0] = 0
            data_t_s = np.sum(data_t)
            dt = np.abs(data_t_s - data_sum)
            if dt < dm:
                dm = dt
                data_seq = data_t
                if dt == 0:
                    break
                
    else:
        data_seq = data
    return data_seq
        


# 根据节点度序列，随机生成社区内部的边，用于构造图结构
# generate graph(intra edges) based on degree sequence
def generate_intra_edge(dd1,div=1):
    dd1 = np.array(dd1,dtype=np.int32)
    dd1[dd1<0] = 0
    dd1_len = len(dd1)
    dd1_p = dd1.reshape(dd1_len,1) * dd1.reshape(1,dd1_len)
    s1 = np.sum(dd1)

    dd1_res = np.zeros([dd1_len,dd1_len],dtype=np.int8)
    if s1 > 0:
        batch_num = int(dd1_len / div)
        begin_id = 0
        for i in range(div):
            if i == div-1:
                batch_n = dd1_len - begin_id
                dd1_r = np.random.randint(0,high=s1,size=(batch_n,dd1_len))
                res = dd1_p[begin_id:,:] - dd1_r
                res[res>0] = 1
                res[res<1] = 0
                dd1_res[begin_id:,:] = res
            else:
                dd1_r = np.random.randint(0,high=s1,size=(batch_num,dd1_len))
                res = dd1_p[begin_id:begin_id+batch_num,:] - dd1_r
                res[res>0] = 1
                res[res<1] = 0
                dd1_res[begin_id:begin_id+batch_num,:] = res
                begin_id = begin_id + batch_num
    
    # make sure the final adjacency matrix is symmetric
    dd1_out = np.triu(dd1_res,1)
    dd1_out = dd1_out + np.transpose(dd1_out)
    return dd1_out

# 计算图的直径（最远两点距离），衡量图的连通紧凑程度。
# calculate the diameter
def cal_diam(mat):
    mat_graph = nx.from_numpy_array(mat,create_using=nx.Graph)
    max_diam = 0
    for com in nx.connected_components(mat_graph):
        com_list = list(com)
        mat_sub = mat[np.ix_(com_list,com_list)]
        sub_g = nx.from_numpy_array(mat_sub,create_using=nx.Graph)
        diam = nx.diameter(sub_g)
        if diam > max_diam:
            max_diam = diam
    return max_diam

# 计算两个节点排序的前 k 个重叠率，常用于种子节点一致性对比
# calculate the overlap 
def cal_overlap(la,lb,k):
    la = la[:k]
    lb = lb[:k]
    la_s = set(la)
    lb_s = set(lb)
    num = len(la_s & lb_s)
    rate = num / k
    return rate

# 计算两个分布的KL 散度，衡量分布差异
# calculate the KL divergence
def cal_kl(A,B): 
    p = A / sum(A)
    q = B / sum(B)
    if A.shape[0] > B.shape[0]:
        q = np.pad(q,(0,p.shape[0]-q.shape[0]),'constant',constant_values=(0,0))
    elif A.shape[0] < B.shape[0]:
        p = np.pad(p,(0,q.shape[0]-p.shape[0]),'constant',constant_values=(0,0))
    kl = p * np.log((p+np.finfo(np.float64).eps)/(q+np.finfo(np.float64).eps))
    kl = np.sum(kl)
    return kl

# 计算相对误差，衡量两组数据的接近程度。
# calculate the RE
def cal_rel(A,B): 
    eps = 0.000000000000001
    A = np.float64(A)
    B = np.float64(B)
    #eps = np.float64(eps)
    res = abs((A-B)/(A+eps))
    return res

# 计算均方误差，衡量预测值与真实值的差距。
# calculate the MSE
def cal_MSE(A,B): 
    res = np.mean((A-B)**2)
    return res

# 计算平均绝对误差，简单直观的误差指标
# calculate the MAE
def cal_MAE(A,B,k=None): 
    if k== None:
        res = np.mean(abs(A-B))
    else:
        a = np.array(A[:k])
        b = np.array(B[:k])
        res = np.mean(abs(a-b))
    return res

# 把邻接矩阵还原成原始节点 ID，并保存为边列表文本文件。
def write_edge_txt(mat0,mid,file_name):
    a0 = np.where(mat0==1)[0]
    a1 = np.where(mat0==1)[1]
    with open(file_name,'w+') as f:
        for i in range(len(a0)):
            f.write('%d\t%d\n'%(mid[a0[i]],mid[a1[i]]))

# 实现一个优先队列（堆），用于贪心算法选择最优节点
class PriorityQueue(object):
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])

# 影响力最大化算法：用度折扣策略选出k 个最有影响力的种子节点。
def degreeDiscountIC(G, k, p=0.01):

    S = []
    dd = PriorityQueue() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S
# 执行独立级联模型（IC 模型），模拟种子节点在图中的传播扩散过程。
def runIC (G, S, p = 0.01):

    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes

    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T
# 从图文件中读取图，直接选出指定数量的影响力最大种子节点
def find_seed(graph_path,seed_size=20):
    
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
        
    
    S = degreeDiscountIC(G, seed_size)
    return S


# 输入种子节点，多次模拟传播，返回平均扩散规模，评估影响力大小。
def cal_spread(graph_path,S_all,p=0.01,seed_size=20,iterations=100):
    
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            # print('u:%s,v:%s'%(u,v))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
           

    #calculate initial set
    
    if seed_size <= len(S_all):
        S = S_all[:seed_size]
    else:
        print('seed_size is too large.')
        S = S_all

    
    avg = 0
    for i in range(iterations):
        T = runIC(G, S, p)
        avg += float(len(T))/iterations

    avg_final = int(round(avg))

    return avg_final



# 新加函数，后处理
def post_process_prune(mat2, pvs, dd_s, ev_mat, comm_n):
    """
    后处理剪枝：将合成图中超出噪声目标的多余边删除
    
    mat2   : 合成图邻接矩阵
    pvs    : 每个社区的节点列表
    dd_s   : 每个社区的目标度序列（已加噪后处理过的）
    ev_mat : 社区间目标边数矩阵（已加噪后处理过的）
    comm_n : 社区数量
    """
    mat2 = mat2.copy().astype(np.int8)

    # ---------- Part 1: 社区内部剪枝 ----------
    # 目标：每个节点在其社区内的度 <= dd_s中对应的目标值
    for i in range(comm_n):
        nodes = pvs[i]
        target_deg = dd_s[i]  # 长度与nodes相同

        for local_idx, node in enumerate(nodes):
            # 当前节点在社区内的实际度
            intra_neighbors = [
                v for v in nodes
                if v != node and mat2[node, v] == 1
            ]
            actual_deg = len(intra_neighbors)
            target = int(target_deg[local_idx])

            if actual_deg > target:
                # 随机删除多余的边
                n_remove = actual_deg - target
                remove_candidates = intra_neighbors.copy()
                np.random.shuffle(remove_candidates)
                for v in remove_candidates[:n_remove]:
                    mat2[node, v] = 0
                    mat2[v, node] = 0

    # ---------- Part 2: 社区间剪枝 ----------
    # 目标：每对社区间的实际边数 <= ev_mat[i,j]
    for i in range(comm_n):
        for j in range(i + 1, comm_n):
            pi = pvs[i]
            pj = pvs[j]
            target_ev = int(ev_mat[i, j])

            # 找出当前社区间所有边
            actual_edges = [
                (u, v)
                for u in pi for v in pj
                if mat2[u, v] == 1
            ]
            actual_ev = len(actual_edges)

            if actual_ev > target_ev:
                n_remove = actual_ev - target_ev
                remove_idx = np.random.choice(
                    len(actual_edges), n_remove, replace=False
                )
                for idx in remove_idx:
                    u, v = actual_edges[idx]
                    mat2[u, v] = 0
                    mat2[v, u] = 0

    return mat2


def post_process_edge_swap(mat2, pvs, comm_n, n_iter_ratio=0.5):
    """
    在不改变度序列的前提下，通过边交换改善 cc 和 modularity
    
    策略：找到社区内"孤立边"(u,v)，将其替换为能构成三角形的边(u,w)
    - 孤立边：(u,v) 之间公共邻居为0，说明这条边"突兀"
    - 候选替换：(u,w) 有公共邻居，且 w 在同一社区
    
    mat2      : 合成图邻接矩阵
    pvs       : 社区节点列表
    comm_n    : 社区数量
    n_iter_ratio : 每个社区尝试交换次数占社区边数的比例
    """
    mat2 = mat2.copy().astype(np.int8)
    n = mat2.shape[0]
    
    for ci in range(comm_n):
        nodes = np.array(pvs[ci])
        if len(nodes) < 4:
            continue
        
        # 获取社区内子矩阵
        sub = mat2[np.ix_(nodes, nodes)]
        
        # 当前社区内所有边
        rows, cols_idx = np.where(np.triu(sub, 1) > 0)
        intra_edges = list(zip(rows, cols_idx))
        
        if len(intra_edges) == 0:
            continue
        
        n_iter = max(1, int(len(intra_edges) * n_iter_ratio))
        
        for _ in range(n_iter):
            # 随机选一条社区内边 (u_local, v_local)
            edge_idx = np.random.randint(len(intra_edges))
            u_local, v_local = intra_edges[edge_idx]
            
            # 计算 u_local 和 v_local 的公共邻居数
            u_neighbors = set(np.where(sub[u_local] > 0)[0])
            v_neighbors = set(np.where(sub[v_local] > 0)[0])
            common = u_neighbors & v_neighbors
            
            # 只对"孤立边"（无公共邻居）尝试替换
            if len(common) > 0:
                continue
            
            # 找 u_local 的邻居中，有没有与某个非邻居有公共邻居
            # 候选 w：与 u_local 不相连，但与 u_local 的某邻居相连
            u_non_neighbors = set(range(len(nodes))) - u_neighbors - {u_local}
            
            best_w = None
            best_score = 0
            
            # 在候选中找公共邻居最多的 w
            for w_local in u_non_neighbors:
                w_neighbors = set(np.where(sub[w_local] > 0)[0])
                score = len(u_neighbors & w_neighbors)  # u和w的公共邻居数
                if score > best_score:
                    best_score = score
                    best_w = w_local
            
            if best_w is None or best_score == 0:
                continue
            
            # 执行交换：删除 (u,v)，添加 (u,w)
            # 度序列不变（u 的度不变，v 少一条边，w 多一条边）
            # 为了严格保持度序列，需要同时找 v 的补偿
            # 简化版：直接交换，接受轻微度变化
            u_global = nodes[u_local]
            v_global = nodes[v_local]
            w_global = nodes[best_w]
            
            # 删除孤立边，添加有三角结构的边
            sub[u_local, v_local] = 0
            sub[v_local, u_local] = 0
            sub[u_local, best_w] = 1
            sub[best_w, u_local] = 1
            
            mat2[u_global, v_global] = 0
            mat2[v_global, u_global] = 0
            mat2[u_global, w_global] = 1
            mat2[w_global, u_global] = 1
            
            # 更新边列表
            intra_edges[edge_idx] = (u_local, best_w)
    
    # ---- 跨社区边清理：删除模块度贡献为负的跨社区边 ----
    m_total = np.sum(mat2) / 2
    if m_total == 0:
        return mat2
    
    degree = np.sum(mat2, axis=1)
    
    for ci in range(comm_n):
        for cj in range(ci + 1, comm_n):
            pi = pvs[ci]
            pj = pvs[cj]
            
            # 找所有跨社区边
            cross_edges = [
                (u, v) for u in pi for v in pj
                if mat2[u, v] == 1
            ]
            
            remove_list = []
            for (u, v) in cross_edges:
                # 模块度贡献：实际连接 - 期望连接
                # 跨社区边的模块度贡献为负（期望值）
                expected = (degree[u] * degree[v]) / (2 * m_total)
                # 贡献为 1/2m * (A_uv - k_u*k_v/2m)
                # 对于跨社区边，A_uv=1，贡献为正值
                # 但从模块度角度，跨社区连接降低了模块度
                # 删除高期望值的跨社区边（本来就"应该"连接的，保留；不该连的，删）
                if expected > 1.0:  # 期望边数>1说明这条边很"自然"，保留
                    continue
                remove_list.append((u, v))
            
            # 随机删除一部分（避免过度删除破坏度分布）
            if len(remove_list) > 0:
                n_remove = max(0, len(cross_edges) - max(1, len(cross_edges) // 2))
                if n_remove > len(remove_list):
                    n_remove = len(remove_list)
                chosen = np.random.choice(len(remove_list), n_remove, replace=False)
                for idx in chosen:
                    u, v = remove_list[idx]
                    mat2[u, v] = 0
                    mat2[v, u] = 0
    
    return mat2







