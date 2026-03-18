import numpy as np
import matplotlib.pyplot as plt
import os  # 用于创建文件夹

# ===================== 数据1（你原来的数据） =====================

save_dir = "./charts/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name = "Chamelon"
eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

all_nmi_arr= [np.float64(0.1110612118369861), np.float64(0.1818985870850439), np.float64(0.22088691192112905), np.float64(0.2292952954529235), np.float64(0.25913152202119843), np.float64(0.2816141110976641), np.float64(0.26636613397444336)]
all_evc_overlap= [np.float64(0.10454545454545454), np.float64(0.7090909090909092), np.float64(0.8227272727272729), np.float64(0.8545454545454547), np.float64(0.8999999999999998), np.float64(0.8818181818181816), np.float64(0.890909090909091)]
all_evc_MAE= [np.float64(0.025960044534886227), np.float64(0.007715436928568695), np.float64(0.0021008341274811365), np.float64(0.002257260987425339), np.float64(0.0028615100367108803), np.float64(0.0021801249937592774), np.float64(0.003520006290335854)]
all_deg_kl= [np.float64(2.355720322260782), np.float64(1.6612702929836327), np.float64(1.260225706819194), np.float64(1.1966983303654635), np.float64(1.0839197916723988), np.float64(1.1210012180224527), np.float64(1.0455438888419306)]
all_diam_rel= [np.float64(0.5454545454545453), np.float64(0.509090909090909), np.float64(0.42727272727272725), np.float64(0.41818181818181815), np.float64(0.39090909090909093), np.float64(0.30909090909090897), np.float64(0.3454545454545454)]
all_cc_rel= [np.float64(0.8458845482691132), np.float64(0.1685981589428842), np.float64(0.10627183364603736), np.float64(0.09965077711627111), np.float64(0.07609991107920681), np.float64(0.0571345802719654), np.float64(0.060784705081233514)]
all_mod_rel= [np.float64(0.5842826883686449), np.float64(0.3407112793374269), np.float64(0.26798304821850494), np.float64(0.2085764110046882), np.float64(0.18339000535120617), np.float64(0.18924091360753215), np.float64(0.21180063143220437)]



nmi_1 = all_nmi_arr
overlap_1 =all_evc_overlap
mae_1 =all_evc_MAE
deg_kl_1 =all_deg_kl
diam_1 =all_diam_rel
cc_1 =all_cc_rel
mod_1 =all_mod_rel

# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.19641698107941002), np.float64(0.20789139243600055), np.float64(0.21533615863693084), np.float64(0.24523750222970947), np.float64(0.22422012877643188), np.float64(0.2567835069555277), np.float64(0.23531529257557163)]
all_evc_overlap= [np.float64(0.7136363636363636), np.float64(0.8045454545454545), np.float64(0.8545454545454545), np.float64(0.8727272727272727), np.float64(0.8454545454545455), np.float64(0.8545454545454545), np.float64(0.8772727272727272)]
all_evc_MAE= [np.float64(0.006259273384191942), np.float64(0.0018545251815168615), np.float64(0.0013310956204373781), np.float64(0.0013648222750258046), np.float64(0.0011886007427862875), np.float64(0.001286940391844667), np.float64(0.0012193906344554139)]
all_deg_kl= [np.float64(1.4324070111790739), np.float64(1.1519356870999673), np.float64(1.0283868486381622), np.float64(1.111977913476211), np.float64(1.0259412156205618), np.float64(1.0371198583263594), np.float64(1.002509229359971)]
all_diam_rel= [np.float64(0.44545454545454544), np.float64(0.42727272727272714), np.float64(0.42727272727272725), np.float64(0.3909090909090909), np.float64(0.4), np.float64(0.3545454545454545), np.float64(0.39999999999999997)]
all_cc_rel= [np.float64(0.031378299437037856), np.float64(0.10288210379622051), np.float64(0.11314062885422112), np.float64(0.10610187227795079), np.float64(0.08482082127089137), np.float64(0.09054264899267314), np.float64(0.06290000566340168)]
all_mod_rel= [np.float64(0.21400162352583685), np.float64(0.17625230869534922), np.float64(0.15727669587795395), np.float64(0.1360515909432028), np.float64(0.13185031994681198), np.float64(0.11345620251525293), np.float64(0.12002594318087126)]
# ===================== 指标配对 =====================

nmi_2 = all_nmi_arr
overlap_2 =all_evc_overlap
mae_2 =all_evc_MAE
deg_kl_2 =all_deg_kl
diam_2 =all_diam_rel
cc_2 =all_cc_rel
mod_2 =all_mod_rel

metrics = [
    ("NMI", nmi_1, nmi_2),
    ("EVC Overlap", overlap_1, overlap_2),
    ("EVC MAE", mae_1, mae_2),
    ("Degree KL", deg_kl_1, deg_kl_2),
    ("Diameter Relative", diam_1, diam_2),
    ("Clustering Coefficient Relative", cc_1, cc_2),
    ("Modularity Relative", mod_1, mod_2)
]

# ===================== 绘图 =====================
plt.rcParams['font.size'] = 12
color1 = '#2E86AB'  # 数据1颜色：蓝色
color2 = '#F95738'  # 数据2颜色：红色

for name, data1, data2 in metrics:
    plt.figure(figsize=(7, 4))
    
    # 画两条线
    plt.plot(eps, data1, marker='o', linewidth=2, markersize=6, color=color1, label='normal')
    plt.plot(eps, data2, marker='s', linewidth=2, markersize=6, color=color2, label='Optimization')
    
    plt.title(f"{dataset_name} - {name} vs ε", fontweight='bold')
    plt.xlabel("ε")
    plt.ylabel(name)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高清300dpi
    print(f"已保存: {save_path}")

    plt.show()