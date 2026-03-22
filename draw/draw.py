import numpy as np
import matplotlib.pyplot as plt
import os  # 用于创建文件夹

# ===================== 数据1（你原来的数据） =====================

save_dir = "./charts/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name = "Chamelon"
eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

all_nmi_arr= [np.float64(0.10526870981676559), np.float64(0.16857673701970527), np.float64(0.22277931721013577), np.float64(0.2433260953880237), np.float64(0.2634230492279347), np.float64(0.27653664804611816), np.float64(0.26831634877029376)]
all_evc_overlap= [np.float64(0.15), np.float64(0.5863636363636364), np.float64(0.8454545454545455), np.float64(0.8227272727272729), np.float64(0.8636363636363636), np.float64(0.8818181818181816), np.float64(0.8818181818181821)] 
all_evc_MAE= [np.float64(0.02397476386084223), np.float64(0.007920292233807394), np.float64(0.001846043957539551), np.float64(0.0026364400338896024), np.float64(0.0023059598326851077), np.float64(0.002526379917922636), np.float64(0.003120036097720605)]
all_deg_kl= [np.float64(2.358138791906158), np.float64(1.709091395278491), np.float64(1.1884001971623517), np.float64(1.1456465222005425), np.float64(1.133057407428656), np.float64(1.0449751568896866), np.float64(0.9962525032591498)]
all_diam_rel= [np.float64(0.5454545454545453), np.float64(0.48181818181818165), np.float64(0.39090909090909093), np.float64(0.44545454545454544), np.float64(0.3818181818181817), np.float64(0.3545454545454545), np.float64(0.32727272727272727)]
all_cc_rel= [np.float64(0.8236013406769389), np.float64(0.24115588687731132), np.float64(0.08336667350628123), np.float64(0.06852707229124066), np.float64(0.07331441177652429), np.float64(0.04752155845381608), np.float64(0.08066886900598663)]
all_mod_rel= [np.float64(0.5659459731137823), np.float64(0.37044396353984155), np.float64(0.22459102438326978), np.float64(0.21469693628019543), np.float64(0.19143063367803728), np.float64(0.16067173993489706), np.float64(0.2318674349078159)]


nmi_1 = all_nmi_arr
overlap_1 =all_evc_overlap
mae_1 =all_evc_MAE
deg_kl_1 =all_deg_kl
diam_1 =all_diam_rel
cc_1 =all_cc_rel
mod_1 =all_mod_rel

# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.1869253932766368), np.float64(0.16012252293023482), np.float64(0.1980408259939664), np.float64(0.2112305058247647), np.float64(0.24099992799582037), np.float64(0.22426160321912506), np.float64(0.21910803422678593)]
all_evc_overlap= [np.float64(0.45909090909090916), np.float64(0.4454545454545455), np.float64(0.6363636363636364), np.float64(0.5954545454545455), np.float64(0.6681818181818182), np.float64(0.6363636363636364), np.float64(0.6363636363636365)]
all_evc_MAE= [np.float64(0.010725311989388914), np.float64(0.011904602327510718), np.float64(0.0032305732283285774), np.float64(0.004584723678551405), np.float64(0.0029820385775883107), np.float64(0.002287181424467024), np.float64(0.003882441237598092)]       
all_deg_kl= [np.float64(1.8279524337747382), np.float64(1.609580627896553), np.float64(1.4301987905644586), np.float64(1.4030421334076744), np.float64(1.3442782545430971), np.float64(1.264824317051005), np.float64(1.3089075018114782)]
all_diam_rel= [np.float64(0.47272727272727266), np.float64(0.5181818181818182), np.float64(0.509090909090909), np.float64(0.46363636363636357), np.float64(0.43636363636363634), np.float64(0.47272727272727255), np.float64(0.48181818181818165)]
all_cc_rel= [np.float64(0.4080461256272847), np.float64(0.1620621995175383), np.float64(0.12498363293366385), np.float64(0.1682708924357094), np.float64(0.19302097451281844), np.float64(0.13909638234726657), np.float64(0.12644354831034949)]
all_mod_rel= [np.float64(0.38157143752650735), np.float64(0.23344383746539946), np.float64(0.14957519296646649), np.float64(0.14095611922542017), np.float64(0.1236022453056214), np.float64(0.1168302502920475), np.float64(0.14554215091592096)]
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