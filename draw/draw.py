import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录，而不是运行时工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "charts")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("图表将保存到:", save_dir)  # 确认路径
dataset_name = "Chamelon"
eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

all_nmi_arr= [np.float64(0.10526870981676559), np.float64(0.16857673701970527), np.float64(0.22277931721013577), np.float64(0.2433260953880237), np.float64(0.2634230492279347), np.float64(0.27653664804611816), np.float64(0.26831634877029376)]
all_evc_overlap= [np.float64(0.15), np.float64(0.5863636363636364), np.float64(0.8454545454545455), np.float64(0.8227272727272729), np.float64(0.8636363636363636), np.float64(0.8818181818181816), np.float64(0.8818181818181821)] 
all_evc_MAE= [np.float64(0.02397476386084223), np.float64(0.007920292233807394), np.float64(0.001846043957539551), np.float64(0.0026364400338896024), np.float64(0.0023059598326851077), np.float64(0.002526379917922636), np.float64(0.003120036097720605)]
all_deg_kl= [np.float64(2.358138791906158), np.float64(1.709091395278491), np.float64(1.1884001971623517), np.float64(1.1456465222005425), np.float64(1.133057407428656), np.float64(1.0449751568896866), np.float64(0.9962525032591498)]
all_diam_rel= [np.float64(0.5454545454545453), np.float64(0.48181818181818165), np.float64(0.39090909090909093), np.float64(0.44545454545454544), np.float64(0.3818181818181817), np.float64(0.3545454545454545), np.float64(0.32727272727272727)]
all_cc_rel= [np.float64(0.8236013406769389), np.float64(0.24115588687731132), np.float64(0.08336667350628123), np.float64(0.06852707229124066), np.float64(0.07331441177652429), np.float64(0.04752155845381608), np.float64(0.08066886900598663)]
all_mod_rel= [np.float64(0.5659459731137823), np.float64(0.37044396353984155), np.float64(0.22459102438326978), np.float64(0.21469693628019543), np.float64(0.19143063367803728), np.float64(0.16067173993489706), np.float64(0.2318674349078159)]

nmi_1, overlap_1, mae_1, deg_kl_1, diam_1, cc_1, mod_1 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.12927277478162888), np.float64(0.22039620669330046), np.float64(0.23059417148767175), np.float64(0.26728189999274654), np.float64(0.2667787645430639), np.float64(0.27202676186146235), np.float64(0.2807249029771147)]
all_evc_overlap= [np.float64(0.32727272727272727), np.float64(0.6500000000000001), np.float64(0.759090909090909), np.float64(0.8545454545454545), np.float64(0.8181818181818181), np.float64(0.831818181818182), np.float64(0.8727272727272727)]
all_evc_MAE= [np.float64(0.017949918040597347), np.float64(0.005898370329408269), np.float64(0.006711207206148177), np.float64(0.0055874121401625195), np.float64(0.006396112699050584), np.float64(0.005451724638477334), np.float64(0.004229946576010611)]
all_deg_kl= [np.float64(2.8316222041283137), np.float64(1.4718477807678472), np.float64(1.2406959695142734), np.float64(1.2377980016307524), np.float64(1.1937261530256), np.float64(1.1676837107277493), np.float64(1.1494860314924484)]
all_diam_rel= [np.float64(0.4545454545454544), np.float64(0.29999999999999993), np.float64(0.3272727272727272), np.float64(0.23636363636363633), np.float64(0.20909090909090908), np.float64(0.23636363636363633), np.float64(0.2636363636363636)]
all_cc_rel= [np.float64(0.6999143117801654), np.float64(0.19434639166122375), np.float64(0.15844766323028264), np.float64(0.13932897996236523), np.float64(0.16274242665922906), np.float64(0.09653012322772145), np.float64(0.12001308304910956)]
all_mod_rel= [np.float64(0.4935185725098082), np.float64(0.2878296383763083), np.float64(0.2812748922169501), np.float64(0.2698683355213888), np.float64(0.27125006505718785), np.float64(0.21304252681625618), np.float64(0.19264535035594516)]

nmi_2, overlap_2, mae_2, deg_kl_2, diam_2, cc_2, mod_2 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


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