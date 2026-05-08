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
all_nmi_arr= [np.float64(0.2908909016082716), np.float64(0.2987103069185862), np.float64(0.314067719606634), np.float64(0.3247204693351798), np.float64(0.32171812194308363), np.float64(0.35048679609115063), np.float64(0.34858862417135217)]
all_evc_overlap= [np.float64(0.19090909090909092), np.float64(0.6636363636363637), np.float64(0.7454545454545454), np.float64(0.8863636363636364), np.float64(0.8545454545454547), np.float64(0.9045454545454545), np.float64(0.8772727272727273)]
all_evc_MAE= [np.float64(0.04412447860204514), np.float64(0.009241496872684014), np.float64(0.0039002997774071494), np.float64(0.0023940331328327455), np.float64(0.0032202531575302056), np.float64(0.002882452529310005), np.float64(0.002067967344939292)]
all_deg_kl= [np.float64(4.334981432000461), np.float64(2.5530016814851666), np.float64(1.637187817704684), np.float64(1.4538080475276334), np.float64(1.328197769673062), np.float64(1.389320406564931), np.float64(1.2233100048876058)]
all_diam_rel= [np.float64(0.14545454545454545), np.float64(0.09999999999999999), np.float64(0.10909090909090909), np.float64(0.1818181818181818), np.float64(0.20909090909090908), np.float64(0.3), np.float64(0.33636363636363636)]
all_cc_rel= [np.float64(0.04957779454160417), np.float64(0.3468064173872838), np.float64(0.2229681932426239), np.float64(0.12703545188676502), np.float64(0.10467575701997515), np.float64(0.0573954411500476), np.float64(0.04985170564244764)]
all_mod_rel= [np.float64(0.12722280991453894), np.float64(0.26351335042043617), np.float64(0.1874530416173487), np.float64(0.18832007342919857), np.float64(0.19942578664135396), np.float64(0.12106606785357558), np.float64(0.10499972631315502)]



nmi_1, overlap_1, mae_1, deg_kl_1, diam_1, cc_1, mod_1 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.021744460768912923), np.float64(0.26697228754693003), np.float64(0.31640900640200814), np.float64(0.339525589001224), np.float64(0.33168249839286357), np.float64(0.33490247407796775), np.float64(0.3502723489992783)]
all_evc_overlap= [np.float64(0.20909090909090905), np.float64(0.5409090909090909), np.float64(0.8272727272727274), np.float64(0.8818181818181818), np.float64(0.8590909090909091), np.float64(0.859090909090909), np.float64(0.9181818181818182)]
all_evc_MAE= [np.float64(0.03836183750297534), np.float64(0.009073027962143178), np.float64(0.002501977825253009), np.float64(0.001975453390856481), np.float64(0.0035280437278358895), np.float64(0.0033471020992643674), np.float64(0.0034708558577751596)]
all_deg_kl= [np.float64(3.398445301754414), np.float64(2.315291531093883), np.float64(1.4960373904722184), np.float64(1.4079300370892063), np.float64(1.2783947761142598), np.float64(1.2708176970776326), np.float64(1.245977123638417)]
all_diam_rel= [np.float64(0.3727272727272727), np.float64(0.08181818181818182), np.float64(0.13636363636363635), np.float64(0.28181818181818175), np.float64(0.3181818181818181), np.float64(0.2727272727272727), np.float64(0.21818181818181817)]
all_cc_rel= [np.float64(0.45528793861451033), np.float64(0.28235619887496255), np.float64(0.1896920709175776), np.float64(0.1433643279265251), np.float64(0.08145147986053591), np.float64(0.049008932282785723), np.float64(0.08899471637977095)]
all_mod_rel= [np.float64(0.31779145621236804), np.float64(0.19501133571913903), np.float64(0.21243498311628137), np.float64(0.13131788271460804), np.float64(0.203103388528059), np.float64(0.136856391193966), np.float64(0.16574366994979353)]

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