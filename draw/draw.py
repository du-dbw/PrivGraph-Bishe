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

all_nmi_arr= [np.float64(0.10523872122342862), np.float64(0.18201631385213246), np.float64(0.22179437408868896), np.float64(0.23811258043327704), np.float64(0.23898566478945232), np.float64(0.24988371588399044), np.float64(0.26023775730762505)]
all_evc_overlap= [np.float64(0.17272727272727273), np.float64(0.7181818181818181), np.float64(0.8136363636363635), np.float64(0.8454545454545455), np.float64(0.8681818181818182), np.float64(0.8954545454545453), np.float64(0.85)]
all_evc_MAE= [np.float64(0.02209230923017249), np.float64(0.005618313369060798), np.float64(0.002498791621058867), np.float64(0.0026336845112355407), np.float64(0.002759215272566287), np.float64(0.0024736850827629135), np.float64(0.0031899339051606967)]
all_deg_kl= [np.float64(2.2377775365969628), np.float64(1.5416550094583878), np.float64(1.3111725943117463), np.float64(1.1309561132258321), np.float64(1.0399450714934253), np.float64(1.0754773944859601), np.float64(1.0380993339203115)]
all_diam_rel= [np.float64(0.5454545454545453), np.float64(0.48181818181818176), np.float64(0.39090909090909093), np.float64(0.38181818181818183), np.float64(0.41818181818181815), np.float64(0.41818181818181815), np.float64(0.39999999999999997)]
all_cc_rel= [np.float64(0.8368572289676891), np.float64(0.19305410320045427), np.float64(0.1998470881142118), np.float64(0.12273428660896528), np.float64(0.05549090444139397), np.float64(0.08425864354059226), np.float64(0.08569504933399678)]
all_mod_rel= [np.float64(0.6006249964876619), np.float64(0.34017709199121343), np.float64(0.2946281049026968), np.float64(0.22527191798664542), np.float64(0.19093964419115178), np.float64(0.2395541994418727), np.float64(0.1918126211577144)]
nmi_1, overlap_1, mae_1, deg_kl_1, diam_1, cc_1, mod_1 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.014652510632868774), np.float64(0.08071574859574708), np.float64(0.1539515457054395), np.float64(0.1743988398884738), np.float64(0.19119114251546945), np.float64(0.2252864369169283), np.float64(0.23388235502932608)]
all_evc_overlap= [np.float64(0.15), np.float64(0.6681818181818182), np.float64(0.8272727272727272), np.float64(0.8227272727272726), np.float64(0.8545454545454545), np.float64(0.8272727272727274), np.float64(0.85)]
all_evc_MAE= [np.float64(0.019342802507042962), np.float64(0.0034027822441206187), np.float64(0.0020509272307490446), np.float64(0.0023982968895069597), np.float64(0.002703899724132495), np.float64(0.002945902891508408), np.float64(0.002733912478444264)]
all_deg_kl= [np.float64(10.650502624200037), np.float64(4.0515419695855135), np.float64(1.1046117695552264), np.float64(1.057270709468416), np.float64(0.975985672160939), np.float64(0.9286336699979287), np.float64(1.014666864618327)]
all_diam_rel= [np.float64(0.6363636363636364), np.float64(0.5454545454545455), np.float64(0.4545454545454544), np.float64(0.44545454545454544), np.float64(0.41818181818181815), np.float64(0.3818181818181818), np.float64(0.3909090909090909)]
all_cc_rel= [np.float64(0.8475837467896932), np.float64(0.21257773861641094), np.float64(0.056955250236754386), np.float64(0.07689813512673721), np.float64(0.0948388986569396), np.float64(0.08575628067911567), np.float64(0.04283042781497386)]
all_mod_rel= [np.float64(0.6371414426955037), np.float64(0.4393840597641863), np.float64(0.29584542391225105), np.float64(0.2892823489214461), np.float64(0.2907310486242557), np.float64(0.26247212759247673), np.float64(0.20257029571601876)]


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