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
all_nmi_arr= [np.float64(0.013295227200766997), np.float64(0.08117853868148865), np.float64(0.15416388385580732), np.float64(0.1949297971679928), np.float64(0.1879503475568201), np.float64(0.22323118778435128), np.float64(0.23256396725269943)]
all_evc_overlap= [np.float64(0.19545454545454544), np.float64(0.5454545454545455), np.float64(0.8181818181818181), np.float64(0.8636363636363636), np.float64(0.8681818181818182), np.float64(0.8954545454545455), np.float64(0.8954545454545453)]
all_evc_MAE= [np.float64(0.02211329896887334), np.float64(0.006900099930398929), np.float64(0.0024655735159267037), np.float64(0.0025952328342435404), np.float64(0.0028506679155080083), np.float64(0.0024560286013010175), np.float64(0.0020676307780151753)]
all_deg_kl= [np.float64(14.044872758808088), np.float64(6.827744650739414), np.float64(3.6818209772585604), np.float64(1.8474605424274444), np.float64(1.301496774836441), np.float64(1.0342392686590134), np.float64(0.9329887419437958)]
all_diam_rel= [np.float64(0.6363636363636364), np.float64(0.609090909090909), np.float64(0.5545454545454545), np.float64(0.5), np.float64(0.48181818181818165), np.float64(0.42727272727272725), np.float64(0.43636363636363634)]
all_cc_rel= [np.float64(0.8382807283362161), np.float64(0.27006844637634136), np.float64(0.09868482922886719), np.float64(0.08986911991172179), np.float64(0.10865767051044875), np.float64(0.08351149830435937), np.float64(0.06512430237522035)]
all_mod_rel= [np.float64(0.6529255451725932), np.float64(0.4886789540564311), np.float64(0.40969447194464303), np.float64(0.33464506244321907), np.float64(0.3495202383288699), np.float64(0.28667248209085827), np.float64(0.26060115453366184)]

nmi_1, overlap_1, mae_1, deg_kl_1, diam_1, cc_1, mod_1 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.1215283032430419), np.float64(0.21758557277875581), np.float64(0.252560183356882), np.float64(0.27410077367507896), np.float64(0.2920053185233962), np.float64(0.3141355228148705), np.float64(0.2900907464668828)]
all_evc_overlap= [np.float64(0.08636363636363638), np.float64(0.5454545454545454), np.float64(0.7181818181818181), np.float64(0.7863636363636364), np.float64(0.8363636363636363), np.float64(0.8681818181818182), np.float64(0.8363636363636363)]
all_evc_MAE= [np.float64(0.02242207227422584), np.float64(0.00391276392637839), np.float64(0.0038741755825502946), np.float64(0.0037175621938789353), np.float64(0.003046896727291879), np.float64(0.0030073991440860036), np.float64(0.0042373247566549555)]
all_deg_kl= [np.float64(2.508135366705239), np.float64(1.2786822439295693), np.float64(1.2200329868931532), np.float64(1.0898362677501037), np.float64(1.0879689577480494), np.float64(1.1372311916010218), np.float64(1.074243188656917)]
all_diam_rel= [np.float64(0.509090909090909), np.float64(0.3818181818181818), np.float64(0.409090909090909), np.float64(0.3363636363636363), np.float64(0.24545454545454543), np.float64(0.22727272727272724), np.float64(0.3272727272727272)]
all_cc_rel= [np.float64(0.733868077695677), np.float64(0.18563478478205142), np.float64(0.082970369631946), np.float64(0.05534536561602447), np.float64(0.04120273758300055), np.float64(0.047712443113592115), np.float64(0.0974603540478736)]
all_mod_rel= [np.float64(0.5606926345184645), np.float64(0.23523608121364697), np.float64(0.19594802172523257), np.float64(0.15956925456303495), np.float64(0.13811481331618225), np.float64(0.11615356264359926), np.float64(0.19076763925816742)]
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