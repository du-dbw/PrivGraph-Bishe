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
all_nmi_arr= [np.float64(0.3832126499123728), np.float64(0.3618956248295321), np.float64(0.35520631533694474), np.float64(0.35701122501859384), np.float64(0.36775871425087503), np.float64(0.36444087446354667), np.float64(0.3745512299164328)]
all_evc_overlap= [np.float64(0.16363636363636364), np.float64(0.6136363636363636), np.float64(0.8272727272727274), np.float64(0.8409090909090908), np.float64(0.8681818181818182), np.float64(0.8727272727272727), np.float64(0.8863636363636364)]
all_evc_MAE= [np.float64(0.057922619229165286), np.float64(0.009825200801491374), np.float64(0.0031815616516332216), np.float64(0.0023766724615138365), np.float64(0.0038686821826117933), np.float64(0.0018344313219260072), np.float64(0.0026842540349760525)]
all_deg_kl= [np.float64(5.264105343637711), np.float64(2.635100669618585), np.float64(1.7790333612086247), np.float64(1.5646980398231274), np.float64(1.5010362497132044), np.float64(1.2539158311357146), np.float64(1.3431163625762617)]
all_diam_rel= [np.float64(0.5545454545454545), np.float64(0.48181818181818176), np.float64(0.4454545454545453), np.float64(0.4636363636363635), np.float64(0.4636363636363635), np.float64(0.42727272727272725), np.float64(0.4636363636363635)]
all_cc_rel= [np.float64(0.2295932467968662), np.float64(0.3944185592612965), np.float64(0.26362310575825776), np.float64(0.18862063872351678), np.float64(0.13526494984646206), np.float64(0.1019099176988004), np.float64(0.0941939148736056)]
all_mod_rel= [np.float64(0.18990337392785933), np.float64(0.15573858184309924), np.float64(0.20838690863462767), np.float64(0.14628612978106662), np.float64(0.1690438682768478), np.float64(0.09903235008197077), np.float64(0.12391108823218026)]

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