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
all_nmi_arr= [np.float64(0.3027051158948042), np.float64(0.2973242672177777), np.float64(0.3103779148603587), np.float64(0.3286398102222151), np.float64(0.33234293695197), np.float64(0.33114337310824016), np.float64(0.3451292003394804)]
all_evc_overlap= [np.float64(0.1318181818181818), np.float64(0.6363636363636364), np.float64(0.8409090909090908), np.float64(0.8681818181818182), np.float64(0.8863636363636364), np.float64(0.8409090909090908), np.float64(0.8727272727272727)]
all_evc_MAE= [np.float64(0.013338943446827997), np.float64(0.005442145600985836), np.float64(0.004191042654781886), np.float64(0.0029677948750459664), np.float64(0.002221566989725923), np.float64(0.0026366641254971807), np.float64(0.00217838236047062)]
all_deg_kl= [np.float64(3.163721673234063), np.float64(1.7226053745756034), np.float64(1.4500074490937702), np.float64(1.4223073914013467), np.float64(1.2666278737741814), np.float64(1.1753859508459623), np.float64(1.168691570931879)]
all_diam_rel= [np.float64(0.309090909090909), np.float64(0.37272727272727274), np.float64(0.41818181818181815), np.float64(0.41818181818181815), np.float64(0.41818181818181815), np.float64(0.4), np.float64(0.43636363636363634)]
all_cc_rel= [np.float64(0.9127980626849524), np.float64(0.566592349980076), np.float64(0.3292742351496746), np.float64(0.24642810392924738), np.float64(0.1880010594651657), np.float64(0.08718012296373835), np.float64(0.10450822877827222)]
all_mod_rel= [np.float64(0.09789439422439651), np.float64(0.044334640260665664), np.float64(0.10063716615823289), np.float64(0.0885871933069485), np.float64(0.09218981246728428), np.float64(0.17525093009147108), np.float64(0.10193689823270484)]

nmi_1, overlap_1, mae_1, deg_kl_1, diam_1, cc_1, mod_1 = \
    all_nmi_arr, all_evc_overlap, all_evc_MAE, all_deg_kl, all_diam_rel, all_cc_rel, all_mod_rel


# ===================== 数据2（你刚发的新数据） =====================
all_nmi_arr= [np.float64(0.27012975460150523), np.float64(0.28084115779967567), np.float64(0.3100725003328441), np.float64(0.30439034357907807), np.float64(0.33764843854639665), np.float64(0.33662088395961487), np.float64(0.34513579555029417)]
all_evc_overlap= [np.float64(0.23181818181818178), np.float64(0.5727272727272728), np.float64(0.8545454545454545), np.float64(0.8454545454545455), np.float64(0.8727272727272727), np.float64(0.8818181818181818), np.float64(0.8727272727272727)]
all_evc_MAE= [np.float64(0.009860755526844437), np.float64(0.00701408171465841), np.float64(0.003900287121553708), np.float64(0.0031372100672026467), np.float64(0.0021816254196855957), np.float64(0.0026846985575327336), np.float64(0.0021509927700387697)]
all_deg_kl= [np.float64(2.8561883613438477), np.float64(1.7149554329723813), np.float64(1.3895226029760654), np.float64(1.2917417928031227), np.float64(1.211800545283835), np.float64(1.1838565772647924), np.float64(1.198902963383761)]
all_diam_rel= [np.float64(0.27272727272727265), np.float64(0.36363636363636365), np.float64(0.43636363636363634), np.float64(0.43636363636363634), np.float64(0.43636363636363634), np.float64(0.42727272727272725), np.float64(0.40909090909090906)]
all_cc_rel= [np.float64(0.6739196616931176), np.float64(0.4778469678260772), np.float64(0.3052522348194982), np.float64(0.15429963912576983), np.float64(0.16284199997869425), np.float64(0.08829897841275093), np.float64(0.09571201816998029)]
all_mod_rel= [np.float64(0.04212409220657555), np.float64(0.09140788665194638), np.float64(0.06649542586487568), np.float64(0.14107704880339808), np.float64(0.07758565893028986), np.float64(0.11497702638823011), np.float64(0.10251183469471359)]


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