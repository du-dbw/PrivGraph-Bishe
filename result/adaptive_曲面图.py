import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os, glob

# ===== 配置 =====
res_path = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'Chamelon'
epsilon = 2.0
e1_r = 0.3

metrics = ['nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']
metric_names = ['NMI (Community Discovery)', 'Overlap (Eigenvalue Nodes)', 'MAE (Eigenvalue)',
                'KL Divergence (Degree Distribution)', 'RE (Diameter)',
                'RE (Clustering Coefficient)', 'RE (Modularity)']
higher_better = [True, True, False, False, False, False, False]

# ===== 读取数据 =====
pattern = os.path.join(res_path, f'adaptive_{dataset_name}_eps{epsilon}_e1r{e1_r:.2f}_*.csv')
files = glob.glob(pattern)
print(f'搜索: {pattern}')
print(f'找到: {len(files)} 个文件')

if len(files) == 0:
    print('没找到文件！请检查路径和文件名')
    exit()

df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f'总记录数: {len(df_all)}')

grouped = df_all.groupby(['Q_noisy', 'e2_r'])[metrics].mean().reset_index()
print(f'有效组合: {len(grouped)} 个')

# ===== 创建输出文件夹 =====
save_dir = os.path.join(res_path, f'adapt_{dataset_name}_eps{epsilon}')
os.makedirs(save_dir, exist_ok=True)

# ===== 逐指标画图 =====
x_raw = grouped['Q_noisy'].values
y_raw = grouped['e2_r'].values

for idx, (metric, name, hb) in enumerate(zip(metrics, metric_names, higher_better)):
    z_raw = grouped[metric].values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 插值曲面
    xi = np.linspace(x_raw.min(), x_raw.max(), 50)
    yi = np.linspace(y_raw.min(), y_raw.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='cubic')

    cmap = 'coolwarm' if hb else 'coolwarm_r'
    ax.plot_surface(Xi, Yi, Zi, cmap=cmap, alpha=0.85,
                    edgecolor='grey', linewidth=0.3)

    # 数据点
    ax.scatter(x_raw, y_raw, z_raw, c='black', s=20, zorder=5)

    # 最优点
    best_idx = z_raw.argmax() if hb else z_raw.argmin()
    ax.scatter(x_raw[best_idx], y_raw[best_idx], z_raw[best_idx],
               c='red', s=120, marker='*', zorder=10)

    best_row = grouped.iloc[best_idx]
    direction = 'Higher is better ↑' if hb else 'Lower is better ↓'
    ax.set_title(
        f'{name}\n{direction}\n'
        f'Best: Q={best_row["Q_noisy"]:.4f}, '
        f'ε₂/ε={best_row["e2_r"]:.2f}, '
        f'{metric}={z_raw[best_idx]:.4f}',
        fontsize=12, pad=20)

    ax.set_xlabel('Q (Step1 modularity)', fontsize=11, labelpad=10)
    ax.set_ylabel('ε₂ / ε', fontsize=11, labelpad=10)
    ax.set_zlabel(metric, fontsize=11, labelpad=10)
    ax.view_init(elev=25, azim=135)

    # 保存
    save_file = os.path.join(save_dir, f'adapt_{metric}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f'[{idx+1}/7] 已保存: {os.path.basename(save_file)}')

    # 弹出可旋转窗口
    plt.show()

print(f'\n全部图片保存在: {save_dir}')