import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os, glob

# ===== 配置 =====
res_path = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'Chamelon'
e3_r = 0.15

metrics = ['nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']
metric_names = ['NMI (Community Discovery)', 'Overlap (Eigenvalue Nodes)', 'MAE (Eigenvalue)',
                'KL Divergence (Degree Distribution)', 'RE (Diameter)',
                'RE (Clustering Coefficient)', 'RE (Modularity)']
higher_better = [True, True, False, False, False, False, False]

# ===== 读取数据 =====
pattern = os.path.join(res_path, f'fixe3_{dataset_name}_e3r{e3_r:.2f}_*.csv')
files = glob.glob(pattern)
print(f'搜索: {pattern}')
print(f'找到: {len(files)} 个文件')

if len(files) == 0:
    print('没找到文件！请检查路径和文件名')
    exit()

df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f'总记录数: {len(df_all)}')
print(f'eps 范围: {sorted(df_all["eps"].unique())}')
print(f'e1_r 范围: {sorted(df_all["e1_r"].unique())}')

# 按 eps 和 e1_r 分组取平均
grouped = df_all.groupby(['eps', 'e1_r', 'e2_r'])[metrics].mean().reset_index()
print(f'有效组合: {len(grouped)} 个')

# ===== 创建输出文件夹 =====
save_dir = os.path.join(res_path, f'fixe3_{dataset_name}_e3r{e3_r}')
os.makedirs(save_dir, exist_ok=True)

# ===== 逐指标画 3D 图 =====
x_raw = grouped['eps'].values
y_raw = grouped['e1_r'].values

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

    best = grouped.iloc[best_idx]
    direction = 'Higher is better ↑' if hb else 'Lower is better ↓'
    ax.set_title(
        f'{name}\n{direction}\n'
        f'Best: ε={best["eps"]:.1f}, '
        f'e1={best["e1_r"]:.2f}, e2={best["e2_r"]:.2f}, e3={e3_r}, '
        f'{metric}={z_raw[best_idx]:.4f}',
        fontsize=12, pad=20)

    ax.set_xlabel('ε', fontsize=11, labelpad=10)
    ax.set_ylabel('ε₁ / ε', fontsize=11, labelpad=10)
    ax.set_zlabel(metric, fontsize=11, labelpad=10)
    ax.view_init(elev=25, azim=135)

    # 保存
    save_file = os.path.join(save_dir, f'fixe3_{metric}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f'[{idx+1}/7] 已保存: {os.path.basename(save_file)}')

    # 弹出可旋转窗口
    plt.show()

# ===== 额外：2D 折线图，每条线一个 eps =====
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, (metric, name, hb) in enumerate(zip(metrics, metric_names, higher_better)):
    ax = axes[idx]

    for eps_val, grp in grouped.groupby('eps'):
        grp_sorted = grp.sort_values('e1_r')
        ax.plot(grp_sorted['e1_r'], grp_sorted[metric],
                'o-', label=f'ε={eps_val:.1f}', markersize=4, alpha=0.8)

    ax.set_xlabel('ε₁ / ε', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)

    # 添加 e2_r 的副坐标轴标注
    ax2 = ax.twiny()
    e1_ticks = ax.get_xticks()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(e1_ticks)
    ax2.set_xticklabels([f'{round(0.85-x,2):.2f}' for x in e1_ticks], fontsize=7)
    ax2.set_xlabel('ε₂ / ε', fontsize=8)

    direction = '↑' if hb else '↓'
    ax.set_title(f'{name} {direction}', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

for idx in range(len(metrics), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f'{dataset_name}, e3_r={e3_r} (fixed)\nX-axis: e1_r (top: corresponding e2_r)',
             fontsize=13, y=1.02)
plt.tight_layout()
save_file = os.path.join(save_dir, f'fixe3_summary_2D.png')
plt.savefig(save_file, dpi=150, bbox_inches='tight')
print(f'\n[汇总] 已保存: {os.path.basename(save_file)}')
plt.show()

print(f'\n全部图片保存在: {save_dir}')