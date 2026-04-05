import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os, glob

# ===== 配置 =====
res_path = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'Chamelon'
N = 20
target_eps = 2.0

metrics = ['nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']
metric_names = ['NMI (Community Discovery)', 'Overlap (Eigenvalue Nodes)', 'MAE (Eigenvalue)',
                'KL Divergence (Degree Distribution)', 'RE (Diameter)',
                'RE (Clustering Coefficient)', 'RE (Modularity)']
higher_better = [True, True, False, False, False, False, False]

# ===== 读取数据 =====
pattern = os.path.join(res_path, f'{dataset_name}_{N}_*.csv')
files = glob.glob(pattern)
print(f'搜索路径: {pattern}')
print(f'找到文件: {len(files)} 个')

all_results = []
for f in files:
    basename = os.path.basename(f).replace('.csv', '')
    parts = basename.split('_')
    try:
        e1_r = float(parts[2])
        e2_r = float(parts[3])
    except (IndexError, ValueError):
        continue
    e3_r = round(1 - e1_r - e2_r, 2)
    if e3_r < 0.01:
        continue
    df = pd.read_csv(f)
    df_eps = df[df['eps'] == target_eps]
    if len(df_eps) == 0:
        continue
    avg = df_eps[metrics].mean()
    avg['e1_r'], avg['e2_r'], avg['e3_r'] = e1_r, e2_r, e3_r
    all_results.append(avg)

results = pd.DataFrame(all_results)
print(f'有效配置: {len(results)} 组')

if len(results) == 0:
    print('没有读到数据！请检查文件路径和文件名格式')
    exit()

# ===== 逐个指标画图 =====
for idx, (metric, name, hb) in enumerate(zip(metrics, metric_names, higher_better)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = results['e1_r'].values
    y = results['e2_r'].values
    z = results[metric].values

    # 插值
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    mask = (Xi + Yi) <= 0.95
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi[~mask] = np.nan

    cmap = 'coolwarm' if hb else 'coolwarm_r'
    ax.plot_surface(Xi, Yi, Zi, cmap=cmap, alpha=0.85,
                    edgecolor='grey', linewidth=0.3)

    # 数据点
    ax.scatter(x, y, z, c='black', s=20, zorder=5)

    # 最优点
    best_idx = z.argmax() if hb else z.argmin()
    ax.scatter(x[best_idx], y[best_idx], z[best_idx],
               c='red', s=120, marker='*', zorder=10)

    direction = 'Higher is better ↑' if hb else 'Lower is better ↓'
    ax.set_title(f'{name}\n{direction}\n'
                 f'Best: ε₁/ε={x[best_idx]:.2f}, ε₂/ε={y[best_idx]:.2f}, '
                 f'ε₃/ε={1-x[best_idx]-y[best_idx]:.2f}, {metric}={z[best_idx]:.4f}',
                 fontsize=12, pad=20)
    ax.set_xlabel('ε₁/ε', fontsize=11, labelpad=10)
    ax.set_ylabel('ε₂/ε', fontsize=11, labelpad=10)
    ax.set_zlabel(metric, fontsize=11, labelpad=10)
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    save_file = os.path.join(res_path, f'surface3d_{dataset_name}_{metric}_eps{target_eps}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'[{idx+1}/7] 已保存: {os.path.basename(save_file)}')

print('\n全部完成!')