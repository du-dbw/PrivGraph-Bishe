#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画 main_comparison.csv 的 7 指标折线图
每个数据集一张图，2x4 布局（最后一格放图例）
Enron 数据不完整时自动跳过
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =================== 配置 ===================
CSV_PATH  = './results/main_comparison.csv'
FIG_DIR   = './figures'

# 完整跑完一个数据集的 reps × epsilons
N_REPS    = 10
N_EPSILON = 7
ROWS_PER_METHOD_FULL = N_REPS * N_EPSILON  # 70

# 数据集顺序（输出文件名与 caption 都按这个）
DATASETS = ['Chamelon', 'Facebook', 'CA-HepPh', 'Enron']

# 数据集显示名（用于图标题与文件名）
DISPLAY_NAME = {
    'Chamelon': 'Chameleon',
    'Facebook': 'Facebook',
    'CA-HepPh': 'CA-HepPh',
    'Enron':    'Enron',
}
FILE_STEM = {
    'Chamelon': 'chamelon',
    'Facebook': 'facebook',
    'CA-HepPh': 'cahepph',
    'Enron':    'enron',
}

# 7 个指标：(列名, 显示名, 方向标签)
# 方向标签仅做辅助提示，不画箭头
METRICS = [
    ('nmi',         'NMI',                  'higher is better'),
    ('mod_rel',     'Modularity Relative',  'lower is better'),
    ('deg_kl',      'Degree KL',            'lower is better'),
    ('cc_rel',      'CC Relative',          'lower is better'),
    ('evc_overlap', 'EVC Overlap',          'higher is better'),
    ('evc_mae',     'EVC MAE',              'lower is better'),
    ('diam_rel',    'Diameter Relative',    'lower is better'),
]

# 配色
COLORS = {
    'privgraph': '#888888',
    'ours':      '#D62728',
}
MARKERS = {
    'privgraph': 's',
    'ours':      'o',
}
LINESTYLES = {
    'privgraph': '--',
    'ours':      '-',
}
LABELS = {
    'privgraph': 'PrivGraph',
    'ours':      'Ours',
}


# =================== 字体 ===================
def setup_font():
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC',
        'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'Microsoft YaHei', 'SimHei', 'PingFang SC', 'STHeiti',
        'Arial Unicode MS', 'DejaVu Sans',
    ]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            mpl.rcParams['font.sans-serif'] = [c] + mpl.rcParams['font.sans-serif']
            break
    mpl.rcParams['axes.unicode_minus'] = False


setup_font()


# =================== 数据加载 ===================
def load_data(path):
    if not os.path.exists(path):
        print(f'[错误] 找不到 CSV：{path}')
        sys.exit(1)
    df = pd.read_csv(path)
    needed = {'dataset', 'method', 'epsilon', 'rep'}
    needed.update(m[0] for m in METRICS)
    missing = needed - set(df.columns)
    if missing:
        print(f'[错误] CSV 缺少列：{missing}')
        sys.exit(1)
    return df


def is_complete(df_ds):
    """检查一个数据集是否两种方法都各跑完 10×7=70 行"""
    if df_ds.empty:
        return False, 'no data'
    for m in ['privgraph', 'ours']:
        sub = df_ds[df_ds['method'] == m]
        if len(sub) < ROWS_PER_METHOD_FULL:
            # 进一步看是哪几个 epsilon 不全
            counts = sub.groupby('epsilon').size()
            bad = counts[counts < N_REPS].index.tolist()
            return False, f'{m} 在 ε={bad} 不足 {N_REPS} reps（共 {len(sub)}/{ROWS_PER_METHOD_FULL} 行）'
    return True, 'ok'


def aggregate(df_ds, metric_col):
    """按 method × epsilon 聚合，返回 (eps_list, mean_dict, sem_dict)"""
    eps_list = sorted(df_ds['epsilon'].unique())
    mean_dict, sem_dict = {}, {}
    for method in ['privgraph', 'ours']:
        sub = df_ds[df_ds['method'] == method]
        gp = sub.groupby('epsilon')[metric_col]
        means = gp.mean().reindex(eps_list).values
        # SEM = std / sqrt(n)
        sems = (gp.std(ddof=1) / np.sqrt(gp.count())).reindex(eps_list).values
        # 单个 rep 时 SEM=NaN，画图时按 0 处理
        sems = np.nan_to_num(sems, nan=0.0)
        mean_dict[method] = means
        sem_dict[method] = sems
    return np.array(eps_list), mean_dict, sem_dict


# =================== 画图 ===================
def plot_one_dataset(df_ds, dataset_name):
    """画一个数据集的 7 指标，2×4 布局，第 8 格放图例"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    axes = axes.flatten()

    for idx, (col, title, direction) in enumerate(METRICS):
        ax = axes[idx]
        eps_list, means, sems = aggregate(df_ds, col)

        for method in ['privgraph', 'ours']:
            m = means[method]
            s = sems[method]
            ax.plot(
                eps_list, m,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markersize=7,
                linewidth=1.8,
                label=LABELS[method],
                markerfacecolor=COLORS[method],
                markeredgecolor='white',
                markeredgewidth=0.8,
            )
            ax.fill_between(
                eps_list, m - s, m + s,
                color=COLORS[method], alpha=0.15, linewidth=0,
            )

        ax.set_title(f'{title}\n({direction})', fontsize=11)
        ax.set_xlabel(r'$\varepsilon$', fontsize=11)
        ax.set_xticks(eps_list)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=9)

    # 第 8 格：图例 + 标注
    ax_leg = axes[-1]
    ax_leg.axis('off')
    # 用代理 handle 画图例
    handles = []
    for method in ['privgraph', 'ours']:
        line, = ax_leg.plot(
            [], [],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=10,
            linewidth=2.2,
            label=LABELS[method],
            markerfacecolor=COLORS[method],
            markeredgecolor='white',
            markeredgewidth=1.0,
        )
        handles.append(line)
    ax_leg.legend(
        handles=handles, loc='center',
        fontsize=14, frameon=True, fancybox=True,
        title=DISPLAY_NAME[dataset_name],
        title_fontsize=14,
    )

    fig.suptitle(
        f'PrivGraph vs Ours on {DISPLAY_NAME[dataset_name]} (mean ± SEM, 10 reps)',
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    stem = FILE_STEM[dataset_name]
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'{stem}_overview7.{ext}')
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ 已保存：{FIG_DIR}/{stem}_overview7.{{pdf,png}}')


def print_summary(df_ds, dataset_name):
    """打印该数据集的均值对比表，便于核对"""
    print(f'\n[{DISPLAY_NAME[dataset_name]} 均值对比]')
    rows = []
    for col, title, _ in METRICS:
        priv = df_ds[df_ds['method'] == 'privgraph'][col].mean()
        ours = df_ds[df_ds['method'] == 'ours'][col].mean()
        rows.append({
            'metric': title,
            'PrivGraph': f'{priv:.4f}',
            'Ours':      f'{ours:.4f}',
        })
    df_show = pd.DataFrame(rows)
    print(df_show.to_string(index=False))


# =================== 主流程 ===================
def main():
    print(f'读取：{CSV_PATH}')
    df = load_data(CSV_PATH)
    print(f'  共 {len(df)} 行')

    for ds in DATASETS:
        df_ds = df[df['dataset'] == ds]
        ok, msg = is_complete(df_ds)
        if not ok:
            print(f'\n[{DISPLAY_NAME[ds]}] 数据不完整：{msg}，跳过')
            continue
        print(f'\n[{DISPLAY_NAME[ds]}] 数据完整，开始绘图')
        plot_one_dataset(df_ds, ds)
        print_summary(df_ds, ds)


if __name__ == '__main__':
    main()