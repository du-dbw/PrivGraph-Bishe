#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Enron 数据集上 PrivGraph vs Ours 的折线对比图。

输入：./results/main_comparison.csv
输出：./figures/enron_overview7.pdf  (1×4 子图：NMI / Mod.Rel. / Deg.KL / CC.Rel.)
      ./figures/enron_full.pdf       (2×4 子图：包含 EVC Overlap/MAE 与 Diam.Rel.)

依赖：pandas, numpy, matplotlib
用法：
  python plot_enron.py
  python plot_enron.py --dataset=Facebook   # 也可换数据集
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 全局样式 ----------
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.labelsize':    13,
    'axes.titlesize':    13,
    'legend.fontsize':   11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.alpha':        0.4,
    'lines.linewidth':   2.0,
    'lines.markersize':  7,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,    # 保证 PDF 中字体可编辑
    'ps.fonttype':       42,
})

# 两种方法的颜色 / 样式
STYLE = {
    'privgraph': dict(color='#4C72B0', marker='o', linestyle='--', label='PrivGraph'),
    'ours':      dict(color='#C44E52', marker='s', linestyle='-',  label='Ours'),
}

# 指标定义：列名 → (标题, y 轴方向)
METRICS_CORE = [
    ('nmi',     'NMI',                'higher'),
    ('mod_rel', 'Modularity Relative','lower'),
    ('deg_kl',  'Degree KL',          'lower'),
    ('cc_rel',  'CC Relative',        'lower'),
]

METRICS_FULL = METRICS_CORE + [
    ('evc_overlap', 'EVC Overlap',       'higher'),
    ('evc_mae',     'EVC MAE',           'lower'),
    ('diam_rel',    'Diameter Relative', 'lower'),
]


def load_data(csv_path, dataset):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'找不到结果文件: {csv_path}')
    df = pd.read_csv(csv_path)
    df = df[df['dataset'] == dataset].copy()
    if df.empty:
        raise ValueError(f'{csv_path} 中没有 dataset={dataset} 的数据')
    return df


def aggregate(df, metric):
    """对每个 (method, epsilon) 聚合：均值 + 标准误"""
    g = df.groupby(['method', 'epsilon'])[metric].agg(['mean', 'std', 'count']).reset_index()
    g['sem'] = g['std'] / np.sqrt(g['count'].clip(lower=1))
    return g


def plot_one_metric(ax, df, metric, title, direction, show_legend=False):
    """在单个 ax 上画 PrivGraph vs Ours 折线（带误差带）"""
    agg = aggregate(df, metric)

    for method in ['privgraph', 'ours']:
        sub = agg[agg['method'] == method].sort_values('epsilon')
        if sub.empty:
            continue
        x = sub['epsilon'].values
        y = sub['mean'].values
        yerr = sub['sem'].values
        style = STYLE[method]
        ax.plot(x, y, **style)
        # 误差带（半透明填充）
        ax.fill_between(x, y - yerr, y + yerr,
                        color=style['color'], alpha=0.15, linewidth=0)

    ax.set_xlabel(r'$\varepsilon$')
    arrow = ' ↑' if direction == 'higher' else ' ↓'
    ax.set_ylabel(title + arrow)
    ax.set_title(title + arrow)

    # X 轴刻度
    eps_vals = sorted(df['epsilon'].unique())
    ax.set_xticks(eps_vals)

    if show_legend:
        ax.legend(loc='best', frameon=True, framealpha=0.9)


def plot_overview(df, metrics, out_path, dataset, ncols=4):
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 3.4),
                             squeeze=False)

    for i, (col, title, direction) in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if col not in df.columns:
            ax.text(0.5, 0.5, f'缺少列: {col}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        plot_one_metric(ax, df, col, title, direction,
                        show_legend=(i == 0))

    # 关掉多余子图
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()

    fig.suptitle(f'{dataset}: PrivGraph vs Ours',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[OK] 已保存: {out_path}')


def print_table(df, dataset):
    """顺手打印一份对比表，便于直接拷贝到论文里"""
    print(f'\n=== {dataset} 均值对比表 ===')
    metrics = [m[0] for m in METRICS_FULL if m[0] in df.columns]
    rows = []
    for method in ['privgraph', 'ours']:
        sub = df[df['method'] == method]
        if sub.empty:
            continue
        agg = sub.groupby('epsilon')[metrics].mean()
        # 加一列 mean over eps
        row = agg.mean()
        row.name = f'{method} (mean)'
        rows.append(row)
    if rows:
        out = pd.DataFrame(rows).round(4)
        print(out.to_string())

        # 相对改进
        if {'privgraph (mean)', 'ours (mean)'} <= set(out.index):
            base = out.loc['privgraph (mean)']
            ours = out.loc['ours (mean)']
            print('\n--- 相对 PrivGraph 的相对变化（% ）---')
            improve = (ours - base) / base.abs() * 100
            print(improve.round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='./results/main_comparison.csv',
                        help='主对比 CSV 路径')
    parser.add_argument('--dataset', default='Enron',
                        help='数据集名称 (Enron / Facebook / CA-HepPh / Chamelon)')
    parser.add_argument('--outdir', default='./figures',
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv, args.dataset)

    # 打印汇总表
    print_table(df, args.dataset)

    # 4 个核心指标（论文正文用）
    overview_path = os.path.join(args.outdir,
                                  f'{args.dataset.lower().replace("-", "")}_overview7.pdf')
    plot_overview(df, METRICS_CORE, overview_path, args.dataset, ncols=4)

    # 7 个完整指标（备用 / 附录）
    full_path = os.path.join(args.outdir,
                              f'{args.dataset.lower().replace("-", "")}_full.pdf')
    plot_overview(df, METRICS_FULL, full_path, args.dataset, ncols=4)


if __name__ == '__main__':
    main()
