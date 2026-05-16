#!/usr/bin/env python3
"""
画 Enron 数据集上 PrivGraph vs Ours 的 7 子图对比图。
风格与 Facebook、CA-HepPh 的 overview7 图保持一致。

用法:
    python plot_enron_overview.py --main ./results/main_comparison.csv --out ./figures/enron_overview7.pdf
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 7 个指标的配置 =====
# (列名, 显示标题, 方向标记: 'higher is better' / 'lower is better')
METRICS = [
    ('nmi',          'NMI',                 'higher is better'),
    ('mod_rel',      'Modularity Relative', 'lower is better'),
    ('deg_kl',       'Degree KL',           'lower is better'),
    ('cc_rel',       'CC Relative',         'lower is better'),
    ('evc_overlap',  'EVC Overlap',         'higher is better'),
    ('evc_mae',      'EVC MAE',             'lower is better'),
    ('diam_rel',     'Diameter Relative',   'lower is better'),
]

# ===== 方法样式 =====
METHOD_STYLES = {
    'privgraph': {
        'label':      'PrivGraph',
        'color':      '#999999',
        'linestyle':  '--',
        'marker':     's',
        'markersize': 5,
    },
    'ours': {
        'label':      'Ours',
        'color':      '#E63946',
        'linestyle':  '-',
        'marker':     'o',
        'markersize': 5,
    },
}


def aggregate(df, dataset, method, metric):
    """聚合指定 dataset / method / metric 的均值和 SEM。"""
    sub = df[(df.dataset == dataset) & (df.method == method)].copy()
    grouped = sub.groupby('epsilon')[metric].agg(['mean', 'sem', 'count']).reset_index()
    grouped = grouped.sort_values('epsilon')
    return grouped


def plot_overview(df, dataset, out_path):
    """画 7 子图 + legend，2 行 4 列布局。"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    axes_flat = axes.flatten()

    n_reps_for_title = None

    for i, (metric_col, title, direction) in enumerate(METRICS):
        ax = axes_flat[i]

        for method, style in METHOD_STYLES.items():
            agg = aggregate(df, dataset, method, metric_col)
            if agg.empty:
                continue
            eps = agg['epsilon'].values
            mean = agg['mean'].values
            sem = agg['sem'].values

            if n_reps_for_title is None:
                n_reps_for_title = int(agg['count'].iloc[0])

            ax.plot(
                eps, mean,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=style['markersize'],
                linewidth=1.5,
            )
            ax.fill_between(
                eps, mean - sem, mean + sem,
                color=style['color'], alpha=0.15, linewidth=0,
            )

        # 标题：双行，与你贴的 Facebook 图一致
        ax.set_title(f'{title}\n({direction})', fontsize=10)
        ax.set_xlabel(r'$\varepsilon$', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(labelsize=9)

    # 最后一个格子放 legend
    legend_ax = axes_flat[7]
    legend_ax.axis('off')
    handles = [
        plt.Line2D(
            [0], [0],
            color=s['color'], linestyle=s['linestyle'],
            marker=s['marker'], markersize=s['markersize'],
            linewidth=1.5, label=s['label'],
        )
        for s in METHOD_STYLES.values()
    ]
    legend_ax.legend(
        handles=handles,
        title=dataset,
        loc='center',
        fontsize=11,
        title_fontsize=12,
        frameon=True,
    )

    # 总标题
    n_str = f'{n_reps_for_title} reps' if n_reps_for_title else ''
    fig.suptitle(
        f'PrivGraph vs Ours on {dataset} (mean ± SEM, {n_str})',
        fontsize=12, y=1.00,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    # 同时存一份 PNG 预览
    png_path = os.path.splitext(out_path)[0] + '.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')
    print(f'[saved] {png_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--main', default='./results/main_comparison.csv',
        help='主对比 CSV 路径',
    )
    parser.add_argument(
        '--dataset', default='Enron',
        help='要画的数据集名',
    )
    parser.add_argument(
        '--out', default='./figures/enron_overview7.pdf',
        help='输出 PDF 路径',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.main)
    print(f'[loaded] {args.main}  rows={len(df)}')
    print(f'[filter] dataset={args.dataset}')

    sub = df[df.dataset == args.dataset]
    if sub.empty:
        raise SystemExit(f'未找到 dataset={args.dataset} 的数据')

    print(f'[methods] {sorted(sub.method.unique())}')
    print(f'[epsilons] {sorted(sub.epsilon.unique())}')
    print(f'[reps] {sub.groupby(["method","epsilon"]).size().min()}~'
          f'{sub.groupby(["method","epsilon"]).size().max()}')

    plot_overview(df, args.dataset, args.out)


if __name__ == '__main__':
    main()
