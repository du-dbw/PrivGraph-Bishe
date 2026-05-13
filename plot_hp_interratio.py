#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画 hp_inter_ratio.csv 的敏感性扫描图。
4 个指标横排展示，凸显"社区结构 vs 局部传递性"的 trade-off。
默认值 0.10 用红色高亮。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =================== 配置 ===================
CSV_PATH = './results/hp_inter_ratio.csv'
FIG_DIR  = './figures'
FIG_STEM = 'hp_interratio'

DEFAULT_RATIO = 0.10   # 默认值，用红色高亮

# (列名, 显示名, 方向)
METRICS = [
    ('nmi',     'NMI',                 'higher is better'),
    ('mod_rel', 'Modularity Relative', 'lower is better'),
    ('deg_kl',  'Degree KL',           'lower is better'),
    ('cc_rel',  'CC Relative',         'lower is better'),
]

# 配色
COLOR_LINE    = '#888888'   # 折线灰色
COLOR_OTHER   = '#888888'   # 普通点
COLOR_DEFAULT = '#D62728'   # 默认值 0.10 用红色
SHADE         = '#1F77B4'   # trend 阴影


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


def ci95_halfwidth(x):
    n = len(x)
    if n < 2:
        return 0.0
    se = np.std(x, ddof=1) / np.sqrt(n)
    return 1.96 * se


def load(path):
    if not os.path.exists(path):
        # 兼容上传路径
        alt = '/mnt/user-data/uploads/1778492711049_hp_inter_ratio.csv'
        if os.path.exists(alt):
            return pd.read_csv(alt)
        print(f'[错误] 找不到 CSV：{path}')
        sys.exit(1)
    return pd.read_csv(path)


def plot(df):
    ratios = sorted(df['inter_ratio'].unique())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))

    for idx, (col, name, direction) in enumerate(METRICS):
        ax = axes[idx]

        means, cis = [], []
        for r in ratios:
            vals = df[df['inter_ratio'] == r][col].values
            means.append(vals.mean())
            cis.append(ci95_halfwidth(vals))
        means = np.array(means)
        cis = np.array(cis)

        # 灰色背景折线 + 阴影（trend）
        ax.plot(ratios, means, '-', color=COLOR_LINE, linewidth=1.5, alpha=0.7, zorder=1)
        ax.fill_between(ratios, means - cis, means + cis,
                        color=COLOR_LINE, alpha=0.15, linewidth=0, zorder=0)

        # 默认值高亮（红色五角星），其他点灰色圆
        for r, m, c in zip(ratios, means, cis):
            is_default = (abs(r - DEFAULT_RATIO) < 1e-9)
            if is_default:
                color, marker, size, edge = COLOR_DEFAULT, '*', 280, 'white'
                lw = 1.8
            else:
                color, marker, size, edge = COLOR_OTHER, 'o', 75, COLOR_LINE
                lw = 1.2
            ax.errorbar(r, m, yerr=c, fmt='none',
                        ecolor=color, capsize=4, elinewidth=1.5, zorder=2)
            ax.scatter([r], [m], s=size, color=color, marker=marker,
                       edgecolor=edge, linewidth=lw, zorder=3)

        # 标题 + 方向
        ax.set_title(f'{name}\n({direction})', fontsize=11)
        ax.set_xlabel('inter_ratio', fontsize=11)
        ax.set_xticks(ratios)
        ax.set_xticklabels([f'{r:g}' for r in ratios])
        # 给 y 轴留点上下空间，避免 errorbar 顶到边
        y0, y1 = ax.get_ylim()
        pad = (y1 - y0) * 0.06
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=9)

    # 全图统一图例
    handle_default = plt.Line2D([0], [0], marker='*', color=COLOR_DEFAULT,
                                markerfacecolor=COLOR_DEFAULT, markeredgecolor='white',
                                markersize=14, linewidth=0,
                                label=f'default (inter_ratio = {DEFAULT_RATIO})')
    handle_other = plt.Line2D([0], [0], marker='o', color=COLOR_OTHER,
                              markerfacecolor=COLOR_OTHER, markeredgecolor=COLOR_LINE,
                              markersize=7, linewidth=0,
                              label='other values')
    fig.legend(handles=[handle_default, handle_other],
               loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=10, frameon=True, fancybox=True)

    fig.suptitle(
        r'Sensitivity of inter_ratio on Chameleon ($\varepsilon = 2.0$, 30 reps, mean ± 95% CI)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'{FIG_STEM}.{ext}')
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ 已保存：{FIG_DIR}/{FIG_STEM}.{{pdf,png}}')


def print_summary(df):
    """终端打印一份均值表，便于核对"""
    print('\n[各 inter_ratio 在 4 项指标上的均值 ± 95% CI]')
    ratios = sorted(df['inter_ratio'].unique())
    rows = []
    for r in ratios:
        sub = df[df['inter_ratio'] == r]
        row = {'inter_ratio': r, 'n': len(sub)}
        for col, name, _ in METRICS:
            m = sub[col].mean()
            ci = ci95_halfwidth(sub[col].values)
            row[name] = f'{m:.4f} ± {ci:.4f}'
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    df = load(CSV_PATH)
    print(f'读取 {len(df)} 行')
    plot(df)
    print_summary(df)


if __name__ == '__main__':
    main()