"""
plot_hp_sweeps.py — 基于 30 reps 数据重新生成超参敏感性扫描图

读取:
  ./result/hp_inter.csv
  ./result/hp_intra.csv
  ./result/hp_swap.csv

输出:
  ./result/figures/hp_interratio.pdf
  ./result/figures/hp_intraratio.pdf
  ./result/figures/hp_swapratio.pdf

每张图:
  - 1 × 4 子图 (NMI / Modularity Relative / Degree KL / CC Relative)
  - 主折线 + 95% CI 误差棒
  - 默认值用灰色竖虚线 + 黑边圆圈高亮
  - 与默认值显著不同 (Welch t-test, p < 0.05) 的点用 ★ 标记
  - 控制台同步打印每个 HP 值的均值/CI 与显著差异，便于核对论文文本

依赖: pandas, numpy, matplotlib, scipy
用法: python plot_hp_sweeps.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

# ---- 中文字体设置（Windows 通常自带 SimHei） ----
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42  # 让 PDF 可编辑文字
matplotlib.rcParams['ps.fonttype'] = 42

# ---- 路径 ----
RESULT_DIR = './result'
OUT_DIR = './result/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 实验配置（与 hp_diagnose.py 保持一致）----
EPS = 2.0  # 该 sweep 的隐私预算

SWEEPS = [
    dict(csv='hp_inter.csv', hp_col='inter_ratio',
         default=0.10, out='hp_interratio.pdf'),
    dict(csv='hp_intra.csv', hp_col='intra_ratio',
         default=0.05, out='hp_intraratio.pdf'),
    dict(csv='hp_swap.csv',  hp_col='swap_ratio',
         default=0.30, out='hp_swapratio.pdf'),
]

METRICS = [
    dict(col='nmi',     title=r'NMI $\uparrow$',
         ylabel='NMI',                  color='#D62728'),
    dict(col='mod_rel', title=r'Modularity Relative $\downarrow$',
         ylabel='Modularity Relative',  color='#1F77B4'),
    dict(col='deg_kl',  title=r'Degree KL $\downarrow$',
         ylabel='Degree KL',            color='#2CA02C'),
    dict(col='cc_rel',  title=r'CC Relative $\downarrow$',
         ylabel='CC Relative',          color='#9467BD'),
]

# 颜色：默认点/星号
COLOR_DEFAULT_EDGE = 'black'
COLOR_STAR_FACE = '#FFB400'
COLOR_STAR_EDGE = '#8C5A00'


def compute_stats(df, hp_col, metric_col):
    """返回 [(hp_value, mean, ci95, raw_array), ...]，按 hp_value 升序。"""
    out = []
    for v in sorted(df[hp_col].unique()):
        vals = df.loc[df[hp_col] == v, metric_col].values.astype(float)
        n = len(vals)
        if n < 2:
            mean = float(vals.mean()) if n else 0.0
            ci95 = 0.0
        else:
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1)) / np.sqrt(n)
            ci95 = 1.96 * sem
        out.append((float(v), mean, ci95, vals))
    return out


def plot_sweep(cfg):
    csv_path = os.path.join(RESULT_DIR, cfg['csv'])
    if not os.path.exists(csv_path):
        print(f"  !! {csv_path} 不存在，跳过")
        return

    df = pd.read_csv(csv_path)
    n_per = int(df.groupby(cfg['hp_col']).size().min())
    print(f"\n>>> {cfg['csv']}: 共 {len(df)} 行，每个 HP 值最少 {n_per} reps")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    title = (f"{cfg['hp_col']} 敏感性扫描  "
             f"($\\varepsilon = {EPS}$, {n_per} reps, 误差棒为 95% CI)")
    fig.suptitle(title, fontsize=12, y=1.02)

    for ax, m in zip(axes, METRICS):
        rows = compute_stats(df, cfg['hp_col'], m['col'])
        xs = np.array([r[0] for r in rows])
        means = np.array([r[1] for r in rows])
        cis = np.array([r[2] for r in rows])

        # 主折线 + CI 误差棒
        ax.errorbar(xs, means, yerr=cis, fmt='o-',
                    color=m['color'], ecolor=m['color'],
                    elinewidth=1.2, capsize=4, markersize=7,
                    linewidth=1.5, alpha=0.85, zorder=3)

        # 默认值竖虚线
        ax.axvline(cfg['default'], color='gray', ls=':', lw=1.0,
                   alpha=0.7, zorder=1)

        # 找默认值索引与样本
        default_idx, default_vals = None, None
        for i, (v, mean, ci, vals) in enumerate(rows):
            if abs(v - cfg['default']) < 1e-9:
                default_idx, default_vals = i, vals
                break

        # 高亮默认值点（带黑边）
        if default_idx is not None:
            ax.plot([xs[default_idx]], [means[default_idx]],
                    marker='o', color=m['color'],
                    markeredgecolor=COLOR_DEFAULT_EDGE,
                    markeredgewidth=1.8, markersize=11, zorder=10)

        # 与默认值做 Welch t-test
        sig_pts = []
        if default_vals is not None and len(default_vals) >= 2:
            for i, (v, mean, ci, vals) in enumerate(rows):
                if i == default_idx or len(vals) < 2:
                    continue
                _, p = stats.ttest_ind(vals, default_vals, equal_var=False)
                if p < 0.05:
                    sig_pts.append((i, v, mean, ci, p))

        # y 轴范围（顶部留出 ★ 的位置）
        y_lo = float((means - cis).min())
        y_hi = float((means + cis).max())
        y_range = max(y_hi - y_lo, 1e-6)
        ax.set_ylim(y_lo - 0.08 * y_range, y_hi + 0.25 * y_range)

        # 显著点上方标 ★
        for i, v, mean, ci, p in sig_pts:
            ax.plot([v], [mean + ci + 0.08 * y_range],
                    marker='*',
                    color=COLOR_STAR_FACE,
                    markeredgecolor=COLOR_STAR_EDGE,
                    markeredgewidth=0.6,
                    markersize=20, zorder=15)

        ax.set_xlabel(cfg['hp_col'])
        ax.set_ylabel(m['ylabel'])
        ax.set_title(m['title'], fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{x:g}' for x in xs])
        ax.grid(True, alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 控制台对照打印（方便核对论文里的数字）
        line = "  {:>8}: ".format(m['col']) + ", ".join(
            [f"{v:g}={mn:.3f}±{ci:.3f}" for v, mn, ci, _ in rows])
        print(line)
        if sig_pts:
            sig_str = ", ".join([f"{v:g}(p={p:.3f})"
                                 for _, v, _, _, p in sig_pts])
            print(f"           sig vs default ({cfg['default']:g}): {sig_str}")
        else:
            print(f"           (无与默认值的显著差异)")

    # 全图共用图例
    legend_handles = [
        plt.Line2D([], [], marker='o', ls='None',
                   markerfacecolor='lightgray',
                   markeredgecolor=COLOR_DEFAULT_EDGE,
                   markeredgewidth=1.8, markersize=11,
                   label='默认取值'),
        plt.Line2D([], [], marker='*', ls='None',
                   markerfacecolor=COLOR_STAR_FACE,
                   markeredgecolor=COLOR_STAR_EDGE,
                   markeredgewidth=0.6, markersize=18,
                   label=r'与默认显著不同 ($p<0.05$, Welch t-test)'),
        plt.Line2D([], [], ls=':', color='gray', lw=1.0,
                   label='默认值'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=3,
               frameon=False, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, cfg['out'])
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  -> 已保存 {out_path}")


def main():
    print("=" * 64)
    print("绘制超参敏感性扫描图 (30 reps, 95% CI, Welch t-test)")
    print("=" * 64)
    for cfg in SWEEPS:
        plot_sweep(cfg)
    print(f"\n全部完成，三张 PDF 输出到 {OUT_DIR}/")
    print("如需放进论文，把这三个 PDF 复制到 LaTeX 的图片目录即可。")


if __name__ == '__main__':
    main()