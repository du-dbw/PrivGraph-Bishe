"""
plot_multi_overview.py
======================
读取 main_test.py 跑出来的 multi_<Dataset>.csv（含 PrivGraph / Ours-Full 两种方法、
eps ∈ {0.5,...,3.5}、每组 10 个 exper），画出 1×4 的 overview 图：
    NMI ↑ | Modularity Relative ↓ | Degree KL ↓ | CC Relative ↓

输出风格与 facebook_overview.pdf / ca_hepph_overview.pdf 完全一致：
    - PrivGraph：灰色虚线 + 方块
    - Ours-Full：红色实线 + 圆点
    - 中文大标题 "<Dataset> 数据集"，横轴 "隐私预算 ε"

用法
----
python plot_multi_overview.py multi_Enron.csv               # 自动取 dataset=Enron
python plot_multi_overview.py multi_Enron.csv --name Enron  # 手动指定显示名
python plot_multi_overview.py multi_Enron.csv --out enron_overview.pdf
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


# -------- 与原图一致的全局样式 --------
EPS_LIST = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

METHOD_STYLE = {
    'PrivGraph': dict(
        color='#7f7f7f', linestyle='--', marker='s',
        markersize=6, linewidth=1.6, label='PrivGraph',
    ),
    'Ours-Full': dict(
        color='#d62728', linestyle='-', marker='o',
        markersize=6, linewidth=1.8, label='Ours-Full',
    ),
}

# 4 个子图：(csv 列名, 显示标题, 是否越大越好)
PANELS = [
    ('nmi',      'NMI',                True),
    ('mod_rel',  'Modularity Relative', False),
    ('deg_kl',   'Degree KL',          False),
    ('cc_rel',   'CC Relative',        False),
]


def setup_chinese_font():
    """让 matplotlib 能渲染中文。若系统没装常见中文字体则给出警告。"""
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC',
        'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC',
        'Arial Unicode MS',
    ]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen] + plt.rcParams['font.sans-serif']
    else:
        print('[warn] 没找到中文字体，标题里的中文会变方块。'
              '可装 fonts-noto-cjk 或 fonts-wqy-microhei，或自己加字体到 matplotlib。',
              file=sys.stderr)
    plt.rcParams['axes.unicode_minus'] = False


def aggregate(df: pd.DataFrame, method: str, col: str):
    """对同一 method 在每个 eps 上取均值（多次 exper 平均）。返回 (eps_arr, mean_arr)."""
    sub = df[df['method'] == method]
    g = sub.groupby('eps')[col].mean().reindex(EPS_LIST)
    return np.array(EPS_LIST), g.values


def plot_overview(df: pd.DataFrame, dataset_name: str, out_path: str):
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.6))
    fig.suptitle(f'{dataset_name} 数据集', fontsize=16, y=1.02)

    for ax, (col, title, higher_better) in zip(axes, PANELS):
        arrow = '↑' if higher_better else '↓'
        ax.set_title(f'{title}  {arrow}', fontsize=13)

        for method, style in METHOD_STYLE.items():
            xs, ys = aggregate(df, method, col)
            ax.plot(xs, ys, **style)

        ax.set_xlabel('隐私预算 ε', fontsize=11)
        ax.set_xticks(EPS_LIST)
        ax.grid(True, linestyle=':', alpha=0.5)
        # 去掉顶/右边框，跟参考图一致
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 只在第一个子图里写 y 轴 label "NMI"，并放图例（参考图就是这么放的）
    axes[0].set_ylabel('NMI', fontsize=11)
    axes[0].legend(loc='best', frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    # 顺手再存一张 png 方便预览
    png_path = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'[ok] saved -> {out_path}')
    print(f'[ok] saved -> {png_path}')


def infer_name(csv_path: str) -> str:
    """从 multi_Enron.csv 这种文件名里抠出 'Enron'。抠不到就返回文件名 stem。"""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    m = re.match(r'multi[_\-](.+)', stem, flags=re.IGNORECASE)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='multi_<Dataset>.csv 路径')
    ap.add_argument('--name', default=None, help='图标题里显示的数据集名 (默认从文件名推断)')
    ap.add_argument('--out',  default=None, help='输出 pdf 路径 (默认 <stem>_overview.pdf)')
    args = ap.parse_args()

    setup_chinese_font()

    df = pd.read_csv(args.csv)
    # 简单校验
    needed = {'method', 'eps', 'nmi', 'mod_rel', 'deg_kl', 'cc_rel'}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f'csv 缺少列: {missing}')

    name = args.name or infer_name(args.csv)
    out  = args.out  or f'{name.lower()}_overview.pdf'
    plot_overview(df, name, out)


if __name__ == '__main__':
    main()