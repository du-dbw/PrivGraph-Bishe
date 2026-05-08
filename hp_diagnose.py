"""
hp_diagnose.py — 检查现有 HP 扫描数据的方差，告诉你哪些差距是真的、哪些在噪声里。

用法：python hp_diagnose.py
读取 ./result/hp_inter.csv, hp_intra.csv, hp_swap.csv
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

RESULT_DIR = './result'

SWEEPS = [
    ('hp_inter.csv', 'inter_ratio'),
    ('hp_intra.csv', 'intra_ratio'),
    ('hp_swap.csv',  'swap_ratio'),
]

METRICS = ['nmi', 'mod_rel', 'deg_kl', 'cc_rel']
DIRECTION = {'nmi': 'higher', 'mod_rel': 'lower',
             'deg_kl': 'lower', 'cc_rel': 'lower'}


def diagnose(csv_path, hp_col):
    if not os.path.exists(csv_path):
        print(f"  {csv_path} 不存在，跳过")
        return
    df = pd.read_csv(csv_path)
    print(f"\n{'='*70}\n  {csv_path}  ({hp_col}, n_rows={len(df)})\n{'='*70}")

    for m in METRICS:
        print(f"\n  >>> {m}  ({DIRECTION[m]} better)")
        # Per-HP statistics
        rows = []
        for v, g in df.groupby(hp_col):
            mean = g[m].mean()
            std = g[m].std()
            sem = std / np.sqrt(len(g))
            ci95 = 1.96 * sem
            rows.append((v, mean, std, sem, ci95, len(g)))

        print(f"    {hp_col:>14}  {'mean':>9}  {'std':>9}  "
              f"{'SEM':>9}  {'95%CI±':>9}  {'n':>4}")
        for v, mean, std, sem, ci, n in rows:
            print(f"    {v:>14.3f}  {mean:>9.4f}  {std:>9.4f}  "
                  f"{sem:>9.4f}  {ci:>9.4f}  {n:>4d}")

        # Pairwise tests: which differences are statistically significant?
        vs = sorted(df[hp_col].unique())
        sig_pairs = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a = df[df[hp_col] == vs[i]][m]
                b = df[df[hp_col] == vs[j]][m]
                t, p = stats.ttest_ind(a, b, equal_var=False)
                if p < 0.05:
                    sig_pairs.append((vs[i], vs[j], a.mean() - b.mean(), p))
        if sig_pairs:
            print(f"    显著差异（Welch t-test, p<0.05）：")
            for a, b, d, p in sig_pairs:
                print(f"      {a} vs {b}: Δmean={d:+.4f}, p={p:.4f}")
        else:
            print(f"    ⚠️  所有两两差异在 p<0.05 下均不显著——无法区分这些 HP 值")


if __name__ == '__main__':
    for csv, col in SWEEPS:
        diagnose(os.path.join(RESULT_DIR, csv), col)
    print(f"\n{'='*70}")
    print("说明：")
    print("  - 如果某个指标下 SEM 接近或超过相邻 HP 值的 mean 差距，")
    print("    说明 10 reps 不足以区分这些 HP 值。")
    print("  - 如果显著差异列表为空或很短，说明你需要更多重复。")