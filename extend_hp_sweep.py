"""
extend_hp_sweep.py — 在已有 hp_*.csv 基础上补跑更多重复，提升统计功效。

为什么需要这个脚本
------------------
hp_diagnose.py 显示 mod_rel / cc_rel 在 10 reps 下方差太大，所有两两差异
都不显著。把每个 HP 值补到 30 reps 后：
  - 真实效应（如 inter_ratio 对 mod_rel）应该能浮出水面；
  - 若仍不显著，可在论文里下"无可测量效应"的结论。

设计要点
--------
1. 复用 run_overnight.py 中的 run_trial / checkpoint，零代码重复。
2. **exper 在外层、HP 值在内层** —— 崩溃时各 HP 值进度均衡，不会出现
   "v1 跑完 30 但 v5 还停在 10" 的情况。
3. 已有 (hp, exper) 自动跳过；--target 是"目标总 reps"，不是"再跑多少"。
4. Chameleon 只加载一次，避免重复 precompute_reference。

用法
----
python extend_hp_sweep.py                       # 把三组 sweep 都补到 30 reps
python extend_hp_sweep.py --target 40           # 补到 40
python extend_hp_sweep.py --only inter          # 只补 hp_inter
python extend_hp_sweep.py --only intra swap     # 补 intra 和 swap
python extend_hp_sweep.py --target 50 --only inter   # 单独把 inter 拉到 50

跑完之后建议立刻：python hp_diagnose.py 看新结果。
"""

import os
import time
import argparse
import traceback
import pandas as pd

from main_test import (
    RESULT_DIR,
    INTRA_RATIO, INTER_RATIO, SWAP_RATIO,
    run_trial, load_dataset,
    load_csv, is_done, append_row,
    Progress,
)

# ===================== 三组 sweep 定义（与 run_overnight 一致） =====================
HP_SWEEPS = [
    ('inter', [0.05, 0.10, 0.15, 0.20, 0.30], 'inter_ratio'),
    ('intra', [0.00, 0.05, 0.10, 0.15],        'intra_ratio'),
    ('swap',  [0.0,  0.1,  0.3,  0.5,  0.7],   'swap_ratio'),
]

EPS_FOR_HP = 2.0


def extend_one_sweep(name, sweep_values, hp_key, eps,
                     mat0, n, ref, target_reps):
    """把单个 HP sweep 补到 target_reps。"""
    csv_path = os.path.join(RESULT_DIR, f'hp_{name}.csv')
    df = load_csv(csv_path)

    total = len(sweep_values) * target_reps
    prog = Progress(total, f'hp-{name}')

    # 关键：exper 外层、v 内层 → 即使中途崩溃，所有 HP 值进度也是均衡的
    for exper in range(target_reps):
        for v in sweep_values:
            key = {hp_key: v, 'exper': exper}
            t0 = time.time()
            if is_done(df, key):
                prog.step(0.0, skipped=True)
                continue

            kw = {
                'intra_ratio': INTRA_RATIO,
                'inter_ratio': INTER_RATIO,
                'swap_ratio':  SWAP_RATIO,
            }
            kw[hp_key] = v

            try:
                m = run_trial(mat0, n, ref, eps, 'Ours-Full', **kw)
                row = {**key, **m}
                append_row(csv_path, row)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                # 监控两个最噪的指标，方便实时判断信号有没有出来
                extra = (f"nmi={m['nmi']:.3f}  mod_rel={m['mod_rel']:.3f}  "
                         f"cc_rel={m['cc_rel']:.3f}")
            except Exception as ex:
                print(f"  !! failed {key} -> {ex}")
                traceback.print_exc()
                extra = 'FAILED'

            prog.step(time.time() - t0, extra=extra)


def print_completion_summary(sweeps_to_run, target_reps):
    """打印每个 HP 值当前的 reps 数量，标记哪些还差。"""
    print("\n" + "=" * 60)
    print(f"完成情况（目标 = {target_reps} reps / HP 值）")
    print("=" * 60)
    for name, vals, key in sweeps_to_run:
        path = os.path.join(RESULT_DIR, f'hp_{name}.csv')
        if not os.path.exists(path):
            print(f"  hp_{name}.csv: 不存在")
            continue
        df = pd.read_csv(path)
        print(f"  hp_{name}.csv: 共 {len(df)} 行")
        for v in vals:
            n_v = int((df[key] == v).sum()) if key in df.columns else 0
            tag = "✓" if n_v >= target_reps else "⚠"
            print(f"    {tag}  {key}={v}: {n_v} reps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=30,
                        help='每个 HP 值的目标重复次数 (默认 30)')
    parser.add_argument('--only', nargs='+', default=None,
                        choices=['inter', 'intra', 'swap'],
                        help='只续跑指定的 sweep（可多选），默认全跑')
    parser.add_argument('--eps', type=float, default=EPS_FOR_HP,
                        help='隐私预算 (默认 2.0，需与原扫描一致)')
    parser.add_argument('--dataset', type=str, default='Chamelon',
                        help='数据集 (默认 Chamelon，与原扫描一致)')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    t_begin = time.time()
    print("=" * 60)
    print(f"Extend HP sweep")
    print(f"  target reps : {args.target}")
    print(f"  eps         : {args.eps}")
    print(f"  dataset     : {args.dataset}")
    print(f"  only        : {args.only or 'all (inter/intra/swap)'}")
    print(f"  started at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60, flush=True)

    # 数据只加载一次（Chameleon 上 precompute_reference 不便宜）
    mat0, n, ref = load_dataset(args.dataset)

    sweeps_to_run = HP_SWEEPS
    if args.only:
        sweeps_to_run = [s for s in HP_SWEEPS if s[0] in args.only]

    for name, vals, key in sweeps_to_run:
        print(f"\n##### 续跑 hp_{name}.csv → {args.target} reps #####")
        extend_one_sweep(name, vals, key, args.eps,
                         mat0, n, ref, args.target)

    print_completion_summary(sweeps_to_run, args.target)
    print(f"\n>>> 全部完成，用时 {(time.time() - t_begin) / 60:.1f} min")
    print(">>> 建议立刻运行：python hp_diagnose.py  查看新统计")


if __name__ == '__main__':
    main()