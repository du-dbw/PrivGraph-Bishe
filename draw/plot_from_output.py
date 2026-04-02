# -*- coding: utf-8 -*-
"""
用法：
1. 把原程序控制台输出完整复制到 RAW_TEXT 里
2. 运行：
   python plot_from_output.py
3. 会在当前目录生成：
   parsed_results.csv
   metrics_heatmaps.png
"""

import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1) 把你的原程序输出粘贴到这里
# =========================
RAW_TEXT = r"""
把你的控制台输出完整粘贴到这里，例如：

-----------epsilon=2.0,e1_r=0.1,e2_r=0.1,exper=1/10-------------
...
-----------------------------
dataset: Chamelon
epsilon= 2
all_nmi_arr= [0.5123]
all_evc_overlap= [0.2345]
all_evc_MAE= [0.1023]
all_deg_kl= [0.5566]
all_diam_rel= [0.2100]
all_cc_rel= [0.1200]
all_mod_rel= [0.3300]
All time:12.34s

-----------epsilon=2.0,e1_r=0.1,e2_r=0.2,exper=1/10-------------
...
-----------------------------
dataset: Chamelon
epsilon= 2
all_nmi_arr= [0.5223]
all_evc_overlap= [0.2445]
all_evc_MAE= [0.1123]
all_deg_kl= [0.5266]
all_diam_rel= [0.2200]
all_cc_rel= [0.1100]
all_mod_rel= [0.3500]
All time:11.90s
"""


def extract_float_from_list(text_line: str):
    """
    从类似 'all_nmi_arr= [0.5123]' 里提取 0.5123
    """
    m = re.search(r'=\s*(\[[^\]]*\])', text_line)
    if not m:
        return None
    try:
        arr = ast.literal_eval(m.group(1))
        if isinstance(arr, list) and len(arr) > 0:
            return float(arr[0])
    except Exception:
        pass
    return None


def parse_blocks(raw_text: str):
    """
    按每个参数组合的最终输出块解析结果
    """
    lines = raw_text.splitlines()

    records = []
    current_e1 = None
    current_e2 = None
    current_eps = None

    block_data = None

    header_pattern = re.compile(
        r'-+epsilon=([0-9.]+),e1_r=([0-9.]+),e2_r=([0-9.]+),exper=\d+/\d+-+'
    )

    for line in lines:
        line_strip = line.strip()

        # 识别实验头，记录当前 e1_r / e2_r / epsilon
        hm = header_pattern.match(line_strip)
        if hm:
            current_eps = float(hm.group(1))
            current_e1 = float(hm.group(2))
            current_e2 = float(hm.group(3))
            continue

        # 一个最终汇总块开始
        if line_strip.startswith("dataset:"):
            block_data = {
                "dataset": line_strip.split(":", 1)[1].strip(),
                "epsilon": current_eps,
                "e1_r": current_e1,
                "e2_r": current_e2,
                "all_nmi_arr": None,
                "all_evc_overlap": None,
                "all_evc_MAE": None,
                "all_deg_kl": None,
                "all_diam_rel": None,
                "all_cc_rel": None,
                "all_mod_rel": None,
            }
            continue

        if block_data is None:
            continue

        if line_strip.startswith("epsilon="):
            # 用当前块里的 epsilon，理论上和 header 一致
            try:
                block_data["epsilon"] = float(line_strip.split("=")[1].strip())
            except Exception:
                pass

        elif line_strip.startswith("all_nmi_arr="):
            block_data["all_nmi_arr"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_evc_overlap="):
            block_data["all_evc_overlap"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_evc_MAE="):
            block_data["all_evc_MAE"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_deg_kl="):
            block_data["all_deg_kl"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_diam_rel="):
            block_data["all_diam_rel"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_cc_rel="):
            block_data["all_cc_rel"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("all_mod_rel="):
            block_data["all_mod_rel"] = extract_float_from_list(line_strip)

        elif line_strip.startswith("All time:"):
            # 一个块结束
            if (
                block_data["e1_r"] is not None
                and block_data["e2_r"] is not None
                and block_data["all_nmi_arr"] is not None
            ):
                records.append(block_data)
            block_data = None

    return pd.DataFrame(records)


def draw_heatmap(ax, df, metric, title):
    pivot = df.pivot(index="e1_r", columns="e2_r", values=metric)
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    data = pivot.values

    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("e2_r")
    ax.set_ylabel("e1_r")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in pivot.index])

    # 在格子里写数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    df = parse_blocks(RAW_TEXT)

    if df.empty:
        raise ValueError(
            "没有解析到数据。请检查：\n"
            "1. 是否把完整控制台输出粘贴到 RAW_TEXT\n"
            "2. 输出里是否包含形如 e1_r=0.1,e2_r=0.2 的行\n"
            "3. 输出里是否包含 all_nmi_arr= [...] 这些汇总行"
        )

    # 去重：同一个 (e1_r, e2_r) 只保留最后一个
    df = df.drop_duplicates(subset=["e1_r", "e2_r"], keep="last").copy()

    # 保存解析结果
    df = df.sort_values(["e1_r", "e2_r"]).reset_index(drop=True)
    df.to_csv("parsed_results.csv", index=False, encoding="utf-8-sig")
    print("已保存解析结果: parsed_results.csv")
    print(df)

    metrics_info = [
        ("all_nmi_arr", "NMI"),
        ("all_evc_overlap", "EVC Overlap"),
        ("all_evc_MAE", "EVC MAE"),
        ("all_deg_kl", "Degree KL"),
        ("all_diam_rel", "Diameter Rel"),
        ("all_cc_rel", "Clustering Coef Rel"),
        ("all_mod_rel", "Modularity Rel"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics_info):
        draw_heatmap(axes[i], df, metric, title)

    # 多余子图删掉
    for j in range(len(metrics_info), len(axes)):
        fig.delaxes(axes[j])

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns and len(df) > 0 else "Unknown"
    eps = df["epsilon"].iloc[0] if "epsilon" in df.columns and len(df) > 0 else "Unknown"

    fig.suptitle(f"Dataset={dataset}, epsilon={eps}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("metrics_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("已保存图片: metrics_heatmaps.png")


if __name__ == "__main__":
    main()