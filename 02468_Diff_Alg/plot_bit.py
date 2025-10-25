# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# File: plot_loss_vs_bits.py
# Function: Plot the curve of loss vs. communication bits.

# Fixed folder order:
#   2-bit, 4-bit, Top-50, Top-150, Rand-200, Rand-400, Norm-sgin, ZO

# Communication overhead (in bits) per round for each algorithm type:
  2-bit    : (2+1)*784 + 64
  4-bit    : (4+1)*784 + 64
  Top-50   : 50*64
  Top-150  : 150*64
  Rand-200 : 200*64
  Rand-400 : 400*64
  Norm-sgin: 784 + 64
  ZO       : 784*64
"""

import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Communication bits per round for each algorithm type ----------
BITS_PER_ITER = {
    "2-bit": (2 + 1) * 784 + 64,
    "4-bit": (4 + 1) * 784 + 64,
    "Top-50": 50 * 64,
    "Top-150": 150 * 64,
    "Rand-200": 200 * 64,
    "Rand-400": 400 * 64,
    "Norm-sgin": 784 + 64,
    "ZO": 784 * 64,
}

# ---------- read CSV ----------
def read_iter_overall(csv_path: Path):
    iters, overs = [], []
    if not csv_path.exists():
        return iters, overs
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or []
        lower = [fn.lower() for fn in fns]
        has_iter = "iter" in lower
        iter_key = fns[lower.index("iter")] if has_iter else None
        overall_key = fns[lower.index("overall")] if "overall" in lower else None

        idx = 0
        for row in reader:
            try:
                ov = float(row.get(overall_key, list(row.values())[1]))
            except Exception:
                idx += 1
                continue
            it = float(row.get(iter_key, idx)) if has_iter else float(idx)
            iters.append(it)
            overs.append(ov)
            idx += 1
    return iters, overs

def running_min(values):
    out, cur = [], None
    for v in values:
        cur = v if cur is None or v < cur else cur
        out.append(cur)
    return out

# ---------- Family and Style ----------
def get_family(name: str):
    n = name.lower()
    if "rand" in n: return "rand"
    if "top" in n: return "top"
    if "bit" in n: return "bit"
    if "zo" == n or n.endswith("/zo"): return "zo"
    if "norm-sgin" in n or "nosgin" in n: return "nosgin"
    return "other"

FAMILY_COLORS = {
    "bit":   ["#5AC8E3", "#1f77b4"],    # blue
    "top":   ["#57be8a", "#3FAE88"],    # green
    "rand":  ["#E07A26", "#D55E00"],    # orange
    "nosgin":["#c1a4d8", "#9A9A9A"],
    "zo":    ["#6A51A3"],               # purple
    "other": ["#9467bd"],
}
FAMILY_LINESTYLES = {
    "bit":   ["--", "-"],
    "top":   ["--", "-"],
    "rand":  ["--", "-"],
    "nosgin":["-"],
    "zo":    ["-"],
    "other": ["-"],
}

def choose_style(name: str, counters: dict):
    fam = get_family(name)
    idx = counters.get(fam, 0)
    counters[fam] = idx + 1
    color = FAMILY_COLORS.get(fam, ["#444"])[idx % len(FAMILY_COLORS[fam])]
    ls = FAMILY_LINESTYLES.get(fam, ["-"])[idx % len(FAMILY_LINESTYLES[fam])]
    lw = 1.1 if ls == "-" else 1.1
    return dict(color=color, linestyle=ls, linewidth=lw)

# ---------- Main ----------
def main():
    root = Path("/home/abc/Simulation_nn_robust_attacks-master/02468_Diff_Alg/HICZ_Compressed").resolve()
    out_png = Path("loss_vs_bits.png").resolve()
    out_svg = Path("loss_vs_bits.svg").resolve()

    order = [
        "2-bit", "4-bit",
        "Top-50", "Top-150",
        "Rand-200", "Rand-400",
        "Norm-sgin", "ZO",
    ]

    # Find the actually existing folders
    name_map = {p.name: p for p in root.iterdir() if p.is_dir()}
    ordered = [name_map[n] for n in order if n in name_map]

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.9,
    })

    fig, ax = plt.subplots(figsize=(8.6, 6))
    fam_counters = {}

    for sd in ordered:
        csv_path = sd / "iter_metrics.csv"
        iters, overs = read_iter_overall(csv_path)
        if not iters or not overs:
            continue

        # Cumulative communication bits
        bits_per_iter = BITS_PER_ITER.get(sd.name, 0)
        bits = [i * bits_per_iter for i in iters]

        best_curve = running_min(overs)
        style = choose_style(sd.name, fam_counters)

        ax.plot(
            bits, best_curve,
            label=sd.name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.97,
        )

    # axis
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel(" Communication bits")
    ax.set_ylabel("Loss")
    # ax.set_xscale("log")   
    ax.set_ylim(bottom=10)
    ax.set_xlim(left=0)
    ax.set_xlim(right=1.5e7)

    
    leg = ax.legend(
        loc="upper right", ncol=2,
        frameon=True, framealpha=0.88,
        edgecolor="gray", facecolor="white",
        columnspacing=0.8, handlelength=2.4,
        borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=1.1)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_svg)
    print("[OK] save:", out_png)

if __name__ == "__main__":
    main()
