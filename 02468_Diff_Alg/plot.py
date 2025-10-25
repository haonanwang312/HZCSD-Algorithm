# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_02468_diff_alg_ordered.py
- ( According to the name of the file ):
  2-bit, 4-bit, Top-50, Top-150, Rand-200, Rand-400, Norm-sgin, ZO
-  figsize=(8.8, 6)
- : bit=blue、Top=green、Rand=orange、ZO=purple、Nosgin( Norm-sgin)=light purple
- (2.0), (1.3)
- 
"""

import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

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
            ok = False
            if overall_key is not None:
                try:
                    ov = float(row.get(overall_key, ""))
                    if not math.isnan(ov):
                        ok = True
                except Exception:
                    ok = False
            if not ok:
                try:
                    vals = list(row.values())
                    ov = float(vals[1])
                except Exception:
                    idx += 1
                    continue

            it = float(row.get(iter_key, idx)) if has_iter else float(idx * 10)
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
    if "nosgin" in n or "norm-sgin" in n or "normsgin" in n: return "nosgin"
    return "other"

FAMILY_COLORS = {
    "bit":   ["#5AC8E3", "#1f77b4"],             # blue
    "top":   ["#57be8a", "#3FAE88"],             # green
    "rand":  ["#E07A26", "#D55E00"],             # orange
    "nosgin":["#c1a4d8", "#9A9A9A"],
    "zo":    ["#6A51A3"],                        # purple
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
    colors = FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])
    lss    = FAMILY_LINESTYLES.get(fam, FAMILY_LINESTYLES["other"])
    color = colors[idx % len(colors)]
    ls    = lss[idx % len(lss)]
    lw    = 1.3 if ls == "-" else 1.0  # Thick solid lines, thin dashed lines
    return dict(color=color, linestyle=ls, linewidth=lw)

# ---------- Main ----------
def main():
    root = Path("/home/abc/Simulation_nn_robust_attacks-master/02468_Diff_Alg/HICZ_Compressed").resolve()
    out_png = Path("0-2-4-6-8_ordered.png").resolve()
    out_svg = Path("0-2-4-6-8_ordered.svg").resolve()

    # Fixed order (matches folder names )
    wanted_order = [
        "2-bit", "4-bit",
        "Top-50", "Top-150",
        "Rand-200", "Rand-400",
        "Norm-sgin",
        "ZO",
    ]

    # Construct an ordered list (automatically skip and prompt for non-existent items)
    name_map = {p.name: p for p in root.iterdir() if p.is_dir()}
    ordered = []
    for n in wanted_order:
        if n in name_map:
            ordered.append(name_map[n])
        else:
            print(f"[WARN] Directory not found: {n} (will be skipped)")

    if not ordered:
        raise SystemExit("No valid subdirectories found")

    # Global style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10.8,
        "axes.linewidth": 0.9,
    })

    # Single-column friendly: slightly wider horizontally
    fig, ax = plt.subplots(figsize=(8.8, 6))
    fam_counters = {}

    for sd in ordered:
        csv_path = sd / "iter_metrics.csv"
        iters, overs = read_iter_overall(csv_path)
        if not iters or not overs:
            print(f"[WARN] Skipped: {sd.name} (no valid data)")
            continue

        # Sort and draw 
        pairs = sorted(zip(iters, overs), key=lambda x: x[0])
        iters = [p[0] for p in pairs]
        overs = [p[1] for p in pairs]
        best_curve = running_min(overs)

        style = choose_style(sd.name, fam_counters)
        ax.plot(
            iters, best_curve,
            label=sd.name,
            alpha=0.96,
            drawstyle="steps-post",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    # grid & axis
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    # ax.set_title("0-2-4-6-8")
    ax.set_ylim(bottom=10)
    ax.set_xlim(left=0)
    ax.set_xlim(right=25000)

    leg = ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="gray",
        fancybox=True,
        columnspacing=0.8,
        handlelength=2.5,
        borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=1.1)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_svg)
    print(f"[OK] save:{out_png}")
    print(f"[OK] save:{out_svg}")

if __name__ == "__main__":
    main()


