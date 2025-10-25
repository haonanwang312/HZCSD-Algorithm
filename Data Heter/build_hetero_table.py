# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build heterogeneity summary table from experiment folders.

Usage:
  1) Edit BASE_DIR to your path, e.g.:
     BASE_DIR = "/home/abc/Simulation_nn_robust_attacks-master/Data Homo log 1.5/HICZ_Heterogeneity_Compressed_SaveAdv"
  2) Run:
     python build_hetero_table.py
  3) Output:
     - hetero_overall_table.xlsx  (10 sheets: i=1..10, columns are serial indices)
     - hetero_overall_index.csv   (per-folder mapping: i, serial, combo, best_overall, csv_path)

Notes:
  - We assume folder names are like: labels_0-1-2-3 (order doesn't matter; we'll sort ascending).
  - "Best" overall is the MIN of the column named "overall" (preferred) or similar variants.
    Set TAKE_MIN = False if you want MAX instead.
  - Serial rule: for a fixed i, we enumerate all i-combinations of digits 0..9 in lexicographic order.
    The serial number is the index of that combination in the full list (starting from 0).
"""
import os
import re
import itertools
import pandas as pd

# ====== USER: EDIT THIS PATH ======
BASE_DIR = "/home/abc/Simulation_nn_robust_attacks-master/githup/Data Heter/HICZ_Heterogeneity_Compressed_SaveAdv"

# Whether to take min (True) or max (False) as "best" of the 'overall' metric
TAKE_MIN = True

FOLDER_PREFIX = "labels_"
CSV_FILE = "iter_metrics.csv"

def parse_digits_from_folder(name: str):
    """
    Parse digits from folder name like 'labels_0-1-2' -> [0,1,2].
    Return None if pattern doesn't match.
    """
    if not name.startswith(FOLDER_PREFIX):
        return None
    part = name[len(FOLDER_PREFIX):]
    # Allow trailing or odd names; filter digits separated by '-'
    try:
        digits = [int(x) for x in part.split('-') if x.strip() != ""]
    except ValueError:
        return None
    # sort ascending to have canonical representation
    digits = sorted(set(digits))
    # only allow digits 0..9
    if any((d < 0 or d > 9) for d in digits):
        return None
    if len(digits) == 0:
        return None
    return digits

def pick_overall_column(columns):
    """
    Choose which column to use as 'overall'.
    Priority list: exact 'overall', then case-insensitive matches containing 'overall',
    then variants like 'loss_overall', 'overall_loss'.
    """
    cols = list(columns)
    # exact match
    if "overall" in cols:
        return "overall"
    # case-insensitive contains
    lower_map = {c.lower(): c for c in cols}
    if "overall" in lower_map:
        return lower_map["overall"]
    # common variants preference order
    preferred_variants = ["loss_overall", "overall_loss", "overall_value", "avg_overall", "overall_mean"]
    for v in preferred_variants:
        if v in cols:
            return v
        if v in lower_map:
            return lower_map[v]
    # last resort: any column containing 'overall' substring (case-insensitive)
    for lc, orig in lower_map.items():
        if "overall" in lc:
            return orig
    return None

def read_best_overall(csv_path: str):
    """
    Read iter_metrics.csv and return best overall (min or max). Return None if not found.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Cannot read CSV: {csv_path} ({e})")
        return None
    col = pick_overall_column(df.columns)
    if col is None:
        print(f"[WARN] No 'overall' column in {csv_path}. Columns: {list(df.columns)}")
        return None
    series = pd.to_numeric(df[col], errors='coerce').dropna()
    if series.empty:
        print(f"[WARN] Column '{col}' empty/invalid in {csv_path}")
        return None
    return series.min() if TAKE_MIN else series.max()

def main():
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")
    # Scan folders
    entries = sorted(os.listdir(BASE_DIR))
    # Collect results: per i, map combo(tuple)->best_value
    per_i_results = {i: {} for i in range(1, 11)}
    index_rows = []  # for hetero_overall_index.csv
    
    # Precompute all lexicographic combo lists for 0..9
    all_combos = {i: list(itertools.combinations(range(10), i)) for i in range(1, 11)}
    serial_map = {i: {comb: idx for idx, comb in enumerate(all_combos[i])} for i in range(1, 11)}
    
    found = 0
    for name in entries:
        folder = os.path.join(BASE_DIR, name)
        if not os.path.isdir(folder):
            continue
        digits = parse_digits_from_folder(name)
        if digits is None:
            continue
        i = len(digits)
        if i < 1 or i > 10:
            continue
        combo = tuple(sorted(digits))
        if combo not in serial_map[i]:
            # Shouldn't happen since digits are 0..9 and unique, but guard anyway
            print(f"[WARN] Combo {combo} not in serial_map for i={i}. Skipping.")
            continue
        serial = serial_map[i][combo]
        csv_path = os.path.join(folder, CSV_FILE)
        best = read_best_overall(csv_path)
        if best is None:
            continue
        per_i_results[i][combo] = best
        index_rows.append({
            "folder": name,
            "i": i,
            "serial": serial,
            "combo": "-".join(map(str, combo)),
            "best_overall": best,
            "csv_path": csv_path
        })
        found += 1
    
    print(f"[INFO] Parsed {found} experiment folders under: {BASE_DIR}")
    
    # Write index CSV (flat mapping)
    idx_df = pd.DataFrame(index_rows).sort_values(["i", "serial"]).reset_index(drop=True)
    idx_csv = os.path.join(BASE_DIR, "hetero_overall_index.csv")
    idx_df.to_csv(idx_csv, index=False, encoding="utf-8")
    print(f"[OK] Wrote index CSV: {idx_csv}")
    
    # Build Excel with 10 sheets (one per i)
    xlsx_path = os.path.join(BASE_DIR, "hetero_overall_table.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        for i in range(1, 11):
            # For this i, create a Series indexed by serial with values = best_overall
            combos_for_i = all_combos[i]
            serials = []
            values = []
            # Iterate over all combos in lex order so serial = position
            for serial, combo in enumerate(combos_for_i):
                if combo in per_i_results[i]:
                    serials.append(serial)
                    values.append(per_i_results[i][combo])
            # Build a DataFrame: a single row (i) with columns = serials
            if len(serials) == 0:
                # empty sheet with a note
                empty_df = pd.DataFrame({"note": [f"No data found for i={i} in {BASE_DIR}"]})
                empty_df.to_excel(writer, sheet_name=f"i={i}", index=False)
                continue
            row_df = pd.DataFrame([values], index=[f"i={i}"], columns=serials)
            # Sort columns numerically
            row_df = row_df.reindex(sorted(row_df.columns), axis=1)
            row_df.to_excel(writer, sheet_name=f"i={i}")
    print(f"[OK] Wrote Excel table: {xlsx_path}")
    
    print("\nDone. If some folders are missing, their cells are simply absent in that row.")
    print("Serial definition: lexicographic index of combination among all C(10,i) combos.")
    print("Switch TAKE_MIN to False to record maximum instead of minimum.")

if __name__ == "__main__":
    main()
