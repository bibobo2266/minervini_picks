#!/usr/bin/env python3
"""
scripts/calibrate_registry.py — 用 panel 校準手冊參考線

把每條門檻換算成「全市場實際百分位」，寫回 combo_registry.json 的 calibration 區塊。
只做描述性校準：回答「這條線篩掉多少」，不碰報酬、不動門檻本身。
門檻要不要改是人的決定，程式只負責把實情印出來。

建議半年重跑一次。天天重算會讓昨天的卡片跟今天的卡片不可比。

  python scripts/calibrate_registry.py --start 2021-01-01
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

# 手冊參考線：欄位 -> {門檻: 說明}
REF = {
    "c01_tight":        {2.5: "極緊", 3.0: "壓縮線", 3.5: "中緊", 5.0: "寬幅線"},
    "c01_base_depth":   {15.0: "淺 base", 30.0: "深 base"},
    "c02_tests":        {3: "測頂下限", 5: "測頂上限", 6: "測試過多"},
    "c03_cq":           {0.40: "收低區", 0.70: "中段上緣", 0.85: "收高區"},
    "c05_volratio":     {1.2: "未明顯放量", 1.5: "突破量比", 3.0: "爆量"},
    "c05_contract":     {0.60: "量縮比"},
    "c06_rs":           {0.0: "領先/落後分界", 2.0: "領先參考"},
    "c07_er":           {0.15: "低效率", 0.40: "高效率"},
    "c08_bo_count":     {1: "初期", 3: "成熟上限", 4: "高齡"},
    "c08_gain_from_low": {50.0: "初期漲幅上限", 100.0: "高齡漲幅"},
    "c12_days_since_high": {15: "創高快", 30: "創高慢"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--min-amt", type=float, default=5e7)
    ap.add_argument("--out", default="reports/CALIBRATION.md")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.features}/features_*.parquet"))
    start = pd.Timestamp(args.start)
    cols = list(REF) + ["date", "amount"]
    buf = []
    for f in files:
        d = pd.read_parquet(f, columns=[c for c in cols])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= start]
        if args.min_amt > 0:
            d = d[d["amount"] >= args.min_amt]
        if len(d):
            buf.append(d.drop(columns=["date", "amount"]))
        del d
    if not buf:
        raise SystemExit("panel 裡沒有符合條件的資料")
    p = pd.concat(buf, ignore_index=True)
    del buf

    calib, lines = {}, []
    lines.append(f"# 手冊參考線校準　樣本 {args.start} 起，20 日均額 ≥ {args.min_amt/1e4:.0f} 萬")
    lines.append(f"\n樣本 {len(p):,} 檔-日。只做描述性校準：這條線篩掉多少，"
                 "不判斷它能不能賺錢。\n")
    lines.append("| 欄位 | 門檻 | 說明 | 實際百分位 | 高於它的佔 | 判讀 |")
    lines.append("|---|---|---|---|---|---|")

    for col, refs in REF.items():
        a = p[col].to_numpy(dtype="float64")
        a = a[np.isfinite(a)]
        if not len(a):
            continue
        qs = np.percentile(a, [5, 25, 50, 75, 95])
        calib[col] = {
            "n": int(len(a)),
            "p5": round(float(qs[0]), 4), "p25": round(float(qs[1]), 4),
            "p50": round(float(qs[2]), 4), "p75": round(float(qs[3]), 4),
            "p95": round(float(qs[4]), 4),
            "thresholds": {},
        }
        for t, label in refs.items():
            pct = float(100 * (a <= t).mean())
            verdict = ("⚠️ 過嚴，幾乎不成立" if pct < 5 else
                       "⚠️ 過鬆，篩不掉東西" if pct > 95 else
                       "篩選力弱（近半數）" if 40 < pct < 60 else "合理")
            calib[col]["thresholds"][str(t)] = {"pct": round(pct, 1), "label": label,
                                                "verdict": verdict}
            lines.append(f"| {col} | {t} | {label} | {pct:.1f} | "
                         f"{100-pct:.1f}% | {verdict} |")

    reg = json.load(open(args.registry, encoding="utf-8"))
    reg["calibration"] = {
        "sample_start": args.start, "min_amt": args.min_amt,
        "built_at": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "note": "全市場實際百分位。描述性，不含報酬。半年重跑一次；"
                "重跑會讓新舊卡片的分位不可比，請記錄版本。",
        "fields": calib,
    }
    json.dump(reg, open(args.registry, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n寫回 {args.registry} 的 calibration 區塊，報表 {args.out}")


if __name__ == "__main__":
    main()
