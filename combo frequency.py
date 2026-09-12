#!/usr/bin/env python3
"""
scripts/combo_frequency.py — combo 命中頻率與重疊率

回答兩件事：
  1. 每組 combo 在歷史上多久出現一次（決定學習順序）
  2. 哪幾組總是同時亮（揭穿重複證據）

python scripts/combo_frequency.py --universe universe_defense.csv --start 2015-01-01
輸出：reports/combo_frequency.md + combo_overlap.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combo_core as cc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_defense.csv")
    ap.add_argument("--adj-dir", default="data/adj")
    ap.add_argument("--index", default="data/futures/index_taiex.parquet")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--respect-thesis", action="store_true",
                    help="只統計論點成立日之後（軍工主題請開）")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    reg = cc.load_registry(args.registry)
    uni = pd.read_csv(args.universe, dtype=str)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end) if args.end else None
    years = list(range(start.year, (end.year if end else pd.Timestamp.today().year) + 1))
    idx = cc.load_index(args.index)
    T = cc._T(reg["thresholds"])

    ids = [c["id"] for c in reg["combos"]]
    hit = {i: 0 for i in ids}
    part = {i: 0 for i in ids}
    stock_days = 0
    cooc = pd.DataFrame(0, index=ids, columns=ids, dtype=int)

    for _, u in uni.iterrows():
        sid = str(u["stock_id"])
        px = cc.load_prices(args.adj_dir, [sid], years)
        if px.empty or len(px) < 300:
            print(f"[SKIP] {sid}")
            continue
        feat = cc.compute_features(px, idx, reg)

        lo = start
        if args.respect_thesis:
            th = pd.to_datetime(u.get("thesis_date"), format="mixed", errors="coerce")
            if pd.notna(th):
                lo = max(lo, th)
        m = feat["date"] >= lo
        if end is not None:
            m &= feat["date"] <= end
        if not m.any():
            continue

        res = cc.evaluate_vectorized(feat, reg)
        mwin = m.to_numpy()
        stock_days += int(mwin.sum())
        mats = {cid: (r["match"] & mwin) for cid, r in res.items()}
        for cid, r in res.items():
            hit[cid] += int(mats[cid].sum())
            part[cid] += int((r["partial"] & mwin).sum())
        for a, b in itertools.combinations(ids, 2):
            both = int((mats[a] & mats[b]).sum())
            if both:
                cooc.loc[a, b] += both
                cooc.loc[b, a] += both
        print(f"[OK] {sid} 累計 {stock_days} 檔-日")
        del px, feat

    os.makedirs(args.out, exist_ok=True)
    meta = {c["id"]: c for c in reg["combos"]}
    rows = []
    for i in ids:
        n = hit[i]
        rows.append({
            "combo": i, "類": meta[i]["cls"], "名稱": meta[i]["name"],
            "命中檔-日": n,
            "命中率%": round(100 * n / stock_days, 3) if stock_days else 0,
            "PARTIAL": part[i],
            "gate": meta[i].get("gate", ""),
            "分級": ("死組（<0.05%）" if stock_days and n / stock_days < 0.0005 else
                    "泛用（>5%）" if stock_days and n / stock_days > 0.05 else "可用"),
        })
    freq = pd.DataFrame(rows).sort_values("命中檔-日", ascending=False)

    # Jaccard 重疊
    ov = []
    for a, b in itertools.combinations(ids, 2):
        inter = cooc.loc[a, b]
        if inter == 0:
            continue
        union = hit[a] + hit[b] - inter
        if union > 0:
            ov.append({"A": a, "B": b, "同時命中": int(inter),
                       "Jaccard": round(inter / union, 3),
                       "A名": meta[a]["name"], "B名": meta[b]["name"]})
    ovdf = pd.DataFrame(ov).sort_values("Jaccard", ascending=False) if ov else pd.DataFrame()

    freq.to_csv(os.path.join(args.out, "combo_frequency.csv"), index=False)
    if len(ovdf):
        ovdf.to_csv(os.path.join(args.out, "combo_overlap.csv"), index=False)

    with open(os.path.join(args.out, "combo_frequency.md"), "w", encoding="utf-8") as f:
        f.write(f"# Combo 命中頻率　母體 {len(uni)} 檔／{stock_days} 檔-日\n\n")
        f.write(f"區間 {args.start} ~ {args.end or '最新'}"
                f"{'（只算論點成立日之後）' if args.respect_thesis else ''}\n\n")
        f.write(freq.to_markdown(index=False))
        f.write("\n\n## 重疊率最高的 20 對（Jaccard）\n\n")
        f.write("_同時命中率高 = 講同一件事兩次，不必分別學。_\n\n")
        f.write(ovdf.head(20).to_markdown(index=False) if len(ovdf) else "_無同時命中紀錄_")
        f.write("\n\n## 讀法\n\n")
        f.write("- **死組**：命中率 <0.05%，一年遇不到幾次，先不學。\n")
        f.write("- **泛用**：命中率 >5%，每天一堆，資訊量低。\n")
        f.write("- **Jaccard >0.6**：兩組實質重複，挑一組學就好。\n")
        f.write("- PARTIAL 高的組代表卡在缺值（多半是 C05 高檔換手已 DISABLED）。\n")
    print(f"\n寫入 {args.out}/combo_frequency.md")


if __name__ == "__main__":
    main()
