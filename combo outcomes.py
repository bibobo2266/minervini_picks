#!/usr/bin/env python3
"""
scripts/combo_outcomes.py — combo 命中之後發生什麼

主指標照既定量尺：單筆勝率、平均賺、平均賠、賠率、每筆期望值。
組合層級指標不算。基準用大盤（不用同族群平均——傳產會被族群輪動扣光）。

判準欄位（跑之前就定死，不事後挑）：
  1. 前後半段（2015–2020 vs 2021–）分開列
  2. 逐年正超額比例
  3. 中位數
  4. 前 3 大單筆佔總報酬比例  ← 樂透型的殺手欄位

進出場：T 日收盤判定 → T+1 開盤進場 → T+1+H 收盤出場。
已平倉偏誤：後續資料不足者一律剔除，不論輸贏。

  python scripts/combo_outcomes.py --universe universe_defense.csv --respect-thesis
  python scripts/combo_outcomes.py --all-market --min-amt 50000000 --start 2015-01-01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combo_core as cc  # noqa: E402

HORIZONS = [5, 20, 60]
SPLIT = pd.Timestamp("2021-01-01")


def collect(sid, px, idx, reg, lo, hi):
    feat = cc.compute_features(px, idx, reg)
    res = cc.evaluate_vectorized(feat, reg)

    n = len(feat)
    o = feat["open"].to_numpy()
    c = feat["close"].to_numpy()
    ic = feat["idx_close"].to_numpy()
    dt = feat["date"].to_numpy()

    win = np.ones(n, bool)
    if lo is not None:
        win &= feat["date"].to_numpy() >= np.datetime64(lo)
    if hi is not None:
        win &= feat["date"].to_numpy() <= np.datetime64(hi)

    recs = []
    for cid, m in res.items():
        sig = np.where(m["match"] & win)[0]
        for i in sig:
            e = i + 1                              # T+1 開盤進場
            if e >= n or not np.isfinite(o[e]):
                continue
            for H in HORIZONS:
                x = e + H
                if x >= n:                         # 資料不足 → 剔除，不論輸贏
                    continue
                r = (c[x] / o[e] - 1) * 100
                b = (ic[x] / ic[e] - 1) * 100 if np.isfinite(ic[e]) and ic[e] > 0 else np.nan
                recs.append((cid, sid, dt[i], H, r, r - b))
    del feat, res
    return recs


def agg(df: pd.DataFrame) -> dict:
    r = df["ret"].to_numpy()
    ex = df["excess"].to_numpy()
    w, l = r[r > 0], r[r <= 0]
    tot = r.sum()
    top3 = np.sort(r)[-3:].sum() if len(r) >= 3 else np.nan
    yr = df.assign(y=pd.to_datetime(df["date"]).dt.year).groupby("y")["excess"].mean()
    return {
        "n": len(r),
        "勝率%": round(100 * len(w) / len(r), 1),
        "平均賺%": round(w.mean(), 2) if len(w) else np.nan,
        "平均賠%": round(l.mean(), 2) if len(l) else np.nan,
        "賠率": round(abs(w.mean() / l.mean()), 2) if len(w) and len(l) and l.mean() != 0 else np.nan,
        "每筆期望值%": round(r.mean(), 2),
        "超額期望值%": round(np.nanmean(ex), 2),
        "中位數%": round(float(np.median(r)), 2),
        "前3大佔比%": round(100 * top3 / tot, 1) if np.isfinite(top3) and tot != 0 else np.nan,
        "逐年正超額%": round(100 * (yr > 0).mean(), 0) if len(yr) else np.nan,
        "年數": len(yr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_defense.csv")
    ap.add_argument("--all-market", action="store_true", help="改用 data/universe.parquet 全市場")
    ap.add_argument("--min-amt", type=float, default=50_000_000, help="全市場模式的 20 日均額門檻")
    ap.add_argument("--adj-dir", default="data/adj")
    ap.add_argument("--index", default="data/futures/index_taiex.parquet")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--respect-thesis", action="store_true")
    ap.add_argument("--min-n", type=int, default=20, help="低於此樣本數的 combo 不列入主表")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    reg = cc.load_registry(args.registry)
    idx = cc.load_index(args.index)
    lo = pd.Timestamp(args.start)
    hi = pd.Timestamp(args.end) if args.end else None
    years = list(range(lo.year, (hi.year if hi else pd.Timestamp.today().year) + 1))

    if args.all_market:
        u = pd.read_parquet("data/universe.parquet")
        u = u[~u["stock_id"].astype(str).str.contains(r"[BLRU]$", regex=True)]
        targets = [(str(s), None) for s in u["stock_id"].unique()]
    else:
        uni = pd.read_csv(args.universe, dtype=str)
        targets = [(str(r["stock_id"]),
                    pd.to_datetime(r.get("thesis_date"), format="mixed", errors="coerce")
                    if args.respect_thesis else None)
                   for _, r in uni.iterrows()]

    recs = []
    for k, (sid, th) in enumerate(targets, 1):
        px = cc.load_prices(args.adj_dir, [sid], years)
        if px.empty or len(px) < 300:
            continue
        if args.all_market:
            amt20 = px["amount"].rolling(20).mean() if "amount" in px else None
            if amt20 is None or not np.isfinite(amt20.iloc[-1]) or amt20.iloc[-1] < args.min_amt:
                del px
                continue
        slo = max(lo, th) if (th is not None and pd.notna(th)) else lo
        recs += collect(sid, px, idx, reg, slo, hi)
        del px
        if k % 50 == 0:
            print(f"  ...{k}/{len(targets)}　累計訊號 {len(recs)}")

    if not recs:
        sys.exit("沒有任何訊號。檢查資料路徑、日期區間或 thesis_date。")

    df = pd.DataFrame(recs, columns=["combo", "stock_id", "date", "H", "ret", "excess"])
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "combo_signals.csv"), index=False)

    meta = {c["id"]: c for c in reg["combos"]}
    lines = [f"# Combo 命中後的結果　母體 {len(targets)} 檔　訊號 {len(df)//len(HORIZONS)} 筆",
             f"\n區間 {args.start} ~ {args.end or '最新'}"
             f"{'（只算論點成立日之後）' if args.respect_thesis else ''}",
             "\n進出場：T 收盤判定 → T+1 開盤進場 → T+1+H 收盤出場。"
             "後續資料不足者一律剔除。基準＝大盤。\n"]

    for H in HORIZONS:
        sub = df[df["H"] == H]
        rows = []
        for cid, g in sub.groupby("combo"):
            if len(g) < args.min_n:
                continue
            a = agg(g)
            a.update({"combo": cid, "類": meta[cid]["cls"], "名稱": meta[cid]["name"]})
            for tag, gg in [("前段", g[pd.to_datetime(g["date"]) < SPLIT]),
                            ("後段", g[pd.to_datetime(g["date"]) >= SPLIT])]:
                a[f"{tag}n"] = len(gg)
                a[f"{tag}期望值%"] = round(gg["ret"].mean(), 2) if len(gg) else np.nan
            rows.append(a)
        if not rows:
            lines.append(f"\n## H={H} 日\n\n_無 combo 達到最低樣本數 {args.min_n}。_\n")
            continue
        t = pd.DataFrame(rows).sort_values("每筆期望值%", ascending=False)
        cols = ["combo", "類", "名稱", "n", "勝率%", "平均賺%", "平均賠%", "賠率",
                "每筆期望值%", "超額期望值%", "中位數%", "前3大佔比%", "逐年正超額%",
                "前段n", "前段期望值%", "後段n", "後段期望值%"]
        lines.append(f"\n## H={H} 日\n")
        lines.append(t[cols].to_markdown(index=False))
        lines.append("")

    lines += [
        "\n## 怎麼讀\n",
        "- **中位數為負但期望值為正** → 樂透型。少數幾筆撐起全部，實務上抱不住。",
        "- **前3大佔比 >50%** → 同上，更嚴重。這欄擋掉過 ELEC_MID。",
        "- **前段 n 太小** → 無法做前後半段檢驗。軍工母體結構上就是這樣，不是參數問題。",
        "- **超額期望值 ≈ 0** → 這組只是在標記大盤好日子，沒有個股資訊。",
        "- 樣本 <30 筆一律當方向參考，不當統計結論。\n",
        "_本表是歷史統計，不是勝率預測。60 組帶多重比較風險：不能挑出最漂亮的一組就宣稱可靠。_",
    ]
    with open(os.path.join(args.out, "combo_outcomes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n寫入 {args.out}/combo_outcomes.md　（原始訊號 combo_signals.csv）")


if __name__ == "__main__":
    main()
