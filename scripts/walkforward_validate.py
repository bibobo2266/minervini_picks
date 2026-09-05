r"""
走動式時序切分驗證（walk-forward）

為什麼需要這支：
    我在資金曲線那輪掃了六個股價上限（15/25/40/60/100/200），然後挑了最好的
    25 元回報 CAGR 17.2%。那是用全樣本挑參數、再用全樣本報告績效——
    正是 findings 附錄檢查清單第一條的錯誤：「樣本是不是我自己挑的」。
    這支的存在是為了回答：如果當年只看得到過去的資料，你會挑到什麼參數？
    用那個參數在未來一年的成績是多少？

做法（錨定式擴張窗口）：
    對每一年 Y：
      1. 只用「資料開始 → Y-1 年底」跑參數網格，挑出當時最好的股價上限
      2. 進入 Y 年之後，新進場的部位就用那個上限
      3. Y+1 年重新用「資料開始 → Y 年底」挑一次
    所有 Y 年的成績串起來，就是一條完全沒有偷看未來的資金曲線。

    刻意用「一條連續模擬」而不是「每年獨立跑一次」：
    這個策略的 edge 在右尾，贏家要抱 110–250 天。如果每年底強制平倉重來，
    會系統性砍掉跨年的大贏家，把結果壓低到沒有意義。所以部位照常跨年持有，
    只有「新進場」會套用當年的參數。

比較基準（同一段 OOS 期間）：
    A. 走動式選參數      ← 誠實的數字
    B. 固定無上限        ← 什麼參數都不挑
    C. 固定用全樣本最佳  ← 偷看未來，上限就是它
    D. 等權重市場指數
    A 和 C 的差距 = 我剛才那輪掃參數灌了多少水。
    A 和 B 的差距 = 這個參數到底值不值得挑。

⚠️ 這支仍然沒有解決的事：
    - 網格本身（要掃哪六個值）是我事後定的，這一層的挑選偏差消不掉
    - 隨機抽樣造成路徑差異，所以每組都跑多個種子取平均
    - 25 元以下的股票有多少實際買得到（全額交割、處置、量太薄）仍未查證

用法：
    python scripts/walkforward_validate.py
    python scripts/walkforward_validate.py --seeds 5 --start-year 2019
    python scripts/walkforward_validate.py --grid 2.5x40 --caps 15,25,40,60,100,200
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_backtest import (build_matrices, bench_curve, simulate,
                                metrics, load_etf, OUT_DIR)


def slice_all(mats, i0, i1):
    return [m.iloc[i0:i1 + 1] for m in mats]


def score(C, O, L, B, RAW, capital, pos_pct, max_pos, stop, cap, seeds,
          floor=0.0):
    """回傳多組種子的平均 MAR 與 CAGR。"""
    mars, cagrs = [], []
    for s in range(seeds):
        eq, _tr, _st = simulate(C, O, L, B, RAW, capital, pos_pct, max_pos,
                                stop, seed=s, maxprice=cap, minprice=floor)
        m, _ = metrics(eq, C.index, capital)
        if np.isfinite(m["MAR"]):
            mars.append(m["MAR"])
        cagrs.append(m["CAGR"])
    return (np.mean(mars) if mars else -99), float(np.mean(cagrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000_000)
    ap.add_argument("--stop", type=float, default=12.0)
    ap.add_argument("--grid", default="2.5x40", help="部位%%x最大檔數")
    ap.add_argument("--caps", default="25,30,40,60,100",
                    help="要掃的股價上限，0 = 無上限")
    ap.add_argument("--floors", default="10",
                    help="要掃的股價下限，逗號分隔。10 = 排除水餃股")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--select-seeds", type=int, default=2,
                    help="挑參數階段的種子數（會跑很多次，設少一點）")
    ap.add_argument("--start-year", type=int, default=2019,
                    help="第一個 OOS 年份；之前的年份全部拿來當訓練資料")
    ap.add_argument("--market", default="all", choices=["all", "twse", "tpex"])
    ap.add_argument("--signal", default="simple",
                    choices=["simple", "minervini"],
                    help="simple = 純 250 日新高；minervini = 再加八條趨勢模板")
    ap.add_argument("--breakout-days", type=int, default=250,
                    help="創幾日新高（250 / 500 / 750）")
    args = ap.parse_args()

    pp, mp = args.grid.split("x")
    pos_pct, max_pos = float(pp) / 100, int(mp)
    stop = args.stop / 100
    caps = [float(x) for x in args.caps.split(",")]
    floors = [float(x) for x in args.floors.split(",")]
    combos = [(f, c) for f in floors for c in caps if c == 0 or c > f]

    os.makedirs(OUT_DIR, exist_ok=True)   # 要在任何寫檔之前建好
    C, O, L, U, B, RAW = build_matrices(args.market, args.breakout_days,
                                        args.signal)
    dates = C.index
    years = sorted(set(dates.year))
    oos_years = [y for y in years if y >= args.start_year]
    print(f"資料 {dates[0].date()} → {dates[-1].date()}，"
          f"OOS 年份 {oos_years[0]}–{oos_years[-1]}，"
          f"{args.signal} 訊號 {int(B.values.sum())} 筆\n")

    # ---- 第一步：每年用「當年以前」的資料挑上限 ----
    chosen = {}
    for y in oos_years:
        i1 = int(np.searchsorted(dates, pd.Timestamp(f"{y}-01-01"))) - 1
        if i1 < 400:
            continue
        tr = slice_all([C, O, L, B, RAW], 0, i1)
        best, best_mar = None, -1e9
        line = []
        for fl, cap in combos:
            mar, cagr = score(*tr, args.capital, pos_pct, max_pos, stop,
                              cap, args.select_seeds, floor=fl)
            line.append(f"{int(fl)}-{int(cap) if cap else '∞'}:{mar:.2f}")
            if mar > best_mar:
                best, best_mar = (fl, cap), mar
        chosen[y] = best
        print(f"  訓練到 {y-1} 年底 → 選 {int(best[0])}–"
              f"{int(best[1]) if best[1] else '∞'} 元　(MAR {' '.join(line)})")

    # ---- 第二步：一條連續模擬，只有新進場套用當年參數 ----
    i0 = int(np.searchsorted(dates, pd.Timestamp(f"{oos_years[0]}-01-01")))
    oos = slice_all([C, O, L, B, RAW], i0, len(dates) - 1)
    od = oos[0].index
    dflt = (floors[0], caps[0])
    wf_caps = np.array([chosen.get(d.year, dflt)[1] for d in od], dtype=float)
    wf_floors = np.array([chosen.get(d.year, dflt)[0] for d in od], dtype=float)

    # 全樣本最佳（偷看未來的對照組）
    full_best, full_mar = None, -1e9
    for fl, cap in combos:
        mar, _ = score(C, O, L, B, RAW, args.capital, pos_pct, max_pos,
                       stop, cap, args.select_seeds, floor=fl)
        if mar > full_mar:
            full_best, full_mar = (fl, cap), mar

    fb = f"{int(full_best[0])}–{int(full_best[1]) if full_best[1] else '∞'}"
    runs = {
        "A 走動式選參數（誠實）": (wf_floors, wf_caps),
        f"B 固定 {int(floors[0])} 元以上不設上限": (floors[0], 0.0),
        f"C 固定用全樣本最佳 {fb} 元（偷看未來）": full_best,
    }
    rows = []
    for name, (fl, cap) in runs.items():
        ms = []
        for s in range(args.seeds):
            eq, tr, _st = simulate(*oos, args.capital, pos_pct, max_pos,
                                   stop, seed=s, maxprice=cap, minprice=fl)
            m, ann = metrics(eq, od, args.capital)
            m["交易筆數"] = len(tr)
            ms.append(m)
            if s == 0 and name.startswith("A"):
                pd.Series(eq, index=od).to_csv(
                    os.path.join(OUT_DIR, "equity_walkforward.csv"),
                    header=["equity"])
                wf_ann = ann
        r = pd.DataFrame(ms)
        rows.append({"組別": name, "CAGR": r.CAGR.mean(),
                     "最大回撤": r.最大回撤.mean(), "MAR": r.MAR.mean(),
                     "最差年度": r.最差年度.mean(),
                     "交易筆數": r.交易筆數.mean()})

    yrs = (od[-1] - od[0]).days / 365.25
    for sid in ("0050", "006208"):
        e = load_etf(sid)
        if e is None:
            continue
        e = e.reindex(od).ffill().bfill()
        peak = e.cummax()
        rows.append({"組別": f"D {sid} 買進持有（含息）",
                     "CAGR": ((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100,
                     "最大回撤": (e / peak - 1).min() * 100,
                     "MAR": np.nan, "最差年度": np.nan, "交易筆數": np.nan})
    for i, r in enumerate(rows):
        if r["組別"].startswith("D ") and np.isnan(r["MAR"]) \
                and r["最大回撤"] == r["最大回撤"] and r["最大回撤"]:
            rows[i]["MAR"] = r["CAGR"] / abs(r["最大回撤"])
    bench = bench_curve(C, U)[i0:]
    rows.append({"組別": "E 等權重市場指數（做不到的參考值）",
                 "CAGR": ((bench[-1] / bench[0]) ** (1 / yrs) - 1) * 100,
                 "最大回撤": np.nan, "MAR": np.nan, "最差年度": np.nan,
                 "交易筆數": np.nan})

    res = pd.DataFrame(rows).round(2)
    res.to_csv(os.path.join(OUT_DIR, "walkforward_summary.csv"),
               index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"OOS 期間 {od[0].date()} → {od[-1].date()}　"
          f"起始資金 {args.capital:,.0f}　{args.grid}　停損 -{args.stop:.0f}%　"
          f"種子 {args.seeds} 組")
    print(res.to_string(index=False))
    print("\nA 走動式的逐年報酬 %：")
    print("  " + "  ".join(f"{d.year}:{v*100:.0f}" for d, v in wf_ann.items()))
    print("\n判讀：")
    print("  A 明顯低於 C → 我上一輪的 17.2% 有相當比例是掃參數掃出來的")
    print("  A 不高於 B   → 這個參數不值得挑，固定下限以上全做就好")
    print("  A 不高於 D   → 打不贏直接買 0050，那就不值得做")
    print("  E 只是參考。它每天要對幾百檔再平衡，不是你能買的東西。")


if __name__ == "__main__":
    main()
