r"""
準備名單回放（Watchlist Replay）

從某一天開始，逐個交易日重跑一次 watchlist 的進出判定，把整段歷史的
進榜／觸發／移出全部攤成事件表。

用途是**檢查名單本身**——移出門檻鬆不鬆、名單會不會養殭屍、進榜後才發生
的漲幅佔多少。不是策略績效驗證：名單不管部位大小也不管停損，而 2026 YTD
只涵蓋單一市場環境，看到漂亮的報酬數字不要當成策略可行的證據。

判定邏輯完全來自 watchlist_update.step() 與 report_minervini.scan_today()，
這裡不複製任何一條規則——回測與實跑分岔的話，回測就沒有意義。

as-of 正確性：
  * 流動性門檻在 minervini_core.scan() 裡用 mo.tail(60) 當場算，切片後
    自然是當日母體，不是「今天還活著、今天夠大」的名單。
  * data/adj/ 含下市股，倖存者偏差已由資料層處理。
  * universe.parquet 只用來查名稱／產業，不參與篩選，所以用最新版無妨
    （代價是當年的舊名稱會顯示成現在的名稱）。
  * build_matrices 的 ffill 只往前填，先建全表再切片與逐日重建等價。

用法：
    python scripts/watchlist_replay.py                      # 2026 YTD
    python scripts/watchlist_replay.py --start 2026-01-01 --end 2026-08-31
    python scripts/watchlist_replay.py --start 2015-06-01 --years 13   # 全段
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minervini_core as M                      # noqa: E402
from report_minervini import scan_today         # noqa: E402
from watchlist_update import COLS, empty_wl, step   # noqa: E402

OUT = "out"
FWD = (20, 60)          # 進榜／觸發後幾個交易日回頭看報酬


def replay(start, end, liq, min_tt, years, progress=20):
    """逐日推進，回傳 (事件表, 每日在榜檔數, 收盤矩陣)。"""
    m_full, _ = M.build_matrices(years=years, as_of=end)
    uni = M.load_universe()

    days = m_full["c"].index
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if len(days) == 0:
        raise SystemExit("指定區間內沒有交易日，檢查 --start/--end 或 --years 夠不夠")
    print(f"回放 {len(days)} 個交易日：{days[0].date()} → {days[-1].date()}")

    wl, hist = empty_wl(), empty_wl()
    events, daily_n = [], []

    for i, d in enumerate(days, 1):
        day = d.date()
        m_d = {k: v.loc[:d] for k, v in m_full.items()}
        try:
            scan = scan_today(liq, min_tt, m=m_d, uni=uni)
        except Exception as e:                      # 單日資料不足不該中斷整段
            print(f"  {day} 掃描失敗，跳過：{type(e).__name__} {e}")
            continue
        if scan.empty:
            daily_n.append((day, len(wl)))
            continue

        # verbose=False：逐日重放不需要「冷卻中，不收」那種每日訊息
        new_wl, removed, added, promoted = step(wl, hist, scan, day, verbose=False)

        for a in added:
            events.append(dict(日期=day, 代號=a["代號"], 名稱=a["名稱"],
                               產業=a["產業"], 事件="進榜", 原因=a["進榜狀態"],
                               收盤=a["進榜價"], RS=a["進榜RS"],
                               底部序=a["進榜底部序"], 在榜天數=0))
        for p in promoted:
            sid = p.split()[0]
            r = new_wl[new_wl["代號"] == sid]
            g = r.iloc[0] if len(r) else None
            events.append(dict(日期=day, 代號=sid,
                               名稱=g["名稱"] if g is not None else "",
                               產業=g["產業"] if g is not None else "",
                               事件="觸發", 原因="",
                               收盤=g["最新價"] if g is not None else np.nan,
                               RS=g["最新RS"] if g is not None else np.nan,
                               底部序=np.nan,
                               在榜天數=g["在榜天數"] if g is not None else np.nan))
        for r in removed:
            events.append(dict(日期=day, 代號=r["代號"], 名稱=r["名稱"],
                               產業=r["產業"], 事件="移出", 原因=r["移出原因"],
                               收盤=r["最新價"], RS=r["最新RS"],
                               底部序=np.nan, 在榜天數=r["在榜天數"]))

        if removed:
            hist = pd.concat([hist, pd.DataFrame(removed, columns=COLS)],
                             ignore_index=True)
        wl = new_wl
        daily_n.append((day, len(wl)))

        if progress and i % progress == 0:
            print(f"  {day}  在榜 {len(wl)}　累計事件 {len(events)}")

    ev = pd.DataFrame(events)
    dn = pd.DataFrame(daily_n, columns=["日期", "在榜"])
    return ev, dn, m_full["c"], wl


def add_forward_returns(ev, c):
    """對每筆進榜／觸發，補 +N 交易日的報酬，以及同期全市場中位數當基準。

    基準用母體的橫斷面中位數，不是加權指數——這裡要問的是「名單選出來的
    有沒有比隨便抓一檔好」，不是要複製大盤走勢。
    """
    for n in FWD:
        f = c.shift(-n) / c - 1
        bench = f.median(axis=1)
        col, bcol = f"報酬{n}D", f"基準{n}D"
        vals, bvals = [], []
        for _, e in ev.iterrows():
            d, sid = pd.Timestamp(e["日期"]), e["代號"]
            if e["事件"] == "移出" or d not in f.index or sid not in f.columns:
                vals.append(np.nan)
                bvals.append(np.nan)
                continue
            vals.append(f.at[d, sid] * 100)
            bvals.append(bench.at[d] * 100)
        ev[col] = np.round(vals, 1)
        ev[bcol] = np.round(bvals, 1)
    return ev


TOP_N = 20          # 摘要裡列幾檔；完整名單看 replay_events.csv


def report(ev, dn, wl_end):
    print("\n" + "=" * 56)
    if ev.empty:
        print("整段沒有任何事件——門檻可能太嚴，或區間太短")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    add = ev[ev["事件"] == "進榜"]
    trg = ev[ev["事件"] == "觸發"]
    rmv = ev[ev["事件"] == "移出"]

    print(f"進榜 {len(add)} 次（{add['代號'].nunique()} 檔）、"
          f"觸發 {len(trg)} 次、移出 {len(rmv)} 次、"
          f"期末仍在榜 {len(wl_end)} 檔")
    print(f"在榜檔數：中位數 {dn['在榜'].median():.0f}、"
          f"尖峰 {dn['在榜'].max():.0f}（{dn.loc[dn['在榜'].idxmax(), '日期']}）")

    print("\n移出原因分佈：")
    if len(rmv):
        vc = rmv["原因"].value_counts()
        for k, v in vc.items():
            print(f"  {k:<8} {v:>4}　{v / len(rmv):.0%}")
        rmv_days = pd.to_numeric(rmv["在榜天數"], errors="coerce")
        print(f"  平均在榜 {rmv_days.mean():.0f} 天"
              f"（中位數 {rmv_days.median():.0f}）")
    else:
        print("  無")

    if len(add):
        print(f"\n進榜後曾觸發：{trg['代號'].nunique()} / "
              f"{add['代號'].nunique()} 檔 = "
              f"{trg['代號'].nunique() / add['代號'].nunique():.0%}")

    print("\n進榜／觸發後報酬（中位數 %，括號為同期全市場中位數）：")
    for label, g in (("進榜", add), ("觸發", trg)):
        if g.empty:
            continue
        cells = []
        for n in FWD:
            r, b = g[f"報酬{n}D"].median(), g[f"基準{n}D"].median()
            cells.append(f"{n}D {r:+.1f}（{b:+.1f}）")
        print(f"  {label}　" + "　".join(cells))
    print("提醒：這段數字只說明名單方向，不是策略績效——沒有部位管理與停損，"
          "且區間內市場環境單一。")

    ev["月"] = pd.to_datetime(ev["日期"]).dt.to_period("M").astype(str)
    monthly = (ev.pivot_table(index="月", columns="事件", values="代號",
                              aggfunc="count").fillna(0).astype(int))
    dn2 = dn.copy()
    dn2["月"] = pd.to_datetime(dn2["日期"]).dt.to_period("M").astype(str)
    monthly = monthly.join(dn2.groupby("月")["在榜"].last().rename("月底在榜"))
    print("\n每月：")
    print(monthly.to_string())
    # ---- 族群 ----
    # 看的是「名單有沒有押在同一個族群上」。集中在一兩個產業的話，
    # 名單的分散度是假的——17 檔可能實質只是一個賭注。
    ind = pd.DataFrame(index=sorted(add["產業"].fillna("").unique()))
    ind["進榜"] = add.groupby("產業").size()
    ind["觸發"] = trg.groupby("產業").size()
    ind["移出"] = rmv.groupby("產業").size()
    ind = ind.fillna(0).astype(int)
    ind["觸發率"] = (ind["觸發"] / ind["進榜"].replace(0, np.nan) * 100).round(0)
    ind["報酬60D中位"] = add.groupby("產業")["報酬60D"].median().round(1)
    ind = ind.sort_values("進榜", ascending=False)
    print("\n族群（依進榜次數）：")
    print(ind.head(15).to_string())
    if len(add):
        top_share = ind["進榜"].iloc[0] / len(add)
        print(f"  最大族群佔全部進榜 {top_share:.0%}"
              + ("　← 名單過度集中，分散度是假的" if top_share > 0.35 else ""))

    # ---- 個股 ----
    stock = (add.groupby(["代號", "名稱", "產業"])
             .agg(進榜次數=("日期", "size"),
                  首次進榜=("日期", "min"),
                  報酬60D=("報酬60D", "median")).reset_index())
    trg_n = trg.groupby("代號").size().rename("觸發次數")
    stock = stock.merge(trg_n, on="代號", how="left").fillna({"觸發次數": 0})
    stock["觸發次數"] = stock["觸發次數"].astype(int)
    stock = stock.sort_values(["進榜次數", "報酬60D"], ascending=[False, False])
    print(f"\n個股（前 {TOP_N} 檔，完整名單見 replay_events.csv）：")
    print(stock.head(TOP_N).to_string(index=False))
    rep = stock[stock["進榜次數"] >= 3]
    if len(rep):
        print(f"  進榜 3 次以上的有 {len(rep)} 檔 ← 反覆進出，可能是冷卻期太短")

    print("\n讀法：逾期佔多數 → 進榜門檻太鬆；突破失敗佔多數 → 觸發判定有問題；"
          "階段破壞佔多數 → 進場時機太早；尖峰在榜遠超過每天看得完的檔數 → "
          "要加排序或收緊門檻。")
    print("=" * 56)
    return monthly, ind, stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--liq", type=float, default=5000)
    ap.add_argument("--min-tt", type=int, default=8)
    ap.add_argument("--years", type=int, default=0,
                    help="載入幾年資料。預設自動＝區間長度＋2 年暖身"
                         "（趨勢模板要 250 日、底部序回看 378 日）")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    years = args.years or (end.year - start.year + 3)

    ev, dn, c, wl_end = replay(start, end, args.liq, args.min_tt, years)
    if not ev.empty:
        ev = add_forward_returns(ev, c)
    monthly, ind, stock = report(ev, dn, wl_end)

    os.makedirs(args.out, exist_ok=True)
    written = []
    for name, df, idx in [("replay_events.csv", ev, False),
                          ("replay_monthly.csv", monthly, True),
                          ("replay_industry.csv", ind, True),
                          ("replay_stocks.csv", stock, False)]:
        if df is None or not len(df):
            continue
        p = os.path.join(args.out, name)
        df.to_csv(p, index=idx, encoding="utf-8-sig")
        written.append(p)
    print("\n已寫入：" + "、".join(written))


if __name__ == "__main__":
    main()
