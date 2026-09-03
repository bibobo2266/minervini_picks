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

T+1 開盤價：
  訊號在 T 日收盤產生，你最快 T+1 開盤才買得到。用收盤價當進場價會系統性
  高估績效，中小型股突破隔天跳空 2-5% 是常態。報表同時給兩組數字——
  「收盤起算」是訊號本身的品質，「開盤起算」是你實際拿得到的。
  兩者差距就是跳空成本，那才是決定這套能不能做的數字。

用法：
    python scripts/watchlist_replay.py                      # 2026 YTD
    python scripts/watchlist_replay.py --start 2026-01-01 --end 2026-08-31
    python scripts/watchlist_replay.py --start 2015-06-01 --years 13   # 全段
"""
import argparse
import datetime as dt
import os
import sys

import warnings

import numpy as np
import pandas as pd

# minervini_core 對資料不足的股票會做空切片平均，噪音蓋掉真正的訊息
warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minervini_core as M                      # noqa: E402
from report_minervini import scan_today         # noqa: E402
from watchlist_update import COLS, empty_wl, step   # noqa: E402

OUT = "out"
FWD = (20, 60)          # 進榜／觸發後幾個交易日回頭看報酬


def replay(start, end, liq, min_tt, years, progress=20,
           liq_pct=0, warmup_months=0):
    """逐日推進，回傳 (事件表, 每日在榜檔數, 價格矩陣 dict, 期末名單)。

    warmup_months：實際從 start 往前這麼多個月開始跑，但只回報 start 之後的
    事件。分年跑長期回測時必須設——名單是有狀態的，1/1 從空表開始跑，前兩個月
    的「進榜」會混進大量其實去年就該在榜的股票，把年初的事件數灌水。
    STALE_WEEKS = 8，所以三個月暖身足夠讓狀態收斂。
    """
    m_full, _ = M.build_matrices(years=years, as_of=end)
    uni = M.load_universe()

    warm = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)
            if warmup_months else pd.Timestamp(start))
    days = m_full["c"].index
    days = days[(days >= warm) & (days <= pd.Timestamp(end))]
    if len(days) == 0:
        raise SystemExit("指定區間內沒有交易日，檢查 --start/--end 或 --years 夠不夠")
    print(f"回放 {len(days)} 個交易日：{days[0].date()} → {days[-1].date()}"
          + (f"（其中 {warm.date()} → {pd.Timestamp(start).date()} 為暖身，"
             "事件不列入報表）" if warmup_months else "")
          + (f"　流動性門檻：前 {liq_pct}%" if liq_pct else ""))

    wl, hist = empty_wl(), empty_wl()
    events, daily_n = [], []

    for i, d in enumerate(days, 1):
        day = d.date()
        m_d = {k: v.loc[:d] for k, v in m_full.items()}
        try:
            scan = scan_today(liq, min_tt, m=m_d, uni=uni, liq_pct=liq_pct)
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
            if a["進榜狀態"] == "觸發":
                # 進榜當下就已突破的，也是一次觸發。不補這筆的話
                # 觸發率會漏算——例如 2026-09-02 那批 17 檔有 3 檔是這種。
                events.append(dict(日期=day, 代號=a["代號"], 名稱=a["名稱"],
                                   產業=a["產業"], 事件="觸發", 原因="進榜即觸發",
                                   收盤=a["進榜價"], RS=a["進榜RS"],
                                   底部序=a["進榜底部序"], 在榜天數=0))
        for p in promoted:
            sid = p.split()[0]
            r = new_wl[new_wl["代號"] == sid]
            g = r.iloc[0] if len(r) else None
            # 底部序改用進榜當時的值，不再留白。原本 NaN 會讓「等待後觸發」
            # 那批完全無法做底部序分組——553 筆有底部序的觸發全是進榜即觸發，
            # 分析時看起來像是兩者有差異，其實只是另一組沒有資料。
            events.append(dict(日期=day, 代號=sid,
                               名稱=g["名稱"] if g is not None else "",
                               產業=g["產業"] if g is not None else "",
                               事件="觸發", 原因="等待後觸發",
                               收盤=g["最新價"] if g is not None else np.nan,
                               RS=g["最新RS"] if g is not None else np.nan,
                               底部序=g["進榜底部序"] if g is not None else np.nan,
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
    if warmup_months and len(ev):
        s = pd.Timestamp(start).date()
        ev = ev[pd.to_datetime(ev["日期"]).dt.date >= s].reset_index(drop=True)
        dn = dn[pd.to_datetime(dn["日期"]).dt.date >= s].reset_index(drop=True)
    return ev, dn, m_full, wl


def _pick(mat, ev, scale=100.0):
    """把 date × stock_id 的矩陣依事件的 (日期, 代號) 取值。移出事件留白。"""
    out = []
    for _, e in ev.iterrows():
        d, sid = pd.Timestamp(e["日期"]), e["代號"]
        if e["事件"] == "移出" or d not in mat.index or sid not in mat.columns:
            out.append(np.nan)
        else:
            out.append(mat.at[d, sid] * scale)
    return np.round(out, 1)


def add_forward_returns(ev, c, o=None):
    """對每筆進榜／觸發，補 +N 交易日的報酬，以及同期全市場中位數當基準。

    基準用母體的橫斷面中位數，不是加權指數——這裡要問的是「名單選出來的
    有沒有比隨便抓一檔好」，不是要複製大盤走勢。

    兩組報酬：
      報酬nD    T 日收盤買進 → T+n 日收盤。訊號本身的品質，但買不到。
      報酬nD開  T+1 日開盤買進 → T+n 日收盤。你實際拿得到的。
    差額就是跳空成本。o 為 None（舊 parquet 沒有 open）時只出前者。
    """
    for n in FWD:
        f = c.shift(-n) / c - 1
        ev[f"報酬{n}D"] = _pick(f, ev)
        ev[f"基準{n}D"] = _pick(pd.DataFrame(
            np.repeat(f.median(axis=1).values[:, None], len(f.columns), axis=1),
            index=f.index, columns=f.columns), ev)

    if o is None:
        return ev

    entry = o.shift(-1)                    # T+1 開盤 = 可成交價
    ev["次日開盤"] = _pick(entry, ev, scale=1.0)
    ev["跳空%"] = _pick(entry / c - 1, ev)
    for n in FWD:
        fo = c.shift(-n) / entry - 1
        ev[f"報酬{n}D開"] = _pick(fo, ev)
        ev[f"基準{n}D開"] = _pick(pd.DataFrame(
            np.repeat(fo.median(axis=1).values[:, None], len(fo.columns), axis=1),
            index=fo.index, columns=fo.columns), ev)
    return ev


TOP_N = 20          # 摘要裡列幾檔；完整名單看 replay_events.csv
CHASE_MAX = 5       # 與 watchlist_update.CHASE_MAX 對齊；只用於分組顯示


def _med(g, col):
    return float(g[col].median()) if col in g.columns and g[col].notna().any() \
        else float("nan")


def gap_and_filters(ev):
    """跳空成本 + 三個候選濾網的分組表現。

    這一段是拿來決定要不要打開 watchlist_update 裡那幾個參數的。
    看的是「中位超額」——扣掉同期全市場中位數之後還剩多少。
    """
    trg = ev[ev["事件"] == "觸發"].copy()
    if trg.empty:
        return

    has_open = "報酬60D開" in trg.columns
    print("\n── 跳空成本（訊號在 T 收盤，實際 T+1 開盤才買得到）──")
    if not has_open:
        print("  無 open 資料，跳過。確認 data/adj 的 parquet 有 open 欄位、"
              "且 minervini_core.build_matrices 有讀進來。")
    else:
        g = trg["跳空%"].dropna()
        if len(g):
            print(f"  觸發日→次日開盤跳空：中位 {g.median():+.2f}%、"
                  f"平均 {g.mean():+.2f}%、n={len(g)}")
            print(f"  跳空超過 +3% 的比例：{(g > 3).mean():.0%}"
                  f"、超過 +5%：{(g > 5).mean():.0%}")
        for n in FWD:
            a, b = _med(trg, f"報酬{n}D"), _med(trg, f"報酬{n}D開")
            ea = a - _med(trg, f"基準{n}D")
            eb = b - _med(trg, f"基準{n}D開")
            if a == a:
                print(f"  {n}D 中位：收盤起算 {a:+.1f}（超額 {ea:+.1f}）　"
                      f"開盤起算 {b:+.1f}（超額 {eb:+.1f}）　"
                      f"差 {b - a:+.1f}")
        print("  讀法：如果開盤起算的超額接近 0 或為負，這個週期的訊號是做不出來的，"
              "不是策略無效，是進場價拿不到。")

    rcol = "報酬60D開" if has_open else "報酬60D"
    bcol = "基準60D開" if has_open else "基準60D"
    if rcol not in trg.columns:
        return
    t = trg.dropna(subset=[rcol]).copy()
    if t.empty:
        return
    t["超額"] = t[rcol] - t[bcol]
    tag = "（開盤起算）" if has_open else "（收盤起算，高估）"

    print(f"\n── 候選濾網的分組表現：60D 中位超額 {tag} ──")

    print("  底部序：")
    for k, g in t.dropna(subset=["底部序"]).groupby(
            t["底部序"].clip(upper=4).astype("Int64")):
        print(f"    {k}{'+' if k == 4 else ' '}　n={len(g):>4}　"
              f"中位超額 {g['超額'].median():+.1f}")
    print("    ⚠ 底部序 ≥ LATE_BASE 且未觸發的股票會被 check_exit 擋在名單外，"
          "這一欄是被截斷的樣本，不能用來論證原典對錯。")

    print("  RS：")
    for k, g in t.groupby(pd.cut(t["RS"], [0, 70, 80, 90, 95, 100]),
                          observed=True):
        print(f"    {str(k):<12} n={len(g):>4}　中位超額 {g['超額'].median():+.1f}")

    print("  觸發時在榜天數：")
    bins = [-1, 0, CHASE_MAX, 15, 30, 10**6]
    lab = ["0 進榜即觸發", f"1-{CHASE_MAX} 追價", f"{CHASE_MAX + 1}-15",
           "16-30", "30+"]
    for k, g in t.groupby(pd.cut(t["在榜天數"], bins, labels=lab),
                          observed=True):
        print(f"    {str(k):<14} n={len(g):>4}　中位超額 {g['超額'].median():+.1f}")

    t["月"] = pd.to_datetime(t["日期"]).dt.to_period("M").astype(str)
    if t["月"].nunique() <= 24:
        print("  每月（檢查上面的分組是不是被某一個月帶著走）：")
        for k, g in t.groupby("月"):
            print(f"    {k}　n={len(g):>4}　中位超額 {g['超額'].median():+.1f}")
        print("    單月樣本 < 30 或某月的超額是其他月的數倍時，整份分組表都不可信。")


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
            k = g[f"報酬{n}D"].notna().sum()
            if pd.isna(r):
                cells.append(f"{n}D 無（未來 {n} 日資料還沒發生）")
            else:
                cells.append(f"{n}D {r:+.1f}（{b:+.1f}）n={k}")
        print(f"  {label}　" + "　".join(cells))
    print("提醒：這段數字只說明名單方向，不是策略績效——沒有部位管理與停損，"
          "且區間內市場環境單一。")

    gap_and_filters(ev)

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


def yearly_stability(ev):
    """逐年攤開三個候選濾網，回傳矩陣 DataFrame 並列印。

    這是決定「該不該把濾網寫死」的唯一依據。把多年事件池在一起算一個
    中位數是錯的——觸發事件最多的必然是多頭年，池在一起等於變相只測多頭。
    要看的是**符號的一致性**，不是幅度：某個分組如果 11 年裡有 8 年為負，
    那是規則；只有 1 年為負，那是雜訊。
    """
    trg = ev[ev["事件"] == "觸發"].copy()
    has_open = "報酬60D開" in trg.columns
    rcol = "報酬60D開" if has_open else "報酬60D"
    bcol = "基準60D開" if has_open else "基準60D"
    if rcol not in trg.columns:
        return pd.DataFrame()
    t = trg.dropna(subset=[rcol]).copy()
    if t.empty:
        return pd.DataFrame()
    t["超額"] = t[rcol] - t[bcol]
    t["年"] = pd.to_datetime(t["日期"]).dt.year

    rows = []
    for y, g in t.groupby("年"):
        b1 = g[g["底部序"] == 1]["超額"]
        hi = g[g["RS"] >= 95]["超額"]
        ch = g[(g["在榜天數"] >= 1) & (g["在榜天數"] <= CHASE_MAX)]["超額"]
        rows.append({
            "年": y, "觸發數": len(g),
            "全體超額": round(float(g["超額"].median()), 1),
            "底部序1": round(float(b1.median()), 1) if len(b1) else np.nan,
            "n1": len(b1),
            "RS95+": round(float(hi.median()), 1) if len(hi) else np.nan,
            "nRS": len(hi),
            "追價": round(float(ch.median()), 1) if len(ch) else np.nan,
            "n追": len(ch),
            "跳空": round(float(g["跳空%"].median()), 2) if "跳空%" in g else np.nan,
        })
    yr = pd.DataFrame(rows).set_index("年")
    print("\n── 逐年穩定性"
          + ("（開盤起算）" if has_open else "（收盤起算，高估）") + " ──")
    print(yr.to_string())

    n = len(yr)
    for col, label in [("底部序1", "底部序 1"), ("RS95+", "RS ≥ 95"),
                       ("追價", f"追價（第 1-{CHASE_MAX} 天觸發）")]:
        s = yr[col].dropna()
        if not len(s):
            continue
        neg = int((s < 0).sum())
        verdict = ("符號穩定，可考慮寫死" if neg >= 0.75 * len(s)
                   else "符號不穩，不要寫死" if neg <= 0.5 * len(s)
                   else "傾向為負，但不夠穩，先當標記用")
        print(f"  {label}：{neg}/{len(s)} 年為負　← {verdict}")
    small = yr[yr["觸發數"] < 100].index.tolist()
    if small:
        print(f"  ⚠ 觸發數 < 100 的年份：{small}　這些年的分組數字不要單獨解讀")
    if n < 5:
        print("  ⚠ 只有 {} 年，任何「符號穩定」的判定都言之過早".format(n))
    return yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--liq", type=float, default=5000)
    ap.add_argument("--liq-pct", type=float, default=0,
                    help="百分位流動性門檻：25 = 取成交值前 25%%。跨年度回測"
                         "務必設，否則母體大小會跟著市場成交量漂移")
    ap.add_argument("--warmup-months", type=int, default=0,
                    help="往前多跑幾個月當暖身，事件不列入報表。分年跑長期"
                         "回測時設 3")
    ap.add_argument("--min-tt", type=int, default=8)
    ap.add_argument("--years", type=int, default=0,
                    help="載入幾年資料。預設自動＝區間長度＋2 年暖身"
                         "（趨勢模板要 250 日、底部序回看 378 日）")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    years = args.years or (end.year - start.year + 3
                           + (args.warmup_months + 11) // 12)

    ev, dn, m, wl_end = replay(start, end, args.liq, args.min_tt, years,
                               liq_pct=args.liq_pct,
                               warmup_months=args.warmup_months)
    if not ev.empty:
        ev = add_forward_returns(ev, m["c"], m.get("o"))
    monthly, ind, stock = report(ev, dn, wl_end)
    yearly = yearly_stability(ev) if not ev.empty else pd.DataFrame()

    os.makedirs(args.out, exist_ok=True)
    written = []
    for name, df, idx in [("replay_events.csv", ev, False),
                          ("replay_monthly.csv", monthly, True),
                          ("replay_industry.csv", ind, True),
                          ("replay_stocks.csv", stock, False),
                          ("replay_yearly.csv", yearly, True)]:
        if df is None or not len(df):
            continue
        p = os.path.join(args.out, name)
        df.to_csv(p, index=idx, encoding="utf-8-sig")
        written.append(p)
    print("\n已寫入：" + "、".join(written))


if __name__ == "__main__":
    main()
