r"""
資金曲線回測：把逐筆超額報酬變成有限資本的真實帳戶模擬

為什麼需要這支：
    目前所有結論都是「每筆交易的平均超額報酬」。那個數字看不到三件事——
    複利、滿倉時買不下去的排擠、最大回撤。而市場層級開關（Weinstein）到底
    值不值得做，只能用回撤來判斷：空手一年的超額報酬是 0，看起來沒好處，
    但它避開的回撤在逐筆指標裡完全不會出現。
    所以資金曲線必須先做，它是評估開關的度量衡。

⚠️ 台股的整張約束不是「縮放」：
    一張 = 1000 股。100 萬本金、單筆 10% = 10 萬，股價超過 100 元就連一張
    都買不起。這是硬性的非線性約束，不是把縱軸乘十倍。
    實測：單筆 10 萬買得起整張的訊號佔 71%，而這 71% 的平均超額是 +4.23，
    高於全體的 +3.54——買不到的那群本來就是表現最差的。約束跟 edge 同方向。

⚠️ 零股不能用來解除這個約束：
    盤中零股第一次撮合是 9:10，刻意避開普通交易開盤作業，所以零股結構上
    沒有開盤價；本策略是「隔日開盤買」，零股拿不到那個價。零股又只能限價、
    不能市價，突破跳空時可能根本不成交。整股與零股是獨立市場，價格也不同。
    （2026-12-07 起零股首撮提前到 9:00，但仍是獨立市場，回測仍對不上。）

⚠️ 張數約束要用「實際成交股價」，不是還原股價：
    還原價會把配息多的老股票壓得很低，2016 年的還原價可能只有實際價的一半。
    本程式用 dividend_events.parquet 把還原因子反推回未調整價來算張數。
    損益本身仍用還原價計算（含息總報酬假設）。
    限制：dividend_events 只到 2017-01，2015–2016 的除權息缺一部分，
    那兩年的實際股價會被低估一點，張數會算得寬鬆一些。

滿倉時直接跳過，不排隊：
    排隊會改變訊號本身。5/1 突破、5/20 才買到，那不是「250 日新高次日買」，
    是「17 天前曾經突破的股票現在買」，是另一個策略。

同日訊號多於空位時隨機抽：
    這是刻意的。任何主觀篩選都在隨機丟掉右尾。用 --seeds 跑多組隨機種子，
    看結果分佈而不是單一路徑。

用法：
    python scripts/portfolio_backtest.py                      # 預設矩陣
    python scripts/portfolio_backtest.py --seeds 20
    python scripts/portfolio_backtest.py --grid 10x10 --seeds 50
    python scripts/portfolio_backtest.py --capital 1000000 --stop 12
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data/adj"
OUT_DIR = "data/portfolio"
LOT = 1000                 # 一張 = 1000 股
BREAKOUT_DAYS = 250        # 創幾日新高
REENTRY_GAP = 20           # 同一檔幾個交易日內不重複進場
MAX_HOLD = 250             # 最大持有交易日數
COST = 0.006               # 來回 0.60%（手續費 + 交易稅）
UNIVERSE_PCT = 0.25        # 母體：當日成交值前 25%


def build_matrices(market="all"):
    fs = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))
    if not fs:
        raise SystemExit(f"{DATA_DIR} 底下沒有 prices_adj_*.parquet")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])

    import minervini_core as M
    uni = M.load_universe()
    types = ["twse", "tpex"] if market == "all" else [market]
    listed = set(uni[uni["type"].isin(types)].index.astype(str))
    print(f"  市場別 {market}：{len(listed)} 檔")
    d = d[d["stock_id"].astype(str).str.match(r"^[1-9]\d{3}$")]
    d = d[d["stock_id"].isin(listed) & (d["close"] > 0)]

    C = d.pivot_table(index="date", columns="stock_id", values="close")
    O = d.pivot_table(index="date", columns="stock_id", values="open")
    L = d.pivot_table(index="date", columns="stock_id", values="min")
    T = d.pivot_table(index="date", columns="stock_id", values="Trading_money")

    # 母體：當日成交值前 25%
    U = (T.rank(axis=1, pct=True, ascending=False) <= UNIVERSE_PCT) & C.notna()
    # 250 日新高，且前一日不是新高
    prior = C.shift(1).rolling(BREAKOUT_DAYS, min_periods=BREAKOUT_DAYS).max()
    B = (C > prior) & (C.shift(1) <= prior.shift(1)) & U

    # 還原因子 → 實際成交股價（算張數用）
    F = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    ev_path = os.path.join(DATA_DIR, "dividend_events.parquet")
    if os.path.exists(ev_path):
        ev = pd.read_parquet(ev_path)
        ev["date"] = pd.to_datetime(ev["date"])
        ev = ev[ev["stock_id"].isin(C.columns)].sort_values("date")
        ev["f"] = ev["after_price"] / ev["before_price"]
        for sid, g in ev.groupby("stock_id"):
            # 某日的因子 = 該日之後所有除權息 (after/before) 的連乘
            s = pd.Series(g["f"].values, index=g["date"].values)
            s = s.reindex(C.index, fill_value=1.0)
            F[sid] = s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    else:
        print("  ⚠️ 找不到 dividend_events.parquet，張數會用還原價算，會偏寬鬆")
    RAW = C / F        # 近似的實際成交股價
    return C, O, L, U, B, RAW


def bench_curve(C, U):
    c = C.values
    r = c[1:] / c[:-1] - 1
    m = U.values[:-1] & np.isfinite(r)
    n = m.sum(1)
    daily = np.where(n > 0, np.nansum(np.where(m, r, 0), 1) / np.maximum(n, 1), 0.0)
    return np.concatenate([[1.0], np.cumprod(1 + daily)])


def simulate(C, O, L, B, RAW, capital, pos_pct, max_pos, stop_pct, seed, maxprice=0.0):
    """事件驅動的有限資本模擬。回傳 (每日權益, 交易明細, 統計)。"""
    rng = np.random.default_rng(seed)
    dates = C.index
    sids = np.array(C.columns)
    c, o, lo, raw = C.values, O.values, L.values, RAW.values
    sig_i, sig_j = np.where(B.values)
    sig_by_day = {}
    for i, j in zip(sig_i, sig_j):
        sig_by_day.setdefault(i, []).append(j)

    cash = float(capital)
    pending = 0.0                      # 賣出價款，下一個交易日才可用
    pos = {}                           # j -> dict
    last_entry = {}
    equity = np.zeros(len(dates))
    trades, skipped = [], []
    full_days = invested = 0

    for t in range(len(dates)):
        cash += pending
        pending = 0.0

        # --- 出場（先做，空出來的位置當天不放行，資金隔日才可用）---
        for j in list(pos):
            p = pos[j]
            px = None
            if np.isfinite(lo[t, j]) and lo[t, j] <= p["stop"]:
                op = o[t, j]
                px = op if (np.isfinite(op) and op <= p["stop"]) else p["stop"]
                why = "停損"
            elif t - p["entry_i"] >= MAX_HOLD and np.isfinite(c[t, j]):
                px = c[t, j]
                why = "到期"
            if px is not None:
                gross = px * p["shares"]
                pending += gross * (1 - COST / 2)
                r = px / p["entry_px"] - 1 - COST
                trades.append(dict(sid=sids[j], 進場日=dates[p["entry_i"]],
                                   出場日=dates[t], 進場價=p["entry_px"],
                                   出場價=px, 實際股價=p["raw_px"],
                                   張數=p["shares"] // LOT, 報酬=r * 100,
                                   持有日=t - p["entry_i"], 出場原因=why))
                del pos[j]

        # --- 進場：前一日收盤產生的訊號，今日開盤買 ---
        cands = sig_by_day.get(t - 1, [])
        cands = [j for j in cands
                 if j not in pos and t - last_entry.get(j, -10**9) > REENTRY_GAP
                 and np.isfinite(o[t, j]) and o[t, j] > 0]
        slots = max_pos - len(pos)
        mark = np.array([c[t, j] if np.isfinite(c[t, j]) else 0.0
                         for j in pos], dtype=float)
        cur_eq = cash + float((mark * np.array([pos[j]["shares"]
                                                for j in pos])).sum()) if pos else cash
        budget = cur_eq * pos_pct

        if cands:
            if len(cands) > slots:
                # 訊號多於空位 → 隨機抽，不做主觀篩選
                take = list(rng.choice(cands, size=max(slots, 0), replace=False)) \
                    if slots > 0 else []
                for j in cands:
                    if j not in take:
                        skipped.append((dates[t], sids[j], "滿倉"))
            else:
                take = cands
            for j in take:
                entry_px = o[t, j]
                raw_px = raw[t, j] if np.isfinite(raw[t, j]) and raw[t, j] > 0 else entry_px
                cap = maxprice[t] if hasattr(maxprice, "__len__") else maxprice
                if cap and raw_px > cap:
                    skipped.append((dates[t], sids[j], "超過股價上限"))
                    continue
                lots = int(budget // (raw_px * LOT))
                if lots < 1:
                    skipped.append((dates[t], sids[j], "買不起整張"))
                    continue
                shares = lots * LOT
                cost = entry_px * shares * (1 + COST / 2)
                if cost > cash:
                    lots = int(cash // (entry_px * LOT * (1 + COST / 2)))
                    if lots < 1:
                        skipped.append((dates[t], sids[j], "現金不足"))
                        continue
                    shares = lots * LOT
                    cost = entry_px * shares * (1 + COST / 2)
                cash -= cost
                pos[j] = dict(shares=shares, entry_px=entry_px, raw_px=raw_px,
                              stop=entry_px * (1 - stop_pct), entry_i=t)
                last_entry[j] = t

        mv = sum(pos[j]["shares"] * (c[t, j] if np.isfinite(c[t, j])
                                     else pos[j]["entry_px"]) for j in pos)
        equity[t] = cash + mv
        if len(pos) >= max_pos:
            full_days += 1
        invested += (mv / equity[t]) if equity[t] > 0 else 0

    stats = dict(滿倉日比=full_days / len(dates) * 100,
                 平均投入比=invested / len(dates) * 100,
                 跳過訊號=len(skipped),
                 因滿倉跳過=sum(1 for s in skipped if s[2] == "滿倉"),
                 因買不起跳過=sum(1 for s in skipped if s[2] == "買不起整張"),
                 因超過上限跳過=sum(1 for s in skipped if s[2] == "超過股價上限"))
    return equity, pd.DataFrame(trades), stats


def metrics(equity, dates, capital):
    eq = pd.Series(equity, index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1
    peak = eq.cummax()
    dd = eq / peak - 1
    mdd = dd.min()
    under = (dd < -0.01)
    longest = 0
    run = 0
    for v in under:
        run = run + 1 if v else 0
        longest = max(longest, run)
    ann = eq.resample("YE").last().pct_change()
    ann.iloc[0] = eq.resample("YE").last().iloc[0] / capital - 1
    return dict(期末權益=eq.iloc[-1], CAGR=cagr * 100, 最大回撤=mdd * 100,
                MAR=(cagr / abs(mdd) if mdd else np.nan),
                最差年度=ann.min() * 100, 最長水下交易日=longest), ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--stop", type=float, default=12.0, help="停損 %%")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--market", default="all", choices=["all", "twse", "tpex"],
                    help="⚠️ universe 的 type 是現值，轉板股會被回貼成現在的市場別")
    ap.add_argument("--maxprice", type=float, default=0.0,
                    help="股價上限（元），0 = 只受部位金額限制")
    ap.add_argument("--grid", default="5x20,10x10,20x5",
                    help="部位%%x最大檔數，逗號分隔")
    args = ap.parse_args()

    C, O, L, U, B, RAW = build_matrices(args.market)
    print(f"母體矩陣 {C.shape[0]} 個交易日 × {C.shape[1]} 檔，"
          f"突破訊號 {int(B.values.sum())} 筆")
    bench = bench_curve(C, U)
    yrs = (C.index[-1] - C.index[0]).days / 365.25
    print(f"等權重市場指數 CAGR {(bench[-1] ** (1 / yrs) - 1) * 100:.1f}%"
          f"（{C.index[0].date()} → {C.index[-1].date()}）\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for combo in args.grid.split(","):
        pp, mp = combo.strip().split("x")
        pos_pct, max_pos = float(pp) / 100, int(mp)
        runs = []
        best_eq = None
        for s in range(args.seeds):
            eq, tr, st = simulate(C, O, L, B, RAW, args.capital,
                                  pos_pct, max_pos, args.stop / 100, seed=s,
                                  maxprice=args.maxprice)
            m, ann = metrics(eq, C.index, args.capital)
            m.update(st)
            runs.append(m)
            if s == 0:
                best_eq, best_tr, best_ann = eq, tr, ann
        r = pd.DataFrame(runs)
        row = {"配置": f"{pp}% × {mp} 檔"}
        for k in ["CAGR", "最大回撤", "MAR", "最差年度", "最長水下交易日",
                  "滿倉日比", "平均投入比", "因滿倉跳過", "因買不起跳過",
                  "因超過上限跳過"]:
            row[k] = r[k].mean()
        row["CAGR標準差"] = r["CAGR"].std()
        row["交易筆數"] = len(best_tr)
        rows.append(row)

        pd.Series(best_eq, index=C.index).to_csv(
            os.path.join(OUT_DIR, f"equity_{pp}x{mp}.csv"), header=["equity"])
        best_tr.to_csv(os.path.join(OUT_DIR, f"trades_{pp}x{mp}.csv"),
                       index=False, encoding="utf-8-sig")
        print(f"[{pp}% × {mp} 檔] 完成 {args.seeds} 組隨機種子")
        print("  年報酬 %%：" + "  ".join(
            f"{d.year}:{v * 100:.0f}" for d, v in best_ann.items()))

    res = pd.DataFrame(rows).round(2)
    res.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False,
               encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"起始資金 {args.capital:,.0f}　停損 -{args.stop:.0f}%　"
          f"隨機種子 {args.seeds} 組（取平均）")
    print(res.to_string(index=False))
    print("\n說明：")
    print("  MAR = CAGR / |最大回撤|，越高越好。這是評估市場開關的主要指標。")
    print("  CAGR標準差 = 不同隨機抽樣路徑之間的離散度，越大代表越靠運氣。")
    print("  「因買不起跳過」= 股價 × 1000 超過單筆部位金額的訊號數。")


if __name__ == "__main__":
    main()
