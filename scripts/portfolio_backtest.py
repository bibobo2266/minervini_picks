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
_HIGH = []                 # build_matrices 會把最高價矩陣放進來給 simulate 用
_AGE = []                  # 突破歷史距離（上次高於現價是幾個交易日前）
_SCORE = []                # 外掛的排序分數矩陣（例如月營收 YoY），--score-file 載入
_ATR = []                  # ATR20 / 收盤價
_BREADTH = []              # 母體中收盤價高於 200 日均的比例（逐日）

# --atr-stop / --vol-size：實測全部比固定 -12% 差，預設關閉，保留供複驗。
# --regime：上限 = base × 母體在 200 日均之上的比例（連續映射，不是門檻）。
#   分段門檻要決定回看期、門檻值、幾段、每段曝險、延遲幾天——五個旋鈕、
#   幾百種組合，一定找得到漂亮的一組。連續映射只有「200 日均」一個參數。
#   既有部位不強制平倉，只限制新單。
# ⚠️ 必須同時跑反向（1-breadth）與曝險配對（固定在平均允許檔數）兩個對照，
#    否則分不出「timing 有資訊」與「單純少放錢所以少跌」。

# 滿倉排序（--rank）：訊號多於空位時，用什麼順序決定誰先拿到位置。
#   random  隨機抽（baseline。任何主觀篩選都在隨機丟掉右尾）
#   age     突破歷史距離越久越優先
#   age_rev 反向，當作對照組——如果 age 有效，反向就該明顯變差
# 為什麼是排序不是濾網：750 日新高的逐筆品質確實比 250 日好（前三大贏家
# 完全相同），但硬篩成 ≥750 日會讓平均投入比從 69% 掉到 49%，補曝險又要
# 用集中度換，回撤從 -34% 惡化到 -47%。排序不減少任何訊號，
# participation 不變，資金閒置成本為零。
# ⚠️ 750 日新高是 250 日新高的子集合，兩者不是競爭關係，是巢狀層級。
BREAKOUT_DAYS = 250        # 創幾日新高（可用 --breakout-days 覆寫）
# 逐筆掃描顯示回看窗口越長越好，且單調：
#   60D +0.34 / 120D +1.23 / 250D +2.11 / 350D +2.18 / 500D +2.70 / 750D +3.32
#   （120 個交易日持有期的平均超額 %，同母體同基準，已扣 0.60% 成本）
# 直覺：三年新高代表走出更大的底部，重新定價幅度更大。訊號數也少一半
# （750D 20,806 筆 vs 250D 38,883 筆），對有限資本的容量瓶頸是加分。
# ⚠️ 窗口長度是掃出來的，一定要走 walkforward_validate.py 才算數。
REENTRY_GAP = 20           # 同一檔幾個交易日內不重複進場
MAX_HOLD = 250             # 最大持有交易日數

# 死錢出場（dead money exit）：預設關閉，是待驗證的假說不是已知結論。
#   進場滿 DEAD_DAYS 個交易日、期間最高報酬從未超過 DEAD_GAIN、且目前仍虧損 → 出場。
# 為什麼逐筆回測測不出這條的價值：逐筆層級沒有部位上限，一個磨到期滿的部位
# 結算大概 -1%，對平均超額幾乎沒影響。但在有限資本下它吃掉一個位置整整一年，
# 而同一段時間你因為滿倉跳過了三萬多個訊號。死錢的真實成本是它擋掉的交易，
# 不是它自己虧多少。所以「十五種出場組合差不多」那個結論在這裡不適用。
# ⚠️ 它砍不到大贏家（前十大贏家在第 40 天早就遠超 +5%），但可能砍到
#    盤整很久才發動的股票。這是真實風險，必須用走動式驗證，不能用全樣本挑參數。
COST = 0.006               # 來回 0.60%（手續費 + 交易稅）
UNIVERSE_PCT = 0.25        # 母體：當日成交值前 25%


def load_etf(sid="0050"):
    """讀取 ETF 的還原收盤價當基準（還原價已含息，等同總報酬）。"""
    fs = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))
    out = []
    for f in fs:
        d = pd.read_parquet(f, columns=["date", "stock_id", "close"])
        out.append(d[d["stock_id"] == sid])
    d = pd.concat(out, ignore_index=True)
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").set_index("date")["close"]


def minervini_template(C):
    """Minervini SEPA 趨勢模板（八條）。回傳布林矩陣。

    ① 股價 > 150 日均 且 > 200 日均
    ② 150 日均 > 200 日均
    ③ 200 日均至少上升一個月（用 20 個交易日）
    ④ 50 日均 > 150 日均 且 > 200 日均
    ⑤ 股價 > 50 日均
    ⑥ 股價離 52 週低點至少 30%
    ⑦ 股價離 52 週高點 25% 以內
    ⑧ RS ≥ 70（六個月報酬的橫斷面百分位）

    ⚠️ 原 findings 已證明這八條是負貢獻（逐筆超額 +2.40 vs 笨基準 +3.89），
    這裡重新實作是為了在「25 元以上的可執行區間」重測一次——
    因為那八條要求「離低點遠、離高點近、RS 高」，可能天生避開水餃股，
    所以在受限區間裡兩者的差距未必還是同一個方向。
    """
    ma50 = C.rolling(50).mean()
    ma150 = C.rolling(150).mean()
    ma200 = C.rolling(200).mean()
    lo52 = C.rolling(250, min_periods=250).min()
    hi52 = C.rolling(250, min_periods=250).max()
    rs = (C / C.shift(126) - 1).rank(axis=1, pct=True) * 100
    return ((C > ma150) & (C > ma200) & (ma150 > ma200)
            & (ma200 > ma200.shift(20)) & (ma50 > ma150) & (ma50 > ma200)
            & (C > ma50) & (C >= lo52 * 1.30) & (C >= hi52 * 0.75)
            & (rs >= 70))


def build_matrices(market="all", breakout_days=BREAKOUT_DAYS, signal="simple"):
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
    H = d.pivot_table(index="date", columns="stock_id", values="max")
    LO = d.pivot_table(index="date", columns="stock_id", values="min")
    O = d.pivot_table(index="date", columns="stock_id", values="open")
    L = d.pivot_table(index="date", columns="stock_id", values="min")
    T = d.pivot_table(index="date", columns="stock_id", values="Trading_money")

    # 母體：當日成交值前 25%
    U = (T.rank(axis=1, pct=True, ascending=False) <= UNIVERSE_PCT) & C.notna()
    # 250 日新高，且前一日不是新高
    prior = C.shift(1).rolling(breakout_days, min_periods=breakout_days).max()
    B = (C > prior) & (C.shift(1) <= prior.shift(1)) & U
    if signal == "minervini":
        B = B & minervini_template(C)

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
    # 突破歷史距離：往回找最近一次收盤高於今日收盤，是幾個交易日前。
    # 用擴張窗口的累積最高價分段算，避免逐檔迴圈。
    cv = C.values
    age = np.zeros_like(cv)
    for j in range(cv.shape[1]):
        col = cv[:, j]
        last_hi = -1          # 最近一次「高於當前價」的位置
        run = np.full(len(col), np.nan)
        stack = []            # 單調遞減堆疊，存 (index, price)
        for i, v in enumerate(col):
            if not np.isfinite(v):
                continue
            while stack and stack[-1][1] <= v:
                stack.pop()
            run[i] = (i - stack[-1][0]) if stack else i + 1
            stack.append((i, v))
        age[:, j] = run
    _AGE.clear()
    _AGE.append(age)

    RAW = C / F        # 近似的實際成交股價
    _HIGH.clear()
    _HIGH.append(H.reindex(columns=C.columns).values)
    tr = (H.reindex(columns=C.columns) - LO.reindex(columns=C.columns)) / C
    _ATR.clear()
    _ATR.append(tr.rolling(20, min_periods=10).mean().values)
    ma200 = C.rolling(200, min_periods=200).mean()
    above = ((C > ma200) & U).sum(axis=1)
    tot = (C.notna() & U).sum(axis=1)
    _BREADTH.clear()
    _BREADTH.append((above / tot.replace(0, np.nan)).ffill().fillna(0.5).values)
    return C, O, L, U, B, RAW


def bench_curve(C, U):
    c = C.values
    r = c[1:] / c[:-1] - 1
    m = U.values[:-1] & np.isfinite(r)
    n = m.sum(1)
    daily = np.where(n > 0, np.nansum(np.where(m, r, 0), 1) / np.maximum(n, 1), 0.0)
    return np.concatenate([[1.0], np.cumprod(1 + daily)])


def simulate(C, O, L, B, RAW, capital, pos_pct, max_pos, stop_pct, seed,
             maxprice=0.0, minprice=0.0, dead_days=0, dead_gain=0.05,
             rank="random", atr_stop=0.0, vol_size=False, regime="none"):
    """事件驅動的有限資本模擬。回傳 (每日權益, 交易明細, 統計)。"""
    rng = np.random.default_rng(seed)
    dates = C.index
    sids = np.array(C.columns)
    c, o, lo, raw = C.values, O.values, L.values, RAW.values
    hh = _HIGH[0] if _HIGH else c   # 期間最高價；沒帶就退回用收盤價
    ag = _AGE[0] if _AGE else None  # 突破歷史距離
    sc_ext = _SCORE[0] if _SCORE else None
    atr = _ATR[0] if _ATR else None
    bre = _BREADTH[0] if _BREADTH else None
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
    allowed_sum = 0
    n_regime_skip = 0

    for t in range(len(dates)):
        cash += pending
        pending = 0.0

        # --- 出場（先做，空出來的位置當天不放行，資金隔日才可用）---
        for j in list(pos):
            p = pos[j]
            hi = hh[t, j]
            if np.isfinite(hi) and hi > p["peak"]:
                p["peak"] = hi
            px = None
            if np.isfinite(lo[t, j]) and lo[t, j] <= p["stop"]:
                op = o[t, j]
                px = op if (np.isfinite(op) and op <= p["stop"]) else p["stop"]
                why = "停損"
            elif t - p["entry_i"] >= MAX_HOLD and np.isfinite(c[t, j]):
                px = c[t, j]
                why = "到期"
            elif (dead_days and t - p["entry_i"] >= dead_days
                  and np.isfinite(c[t, j])
                  and p["peak"] / p["entry_px"] - 1 <= dead_gain
                  and c[t, j] < p["entry_px"]):
                px = c[t, j]
                why = "死錢"
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
        eff_max = max_pos
        if regime != "none" and bre is not None and t > 0:
            b = float(bre[t - 1])          # point-in-time：只用 t-1 已知的
            eff_max = int(round(max_pos * (1 - b if regime == "breadth_rev" else b)))
        allowed_sum += eff_max
        slots = max(0, eff_max - len(pos))
        mark = np.array([c[t, j] if np.isfinite(c[t, j]) else 0.0
                         for j in pos], dtype=float)
        cur_eq = cash + float((mark * np.array([pos[j]["shares"]
                                                for j in pos])).sum()) if pos else cash
        budget = cur_eq * pos_pct

        if cands:
            if slots <= 0 and eff_max < max_pos:
                n_regime_skip += len(cands)
            if len(cands) > slots:
                if slots <= 0:
                    take = []
                elif rank == "random":
                    # baseline：隨機抽，不做主觀篩選
                    take = list(rng.choice(cands, size=slots, replace=False))
                else:
                    # 只有在真正發生資本競爭時才排序，訊號本身一個都沒少
                    src = sc_ext if rank.startswith("score") else ag
                    if src is None:
                        take = list(rng.choice(cands, size=slots, replace=False))
                    else:
                        sc = np.array([src[t - 1, j]
                                       if np.isfinite(src[t - 1, j]) else -1e18
                                       for j in cands])
                        if rank.endswith("_rev"):
                            sc = np.where(sc > -1e17, -sc, -1e18)
                        take = [cands[k] for k in np.argsort(-sc)[:slots]]
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
                # 股價下限：25 元以下在台股多為水餃股，常見全額交割與處置。
                # 全額交割要圈存全額價金、券商多有限制；處置股是人工撮合
                # （每 5–20 分鐘一次），而本策略是隔日開盤買，處置期間拿不到
                # 開盤價。這兩個障礙回測完全模擬不了，只能用下限迴避。
                # （下市風險不在此列——還原股價含已下市個股，沒有生存者偏差。）
                flr = minprice[t] if hasattr(minprice, "__len__") else minprice
                if flr and raw_px < flr:
                    skipped.append((dates[t], sids[j], "低於股價下限"))
                    continue
                sd = stop_pct
                if atr_stop and atr is not None and np.isfinite(atr[t - 1, j]):
                    sd = float(np.clip(atr_stop * atr[t - 1, j], 0.08, 0.20))
                bgt = budget * (stop_pct / sd) if vol_size else budget
                bgt = float(np.clip(bgt, budget * 0.5, budget * 2.0))
                lots = int(bgt // (raw_px * LOT))
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
                              stop=entry_px * (1 - sd), entry_i=t,
                              peak=entry_px)
                last_entry[j] = t

        mv = sum(pos[j]["shares"] * (c[t, j] if np.isfinite(c[t, j])
                                     else pos[j]["entry_px"]) for j in pos)
        equity[t] = cash + mv
        if len(pos) >= eff_max:
            full_days += 1
        invested += (mv / equity[t]) if equity[t] > 0 else 0

    tdf = pd.DataFrame(trades)
    stats = dict(平均允許檔數=allowed_sum / len(dates),
                 因節流跳過=n_regime_skip,
                 死錢出場=int((tdf["出場原因"] == "死錢").sum()) if len(tdf) else 0,
                 滿倉日比=full_days / len(dates) * 100,
                 平均投入比=invested / len(dates) * 100,
                 跳過訊號=len(skipped),
                 因滿倉跳過=sum(1 for s in skipped if s[2] == "滿倉"),
                 因買不起跳過=sum(1 for s in skipped if s[2] == "買不起整張"),
                 因超過上限跳過=sum(1 for s in skipped if s[2] == "超過股價上限"),
                 因低於下限跳過=sum(1 for s in skipped if s[2] == "低於股價下限"))
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
    ap.add_argument("--score-file", default="",
                    help="外掛排序分數的 .npy（形狀要跟收盤價矩陣一致）")
    ap.add_argument("--rank", default="random",
                    choices=["random", "age", "age_rev", "score", "score_rev"],
                    help="滿倉時的排序方式（age = 突破歷史距離越久越優先）")
    ap.add_argument("--signal", default="simple",
                    choices=["simple", "minervini"],
                    help="simple = 純 250 日新高；minervini = 再加八條趨勢模板")
    ap.add_argument("--breakout-days", type=int, default=BREAKOUT_DAYS,
                    help="創幾日新高（250 / 500 / 750）")
    ap.add_argument("--dead-days", type=int, default=0,
                    help="死錢出場的觀察天數，0 = 關閉")
    ap.add_argument("--dead-gain", type=float, default=5.0,
                    help="死錢出場的最高漲幅門檻 %%（期間最高從未超過就算死錢）")
    ap.add_argument("--regime", default="none",
                    choices=["none", "breadth", "breadth_rev"],
                    help="市場節流：上限 = base × 母體在 200 日均之上的比例")
    ap.add_argument("--atr-stop", type=float, default=0.0,
                    help="停損距離 = 倍數 × ATR20%%，夾在 8%%–20%%。0 = 固定停損")
    ap.add_argument("--vol-size", action="store_true",
                    help="部位大小 ∝ 1/停損距離（固定元風險）")
    ap.add_argument("--minprice", type=float, default=0.0,
                    help="股價下限（元），迴避水餃股／全額交割／處置股")
    ap.add_argument("--maxprice", type=float, default=0.0,
                    help="股價上限（元），0 = 只受部位金額限制")
    ap.add_argument("--grid", default="5x20,10x10,20x5",
                    help="部位%%x最大檔數，逗號分隔")
    args = ap.parse_args()

    C, O, L, U, B, RAW = build_matrices(args.market, args.breakout_days,
                                        args.signal)
    if args.score_file:
        _SCORE.clear()
        _SCORE.append(np.load(args.score_file))
        print(f"  載入排序分數 {args.score_file}")
    print(f"母體矩陣 {C.shape[0]} 個交易日 × {C.shape[1]} 檔，"
          f"{args.signal} 訊號 {int(B.values.sum())} 筆")
    bench = bench_curve(C, U)
    yrs = (C.index[-1] - C.index[0]).days / 365.25
    print(f"基準（{C.index[0].date()} → {C.index[-1].date()}）")
    print(f"  等權重市場指數 CAGR {(bench[-1] ** (1 / yrs) - 1) * 100:.1f}%"
          "　← 每天把錢平均分給母體所有股票，實務上做不到")
    for sid in ("0050", "006208"):
        e = load_etf(sid)
        if e is not None:
            e = e.reindex(C.index).ffill().dropna()
            n = (e.index[-1] - e.index[0]).days / 365.25
            print(f"  {sid} 含息 CAGR "
                  f"{((e.iloc[-1] / e.iloc[0]) ** (1 / n) - 1) * 100:.1f}%"
                  "　← 你真正的替代方案")
    print()

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
                                  maxprice=args.maxprice,
                                  minprice=args.minprice,
                                  dead_days=args.dead_days,
                                  dead_gain=args.dead_gain / 100,
                                  rank=args.rank, regime=args.regime,
                                  atr_stop=args.atr_stop,
                                  vol_size=args.vol_size)
            m, ann = metrics(eq, C.index, args.capital)
            m.update(st)
            runs.append(m)
            if s == 0:
                best_eq, best_tr, best_ann = eq, tr, ann
        r = pd.DataFrame(runs)
        row = {"配置": f"{pp}% × {mp} 檔"}
        for k in ["CAGR", "最大回撤", "MAR", "最差年度", "最長水下交易日",
                  "滿倉日比", "平均投入比", "平均允許檔數", "因節流跳過",
                  "死錢出場", "因滿倉跳過",
                  "因買不起跳過", "因超過上限跳過", "因低於下限跳過"]:
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
