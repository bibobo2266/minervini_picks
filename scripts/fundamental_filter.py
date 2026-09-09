#!/usr/bin/env python3
"""在 250 日新高訊號上加基本面門檻，比較有加跟沒加。

門檻不是排序器：不挑最好的，只剔掉爛的。
所有基本面欄位一律用 available_date（實際公告日）對齊，不用財報所屬日期。
balance_sheet 沒有 available_date 欄，自己補：Q1→5/15、Q2→8/14、
Q3→11/14、Q4→次年 3/31。用所屬季末直接對價格就是前視偏誤。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_backtest as PB
from cell_backtest import sector_series, size_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END = "2023-12-31"
CELL, SLOTS, STOP, LB = "TECH_LARGE", 10, 0.16, 250
SEEDS = 3


def pub_date(d):
    """財報所屬季末 → 實際公告日"""
    d = pd.to_datetime(d)
    m, y = d.dt.month, d.dt.year
    out = pd.Series(pd.NaT, index=d.index)
    out[m == 3] = pd.to_datetime(y[m == 3].astype(str) + "-05-15")
    out[m == 6] = pd.to_datetime(y[m == 6].astype(str) + "-08-14")
    out[m == 9] = pd.to_datetime(y[m == 9].astype(str) + "-11-14")
    out[m == 12] = pd.to_datetime((y[m == 12] + 1).astype(str) + "-03-31")
    return out


def as_matrix(df, valcol, C):
    """long → 寬矩陣，用 available_date 對齊 C.index，之後 ffill"""
    M = df.pivot_table(index="avail", columns="stock_id", values=valcol,
                       aggfunc="last")
    M = M.reindex(columns=C.columns)
    M = M.reindex(M.index.union(C.index)).sort_index().ffill()
    return M.reindex(C.index)


def build_filters(C):
    f = {}

    # 1) 負債比 = 負債總計 / 資產總計
    bs = pd.read_parquet(f"{ROOT}/data/fundamentals/balance_sheet.parquet")
    bs["stock_id"] = bs["stock_id"].astype(str)
    bs = bs[bs["stock_id"].isin(C.columns) & bs["type"].isin(["Liabilities", "TotalAssets"])]
    w = bs.pivot_table(index=["date", "stock_id"], columns="type",
                       values="value", aggfunc="last").reset_index()
    w = w.dropna(subset=["Liabilities", "TotalAssets"])
    w = w[w["TotalAssets"] > 0]
    w["ratio"] = w["Liabilities"] / w["TotalAssets"]
    w["avail"] = pub_date(w["date"])
    f["負債比<=70%"] = as_matrix(w, "ratio", C) <= 0.70

    # 2) 最近一季稅後淨利 > 0
    fin = pd.read_parquet(f"{ROOT}/data/fundamentals/financials.parquet")
    fin["stock_id"] = fin["stock_id"].astype(str)
    # ⚠️ IncomeAfterTax 只有 1,391 列且只存在於銀行股（2330 完全沒有），
    # 是金融業專用科目。改用 EPS：87,181 列 / 1,898 檔，且已是單季值。
    ni = fin[fin["stock_id"].isin(C.columns) & (fin["type"] == "EPS")].copy()
    ni["avail"] = pd.to_datetime(ni["available_date"])
    f["當季EPS>0"] = as_matrix(ni, "value", C) > 0

    # 3) 月營收：近三個月 YoY 不是全負
    mr = pd.read_parquet(f"{ROOT}/data/fundamentals/month_revenue.parquet")
    mr["stock_id"] = mr["stock_id"].astype(str)
    mr = mr[mr["stock_id"].isin(C.columns)].copy()
    rc = [c for c in mr.columns if "revenue" in c.lower()][0]
    mr["avail"] = pd.to_datetime(mr["available_date"])
    mr = mr.sort_values(["stock_id", "avail"])
    g = mr.groupby("stock_id")[rc]
    mr["yoy"] = g.pct_change(12, fill_method=None)
    mr["neg3"] = (mr.groupby("stock_id")["yoy"]
                    .transform(lambda s: (s < 0).rolling(3, min_periods=3).sum()))
    f["非營收連三衰退"] = as_matrix(mr, "neg3", C) < 3

    # 4) PBR 不在自己五年歷史的最高 10%
    per = pd.concat([pd.read_parquet(x) for x in
                     sorted(glob.glob(f"{ROOT}/data/fundamentals/stock_per_*.parquet"))],
                    ignore_index=True)
    per["stock_id"] = per["stock_id"].astype(str)
    per = per[per["stock_id"].isin(C.columns) & (per["PBR"] > 0)]
    P = per.pivot_table(index=pd.to_datetime(per["date"]), columns="stock_id",
                        values="PBR", aggfunc="last")
    P = P.reindex(index=C.index, columns=C.columns).ffill()
    rank5y = P.rolling(1250, min_periods=250).rank(pct=True)
    f["PBR非歷史前10%貴"] = rank5y <= 0.90
    return f


def main():
    C, O, L, U, B, RAW = PB.build_matrices("all", LB, "simple")
    n = int((C.index <= pd.Timestamp(END)).sum())
    C, O, L, U, B, RAW = (X.iloc[:n] for X in (C, O, L, U, B, RAW))
    for g in (PB._HIGH, PB._ATR, PB._BREADTH, PB._AGE):
        if g:
            g[0] = g[0][:n]

    sec = sector_series(C.columns).values.astype(str)
    lab = size_matrix(C, U).values.astype(str)
    s, z = CELL.split("_")
    cellmask = (sec == s) & (lab == z)
    B0 = B.copy(); B0.values[~cellmask] = False
    Uc = U.copy(); Uc.values[~cellmask] = False
    yrs = (C.index[-1] - C.index[0]).days / 365.25
    cb = PB.bench_curve(C, Uc)
    print(f"\n{CELL}  格內基準 CAGR {(cb[-1]**(1/yrs)-1)*100:.1f}%  "
          f"（進場 {LB} 日新高，停損 {STOP:.0%}，槽位 {SLOTS}）")

    filters = build_filters(C)
    sigmask = B0.values
    for k, v in filters.items():
        vv = v.reindex_like(C)
        pas = np.nansum(vv.values & sigmask) / max(sigmask.sum(), 1)
        print(f"  門檻 {k}: 訊號通過率 {pas:.0%}")

    def run(Bx, tag, slots=SLOTS):
        nsig = int(Bx.values.sum())
        if nsig < 80:
            print(f"{tag:<22}{nsig:>7}   訊號不足")
            return (np.nan, np.nan, np.nan)
        eqs, invs, nts = [], [], []
        for sd in range(SEEDS):
            eq, tr, st = PB.simulate(C, O, L, Bx, RAW, 1_000_000,
                                     1.0 / slots, slots, STOP, sd, minprice=10.0)
            eqs.append(eq); invs.append(st["平均投入比"]); nts.append(len(tr))
        m, _ = PB.metrics(np.mean(eqs, axis=0), C.index, 1_000_000)
        print(f"{tag:<22}{nsig:>7}{np.mean(nts):>7.0f}{np.mean(invs):>8.1f}"
              f"{m['CAGR']:>8.2f}{m['最大回撤']:>9.2f}{m['MAR']:>7.2f}"
              f"{m['最差年度']:>8.2f}")
        return m['CAGR'], m['最大回撤'], m['MAR']

    print(f"\n{'門檻':<22}{'訊號':>7}{'成交':>7}{'投入比':>8}{'CAGR':>8}"
          f"{'MaxDD':>9}{'MAR':>7}{'最差年':>8}")
    run(B0, "無門檻（基準線）")
    allm = None
    for k, v in filters.items():
        vv = v.reindex_like(C).fillna(True)   # 沒資料的不剔除，避免變成資料覆蓋率測試
        Bx = B0.copy(); Bx.values[~vv.values] = False
        run(Bx, k)
        allm = vv if allm is None else (allm & vv)
    Bx = B0.copy(); Bx.values[~allm.values] = False
    run(Bx, "四項全過")

    # 兩項合併：只用單獨測有效、方向又不衝突的兩個
    two = (filters["負債比<=70%"].reindex_like(C).fillna(True)
           & filters["非營收連三衰退"].reindex_like(C).fillna(True))
    Bx = B0.copy(); Bx.values[~two.values] = False
    run(Bx, "負債比+營收（兩項）")
    # 曝險配對：掃槽位讓投入比回到無門檻的水準
    for sl in (8, 9, 11, 12, 13):
        run(Bx, f"  兩項@槽位{sl}", slots=sl)

    # 反向對照：只留被兩項門檻剔除的訊號
    Bx = B0.copy(); Bx.values[two.values] = False
    run(Bx, "反向（兩項剔除的）")

    # 決定性檢驗：同一條槽位軸上比「無門檻」與「負債比」
    # 如果兩條曲線交錯，改善就落在參數雜訊裡，不算發現。
    print("\n=== 槽位掃描對照（每格 3 顆種子）===")
    print("同一條槽位軸上比，兩條曲線交錯就表示落在參數雜訊裡，不算發現。")
    variants = {"負債比": "負債比<=70%", "營收": "非營收連三衰退",
                "EPS": "當季EPS>0"}
    curves = {"無門檻": []}
    for k in variants:
        curves[k] = []
    for sl in (7, 8, 9, 10, 11, 12, 14, 16):
        r = run(B0, f"無門檻@{sl}", slots=sl)
        curves["無門檻"].append(r)
        for tag, key in variants.items():
            Bv = B0.copy()
            mm = filters[key].reindex_like(C).fillna(True)
            Bv.values[~mm.values] = False
            r = run(Bv, f"  {tag}@{sl}", slots=sl)
            curves[tag].append(r)

    print("\n=== 彙總（八個槽位設定的平均）===")
    print(f"{'門檻':<12}{'CAGR均':>9}{'回撤均':>9}{'MAR均':>8}"
          f"{'MAR贏幾次':>11}{'CAGR贏幾次':>12}")
    base = curves["無門檻"]
    for tag, cs in curves.items():
        cg = np.mean([c[0] for c in cs]); dd = np.mean([c[1] for c in cs])
        mr = np.mean([c[2] for c in cs])
        if tag == "無門檻":
            print(f"{tag:<12}{cg:>9.2f}{dd:>9.2f}{mr:>8.3f}{'—':>11}{'—':>12}")
            continue
        wm = sum(1 for a, b in zip(cs, base) if a[2] > b[2])
        wc = sum(1 for a, b in zip(cs, base) if a[0] > b[0])
        print(f"{tag:<12}{cg:>9.2f}{dd:>9.2f}{mr:>8.3f}"
              f"{wm:>9}/8{wc:>10}/8")


if __name__ == "__main__":
    main()
