#!/usr/bin/env python3
"""按市值層與產業切格，跑 portfolio_backtest.py 現成的 simulate/bench_curve。

不重寫引擎、不重寫基準 —— handover 說自己重寫基準是這專案犯過三次的錯。
這支只做一件事：把訊號矩陣 B 遮成單一格，其餘原封不動交給現成函式。

用法：
  python scripts/cell_backtest.py --cells LARGE,MID,SMALL
  python scripts/cell_backtest.py --cells TECH_LARGE,TECH_MID,TRAD_LARGE
  python scripts/cell_backtest.py --cells LARGE --minprice 10 --maxprice 30
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from portfolio_backtest import build_matrices, simulate, bench_curve, metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TECH = {'半導體業','電子零組件業','電腦及週邊設備業','光電業','電子工業',
        '通信網路業','其他電子業','其他電子類','電子通路業'}
TRAD = {'水泥工業','鋼鐵工業','塑膠工業','航運業','橡膠工業','電器電纜',
        '玻璃陶瓷','造紙工業','紡織纖維','化學工業','汽車工業','建材營造'}
FIN = {'金融保險','金融業'}


def sector_series(cols):
    u = pd.read_parquet(f"{ROOT}/data/universe.parquet")
    u['stock_id'] = u['stock_id'].astype(str)
    m = {}
    for sid, ind in zip(u['stock_id'], u['industry_category']):
        if ind in TECH: m[sid] = 'TECH'
        elif ind in TRAD: m[sid] = 'TRAD'
        elif ind in FIN: m[sid] = 'FIN'
    return pd.Series([m.get(c) for c in cols], index=cols)


def size_matrix(C, U):
    """逐季重算市值三分位。回傳與 C 同形狀的字串矩陣（LARGE/MID/SMALL/NaN）。

    每季重算一次，不逐日重算 —— 逐日會產生大量假換手。
    """
    fs = sorted(glob.glob(f"{ROOT}/data/fundamentals/market_value_*.parquet"))
    if not fs:
        raise SystemExit("找不到 market_value_*.parquet")
    mv = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    mv['stock_id'] = mv['stock_id'].astype(str)
    mv = mv[mv['stock_id'].str.len() == 4]
    mv['date'] = pd.to_datetime(mv['date'].astype(str).str.slice(0, 10))
    mv = mv.drop_duplicates(['date', 'stock_id'])
    M = mv.pivot_table(index='date', columns='stock_id', values='market_value')
    M = M.reindex(index=C.index, columns=C.columns).ffill()

    # 每季第一個交易日重新分層，季內固定
    q = pd.Series(C.index, index=C.index).dt.to_period('Q')
    first = ~q.duplicated()
    lab = pd.DataFrame(index=C.index, columns=C.columns, dtype=object)
    cur = None
    for i, dt in enumerate(C.index):
        if first.iloc[i]:
            row = M.iloc[i].where(U.iloc[i])
            r = row.rank(pct=True)
            cur = pd.Series(np.select([r > 2/3, r > 1/3, r.notna()],
                                      ['LARGE', 'MID', 'SMALL'], default=None),
                            index=C.columns)
        lab.iloc[i] = cur.values
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="LARGE,MID,SMALL")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--pos", type=float, default=4.0)
    ap.add_argument("--maxpos", type=int, default=25)
    ap.add_argument("--stop", type=float, default=12.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--minprice", type=float, default=10.0)
    ap.add_argument("--maxprice", type=float, default=0.0)
    ap.add_argument("--match-exposure", type=float, default=0.0,
                    help="目標平均投入比%%。給了就掃 maxpos 找最接近的設定，"
                         "讓各格在相同曝險下比較（部位%%= 100/maxpos）")
    args = ap.parse_args()

    C, O, L, U, B, RAW = build_matrices("all", 250, "simple")
    print(f"母體矩陣 {C.shape[0]} × {C.shape[1]}，總訊號 {int(B.values.sum())} 筆")

    sec = sector_series(C.columns)
    lab = size_matrix(C, U)
    labv = lab.values.astype(str)
    secv = sec.values.astype(str)

    yrs = (C.index[-1] - C.index[0]).days / 365.25
    gb = bench_curve(C, U)
    print(f"全母體等權基準 CAGR {(gb[-1]**(1/yrs)-1)*100:.1f}%  "
          f"（文件 16.4%，對不上就停）\n")

    print(f"{'格':<14}{'訊號':>8}{'槽位':>6}{'投入比':>8}{'成交':>7}{'CAGR':>8}"
          f"{'MaxDD':>9}{'MAR':>7}{'最差年':>8}{'格內基準':>10}")
    for cell in args.cells.split(","):
        cell = cell.strip()
        if "_" in cell:
            s, z = cell.split("_")
            mask = (secv == s) & (labv == z)
        elif cell in ("LARGE", "MID", "SMALL"):
            mask = (labv == cell)
        else:
            mask = (secv == cell)

        Bc = B.copy()
        Bc.values[~mask] = False
        Uc = U.copy()
        Uc.values[~mask] = False
        nsig = int(Bc.values.sum())
        if nsig < 100:
            print(f"{cell:<14}{nsig:>8}   訊號不足，跳過")
            continue

        def run(maxpos):
            pos_pct = 1.0 / maxpos
            eqs_, sts_, trs_ = [], [], []
            for sd in range(args.seeds):
                eq_, tr_, st_ = simulate(C, O, L, Bc, RAW, args.capital,
                                         pos_pct, maxpos, args.stop / 100, sd,
                                         maxprice=args.maxprice,
                                         minprice=args.minprice)
                eqs_.append(eq_); sts_.append(st_); trs_.append(tr_)
            return (np.mean(eqs_, axis=0),
                    np.mean([x['平均投入比'] for x in sts_]),
                    np.mean([len(t) for t in trs_]))

        if args.match_exposure > 0:
            best = None
            for mp in (5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80):
                e_, inv_, nt_ = run(mp)
                d_ = abs(inv_ - args.match_exposure)
                if best is None or d_ < best[0]:
                    best = (d_, mp, e_, inv_, nt_)
            _, used_pos, eq, inv, ntr = best
        else:
            used_pos = args.maxpos
            eq, inv, ntr = run(args.maxpos)
        m, _ = metrics(eq, C.index, args.capital)
        cb = bench_curve(C, Uc)
        cbc = (cb[-1] ** (1 / yrs) - 1) * 100
        print(f"{cell:<14}{nsig:>8}{used_pos:>6}{inv:>8.1f}{ntr:>7.0f}"
              f"{m['CAGR']:>8.2f}{m['最大回撤']:>9.2f}{m['MAR']:>7.2f}"
              f"{m['最差年度']:>8.2f}{cbc:>10.1f}")


if __name__ == "__main__":
    main()
