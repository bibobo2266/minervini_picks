#!/usr/bin/env python3
"""各格的進出場參數掃描：回看窗口 × 停損幅度。

槽位固定用曝險配對算出來的值，讓同一格內的參數比較不受曝險影響。
封存段 2024-01 之後截掉，只用探索+確認段。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_backtest as PB
from cell_backtest import sector_series, size_matrix

END = "2023-12-31"          # 封存段之後截掉
SEEDS = 2
# 各格的槽位：曝險配對到約 50% 投入比時算出來的值
CELLS = {"TECH_LARGE": 10, "TECH_MID": 16, "FIN_LARGE": 10,
         "TRAD_LARGE": 8, "TRAD_MID": 8}
LOOKBACK = [120, 250, 500]
STOPS = [8, 12, 16, 20]


def main():
    rows = []
    for lb in LOOKBACK:
        C, O, L, U, B, RAW = PB.build_matrices("all", lb, "simple")
        keep = C.index <= pd.Timestamp(END)
        n = int(keep.sum())
        C, O, L, U, B, RAW = (X.iloc[:n] for X in (C, O, L, U, B, RAW))
        for g in (PB._HIGH, PB._ATR, PB._BREADTH, PB._AGE):
            if g:
                g[0] = g[0][:n]
        sec = sector_series(C.columns).values.astype(str)
        lab = size_matrix(C, U).values.astype(str)
        yrs = (C.index[-1] - C.index[0]).days / 365.25

        for cell, slots in CELLS.items():
            s, z = cell.split("_")
            mask = (sec == s) & (lab == z)
            Bc = B.copy(); Bc.values[~mask] = False
            Uc = U.copy(); Uc.values[~mask] = False
            nsig = int(Bc.values.sum())
            if nsig < 100:
                continue
            cb = PB.bench_curve(C, Uc)
            cbc = (cb[-1] ** (1 / yrs) - 1) * 100
            for stop in STOPS:
                eqs, invs, nts = [], [], []
                for sd in range(SEEDS):
                    eq, tr, st = PB.simulate(C, O, L, Bc, RAW, 1_000_000,
                                             1.0 / slots, slots, stop / 100,
                                             sd, minprice=10.0)
                    eqs.append(eq); invs.append(st['平均投入比'])
                    nts.append(len(tr))
                m, _ = PB.metrics(np.mean(eqs, axis=0), C.index, 1_000_000)
                rows.append(dict(cell=cell, lb=lb, stop=stop, nsig=nsig,
                                 trades=np.mean(nts), inv=np.mean(invs),
                                 cagr=m['CAGR'], dd=m['最大回撤'],
                                 mar=m['MAR'], worst=m['最差年度'], bench=cbc))
                print(f"  {cell:<12} lb={lb:<4} stop={stop:<3} "
                      f"CAGR {m['CAGR']:6.2f}  DD {m['最大回撤']:7.2f}  "
                      f"MAR {m['MAR']:5.2f}  投入 {np.mean(invs):4.1f}%",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/home/claude/param_sweep.csv", index=False)
    print("\n\n########## 各格最佳參數（依 MAR）##########")
    for cell in CELLS:
        d = df[df.cell == cell]
        if d.empty:
            continue
        b = d.loc[d.mar.idxmax()]
        print(f"\n[{cell}]  格內基準 CAGR {b.bench:.1f}%")
        print(f"{'回看':>6}{'停損':>6}{'CAGR':>8}{'MaxDD':>9}{'MAR':>7}{'最差年':>8}")
        for _, r in d.sort_values('mar', ascending=False).head(5).iterrows():
            print(f"{int(r.lb):>6}{int(r.stop):>5}%{r.cagr:>8.2f}"
                  f"{r.dd:>9.2f}{r.mar:>7.2f}{r.worst:>8.2f}")


if __name__ == "__main__":
    main()
