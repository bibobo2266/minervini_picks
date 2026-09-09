#!/usr/bin/env python3
"""把組合搜尋找到的配方放進真實資金曲線，看 25 槽位限制下還剩多少。

訊號層級的期望值不等於可下單的績效：槽位滿了訊號就進不來。
TECH_LARGE 的 250 日新高在 4%×25 設定下，91% 的訊號被「因滿倉跳過」擋掉。

含槽位掃描（第33條）：同一條槽位軸上比配方與基準線，交錯就是雜訊。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_backtest as PB
from cell_backtest import sector_series, size_matrix

SEEDS = 3
CAPITAL = 1_000_000


def build_signals(C, O, L, RAW):
    """回傳各種訊號矩陣。條件用 T 日收盤決定，引擎會在 T+1 開盤進場。"""
    c = C
    ret = c.pct_change(fill_method=None)
    rv20 = ret.rolling(20, min_periods=10).std()
    rv60 = ret.rolling(60, min_periods=30).std()
    vc = rv20 / (rv60 + 1e-9)

    gap = (O / c.shift(1) - 1) > 0.03
    hi60 = c.shift(1).rolling(60, min_periods=60).max()
    brk60 = (c > hi60) & (c.shift(1) <= hi60.shift(1))
    hi250 = c.shift(1).rolling(250, min_periods=250).max()
    brk250 = (c > hi250) & (c.shift(1) <= hi250.shift(1))

    return {
        "250日新高（現行基準）": brk250,
        "60日新高": brk60,
        "跳空3%": gap,
        "跳空3% ∧ 波動放大>1.25": gap & (vc > 1.25),
        "60日新高 ∧ 壓縮<0.60": brk60 & (vc < 0.60),
        "60日新高 ∧ 壓縮<0.75": brk60 & (vc < 0.75),
        "250日新高 ∧ 壓縮<0.75": brk250 & (vc < 0.75),
    }


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "TECH_LARGE"
    C, O, L, U, B, RAW = PB.build_matrices("all", 250, "simple")
    sec = sector_series(C.columns).values.astype(str)
    lab = size_matrix(C, U).values.astype(str)
    s, z = cell.split("_")
    cellmask = (sec == s) & (lab == z)
    yrs = (C.index[-1] - C.index[0]).days / 365.25
    Uc = U.copy(); Uc.values[~cellmask] = False
    cb = PB.bench_curve(C, Uc)
    print(f"\n{cell}　格內等權基準 CAGR {(cb[-1]**(1/yrs)-1)*100:.1f}%"
          f"　（0050 含息 21.4%）")

    sigs = build_signals(C, O, L, RAW)

    def run(Bx, slots):
        eqs, invs, nts, fulls = [], [], [], []
        for sd in range(SEEDS):
            eq, tr, st = PB.simulate(C, O, L, Bx, RAW, CAPITAL,
                                     1.0 / slots, slots, 0.12, sd,
                                     minprice=10.0)
            eqs.append(eq); invs.append(st['平均投入比']); nts.append(len(tr))
            fulls.append(st.get('因滿倉跳過', 0))
        m, _ = PB.metrics(np.mean(eqs, axis=0), C.index, CAPITAL)
        return m, np.mean(invs), np.mean(nts), np.mean(fulls)

    print(f"\n{'配方':<26}{'訊號':>7}{'槽位':>5}{'成交':>6}{'投入比':>7}"
          f"{'滿倉跳過':>9}{'CAGR':>7}{'MaxDD':>8}{'MAR':>6}{'最差年':>8}")
    curves = {}
    for name, sig in sigs.items():
        Bx = sig.fillna(False) & pd.DataFrame(cellmask, index=C.index,
                                              columns=C.columns)
        Bx = Bx.astype(bool)
        n = int(Bx.values.sum())
        if n < 100:
            print(f"{name:<26}{n:>7}   訊號不足"); continue
        row = []
        for slots in (6, 8, 10, 12, 16, 20, 25):
            m, inv, nt, full = run(Bx, slots)
            row.append(m['MAR'])
            if slots == 10:
                print(f"{name:<26}{n:>7}{slots:>5}{nt:>6.0f}{inv:>7.1f}"
                      f"{full:>9.0f}{m['CAGR']:>7.2f}{m['最大回撤']:>8.2f}"
                      f"{m['MAR']:>6.2f}{m['最差年度']:>8.2f}")
        curves[name] = row

    print(f"\n=== 槽位掃描 MAR（第33條：交錯就是雜訊）===")
    print(f"{'配方':<26}" + "".join(f"{x:>7}" for x in
                                    (6, 8, 10, 12, 16, 20, 25)) + f"{'平均':>8}")
    base = curves.get("250日新高（現行基準）")
    for name, row in curves.items():
        line = f"{name:<26}" + "".join(f"{v:>7.2f}" for v in row)
        line += f"{np.mean(row):>8.3f}"
        if base and name != "250日新高（現行基準）":
            wins = sum(1 for a, b in zip(row, base) if a > b)
            line += f"　贏 {wins}/{len(row)}"
        print(line)


if __name__ == "__main__":
    main()
