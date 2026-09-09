#!/usr/bin/env python3
"""出場工具比較 —— 主指標是單筆勝率／賠率／期望值，不是 CAGR。

期望值 = 勝率 × 平均獲利 − 敗率 × 平均虧損，且引擎的「報酬」欄已扣 0.60% 來回成本。
勝率高不等於賺錢：停利設很緊、不設停損，勝率可以拉到 90% 而穩定賠錢，
所以三個數字要一起看。

只在兩格上測（TECH_LARGE、FIN_LARGE），其餘格已在 CROSS_SECTOR_FINDINGS 淘汰。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_backtest as PB
from cell_backtest import sector_series, size_matrix

END = "2023-12-31"
SEEDS = 3
CELLS = {"TECH_LARGE": dict(slots=10, stop=0.16, lb=250),
         "FIN_LARGE":  dict(slots=10, stop=0.12, lb=120)}


def exits_for(C):
    cl = C
    out = [("固定停損＋抱到期（基準線）", None)]
    for n in (50, 100, 150, 200):
        M = cl.rolling(n, min_periods=n // 2).mean().values
        out.append((f"跌破 MA{n}", dict(kind="ma_break", M=M)))
    return out, cl


def main():
    for cell, cfg in CELLS.items():
        C, O, L, U, B, RAW = PB.build_matrices("all", cfg["lb"], "simple")
        n = int((C.index <= pd.Timestamp(END)).sum())
        C, O, L, U, B, RAW = (X.iloc[:n] for X in (C, O, L, U, B, RAW))
        for g in (PB._HIGH, PB._ATR, PB._BREADTH, PB._AGE):
            if g:
                g[0] = g[0][:n]
        sec = sector_series(C.columns).values.astype(str)
        lab = size_matrix(C, U).values.astype(str)
        s, z = cell.split("_")
        Bc = B.copy(); Bc.values[~((sec == s) & (lab == z))] = False

        # 出場規則：均線跌破 / Donchian 低點 / 固定停利 / 時間出場
        rules = [("固定停損＋抱到期（基準線）", None)]
        for nn in (50, 100, 150, 200):
            rules.append((f"跌破 MA{nn}",
                          dict(kind="ma_break",
                               M=C.rolling(nn, min_periods=nn // 2).mean().values)))
        low = pd.DataFrame(L.values, index=L.index, columns=L.columns)
        for nn in (20, 50):
            rules.append((f"跌破 Donchian{nn}",
                          dict(kind="donchian",
                               M=low.rolling(nn, min_periods=nn // 2).min().values)))
        for tp in (0.20, 0.30, 0.50):
            rules.append((f"停利 +{int(tp*100)}%",
                          dict(kind="take_profit", param=tp)))
        for td in (60, 120):
            rules.append((f"時間出場 {td} 日", dict(kind="time_exit", param=td)))

        print(f"\n{'='*104}")
        print(f"{cell}  進場 {cfg['lb']} 日新高，停損 -{cfg['stop']:.0%}，"
              f"槽位 {cfg['slots']}，訊號 {int(Bc.values.sum())} 筆　"
              f"（2015-06 ~ {END}，封存段未動）")
        print(f"{'出場規則':<24}{'筆數':>6}{'持有':>6}{'勝率':>7}{'平均賺':>8}"
              f"{'平均賠':>8}{'賠率':>6}{'期望值':>8}{'CAGR':>7}{'MaxDD':>8}{'MAR':>6}")
        for name, ex in rules:
            PB._EXITX.clear()
            if ex:
                PB._EXITX.append(ex)
            eqs, trs = [], []
            for sd in range(SEEDS):
                eq, tr, st = PB.simulate(C, O, L, Bc, RAW, 1_000_000,
                                         1.0 / cfg["slots"], cfg["slots"],
                                         cfg["stop"], sd, minprice=10.0)
                eqs.append(eq); trs.append(tr)
            t = pd.concat(trs, ignore_index=True)
            if len(t) < 30:
                print(f"{name:<24}{len(t):>6}  交易太少"); continue
            r = t["報酬"]
            win = r > 0
            aw = r[win].mean(); al = r[~win].mean()
            odds = aw / abs(al) if al else np.nan
            m, _ = PB.metrics(np.mean(eqs, axis=0), C.index, 1_000_000)
            print(f"{name:<24}{len(t)//SEEDS:>6}{t['持有日'].mean():>6.0f}"
                  f"{win.mean()*100:>6.1f}%{aw:>8.1f}{al:>8.1f}{odds:>6.2f}"
                  f"{r.mean():>8.2f}{m['CAGR']:>7.2f}{m['最大回撤']:>8.2f}"
                  f"{m['MAR']:>6.2f}")
        PB._EXITX.clear()


if __name__ == "__main__":
    main()
