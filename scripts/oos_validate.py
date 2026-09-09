#!/usr/bin/env python3
"""封存段驗證：2024-01 ~ 2026-09。

只驗一件事：TECH_LARGE 偏好長回看＋寬停損、FIN_LARGE 偏好短回看＋緊停損，
這個「方向相反」的結論在完全沒看過的資料上還成不成立。

做法：矩陣仍用全期（指標要暖身），但訊號矩陣 B 只在 2024-01 之後開放，
權益曲線只取封存段那一段算指標。單筆統計本身與期間長度無關。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_backtest as PB
from cell_backtest import sector_series, size_matrix

OOS = "2024-01-01"
SEEDS = 3
CELLS = {"TECH_LARGE": 10, "FIN_LARGE": 10}
LOOKBACK = [120, 250, 500]
STOPS = [8, 12, 16, 20]


def main():
    res = []
    for lb in LOOKBACK:
        C, O, L, U, B, RAW = PB.build_matrices("all", lb, "simple")
        sec = sector_series(C.columns).values.astype(str)
        lab = size_matrix(C, U).values.astype(str)
        oos = C.index >= pd.Timestamp(OOS)
        i0 = int(np.argmax(oos))
        yrs = (C.index[-1] - C.index[i0]).days / 365.25

        for cell, slots in CELLS.items():
            s, z = cell.split("_")
            Bc = B.copy()
            Bc.values[~((sec == s) & (lab == z))] = False
            Bc.values[~oos] = False              # 封存段之前不開放進場
            nsig = int(Bc.values.sum())
            for stop in STOPS:
                eqs, trs = [], []
                for sd in range(SEEDS):
                    eq, tr, st = PB.simulate(C, O, L, Bc, RAW, 1_000_000,
                                             1.0 / slots, slots, stop / 100,
                                             sd, minprice=10.0)
                    eqs.append(eq); trs.append(tr)
                eq = np.mean(eqs, axis=0)[i0:]
                t = pd.concat(trs, ignore_index=True)
                if len(t) < 15:
                    continue
                r = t["報酬"]; w = r > 0
                aw = r[w].mean(); al = r[~w].mean()
                peak = pd.Series(eq).cummax()
                mdd = (pd.Series(eq) / peak - 1).min() * 100
                cagr = ((eq[-1] / eq[0]) ** (1 / yrs) - 1) * 100
                res.append(dict(cell=cell, lb=lb, stop=stop, nsig=nsig,
                                n=len(t) // SEEDS, win=w.mean() * 100,
                                aw=aw, al=al,
                                odds=aw / abs(al) if al else np.nan,
                                exp=r.mean(), cagr=cagr, dd=mdd,
                                mar=cagr / abs(mdd) if mdd else np.nan))
                print(f"  {cell:<12} lb={lb:<4} stop={stop:<3} n={len(t)//SEEDS:<4} "
                      f"勝率 {w.mean()*100:5.1f}%  期望值 {r.mean():6.2f}  "
                      f"MAR {res[-1]['mar']:5.2f}", flush=True)

    df = pd.DataFrame(res)
    df.to_csv("/home/claude/oos.csv", index=False)
    for cell in CELLS:
        d = df[df.cell == cell]
        if d.empty:
            continue
        print(f"\n===== {cell}　封存段 {OOS} ~ 2026-09 =====")
        for metric, title in (("exp", "每筆期望值 %"), ("win", "勝率 %"),
                              ("mar", "MAR")):
            print(f"\n{title}（列=回看, 欄=停損）")
            print(d.pivot(index="lb", columns="stop",
                          values=metric).round(2).to_string())


if __name__ == "__main__":
    main()
