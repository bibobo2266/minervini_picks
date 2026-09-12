#!/usr/bin/env python3
"""傳產均值回歸 —— 條件式動態出場引擎。

與 probe_vcp.py 的差異在出場：那支是趨勢追蹤（固定持有 H 天 + 固定停損），
這支是狀態機逐日掃描，三種出場條件先到先算：

  1. 停損     盤中跌破進場價 x (1 - stop)  → 以 min(當日開盤, 停損價) 成交
  2. 回歸達成 收盤站回 MA20                → 次日開盤平倉
  3. 時間停損 進場後第 MAXHOLD 個交易日     → 當日開盤平倉

為什麼不能沿用固定持有：均值回歸是「橡皮筋彈回」，反彈又急又猛，站回均線後
若景氣未反轉往往掉頭破底。固定抱 60 天等於賺到 10% 反彈卻硬抱到下一波破底
觸發停損。

主指標仍是超額期望值 = 每筆期望值 − 同日同格全體平均報酬，但**基準也跑同一套
動態出場**。這點很重要：跌深的日子全族群都在反彈，若拿固定持有的基準去比動態
出場的訊號，超額會被系統性高估。

配方（裸觸發，不加條件B —— 先確認承重牆在不在）
  族群   傳產（MAT 原物料 + IND 工業運輸 + CONS 內需消費）
  母體   還原收盤價 >= 10 元、20 日均成交金額 >= 2,000 萬
  觸發A  Bias20 = 收盤 / MA20 - 1 < -10%（訊號當日）→ T+1 開盤買進
  成本   0.80% 來回

避坑：進場點須有 MAXHOLD + 2 個交易日的後續資料；母體遮罩不砍列。
"""
import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.008
LIQ_FLOOR = 2e7        # 傳產流動性較差，地板降到 2,000 萬
BIAS_TH = -0.10
NOREPEAT = 20
END = "2026-09-08"
SPLIT_YEAR = 2020

TRAD = {
    'MAT':  {'鋼鐵工業', '塑膠工業', '化學工業', '水泥工業', '造紙工業',
             '玻璃陶瓷', '橡膠工業'},
    'IND':  {'電機機械', '電器電纜', '汽車工業', '建材營造', '航運業',
             '油電燃氣業', '綠能環保', '綠能環保類'},
    'CONS': {'食品工業', '貿易百貨', '觀光事業', '紡織纖維', '運動休閒',
             '居家生活類'},
}


def sector_ids(cats):
    u = pd.read_parquet(f"{ROOT}/data/universe.parquet")
    u['stock_id'] = u['stock_id'].astype(str)
    u = u[(u['stock_id'].str.len() == 4) & (u['type'].isin(['twse', 'tpex']))]
    return set(u.loc[u['industry_category'].isin(cats), 'stock_id'])


def load(ids):
    parts = []
    for f in sorted(glob.glob(f"{ROOT}/data/adj/prices_adj_*.parquet")):
        d = pd.read_parquet(f, columns=['date', 'stock_id', 'open', 'min',
                                        'close', 'Trading_money'])
        d['stock_id'] = d['stock_id'].astype(str)
        d = d[d['stock_id'].isin(ids)]
        d['date'] = d['date'].astype(str).str.slice(0, 10)
        parts.append(d)
        del d
    p = pd.concat(parts, ignore_index=True)
    del parts
    p = p[p['date'] <= END].sort_values(['stock_id', 'date']).reset_index(drop=True)
    for c in ('open', 'min', 'close'):
        p[c] = p[c].astype('float64')
    sid = p['stock_id']
    ma20 = p['close'].groupby(sid, sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).mean())
    p['ma20'] = ma20
    p['bias20'] = p['close'] / ma20 - 1
    p['liq_ok'] = (p['Trading_money'].groupby(sid, sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).mean()) >= LIQ_FLOOR).fillna(False)
    p['px_ok'] = p['close'] >= 10
    del p['Trading_money']
    dates = sorted(p['date'].unique())
    p['_i'] = p['date'].map({d: i for i, d in enumerate(dates)})
    return p.reset_index(drop=True), len(dates)


def returns_dyn(p, union, cutoff_i, stop, maxhold):
    """狀態機出場。回傳 (報酬%, 持有天數, 出場原因碼)。
    原因碼 0=停損 1=站回MA20 2=時間到"""
    n_all = len(p)
    R = np.full(n_all, np.nan)
    HOLD = np.full(n_all, np.nan)
    WHY = np.full(n_all, -1, dtype=np.int8)
    oa = p['open'].values
    la = p['min'].values
    ca = p['close'].values
    ma = p['ma20'].values
    da = p['_i'].values
    for _, d in p.groupby('stock_id', sort=False):
        gi = d.index.values
        o, lo, c, m20, di = oa[gi], la[gi], ca[gi], ma[gi], da[gi]
        u = union[gi]
        n = len(gi)
        for k in range(n - 1):
            if not u[k] or di[k] > cutoff_i:
                continue
            e = o[k + 1]
            if not np.isfinite(e) or e <= 0:
                continue
            sp = e * (1 - stop)
            last = min(k + 1 + maxhold, n - 1)
            r = None
            for j in range(k + 1, last + 1):
                if lo[j] <= sp:                       # 1. 盤中停損
                    r = (min(o[j], sp) / e - 1)
                    HOLD[gi[k]] = j - k
                    WHY[gi[k]] = 0
                    break
                if np.isfinite(m20[j]) and c[j] >= m20[j]:   # 2. 收盤站回 MA20
                    x = min(j + 1, n - 1)
                    r = o[x] / e - 1
                    HOLD[gi[k]] = x - k
                    WHY[gi[k]] = 1
                    break
            if r is None:                              # 3. 時間停損
                r = o[last] / e - 1
                HOLD[gi[k]] = last - k
                WHY[gi[k]] = 2
            R[gi[k]] = (r - COST) * 100
    return R, HOLD, WHY


def union_for(sig, valid, di, cutoff_i):
    days = np.unique(di[sig & (di <= cutoff_i)])
    return (valid & np.isin(di, days)) | sig


def daily_base(R, valid, di, nd):
    fin = np.isfinite(R) & valid
    s_ = np.bincount(di[fin], weights=R[fin], minlength=nd)
    c_ = np.bincount(di[fin], minlength=nd)
    return np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)


def norepeat(sig, R_ref, di, sids):
    hit = np.flatnonzero(sig & np.isfinite(R_ref))
    out, last_s, last_p = [], None, -10**9
    for k in hit:
        if sids[k] != last_s:
            last_s, last_p = sids[k], -10**9
        if di[k] - last_p < NOREPEAT:
            continue
        last_p = di[k]
        out.append(k)
    return np.array(out, dtype=np.int64)


def stats(a, b):
    w = a > 5.0
    aw = a[w].mean() if w.any() else 0.0
    al = a[~w].mean() if (~w).any() else 0.0
    return dict(n=len(a), win=w.mean() * 100, aw=aw, al=al,
                odds=aw / abs(al) if al else np.nan,
                exp=a.mean(), base=np.nanmean(b), exc=a.mean() - np.nanmean(b))


def run(sectors, stop, maxhold, label, p, ndates, verbose=True):
    ids_by = {k: sector_ids(v) for k, v in TRAD.items()}
    want = set()
    for s in sectors:
        want |= ids_by[s]
    di, sids = p['_i'].values, p['stock_id'].values
    yr = p['date'].str.slice(0, 4).astype(int).values
    nd = ndates
    cutoff = nd - maxhold - 3
    inset = p['stock_id'].isin(want).values
    valid = p['px_ok'].values & p['liq_ok'].values & inset
    sig = valid & (p['bias20'].values < BIAS_TH)
    R, HOLD, WHY = returns_dyn(p, union_for(sig, valid, di, cutoff),
                               cutoff, stop, maxhold)
    db = daily_base(R, valid, di, nd)
    sel = norepeat(sig, R, di, sids)
    a = R[sel]
    m = np.isfinite(a)
    a, k = a[m], sel[m]
    if len(a) < 50:
        print(f"{label:<22}{len(a):>6}  樣本不足")
        return None
    b = db[di[k]]
    st = stats(a, b)
    y = yr[k]
    h = []
    for msk in (y <= SPLIT_YEAR, y >= SPLIT_YEAR + 1):
        h.append(a[msk].mean() - np.nanmean(b[msk]) if msk.sum() >= 25 else np.nan)
    tot = pos = 0
    for Y in sorted(set(y)):
        mm = y == Y
        if mm.sum() < 5:
            continue
        tot += 1
        pos += (a[mm].mean() - np.nanmean(b[mm])) > 0
    t3 = np.sort(a)[-3:].sum() / a.sum() * 100 if a.sum() > 0 else np.nan
    print(f"{label:<22}{st['n']:>6}{st['win']:>6.1f}%{st['odds']:>6.2f}"
          f"{st['exp']:>+8.2f}{st['base']:>+7.2f}{st['exc']:>+8.2f}"
          f"{h[0]:>+8.2f}{h[1]:>+8.2f}{np.median(a):>+7.1f}{t3:>6.1f}%"
          f"{pos:>4}/{tot}")
    if verbose:
        w = WHY[k]
        hd = HOLD[k]
        print(f"{'':22}出場：停損 {(w==0).mean()*100:.0f}% ｜ "
              f"站回MA20 {(w==1).mean()*100:.0f}% ｜ 時間到 {(w==2).mean()*100:.0f}%"
              f" ｜ 平均持有 {np.nanmean(hd):.1f} 天")
    return a, k, b, y


HDR = (f"{'配方':<22}{'n':>6}{'勝率':>7}{'賠率':>6}{'期望值':>8}{'基準':>7}"
       f"{'超額':>8}{'前半':>8}{'後半':>8}{'中位':>7}{'前3筆':>7}{'正年':>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=float, default=0.08)
    ap.add_argument("--maxhold", type=int, default=20)
    args = ap.parse_args()
    allids = set()
    for v in TRAD.values():
        allids |= sector_ids(v)
    p, nd = load(allids)
    print(f"傳產母體 {p['stock_id'].nunique()} 檔 / {len(p):,} 列\n")

    print(f"=== 分族群（停損-{args.stop:.0%}, 時間停損 {args.maxhold} 日）===\n{HDR}")
    for s in ('MAT', 'IND', 'CONS'):
        run([s], args.stop, args.maxhold, s, p, nd)
    run(['MAT', 'IND', 'CONS'], args.stop, args.maxhold, '傳產全體', p, nd)

    print(f"\n=== 停損掃描（傳產全體, 時間停損 {args.maxhold} 日）===\n{HDR}")
    for stop in (0.05, 0.08, 0.10, 0.12):
        run(['MAT', 'IND', 'CONS'], stop, args.maxhold, f"停損-{stop:.0%}",
            p, nd, verbose=False)

    print(f"\n=== 時間停損掃描（傳產全體, 停損-{args.stop:.0%}）===\n{HDR}")
    for mh in (10, 20, 40, 60):
        run(['MAT', 'IND', 'CONS'], args.stop, mh, f"時間停損 {mh} 日",
            p, nd, verbose=False)


if __name__ == "__main__":
    main()
