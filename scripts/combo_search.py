#!/usr/bin/env python3
"""格內組合搜尋：條件A（觸發）∧ 條件B（狀態/資訊/情境）。

主指標是每筆期望值，不是 CAGR。
不用組合回測引擎——那會被 25 個槽位擋掉 97% 的訊號，看不到訊號本身的品質。
這裡直接對「每一個訊號」算未來報酬：T+1 開盤買，-N% 停損（盤中觸價），
最長抱 250 天或到資料結束，扣 0.80% 來回成本。

避坑：
  第39條 已平倉偏誤 —— 訊號距資料結束不足 250 天者整筆剔除，不論輸贏。
  第33條 先量雜訊 —— 對照組是同格同期隨機抽同樣筆數的訊號，跑 200 次取分佈。
  第34條 成員資格與報酬不能同日 —— 所有條件用 T 日收盤決定，T+1 開盤進場。
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.008
MAXHOLD = 250
NOREPEAT = 20

TECH = {'半導體業','電子零組件業','電腦及週邊設備業','光電業','電子工業',
        '通信網路業','其他電子業','其他電子類','電子通路業'}
TRAD = {'水泥工業','鋼鐵工業','塑膠工業','航運業','橡膠工業','電器電纜',
        '玻璃陶瓷','造紙工業','紡織纖維','化學工業','汽車工業','建材營造'}
FIN = {'金融保險','金融業'}


def load(end, only_sector=None):
    u = pd.read_parquet(f"{ROOT}/data/universe.parquet")
    u['stock_id'] = u['stock_id'].astype(str)
    u = u[(u['stock_id'].str.len() == 4) & (u['type'].isin(['twse', 'tpex']))]
    smap = {}
    for sid, ind in zip(u['stock_id'], u['industry_category']):
        if ind in TECH: smap[sid] = 'TECH'
        elif ind in TRAD: smap[sid] = 'TRAD'
        elif ind in FIN: smap[sid] = 'FIN'
    if only_sector:
        smap = {k: v for k, v in smap.items() if v == only_sector}

    parts = []
    for f in sorted(glob.glob(f"{ROOT}/data/adj/prices_adj_*.parquet")):
        d = pd.read_parquet(f, columns=['date','stock_id','open','max','min',
                                        'close','Trading_Volume','Trading_money'])
        d['stock_id'] = d['stock_id'].astype(str)
        d = d[d['stock_id'].isin(smap)]
        d['date'] = d['date'].astype(str).str.slice(0, 10)
        parts.append(d)
    p = pd.concat(parts, ignore_index=True)
    p = p[p['date'] <= end].sort_values(['stock_id', 'date']).reset_index(drop=True)
    for c in ('open', 'max', 'min', 'close'):
        p[c] = p[c].astype('float64')
    p['sector'] = p['stock_id'].map(smap)
    return p


def add_size(p):
    fs = sorted(glob.glob(f"{ROOT}/data/fundamentals/market_value_*.parquet"))
    mv = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    mv['stock_id'] = mv['stock_id'].astype(str)
    mv = mv[mv['stock_id'].str.len() == 4]
    mv['date'] = mv['date'].astype(str).str.slice(0, 10)
    mv = mv.drop_duplicates(['date', 'stock_id'])
    p = p.merge(mv[['date', 'stock_id', 'market_value']], on=['date', 'stock_id'],
                how='left')
    p['market_value'] = p.groupby('stock_id')['market_value'].ffill()
    p['q'] = p['date'].str.slice(0, 7)
    # 每季分層：用該季第一個月的市值排名，季內固定
    p['qtr'] = p['date'].str.slice(0, 4) + "Q" + \
        ((p['date'].str.slice(5, 7).astype(int) - 1) // 3 + 1).astype(str)
    first = p.groupby(['qtr', 'stock_id'])['market_value'].first().reset_index()
    first['r'] = first.groupby('qtr')['market_value'].rank(pct=True)
    first['size'] = np.select([first.r > 2/3, first.r > 1/3, first.r.notna()],
                              ['LARGE', 'MID', 'SMALL'], default=None)
    return p.merge(first[['qtr', 'stock_id', 'size']], on=['qtr', 'stock_id'],
                   how='left')


def add_features(p):
    sid = p['stock_id']
    c, h, l, v, mo = p['close'], p['max'], p['min'], p['Trading_Volume'], p['Trading_money']

    def T(s, fn):
        return s.groupby(sid, sort=False).transform(fn)

    # ---- 觸發類（條件A 用）----
    for n in (60, 120, 250):
        hi = T(c, lambda s, n=n: s.shift(1).rolling(n, min_periods=n).max())
        p[f'brk{n}'] = (c > hi) & (T(c, lambda s: s.shift(1)) <= T(hi, lambda s: s.shift(1)))
    prevhi = T(c, lambda s: s.shift(1).rolling(250, min_periods=250).max())
    p['gap'] = (p['open'] / T(c, lambda s: s.shift(1)) - 1) > 0.03
    vma = T(v, lambda s: s.rolling(20, min_periods=10).mean())
    p['volspike'] = (v / (vma + 1e-9)) > 2.0
    ma20 = T(c, lambda s: s.rolling(20, min_periods=10).mean())
    was_above = T(c, lambda s: s.shift(1)) > T(ma20, lambda s: s.shift(1))
    p['pullback'] = ((c / prevhi > 0.90) & (c / prevhi < 1.0)
                     & (l <= ma20) & (c > ma20) & was_above)

    # ---- 狀態類（條件B 用）----
    for n in (50, 150, 200):
        p[f'ma{n}'] = T(c, lambda s, n=n: s.rolling(n, min_periods=n // 2).mean())
    p['stack'] = (c > p['ma50']) & (p['ma50'] > p['ma150']) & (p['ma150'] > p['ma200'])
    p['ma200up'] = p['ma200'] > T(p['ma200'], lambda s: s.shift(20))
    hi250 = T(c, lambda s: s.rolling(250, min_periods=120).max())
    p['near_high'] = (c / hi250) > 0.90
    ret = T(c, lambda s: s.pct_change(fill_method=None))
    rv20 = ret.groupby(sid, sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())
    rv60 = ret.groupby(sid, sort=False).transform(lambda s: s.rolling(60, min_periods=30).std())
    p['squeeze'] = (rv20 / (rv60 + 1e-9)) < 0.75
    p['loosen'] = (rv20 / (rv60 + 1e-9)) > 1.25
    tr = pd.concat([(h - l), (h - T(c, lambda s: s.shift())).abs(),
                    (l - T(c, lambda s: s.shift())).abs()], axis=1).max(axis=1)
    atr = tr.groupby(sid, sort=False).transform(lambda s: s.rolling(14, min_periods=7).mean()) / c
    p['lowvol'] = atr < atr.groupby(p['date']).transform('median')
    p['highvol'] = ~p['lowvol']

    # ---- 波動壓縮門檻掃描（0.75 是拍的，要看是平原還是單點）----
    vc = rv20 / (rv60 + 1e-9)
    for th in (60, 70, 75, 80, 90):
        p[f'sq{th}'] = vc < th / 100

    # ---- 檯面上的技術指標，這次全部當「條件B」測 ----
    dl = T(c, lambda s: s.diff())
    ru = dl.clip(lower=0).groupby(sid, sort=False).transform(
        lambda s: s.ewm(alpha=1/14, min_periods=14).mean())
    rd = (-dl).clip(lower=0).groupby(sid, sort=False).transform(
        lambda s: s.ewm(alpha=1/14, min_periods=14).mean())
    rsi = 100 - 100 / (1 + ru / (rd + 1e-9))
    p['rsi_hi'] = rsi > 70
    p['rsi_mid'] = (rsi >= 50) & (rsi <= 70)
    p['rsi_lo'] = rsi < 50

    ef = T(c, lambda s: s.ewm(span=12, min_periods=12).mean())
    es = T(c, lambda s: s.ewm(span=26, min_periods=26).mean())
    dif = ef - es
    dea = dif.groupby(sid, sort=False).transform(
        lambda s: s.ewm(span=9, min_periods=9).mean())
    p['macd_pos'] = (dif - dea) > 0
    p['macd_above0'] = dif > 0

    h9 = T(h, lambda s: s.rolling(9, min_periods=5).max())
    l9 = T(l, lambda s: s.rolling(9, min_periods=5).min())
    p['kd_hi'] = ((c - l9) / (h9 - l9 + 1e-9) * 100) > 80
    p['kd_mid'] = (((c - l9) / (h9 - l9 + 1e-9) * 100) >= 50) & (~p['kd_hi'])

    dmp = (h - T(h, lambda s: s.shift())).clip(lower=0)
    dmm = (T(l, lambda s: s.shift()) - l).clip(lower=0)
    dmp = dmp.where(dmp > dmm, 0.0); dmm = dmm.where(dmm > dmp, 0.0)
    atr14 = tr.groupby(sid, sort=False).transform(
        lambda s: s.rolling(14, min_periods=7).mean())
    pdi = 100 * dmp.groupby(sid, sort=False).transform(
        lambda s: s.rolling(14, min_periods=7).mean()) / (atr14 + 1e-9)
    mdi = 100 * dmm.groupby(sid, sort=False).transform(
        lambda s: s.rolling(14, min_periods=7).mean()) / (atr14 + 1e-9)
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)).groupby(
        sid, sort=False).transform(lambda s: s.rolling(14, min_periods=7).mean())
    p['adx_trend'] = adx > 25
    p['adx_flat'] = adx <= 25

    hh20 = T(h, lambda s: s.rolling(20, min_periods=10).max())
    ll20 = T(l, lambda s: s.rolling(20, min_periods=10).min())
    p['donch_top'] = ((c - ll20) / (hh20 - ll20 + 1e-9)) > 0.8

    vma2 = T(v, lambda s: s.rolling(20, min_periods=10).mean())
    obv = (np.sign(dl.fillna(0)) * v).groupby(sid, sort=False).cumsum()
    p['obv_up'] = (obv.groupby(sid, sort=False).transform(
        lambda s: s.diff(20)) / (vma2 * 20 + 1e-9)) > 0.1

    p['roc60_pos'] = T(c, lambda s: s.pct_change(60, fill_method=None)) > 0
    p['volratio_hi'] = (v / (vma2 + 1e-9)) > 1.5

    # ---- 情境類 ----
    above = (c > p['ma200'])
    p['sector_breadth'] = above.groupby([p['date'], p['sector']]).transform('mean')
    p['sec_riskon'] = p['sector_breadth'] > 0.5
    p['squeeze'] = p['sq75']
    for c in ('ma50', 'ma150', 'ma200', 'sector_breadth'):
        del p[c]
    for c in p.columns:
        if p[c].dtype == bool:
            continue
        if p[c].dtype == 'float64' and c not in ('open', 'max', 'min', 'close',
                                                 'Trading_money', 'market_value'):
            p[c] = p[c].astype('float32')
    return p


def add_info(p):
    """資訊類條件。一律用 available_date 對齊。"""
    ids = set(p['stock_id'])
    # 月營收加速：近3月YoY均值 > 前3月YoY均值
    mr = pd.read_parquet(f"{ROOT}/data/fundamentals/month_revenue.parquet")
    mr['stock_id'] = mr['stock_id'].astype(str)
    mr = mr[mr['stock_id'].isin(ids)].copy()
    rc = [c for c in mr.columns if 'revenue' in c.lower()][0]
    mr['date'] = mr['available_date'].astype(str).str.slice(0, 10)
    mr = mr.sort_values(['stock_id', 'date'])
    g = mr.groupby('stock_id')[rc]
    mr['yoy'] = g.pct_change(12, fill_method=None)
    a = mr.groupby('stock_id')['yoy'].transform(lambda s: s.rolling(3, min_periods=3).mean())
    b = mr.groupby('stock_id')['yoy'].transform(
        lambda s: s.shift(3).rolling(3, min_periods=3).mean())
    mr['rev_accel'] = a > b
    mr['rev_pos'] = mr['yoy'] > 0
    p = p.merge(mr[['date', 'stock_id', 'rev_accel', 'rev_pos']],
                on=['date', 'stock_id'], how='left')

    # EPS 轉正 / 連兩季成長
    fin = pd.read_parquet(f"{ROOT}/data/fundamentals/financials.parquet")
    fin['stock_id'] = fin['stock_id'].astype(str)
    e = fin[(fin['type'] == 'EPS') & fin['stock_id'].isin(ids)].copy()
    e['date'] = e['available_date'].astype(str).str.slice(0, 10)
    e = e.sort_values(['stock_id', 'date'])
    e['eps_pos'] = e['value'] > 0
    e['eps_up'] = e.groupby('stock_id')['value'].transform(
        lambda s: (s.diff() > 0).rolling(2, min_periods=2).sum() == 2)
    p = p.merge(e[['date', 'stock_id', 'eps_pos', 'eps_up']],
                on=['date', 'stock_id'], how='left')

    # 負債比
    bs = pd.read_parquet(f"{ROOT}/data/fundamentals/balance_sheet.parquet")
    bs['stock_id'] = bs['stock_id'].astype(str)
    bs = bs[bs['stock_id'].isin(ids) & bs['type'].isin(['Liabilities', 'TotalAssets'])]
    w = bs.pivot_table(index=['date', 'stock_id'], columns='type',
                       values='value', aggfunc='last').reset_index()
    w = w.dropna()
    w = w[w['TotalAssets'] > 0]
    d = pd.to_datetime(w['date'])
    m, y = d.dt.month, d.dt.year
    av = pd.Series(pd.NaT, index=w.index)
    av[m == 3] = pd.to_datetime(y[m == 3].astype(str) + "-05-15")
    av[m == 6] = pd.to_datetime(y[m == 6].astype(str) + "-08-14")
    av[m == 9] = pd.to_datetime(y[m == 9].astype(str) + "-11-14")
    av[m == 12] = pd.to_datetime((y[m == 12] + 1).astype(str) + "-03-31")
    w['date'] = av.dt.strftime('%Y-%m-%d')
    w['lowdebt'] = (w['Liabilities'] / w['TotalAssets']) <= 0.50
    p = p.merge(w[['date', 'stock_id', 'lowdebt']], on=['date', 'stock_id'], how='left')

    # 法人：外資/投信 近5日連續淨買超
    parts = []
    for f in sorted(glob.glob(f"{ROOT}/data/inst/stock_inst_*.parquet")):
        d2 = pd.read_parquet(f)
        d2['stock_id'] = d2['stock_id'].astype(str)
        d2 = d2[d2['stock_id'].str.len() == 4]
        d2['date'] = d2['date'].astype(str).str.slice(0, 10)
        parts.append(d2[d2['stock_id'].isin(ids)])
    inst = pd.concat(parts, ignore_index=True).drop_duplicates(['date', 'stock_id'])
    inst = inst.sort_values(['stock_id', 'date'])
    for who, tag in (('foreign', 'fbuy5'), ('trust', 'tbuy5')):
        inst[tag] = inst.groupby('stock_id')[who].transform(
            lambda s: (s > 0).rolling(5, min_periods=5).sum() >= 4)
    p = p.merge(inst[['date', 'stock_id', 'fbuy5', 'tbuy5']],
                on=['date', 'stock_id'], how='left')

    for c in ('rev_accel', 'rev_pos', 'eps_pos', 'eps_up', 'lowdebt',
              'fbuy5', 'tbuy5'):
        p[c] = p.groupby('stock_id')[c].ffill().fillna(False).astype(bool)
    return p


def precompute(p, stop, cutoff_i, union):
    """對所有觸發日算一次未來報酬，之後各組合只做遮罩，不再重走價格。

    未套用 20 日不重複進場 —— 那要在組合層級套（不同組合選到的訊號不同）。
    """
    R = np.full(len(p), np.nan)
    idx = np.arange(len(p))
    for stock, d in p.groupby('stock_id', sort=False):
        gi = idx[p['stock_id'].values == stock] if False else d.index.values
        o = d['open'].values; lo = d['min'].values; di = d['_i'].values
        u = union[d.index.values]
        n = len(d)
        for k in range(n - 1):
            if not u[k] or di[k] > cutoff_i:
                continue
            e = o[k + 1]
            if not np.isfinite(e) or e <= 0:
                continue
            end = min(k + 1 + MAXHOLD, n - 1)
            sp = e * (1 - stop)
            r = o[end] / e - 1
            for j2 in range(k + 1, end + 1):
                if lo[j2] <= sp:
                    r = (min(o[j2], sp) / e - 1) if o[j2] <= sp else -stop
                    break
            R[gi[k]] = (r - COST) * 100
    return R


def apply_norepeat(p, sig, R):
    """套 20 日不重複進場，回傳選中訊號的報酬陣列。"""
    out = []
    pos = p['_i'].values
    sids = p['stock_id'].values
    hit = np.flatnonzero(sig & np.isfinite(R))
    last_sid = None
    last_pos = -10**9
    for k in hit:
        if sids[k] != last_sid:
            last_sid = sids[k]; last_pos = -10**9
        if pos[k] - last_pos < NOREPEAT:
            continue
        last_pos = pos[k]
        out.append(R[k])
    return np.array(out)


def apply_norepeat_y(p, sig, R, yr):
    out, ys = [], []
    pos = p['_i'].values; sids = p['stock_id'].values
    hit = np.flatnonzero(sig & np.isfinite(R))
    last_sid, last_pos = None, -10**9
    for k in hit:
        if sids[k] != last_sid:
            last_sid = sids[k]; last_pos = -10**9
        if pos[k] - last_pos < NOREPEAT:
            continue
        last_pos = pos[k]
        out.append(R[k]); ys.append(yr[k])
    return np.array(out), np.array(ys)


def summarize(a):
    if len(a) < 30:
        return None
    w = a > 0
    aw = a[w].mean() if w.any() else 0.0
    al = a[~w].mean() if (~w).any() else 0.0
    return dict(n=len(a), win=w.mean() * 100, aw=aw, al=al,
                odds=aw / abs(al) if al else np.nan, exp=a.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="TECH_LARGE")
    ap.add_argument("--stop", type=float, default=0.12)
    ap.add_argument("--end", default="2026-09-08")
    args = ap.parse_args()

    print("載入…", flush=True)
    p = load(args.end, only_sector=args.cell.split("_")[0])
    p = add_size(p)
    p = add_features(p)
    p = add_info(p)
    p['liq'] = p.groupby('date')['Trading_money'].rank(pct=True)
    dates = sorted(p['date'].unique())
    dpos = {d: i for i, d in enumerate(dates)}
    p['_i'] = p['date'].map(dpos)
    cutoff_i = len(dates) - MAXHOLD - 2

    s, z = args.cell.split("_")
    pool = (p['close'] >= 10) & (p['liq'] >= 0.75)
    # ⚠️ 只能拿來判斷「這個訊號算不算數」，絕對不能拿來砍列。
    # 砍列會讓每檔的時間序列出現空洞，往前走 250「列」就不是 250 個交易日，
    # 一檔進進出出母體的股票可能橫跨五年，報酬嚴重虛胖（踩過一次）。
    p['_valid'] = pool & (p['sector'] == s) & (p['size'] == z)
    p['_cell'] = True
    print(f"{args.cell}：母體內 {int(p['_valid'].sum()):,} 個股票日 / "
          f"{p.loc[p['_valid'],'stock_id'].nunique()} 檔，"
          f"停損 -{args.stop:.0%}，成本 {COST:.1%}", flush=True)

    TRIG = {"創60日新高": "brk60", "創120日新高": "brk120", "創250日新高": "brk250",
            "跳空3%": "gap", "爆量2倍": "volspike", "突破後回踩MA20": "pullback"}
    COND = {"（無）": None,
            "均線多頭排列": "stack", "MA200上升": "ma200up", "距高點10%內": "near_high",
            "波動放大": "loosen", "低波動": "lowvol", "高波動": "highvol",
            "壓縮<0.60": "sq60", "壓縮<0.70": "sq70", "壓縮<0.75": "sq75",
            "壓縮<0.80": "sq80", "壓縮<0.90": "sq90",
            "RSI>70": "rsi_hi", "RSI 50-70": "rsi_mid", "RSI<50": "rsi_lo",
            "MACD柱>0": "macd_pos", "MACD DIF>0": "macd_above0",
            "KD>80": "kd_hi", "KD 50-80": "kd_mid",
            "ADX>25": "adx_trend", "ADX<=25": "adx_flat",
            "Donchian上緣20%": "donch_top", "OBV上升": "obv_up",
            "ROC60>0": "roc60_pos", "量比>1.5": "volratio_hi",
            "產業breadth>50%": "sec_riskon",
            "月營收加速": "rev_accel", "月營收YoY>0": "rev_pos",
            "EPS>0": "eps_pos", "EPS連兩季成長": "eps_up",
            "負債比<=50%": "lowdebt",
            "外資連買": "fbuy5", "投信連買": "tbuy5"}

    p = p.reset_index(drop=True)
    union = np.zeros(len(p), dtype=bool)
    for tcol in TRIG.values():
        union |= p[tcol].fillna(False).values
    union &= p['_valid'].values
    print(f"觸發日合計 {int(union.sum()):,} 個，預算未來報酬…", flush=True)
    R = precompute(p, args.stop, cutoff_i, union)
    print(f"完成，有效 {int(np.isfinite(R).sum()):,} 筆", flush=True)

    rows = []
    for tname, tcol in TRIG.items():
        base = p[tcol].fillna(False).values & p['_valid'].values
        for cname, ccol in COND.items():
            sig = base if ccol is None else (base & p[ccol].fillna(False).values)
            st = summarize(apply_norepeat(p, sig, R))
            if st is None:
                continue
            st.update(trig=tname, cond=cname)
            rows.append(st)
            print(f"  {tname:<14}∧ {cname:<14} n={st['n']:<5} "
                  f"勝率{st['win']:5.1f}%  賠率{st['odds']:5.2f}  "
                  f"期望值{st['exp']:+7.2f}", flush=True)

    df = pd.DataFrame(rows).sort_values("exp", ascending=False)
    df.to_csv(f"/home/claude/combo_{args.cell}.csv", index=False)
    print(f"\n########## {args.cell} 前 20 名（依每筆期望值）##########")
    print(f"{'觸發':<14}{'條件':<16}{'筆數':>6}{'勝率':>7}{'平均賺':>8}"
          f"{'平均賠':>8}{'賠率':>6}{'期望值':>8}")
    for _, r in df.head(20).iterrows():
        print(f"{r.trig:<14}{r['cond']:<16}{r.n:>6}{r.win:>6.1f}%{r.aw:>8.1f}"
              f"{r.al:>8.1f}{r.odds:>6.2f}{r.exp:>+8.2f}")

    # ---- 雜訊檢定（第33條）：從同觸發的無條件訊號隨機抽同樣筆數，跑 200 次 ----
    print("\n--- 隨機對照檢定（同觸發無條件訊號隨機抽同樣筆數，200 次）---")
    print(f"{'觸發':<14}{'條件':<16}{'n':>6}{'期望值':>8}{'隨機均':>8}"
          f"{'隨機P95':>9}{'百分位':>8}")
    rng = np.random.default_rng(20260908)
    top = df.head(12)
    for _, r in top.iterrows():
        if r['cond'] == "（無）":
            continue
        base = p[TRIG[r.trig]].fillna(False).values & p['_valid'].values
        pool_r = apply_norepeat(p, base, R)
        if len(pool_r) <= r.n:
            continue
        draws = np.array([rng.choice(pool_r, size=int(r.n), replace=False).mean()
                          for _ in range(200)])
        pct = (draws < r.exp).mean() * 100
        print(f"{r.trig:<14}{r['cond']:<16}{r.n:>6}{r.exp:>+8.2f}"
              f"{draws.mean():>+8.2f}{np.percentile(draws, 95):>+9.2f}{pct:>7.0f}%")

    # ---- 逐年分佈：檢查是否集中在少數年份 ----
    print("\n--- 前 5 名的逐年期望值（n<10 的年份標 · ）---")
    yr = p['date'].str.slice(0, 4).values
    for _, r in df.head(5).iterrows():
        if r['cond'] == "（無）":
            continue
        base = p[TRIG[r.trig]].fillna(False).values & p['_valid'].values
        ccol = COND[r['cond']]
        sig = base if ccol is None else (base & p[ccol].fillna(False).values)
        rr, yy = apply_norepeat_y(p, sig, R, yr)
        line = f"{r.trig}∧{r['cond']:<12}"
        pos_yrs = 0; tot_yrs = 0
        for y in sorted(set(yy)):
            m = yy == y
            if m.sum() < 10:
                line += f" {y[2:]}:·"
                continue
            tot_yrs += 1
            v = rr[m].mean()
            pos_yrs += v > 0
            line += f" {y[2:]}:{v:+.0f}"
        print(line + f"   → {pos_yrs}/{tot_yrs} 年為正")

    # ---- 剔除 2025：確認不是單一年份撐起來的 ----
    print("\n--- 剔除 2025 進場後（與同觸發無條件比）---")
    print(f"{'觸發':<12}{'條件':<14}{'n':>6}{'全期':>8}{'剔2025':>9}"
          f"{'無條件剔25':>11}{'改善':>8}")
    for _, r in df.head(10).iterrows():
        if r['cond'] == "（無）":
            continue
        base = p[TRIG[r.trig]].fillna(False).values & p['_valid'].values
        ccol = COND[r['cond']]
        sig = base & p[ccol].fillna(False).values
        rr, yy = apply_norepeat_y(p, sig, R, yr)
        br, by = apply_norepeat_y(p, base, R, yr)
        m = yy != '2025'; bm = by != '2025'
        if m.sum() < 30:
            continue
        print(f"{r.trig:<12}{r['cond']:<14}{int(m.sum()):>6}{r.exp:>+8.2f}"
              f"{rr[m].mean():>+9.2f}{br[bm].mean():>+11.2f}"
              f"{rr[m].mean()-br[bm].mean():>+8.2f}")

    b = df[df['cond'] == "（無）"].set_index("trig")["exp"]
    print("\n--- 相對於同觸發、無條件的改善 ---")
    df["lift"] = df.apply(lambda r: r.exp - b.get(r.trig, np.nan), axis=1)
    for _, r in df.sort_values("lift", ascending=False).head(15).iterrows():
        if r['cond'] == "（無）":
            continue
        print(f"{r.trig:<14}{r['cond']:<16}n={r.n:<5} 期望值{r.exp:+7.2f}  "
              f"改善{r.lift:+7.2f}")


if __name__ == "__main__":
    main()
