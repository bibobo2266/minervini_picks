#!/usr/bin/env python3
"""ELEC_LARGE VCP 突破配方 —— 全套驗證，一鍵重現。

配方
  族群   ELEC_LARGE（電子硬體大型股）
  母體   還原收盤價 >= 10 元、20日均成交金額 >= 3,000 萬、族群內每季市值前 33%
  觸發A  收盤創 60 日新高（前一日非新高）→ T+1 開盤買進
  條件B  rv20 / rv60 < 0.60（訊號當日）
  出場   盤中觸及 -12% 停損，或抱滿 60 個交易日
  成本   0.80% 來回

主指標
  超額期望值 = 每筆期望值 − 同日同格全體平均報酬。
  日期對齊基準修掉「訊號集中在多頭好日子」的擇時運氣。單一純量基準
  （全期隨機抽樣）做不到——訊號在時間軸上不是均勻散佈的。實測：均勻
  抽樣基準 +13.11，日期對齊後 +24~35，那個差距就是擇時運氣本身。

避坑（沿用交接規則）
  第39條 已平倉偏誤 —— 進場點須有 250 個交易日的後續資料，不論輸贏一律
         剔除不足者。四個持有期共用同一個 cutoff，樣本跨 H 相同才可比。
  第34條 成員資格與報酬不能同日 —— 條件用 T 日收盤判定，T+1 開盤進場。
  母體遮罩只用來判斷「訊號算不算數」，絕不拿來砍列。砍列會讓個股時間
         序列出現空洞，往前走 250「列」就不是 250 個交易日。

模式
  --mode audit --recipe vcp       卡片一四件套（預設）
  --mode audit --recipe bare250   卡片二四件套（裸 250 日新高，無壓縮門檻）
  --mode audit --recipe rev       卡片三四件套（創 60 日新高 ∧ 月營收 YoY>20%）
  --mode triggers  五種觸發 A 的裸基座比較 + 形態分解
  --mode grid      22 格跨族群掃描（單一配方）
  --mode overlap   三張卡的訊號重疊、月報酬相關性與活躍度（第八節）
  --mode holdout   真樣本外驗證：把已平倉要求從 250 天降到 60 天，解放
                   2025-08-27 之後從未進入任何計算的區間

記憶體：3GB 環境可跑。一次只持有一個族群的行情表，算完立即釋放；不碰
        fundamentals（本配方全是純價格特徵）。基準只算訊號日，不算全母體。
"""
import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.008          # 來回成本
STOP = 0.12           # 預設停損
VC_TH = 0.60          # 波動壓縮門檻
HOLDS = (20, 60, 120, 250)
MAXH = max(HOLDS)
NOREPEAT = 20         # 同一檔 20 個交易日內不重複進場
LIQ_FLOOR = 3e7       # 20日均成交金額地板（台幣）
END = "2026-09-08"
SPLIT_YEAR = 2020     # 前半段 <= 2020，後半段 >= 2021

SECTORS = {
    'SEMI': {'半導體業'},
    'ELEC': {'電子工業', '電子零組件業', '光電業', '電腦及週邊設備業',
             '其他電子業', '其他電子類', '電子通路業'},
    'ICT':  {'通信網路業', '資訊服務業', '電子商務業', '數位雲端', '數位雲端類'},
    'MAT':  {'鋼鐵工業', '塑膠工業', '化學工業', '水泥工業', '造紙工業',
             '玻璃陶瓷', '橡膠工業'},
    'IND':  {'電機機械', '電器電纜', '汽車工業', '建材營造', '航運業',
             '油電燃氣業', '綠能環保', '綠能環保類'},
    'BIO':  {'生技醫療業', '化學生技醫療', '農業科技業'},
    'CONS': {'食品工業', '貿易百貨', '觀光事業', '紡織纖維', '運動休閒',
             '居家生活類'},
    'FIN':  {'金融保險', '金融業'},
}
SIZE_OVERRIDE = {'FIN': ['LARGE']}   # 金融 MID/SMALL 結構性跳過


# ---------------------------------------------------------------- 資料載入

def sector_ids(sec):
    u = pd.read_parquet(f"{ROOT}/data/universe.parquet")
    u['stock_id'] = u['stock_id'].astype(str)
    u = u[(u['stock_id'].str.len() == 4) & (u['type'].isin(['twse', 'tpex']))]
    return set(u.loc[u['industry_category'].isin(SECTORS[sec]), 'stock_id'])


def load_sector(ids):
    parts = []
    for f in sorted(glob.glob(f"{ROOT}/data/adj/prices_adj_*.parquet")):
        d = pd.read_parquet(f, columns=['date', 'stock_id', 'open', 'max',
                                        'min', 'close', 'Trading_money'])
        d['stock_id'] = d['stock_id'].astype(str)
        d = d[d['stock_id'].isin(ids)]
        d['date'] = d['date'].astype(str).str.slice(0, 10)
        parts.append(d)
        del d
    p = pd.concat(parts, ignore_index=True)
    del parts
    p = p[p['date'] <= END].sort_values(['stock_id', 'date']).reset_index(drop=True)
    for c in ('open', 'max', 'min', 'close'):
        p[c] = p[c].astype('float64')
    return p


def add_size(p, ids):
    """族群內每季市值三分位。用該季第一個月的市值排名，季內固定。"""
    parts = []
    for f in sorted(glob.glob(f"{ROOT}/data/fundamentals/market_value_*.parquet")):
        m = pd.read_parquet(f)
        m['stock_id'] = m['stock_id'].astype(str)
        parts.append(m[m['stock_id'].isin(ids)])
        del m
    mv = pd.concat(parts, ignore_index=True)
    del parts
    mv['date'] = mv['date'].astype(str).str.slice(0, 10)
    mv = mv.drop_duplicates(['date', 'stock_id'])
    p = p.merge(mv[['date', 'stock_id', 'market_value']],
                on=['date', 'stock_id'], how='left')
    del mv
    p['market_value'] = p.groupby('stock_id')['market_value'].ffill()
    p['qtr'] = p['date'].str.slice(0, 4) + "Q" + \
        ((p['date'].str.slice(5, 7).astype(int) - 1) // 3 + 1).astype(str)
    first = p.groupby(['qtr', 'stock_id'])['market_value'].first().reset_index()
    first['r'] = first.groupby('qtr')['market_value'].rank(pct=True)
    first['size'] = np.select([first.r > 2/3, first.r > 1/3, first.r.notna()],
                              ['LARGE', 'MID', 'SMALL'], default=None)
    p = p.merge(first[['qtr', 'stock_id', 'size']], on=['qtr', 'stock_id'],
                how='left')
    del p['market_value'], p['qtr']
    return p


def add_features(p, full=False):
    """只算需要的特徵。full=True 時多算 brk120 / pullback / gap（觸發比較用）。"""
    sid = p['stock_id']
    c, lo = p['close'], p['min']

    def T(s, fn):
        return s.groupby(sid, sort=False).transform(fn)

    wins = (60, 120, 250) if full else (60, 250)
    for n in wins:
        hi = T(c, lambda s, n=n: s.shift(1).rolling(n, min_periods=n).max())
        p[f'brk{n}'] = ((c > hi) & (T(c, lambda s: s.shift(1))
                                    <= T(hi, lambda s: s.shift(1)))).fillna(False)
        del hi
    if full:
        prevhi = T(c, lambda s: s.shift(1).rolling(250, min_periods=250).max())
        ma20 = T(c, lambda s: s.rolling(20, min_periods=10).mean())
        p['pullback'] = ((c / prevhi > 0.90) & (c / prevhi < 1.0)
                         & (lo <= ma20) & (c > ma20)
                         & (T(c, lambda s: s.shift(1))
                            > T(ma20, lambda s: s.shift(1)))).fillna(False)
        p['gap'] = ((p['open'] / T(c, lambda s: s.shift(1)) - 1) > 0.03).fillna(False)
        del prevhi, ma20

    ret = T(c, lambda s: s.pct_change(fill_method=None))
    rv20 = T(ret, lambda s: s.rolling(20, min_periods=10).std())
    rv60 = T(ret, lambda s: s.rolling(60, min_periods=30).std())
    p['vc'] = (rv20 / (rv60 + 1e-9)).astype('float32')
    del ret, rv20, rv60

    # 絕對流動性地板。不用族群內百分位——那等於在同產業挑最大的，交集下來
    # MID/SMALL 必然歸零，看到的「只在大型股有效」會是母體定義造出來的假象。
    # （實測：改成絕對門檻後 MID/SMALL 仍然稀疏，真正的瓶頸是壓縮條件本身。）
    p['liq_ok'] = (T(p['Trading_money'],
                     lambda s: s.rolling(20, min_periods=10).mean())
                   >= LIQ_FLOOR).fillna(False)
    p['px_ok'] = c >= 10
    del p['max'], p['close'], p['Trading_money']
    gc.collect()
    return p


def add_revenue(p, th=0.20):
    """月營收 YoY 門檻。用 available_date 對齊（實際可取得日），不是財報期別。"""
    mr = pd.read_parquet(f"{ROOT}/data/fundamentals/month_revenue.parquet")
    mr['stock_id'] = mr['stock_id'].astype(str)
    mr = mr[mr['stock_id'].isin(set(p['stock_id']))].copy()
    col = [c for c in mr.columns if 'revenue' in c.lower()][0]
    mr['date'] = mr['available_date'].astype(str).str.slice(0, 10)
    mr = mr.sort_values(['stock_id', 'date'])
    mr['rev_ok'] = mr.groupby('stock_id')[col].pct_change(
        12, fill_method=None) > th
    p = p.merge(mr[['date', 'stock_id', 'rev_ok']],
                on=['date', 'stock_id'], how='left')
    del mr
    p['rev_ok'] = p.groupby('stock_id')['rev_ok'].ffill().fillna(False).astype(bool)
    gc.collect()
    return p


def index_dates(p):
    dates = sorted(p['date'].unique())
    p['_i'] = p['date'].map({d: i for i, d in enumerate(dates)})
    p = p.reset_index(drop=True)
    return p, len(dates) - MAXH - 2


# ---------------------------------------------------------------- 報酬計算

def returns4(p, union, cutoff_i, stop=STOP):
    """一趟迴圈算出四個持有期。先找第一次觸停損的位置，四個 H 共用。"""
    R = {H: np.full(len(p), np.nan) for H in HOLDS}
    oa, la, da = p['open'].values, p['min'].values, p['_i'].values
    for _, d in p.groupby('stock_id', sort=False):
        gi = d.index.values
        o, lo, di = oa[gi], la[gi], da[gi]
        u = union[gi]
        n = len(gi)
        for k in range(n - 1):
            if not u[k] or di[k] > cutoff_i:
                continue
            e = o[k + 1]
            if not np.isfinite(e) or e <= 0:
                continue
            sp = e * (1 - stop)
            far = min(k + 1 + MAXH, n - 1)
            j0, r_stop = -1, 0.0
            for j in range(k + 1, far + 1):
                if lo[j] <= sp:
                    j0 = j
                    r_stop = (min(o[j], sp) / e - 1) if o[j] <= sp else -stop
                    break
            for H in HOLDS:
                end = min(k + 1 + H, n - 1)
                r = r_stop if (0 <= j0 <= end) else (o[end] / e - 1)
                R[H][gi[k]] = (r - COST) * 100
    return R


def union_for(sig, valid, di, cutoff_i):
    """基準只需要「有訊號的那些日子」的同格全體，不必算全母體。"""
    days = np.unique(di[sig & (di <= cutoff_i)])
    return (valid & np.isin(di, days)) | sig


def daily_base(Rh, valid, di, nd):
    fin = np.isfinite(Rh) & valid
    s_ = np.bincount(di[fin], weights=Rh[fin], minlength=nd)
    c_ = np.bincount(di[fin], minlength=nd)
    return np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)


def norepeat(sig, R_ref, di, sids):
    """20 日不重複進場。R_ref 用 H=250 的報酬，確保四個 H 的樣本一致。"""
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
    """勝率以 >+5%（已扣成本）為界，平均賺/平均賠同界，賠率分子分母一致。"""
    w = a > 5.0
    aw = a[w].mean() if w.any() else 0.0
    al = a[~w].mean() if (~w).any() else 0.0
    return dict(n=len(a), win=w.mean() * 100, aw=aw, al=al,
                odds=aw / abs(al) if al else np.nan,
                exp=a.mean(), base=np.nanmean(b), exc=a.mean() - np.nanmean(b))


def prep(sec, full=False, revenue=False):
    ids = sector_ids(sec)
    p = load_sector(ids)
    p = add_size(p, ids)
    p = add_features(p, full=full)
    if revenue:
        p = add_revenue(p)
    return index_dates(p)


# ---------------------------------------------------------------- 模式

def mode_audit(sec='ELEC', size='LARGE', recipe='vcp'):
    p, cutoff = prep(sec, revenue=(recipe == 'rev'))
    di, sids = p['_i'].values, p['stock_id'].values
    yr = p['date'].str.slice(0, 4).astype(int).values
    nd = di.max() + 1
    valid = p['px_ok'].values & p['liq_ok'].values & (p['size'].values == size)
    vc = p['vc'].values
    if recipe == 'rev':
        brk = valid & p['brk60'].values
        sig = brk & p['rev_ok'].values
        title = "創60日新高 ∧ 月營收YoY>20%"
    elif recipe == 'bare250':
        brk = valid & p['brk250'].values
        sig = brk
        title = "創250日新高（裸，無壓縮門檻）"
    else:
        brk = valid & p['brk60'].values
        sig = brk & (vc < VC_TH)
        title = f"創60日新高 ∧ 壓縮<{VC_TH}"
    print(f"### 配方：{title}")
    print(f"### {sec}_{size} 母體 {pd.Series(sids[valid]).nunique()} 檔 / "
          f"{valid.sum():,} 股票日\n")

    if recipe == 'bare250':
        # 裸配方沒有門檻可掃，只比對前後半段
        R = returns4(p, union_for(brk, valid, di, cutoff), cutoff)
        db = daily_base(R[60], valid, di, nd)
        print(f"=== 1. 前後半段（H=60, 停損-{STOP:.0%}）===")
        sel = norepeat(sig, R[250], di, sids)
        for lab, msk in ((f"前 ~{SPLIT_YEAR}", yr[sel] <= SPLIT_YEAR),
                         (f"後 {SPLIT_YEAR+1}~", yr[sel] >= SPLIT_YEAR + 1)):
            k = sel[msk]
            a = R[60][k]
            m = np.isfinite(a)
            st = stats(a[m], db[di[k[m]]])
            print(f"{lab:<10}n={st['n']:<6} 勝率{st['win']:5.1f}%  賠率{st['odds']:5.2f}  "
                  f"期望值{st['exp']:+6.2f}  基準{st['base']:+6.2f}  超額{st['exc']:+6.2f}")
        del R
        gc.collect()
        return _audit_tail(p, valid, sig, di, sids, yr, nd, cutoff)

    R = returns4(p, union_for(brk, valid, di, cutoff), cutoff)
    db = daily_base(R[60], valid, di, nd)
    if recipe == 'rev':
        print(f"=== 1. 前後半段（H=60, 停損-{STOP:.0%}）===")
        sel = norepeat(sig, R[250], di, sids)
        for lab, msk in ((f"前 ~{SPLIT_YEAR}", yr[sel] <= SPLIT_YEAR),
                         (f"後 {SPLIT_YEAR+1}~", yr[sel] >= SPLIT_YEAR + 1)):
            k = sel[msk]
            a = R[60][k]
            m = np.isfinite(a)
            st = stats(a[m], db[di[k[m]]])
            print(f"{lab:<10}n={st['n']:<6} 勝率{st['win']:5.1f}%  賠率{st['odds']:5.2f}  "
                  f"期望值{st['exp']:+6.2f}  基準{st['base']:+6.2f}  超額{st['exc']:+6.2f}")
        del R
        gc.collect()
        return _audit_tail(p, valid, sig, di, sids, yr, nd, cutoff)

    print(f"=== 1. 壓縮門檻 x 前後半段（H=60, 停損-{STOP:.0%}）===")
    print(f"{'門檻':>8}{'前n':>6}{'前勝率':>8}{'前超額':>8}   "
          f"{'後n':>6}{'後勝率':>8}{'後超額':>8}")
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, None):
        s = brk if t is None else (brk & (vc < t))
        sel = norepeat(s, R[250], di, sids)
        cells = []
        for msk in (yr[sel] <= SPLIT_YEAR, yr[sel] >= SPLIT_YEAR + 1):
            k = sel[msk]
            a = R[60][k]
            m = np.isfinite(a)
            if m.sum() < 15:
                cells.append(None)
                continue
            st = stats(a[m], db[di[k[m]]])
            cells.append(f"{st['n']:>6}{st['win']:>7.1f}%{st['exc']:>+8.2f}")
        lab = "純突破" if t is None else f"<{t:.2f}"
        blank = f"{'—':>6}{'—':>8}{'—':>8}"
        print(f"{lab:>8}{cells[0] or blank}   {cells[1] or blank}")

    return _audit_tail(p, valid, sig, di, sids, yr, nd, cutoff)


def _audit_tail(p, valid, sig, di, sids, yr, nd, cutoff):
    """停損掃描 + 逐年 + 剔除最強年。兩種配方共用。"""
    U = union_for(sig, valid, di, cutoff)
    print(f"\n=== 2. 停損掃描（H=60）===")
    print(f"{'停損':>6}{'n':>6}{'勝率':>8}{'賠率':>7}{'期望值':>9}"
          f"{'基準':>8}{'超額':>8}")
    keep = None
    for stop in (0.08, 0.10, 0.12, 0.15, 0.20):
        R2 = returns4(p, U, cutoff, stop)
        d2 = daily_base(R2[60], valid, di, nd)
        sel = norepeat(sig, R2[250], di, sids)
        a = R2[60][sel]
        m = np.isfinite(a)
        st = stats(a[m], d2[di[sel[m]]])
        print(f"{-stop:>6.0%}{st['n']:>6}{st['win']:>7.1f}%{st['odds']:>7.2f}"
              f"{st['exp']:>+9.2f}{st['base']:>+8.2f}{st['exc']:>+8.2f}")
        if abs(stop - STOP) < 1e-9:
            keep = (a[m], sel[m], d2[di[sel[m]]])
        del R2
        gc.collect()

    a, k, b = keep
    y = yr[k]
    print(f"\n=== 3. 逐年（停損-{STOP:.0%}, H=60）===")
    print(f"{'年':>6}{'n':>5}{'勝率':>8}{'賠率':>7}{'期望值':>9}"
          f"{'基準':>8}{'超額':>8}")
    tot = pos = 0
    for Y in sorted(set(y)):
        mm = y == Y
        if mm.sum() < 5:
            print(f"{Y:>6}{mm.sum():>5}   樣本不足，不計入")
            continue
        st = stats(a[mm], b[mm])
        tot += 1
        pos += st['exc'] > 0
        print(f"{Y:>6}{st['n']:>5}{st['win']:>7.1f}%{st['odds']:>7.2f}"
              f"{st['exp']:>+9.2f}{st['base']:>+8.2f}{st['exc']:>+8.2f}")
    print(f"→ 正超額 {pos}/{tot} 年（n<5 的年份不計入）")

    contrib = {Y: (a[y == Y].mean() - np.nanmean(b[y == Y])) * (y == Y).sum()
               for Y in set(y) if (y == Y).sum() >= 5}
    best = max(contrib, key=contrib.get)
    print(f"\n=== 4. 剔除貢獻最大年（{best}）===")
    for lab, st in (("全期", stats(a, b)),
                    (f"剔{best}", stats(a[y != best], b[y != best]))):
        print(f"{lab:<8}n={st['n']:<5} 勝率{st['win']:5.1f}%  "
              f"平均賺{st['aw']:+7.2f}  平均賠{st['al']:+6.2f}  "
              f"賠率{st['odds']:5.2f}  期望值{st['exp']:+6.2f}  "
              f"基準{st['base']:+6.2f}  超額{st['exc']:+6.2f}")
    print(f"\n進場檔數 {pd.Series(sids[k]).nunique()} 檔，"
          f"年均 {len(a)/len(set(y)):.1f} 筆")
    q = np.percentile(a, [10, 25, 50, 75, 90])
    print(f"報酬分位 p10={q[0]:+.1f} p25={q[1]:+.1f} 中位={q[2]:+.1f} "
          f"p75={q[3]:+.1f} p90={q[4]:+.1f} 最大={a.max():+.1f}")
    print(f"前 3 大單筆佔總報酬 {np.sort(a)[-3:].sum()/a.sum()*100:.1f}%")


def mode_triggers(sec='ELEC', size='LARGE'):
    p, cutoff = prep(sec, full=True)
    di, sids = p['_i'].values, p['stock_id'].values
    yr = p['date'].str.slice(0, 4).astype(int).values
    nd = di.max() + 1
    valid = p['px_ok'].values & p['liq_ok'].values & (p['size'].values == size)
    vc = p['vc'].values
    TRIG = {'創60日新高': 'brk60', '創120日新高': 'brk120',
            '創250日新高': 'brk250', '突破後回踩MA20': 'pullback',
            '跳空3%': 'gap'}
    anyt = np.zeros(len(p), bool)
    for col in TRIG.values():
        anyt |= (valid & p[col].values)
    R = returns4(p, union_for(anyt, valid, di, cutoff), cutoff)
    db = daily_base(R[60], valid, di, nd)

    def row(name, sig):
        sel = norepeat(sig, R[250], di, sids)
        a = R[60][sel]
        m = np.isfinite(a)
        if m.sum() < 30:
            print(f"{name:<22}{m.sum():>6}  樣本不足")
            return
        a, k = a[m], sel[m]
        st = stats(a, db[di[k]])
        y = yr[k]
        halves = []
        for msk in (y <= SPLIT_YEAR, y >= SPLIT_YEAR + 1):
            halves.append(a[msk].mean() - np.nanmean(db[di[k[msk]]])
                          if msk.sum() >= 25 else np.nan)
        print(f"{name:<22}{st['n']:>6}{st['win']:>6.1f}%{st['odds']:>6.2f}"
              f"{st['exp']:>+8.2f}{st['base']:>+7.2f}{st['exc']:>+8.2f}"
              f"{halves[0]:>+9.2f}{halves[1]:>+9.2f}")

    hdr = (f"{'觸發':<22}{'n':>6}{'勝率':>7}{'賠率':>6}{'期望值':>8}"
           f"{'基準':>7}{'超額':>8}{'前半':>9}{'後半':>9}")
    print(f"=== 裸基座（H=60, 停損-{STOP:.0%}）===\n{hdr}")
    for name, col in TRIG.items():
        row(name, valid & p[col].values)
    print(f"\n=== 疊上壓縮 <{VC_TH} ===\n{hdr}")
    for name, col in TRIG.items():
        row(name, valid & p[col].values & (vc < VC_TH))
    print(f"\n=== 形態分解：創60日新高 是否同時創250日新高 ===\n{hdr}")
    b60 = valid & p['brk60'].values
    for lab, extra in (("創60且創250", p['brk250'].values),
                       ("創60非創250", ~p['brk250'].values)):
        row(lab + " 裸", b60 & extra)
        row(lab + f" ∧壓縮<{VC_TH}", b60 & extra & (vc < VC_TH))


def mode_holdout(sec='ELEC', size='LARGE'):
    """兩張卡都只抱 60 天，本來就不需要 250 天窗口。把已平倉要求降到 60 天，
    2025-08-27 之後的區間就成為一段從未被觀察過的樣本外資料。"""
    p, cutoff = prep(sec, revenue=True)
    di, sids = p['_i'].values, p['stock_id'].values
    nd = di.max() + 1
    dates = np.array(sorted(p['date'].unique()))
    cut60 = nd - 60 - 2
    print(f"訓練期進場截止 {dates[cutoff]}（需 250 天）")
    print(f"樣本外進場區間 {dates[cutoff+1]} ~ {dates[cut60]}"
          f"（{cut60-cutoff} 個交易日）\n")
    valid = p['px_ok'].values & p['liq_ok'].values & (p['size'].values == size)
    cards = {'卡片一 VCP': valid & p['brk60'].values & (p['vc'].values < VC_TH),
             '卡片二 裸250': valid & p['brk250'].values,
             '卡片三 營收': valid & p['brk60'].values & p['rev_ok'].values}
    both = np.zeros(len(p), bool)
    for _s in cards.values():
        both |= _s
    R = returns4(p, union_for(both, valid, di, cut60), cut60)
    db = daily_base(R[60], valid, di, nd)
    print(f"{'卡片':<14}{'區間':<8}{'n':>6}{'勝率':>8}{'賠率':>7}"
          f"{'期望值':>9}{'基準':>8}{'超額':>8}")
    for name, sig in cards.items():
        sel = norepeat(sig & (di <= cut60), R[60], di, sids)
        a = R[60][sel]
        m = np.isfinite(a)
        a, k = a[m], sel[m]
        b = db[di[k]]
        tr = di[k] <= cutoff
        for lab, msk in (('訓練期', tr), ('樣本外', ~tr)):
            if msk.sum() < 5:
                print(f"{name:<14}{lab:<8}{msk.sum():>6}  樣本不足，無法判定")
                continue
            st = stats(a[msk], b[msk])
            print(f"{name:<14}{lab:<8}{st['n']:>6}{st['win']:>7.1f}%"
                  f"{st['odds']:>7.2f}{st['exp']:>+9.2f}{st['base']:>+8.2f}"
                  f"{st['exc']:>+8.2f}")
        ho = ~tr
        if ho.sum() >= 5:
            print(f"{'':14}樣本外 {pd.Series(sids[k[ho]]).nunique()} 檔，"
                  f"中位{np.median(a[ho]):+.1f}，最大{a[ho].max():+.1f}，"
                  f"最小{a[ho].min():+.1f}")


def mode_overlap(sec='ELEC', size='LARGE'):
    """量測三張卡彼此的重疊與相關性。

    早先曾憑「卡片三的條件不在價格上」推論它與另外兩張正交 —— 實測相反：
    卡二與卡三重疊 45%、相關 0.881。能創 250 日新高的股票多半也剛創 60 日
    新高，而營收成長的公司才會一路創高，兩個條件在現實中高度共現。
    涉及「這兩個東西相不相關」的判斷要量，不要推。
    """
    p, cutoff = prep(sec, revenue=True)
    di, sids = p['_i'].values, p['stock_id'].values
    nd = di.max() + 1
    dates = np.array(sorted(p['date'].unique()))
    valid = p['px_ok'].values & p['liq_ok'].values & (p['size'].values == size)
    cards = {
        '卡一VCP': valid & p['brk60'].values & (p['vc'].values < VC_TH),
        '卡二裸250': valid & p['brk250'].values,
        '卡三營收': valid & p['brk60'].values & p['rev_ok'].values,
    }
    both = np.zeros(len(p), bool)
    for _s in cards.values():
        both |= _s
    R = returns4(p, union_for(both, valid, di, cutoff), cutoff)
    db = daily_base(R[60], valid, di, nd)
    S = {k: norepeat(v, R[250], di, sids) for k, v in cards.items()}
    K = list(cards)

    print("=== 進場點重疊（完全相同的 股票x日期）===")
    for i in range(len(K)):
        for j in range(i + 1, len(K)):
            a, b = set(S[K[i]]), set(S[K[j]])
            ov = len(a & b)
            print(f"{K[i]} ∩ {K[j]}: {ov} 筆  "
                  f"（占{K[i]} {ov/max(len(a),1)*100:.1f}%，"
                  f"占{K[j]} {ov/max(len(b),1)*100:.1f}%）")
    tri = set(S[K[0]]) & set(S[K[1]]) & set(S[K[2]])
    print(f"三張同時: {len(tri)} 筆")

    inter = np.zeros(len(p), bool)
    inter[list(set(S['卡一VCP']) & set(S['卡二裸250']))] = True
    if inter.sum() >= 30:
        a = R[60][inter]
        m = np.isfinite(a)
        st = stats(a[m], db[di[np.flatnonzero(inter)[m]]])
        print(f"\n卡一∩卡二 交集: n={st['n']} 勝率{st['win']:.1f}% "
              f"賠率{st['odds']:.2f} 超額{st['exc']:+.2f}")

    mo = pd.Series(dates).str.slice(0, 7).values
    ser = {}
    for k in K:
        sel = S[k]
        d = pd.DataFrame({'m': mo[di[sel]], 'r': R[60][sel]}).dropna()
        ser[k] = d.groupby('m')['r'].agg(['count', 'mean'])
    months = [m for m in sorted(set(mo)) if m <= dates[cutoff][:7]]
    cnt = pd.DataFrame({k: ser[k]['count'].reindex(months).fillna(0) for k in K})
    print(f"\n=== 活躍度（{len(months)} 個月）===")
    print(f"{'卡片':<12}{'有訊號月數':>10}{'占比':>8}{'月均筆數':>9}")
    for k in K:
        n = int((cnt[k] > 0).sum())
        print(f"{k:<12}{n:>10}{n/len(months)*100:>7.0f}%{cnt[k].mean():>9.1f}")
    print(f"三張都空手: {int((cnt == 0).all(axis=1).sum())} / {len(months)}")

    print("\n=== 月報酬相關係數 ===")
    rr = pd.DataFrame({k: ser[k]['mean'].reindex(months) for k in K})
    print(rr.corr().round(3).to_string())


def mode_grid():
    rows = []
    for sec in SECTORS:
        p, cutoff = prep(sec)
        di, sids = p['_i'].values, p['stock_id'].values
        nd = di.max() + 1
        print(f"[{sec}] {len(p):,} 列", flush=True)
        for z in SIZE_OVERRIDE.get(sec, ['LARGE', 'MID', 'SMALL']):
            valid = p['px_ok'].values & p['liq_ok'].values \
                & (p['size'].values == z)
            sig = valid & p['brk60'].values & (p['vc'].values < VC_TH)
            if (sig & (di <= cutoff)).sum() < 30:
                print(f"  {sec}_{z}: 訊號不足 30，跳過", flush=True)
                continue
            R = returns4(p, union_for(sig, valid, di, cutoff), cutoff)
            sel = norepeat(sig, R[250], di, sids)
            if len(sel) < 30:
                print(f"  {sec}_{z}: 去重後不足 30，跳過", flush=True)
                continue
            rec = {'cell': f"{sec}_{z}"}
            for H in HOLDS:
                db = daily_base(R[H], valid, di, nd)
                a = R[H][sel]
                m = np.isfinite(a)
                st = stats(a[m], db[di[sel[m]]])
                for key in ('n', 'win', 'odds', 'exp', 'base', 'exc'):
                    rec[f'{key}{H}'] = st[key]
            rows.append(rec)
            print(f"  {sec}_{z}: n={rec['n60']} "
                  + " ".join(f"H{H}超額{rec[f'exc{H}']:+.2f}" for H in HOLDS),
                  flush=True)
            del R
            gc.collect()
        del p
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(f"{os.environ.get('OUT_DIR', '.')}/probe_vcp_grid.csv", index=False)
    print(f"\n{'格':<13}{'n':>5}{'勝率':>7}{'賠率':>6}{'期望值':>8}"
          f"{'基準':>7}{'超額':>8}   (H=60)")
    for _, r in df.sort_values('exc60', ascending=False).iterrows():
        thin = "" if r.n60 >= 100 else "  ·薄"
        print(f"{r.cell:<13}{r.n60:>5}{r.win60:>6.1f}%{r.odds60:>6.2f}"
              f"{r.exp60:>+8.2f}{r.base60:>+7.2f}{r.exc60:>+8.2f}{thin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="audit",
                    choices=["audit", "triggers", "grid",
                             "holdout", "overlap"])
    ap.add_argument("--sector", default="ELEC")
    ap.add_argument("--size", default="LARGE")
    ap.add_argument("--recipe", default="vcp", choices=["vcp", "bare250"])
    args = ap.parse_args()
    if args.mode == "grid":
        mode_grid()
    elif args.mode == "overlap":
        mode_overlap(args.sector, args.size)
    elif args.mode == "holdout":
        mode_holdout(args.sector, args.size)
    elif args.mode == "triggers":
        mode_triggers(args.sector, args.size)
    else:
        mode_audit(args.sector, args.size, args.recipe)


if __name__ == "__main__":
    main()
