"""
Minervini SEPA 掃描的核心邏輯（無 UI 相依）。

從 app_minervini.py 抽出來，讓 Streamlit app 與 GitHub Actions 的
每日報表腳本共用同一份判斷。跟 stages_core.py / teach_core.py 同一個做法。
"""
import datetime as dt
import os
import io
import re

from functools import lru_cache

import numpy as np
import pandas as pd
import requests


REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
ADJ_URL = RAW + "/data/adj/prices_adj_{year}.parquet"
UNIVERSE_URL = RAW + "/data/universe.parquet"
YEARS_DEFAULT = 3          # 日常只需要近幾年：200日均線 + 52週高低最多用到一年多
FINMIND = "https://api.finmindtrade.com/api/v4/data"

MA_WEEKS = 30
DEFAULT_RISK_PCT = 1.25
DEFAULT_LIQ = 5000          # 萬元，60 日均額門檻

STATES = {
    "觸發": ("🟢", "突破樞紐 + 量價確認"),
    "準備": ("🟠", "VCP 接近完成，樞紐已能辨識"),
    "觀察": ("🟡", "基本條件好，但 Base/VCP 未完成"),
    "淘汰": ("🔴", "前面資格不合格"),
}

# ------------------------------------------------------------------ 資料載入


def _read_parquet_local_first(local: str, url: str) -> pd.DataFrame:
    """本機有就讀本機。Actions 與 Streamlit Cloud 都會 checkout 整個 repo，
    直接讀檔比打 raw URL 快，也避開 raw 的快取延遲——排程剛 commit 完
    馬上跑報表的話，raw 還會回舊檔。"""
    if os.path.exists(local):
        return pd.read_parquet(local)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))

@lru_cache(maxsize=None)
def load_parquet(url: str) -> pd.DataFrame:
    """本機優先：URL 尾巴對應到 repo 內同名路徑就直接讀檔。"""
    for key in ("/data/", ):
        if key in url:
            local = "data/" + url.split(key, 1)[1]
            if os.path.exists(local):
                return pd.read_parquet(local)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))



@lru_cache(maxsize=None)
def load_prices(years: int = YEARS_DEFAULT, as_of=None) -> pd.DataFrame:
    """讀還原股價。分年存檔，只載需要的年份——十一年共 169MB，全載會拖垮
    Streamlit Cloud 的記憶體，而日常掃描最多只用到一年多的歷史。"""
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    yrs = list(range(end.year - years + 1, end.year + 1))
    parts = []
    for y in yrs:
        try:
            parts.append(load_parquet(ADJ_URL.format(year=y)))
        except Exception:
            pass                                # 該年份檔不存在就跳過
    if not parts:
        raise RuntimeError("讀不到任何 data/adj/ 年份檔")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if as_of:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    return df


@lru_cache(maxsize=None)
def build_matrices(years: int = YEARS_DEFAULT, as_of=None):
    """把長表轉成 date × stock_id 的寬矩陣，之後所有計算都向量化。"""
    df = load_prices(years, as_of)
    # 還原股價的母體含 ETF、興櫃、下市股（2841 檔）。四碼、開頭非 0 才是普通股。
    df = df[df["stock_id"].astype(str).str.match(r"^[1-9]\d{3}$")]
    m = {}
    cols = [("c", "close"), ("h", "max"), ("l", "min"),
            ("v", "Trading_Volume"), ("mo", "Trading_money")]
    # 開盤價：判斷邏輯完全不用它，但回測需要——訊號在 T 日收盤產生，
    # 實際買得到的是 T+1 開盤。用收盤價當進場價會系統性高估策略績效。
    # 舊年份的 parquet 可能沒有這欄，所以是條件加入而不是硬寫死。
    if "open" in df.columns:
        cols.append(("o", "open"))
    for k, col in cols:
        m[k] = df.pivot(index="date", columns="stock_id", values=col).sort_index()
    # 停牌日會出現整列 0。這些 0 會把 52 週低點壓成 0（讓趨勢模板第 6 條
    # 永遠通過）、拉低均線、算出 -100% 的漲跌幅。一律當成缺值處理。
    bad = m["c"] <= 0
    for k in ("c", "h", "l", "o"):
        if k in m:
            m[k] = m[k].mask(bad).ffill()      # 價格沿用前一交易日
    for k in ("v", "mo"):
        m[k] = m[k].mask(bad)                  # 量與額留白，不要用 0 拉低均量
    return m, df


@lru_cache(maxsize=None)
def load_universe() -> pd.DataFrame:
    return load_parquet(UNIVERSE_URL).set_index("stock_id")


@lru_cache(maxsize=None)
def finmind(dataset: str, token: str, **kw) -> pd.DataFrame:
    """單次 FinMind 呼叫，失敗回空表。"""
    try:
        r = requests.get(FINMIND, params={"dataset": dataset, "token": token, **kw},
                         timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.DataFrame(r.json().get("data", []))
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------------ 型態偵測

def zigzag(h, l, pct=6.0):
    """交錯轉折點 [(idx, price, 'H'|'L')]，回檔/反彈超過 pct% 才算一個轉折。"""
    n = len(h)
    if n < 10:
        return []
    piv, d, ei, ep = [], 0, 0, h[0]
    for i in range(1, n):
        if d >= 0 and h[i] > ep:
            ei, ep = i, h[i]
        if d < 0 and l[i] < ep:
            ei, ep = i, l[i]
        if d >= 0:
            if l[i] <= ep * (1 - pct / 100):
                piv.append((ei, ep, "H")); d = -1; ei, ep = i, l[i]
        else:
            if h[i] >= ep * (1 + pct / 100):
                piv.append((ei, ep, "L")); d = 1; ei, ep = i, h[i]
    piv.append((ei, ep, "H" if d >= 0 else "L"))
    return piv


def vcp_foot(px: pd.DataFrame, pct=6.0):
    """VCP 足跡。回傳 dict：footprint 字串、收縮次數、最右跌幅、是否收縮、樞紐價。"""
    out = dict(foot="", n=0, first=None, last=None, ok=False, pivot=None, near=False)
    b = px.tail(120)
    if len(b) < 40:
        return out
    h, l = b["max"].values, b["min"].values
    piv = zigzag(h, l, pct)
    depths, starts = [], []
    for i in range(len(piv) - 1):
        if piv[i][2] == "H" and piv[i + 1][2] == "L":
            depths.append(round((piv[i][1] - piv[i + 1][1]) / piv[i][1] * 100, 1))
            starts.append(piv[i][0])
    price = float(b["close"].iloc[-1])
    pivot = round(float(b["max"].tail(20).max()), 2)
    out.update(pivot=pivot, near=bool(price >= pivot * 0.97))
    if len(depths) < 2:
        out.update(n=len(depths))
        return out
    depths, starts = depths[-4:], starts[-4:]
    wk = max(2, (len(b) - min(starts)) // 5)
    # 主判定用振幅收縮（比轉折點計數穩健）：把底部切三段比日振幅
    a = ((b["max"] - b["min"]) / b["close"] * 100).tail(60).values
    seg = len(a) // 3
    s1, s2, s3 = a[:seg].mean(), a[seg:2 * seg].mean(), a[2 * seg:].mean()
    vol = b["Trading_Volume"].tail(60).values
    dry = vol[-10:].mean() <= vol[:-10].mean() * 1.05
    ok = bool(s3 <= s1 * 0.75 and s3 <= s2 * 1.05 and s3 <= 6.0 and dry)
    out.update(foot=f"{wk}W-{depths[0]:.0f}/{depths[-1]:.0f}-{len(depths)}T",
               n=len(depths), first=depths[0], last=depths[-1], ok=ok)
    return out


def base_count(px: pd.DataFrame, thr=12.0, min_bars=15, lookback=378):
    """底部序號。自峰值回檔 ≥thr% 且持續 ≥min_bars 根，再創新高 = 完成一個底部。
    書裡：第 1、2 底最佳，第 3 底尚可，第 4、5 底已邁入後期。"""
    b = px.tail(lookback)
    c, h = b["close"].values, b["max"].values
    if len(c) < 60:
        return 0
    peak, pk_i, n, in_base, bs = h[0], 0, 0, False, 0
    for i in range(len(c)):
        if h[i] > peak:
            if in_base and (i - bs) >= min_bars:
                n += 1
            in_base, peak, pk_i = False, h[i], i
        elif not in_base and c[i] <= peak * (1 - thr / 100):
            in_base, bs = True, pk_i
    return min(6, n + (1 if in_base else 0))


def worst_drop(px: pd.DataFrame, lookback=252):
    """單日最大跌幅 / 今日跌幅 / 量能倍數 / 是否觸發爆量最大跌勢警訊。
    書裡 CROX 案例：第 2 階段以來的單日最大跌勢 + 爆量 = 賣訊，即使盈餘很好。"""
    b = px.tail(lookback)
    r = b["close"].pct_change() * 100
    v = b["Trading_Volume"]
    worst, today = r.min(), r.iloc[-1]
    volx = v.iloc[-1] / v.iloc[-51:-1].mean() if len(v) > 51 else 1.0
    alarm = bool(today <= worst * 0.95 and today < -4 and volx > 1.8)
    return round(float(worst), 1), round(float(today), 1), round(float(volx), 2), alarm


def stage_of(px: pd.DataFrame):
    """Weinstein 階段（30 週線 + 斜率）。"""
    w = px["close"].resample("W-FRI").last().dropna()
    if len(w) < MA_WEEKS + 5:
        return None, None, None
    ma = w.rolling(MA_WEEKS).mean()
    price, ma_now = w.iloc[-1], ma.iloc[-1]
    slope = ma.diff(4).iloc[-1]
    above = price > ma_now
    stage = 2 if (above and slope > 0) else 4 if (not above and slope < 0) \
        else 3 if above else 1
    return stage, round(float(ma_now), 2), not above


def find_breakout(px, lookback=90):
    """自動找最近一次突破日：收盤創 20 日新高且量 > 近50日均量 1.3 倍。回傳 index 位置或 None"""
    b = px.tail(lookback)
    c, v = b["close"].values, b["Trading_Volume"].values
    n = len(c)
    if n < 30: return None
    hi20 = pd.Series(c).rolling(20).max().shift(1).values
    va = pd.Series(v).rolling(50, min_periods=20).mean().shift(1).values
    hits = [i for i in range(20, n) if c[i] > (hi20[i] or 1e18) and v[i] > (va[i] or 1e18) * 1.15]
    if not hits:  # 放寬：不看量，只要創 20 日新高
        hits = [i for i in range(20, n) if c[i] > (hi20[i] or 1e18)]
    if not hits:
        return None
    # 連續創新高視為同一段漲勢，取這一段的「起點」而不是最後一天
    start = hits[-1]
    for a, b_ in zip(hits, hits[1:]):
        pass
    for k in range(len(hits) - 1, 0, -1):
        if hits[k] - hits[k - 1] <= 6:
            start = hits[k - 1]
        else:
            break
    return len(px) - n + start


def tennis(px, bo_pos=None):
    """網球 vs 雞蛋。bo_pos = 突破日在 px 中的整數位置。"""
    out = dict(judge="⏳觀察中", days=None, peak=None, pull=None,
               pull_days=None, volr=None, newhigh=False)
    if bo_pos is None:
        bo_pos = find_breakout(px)
    if bo_pos is None:
        out["judge"] = "—無突破"
        return out
    bo_pos = min(bo_pos, len(px) - 1)
    seg = px.iloc[bo_pos:]
    c = seg["close"].values; v = seg["Trading_Volume"].values
    days = len(seg) - 1
    out["days"] = days
    bo_price = c[0]
    peak_i = int(np.argmax(c)); peak = c[peak_i]
    price = c[-1]
    out["peak"] = round((peak / bo_price - 1) * 100, 1)
    out["pull"] = round((price / peak - 1) * 100, 1)
    out["pull_days"] = len(c) - 1 - peak_i
    out["newhigh"] = bool(peak_i == len(c) - 1)

    # 上漲段量 vs 回檔段量
    up_v = v[:peak_i + 1].mean() if peak_i >= 1 else v[0]
    dn_v = v[peak_i + 1:].mean() if peak_i + 1 < len(v) else np.nan
    out["volr"] = None if np.isnan(dn_v) or up_v <= 0 else round(dn_v / up_v, 2)

    if days < 5:
        return out
    if days > 25:      # 突破太久，網球行為只描述突破後幾天到一兩週
        out["judge"] = "—突破已久"
        return out
    if out["newhigh"] or (out["pull"] >= -8 and out["pull_days"] <= 10
                          and (out["volr"] is None or out["volr"] < 1.0)):
        out["judge"] = "🎾網球"
    elif (out["pull"] <= -12 or out["pull_days"] > 15
          or (out["volr"] is not None and out["volr"] >= 1.2)):
        out["judge"] = "🥚雞蛋"
    else:
        out["judge"] = "⏳觀察中"
    return out

# ------------------------------------------------------------------ 主掃描

def scan(m, liq_wan: float, min_days: int = 250, liq_pct: float = 0):
    """向量化計算全母體的趨勢模板、RS、量比。回傳 DataFrame。

    流動性門檻兩種模式：
      liq_pct = 0  絕對金額（liq_wan 萬元）。每日實跑用這個，直覺、可解釋。
      liq_pct > 0  取成交值前 liq_pct% 的股票，門檻隨市場呼吸。

    跨年度回測一定要用百分位。全市場日均成交值 2015 年約 1,013 億、
    2026 年約 12,309 億，差 12 倍；固定的 5,000 萬在 2015 篩出 355 檔
    （前兩成），在 2026 篩出 784 檔（前三成）。把兩者的事件池在一起算
    中位數，等於把兩種不同質的母體混著看，得出的分組差異可能只是
    母體組成改變的副作用。
    """
    c, h, l, v, mo = m["c"], m["h"], m["l"], m["v"], m["mo"]
    avg60 = mo.tail(60).mean()
    enough = c.notna().sum() >= min_days
    if liq_pct:
        # 分位數只在「歷史夠長」的股票裡算。把新股與長期停牌的殭屍
        # 一起丟進去算分位，會把門檻拉低，等於偷偷放寬。
        elig = avg60[enough & avg60.notna()]
        if elig.empty:
            return pd.DataFrame()
        keep = enough & (avg60 >= elig.quantile(1 - liq_pct / 100))
    else:
        keep = (avg60 > liq_wan * 1e4) & enough
    ids = keep[keep].index
    if len(ids) == 0:
        return pd.DataFrame()
    c, h, l, v, mo = [x[ids] for x in (c, h, l, v, mo)]

    ma50 = c.rolling(50).mean()
    ma150 = c.rolling(150).mean()
    ma200 = c.rolling(200).mean()
    px = c.iloc[-1]
    m50, m150, m200, m200_1mo = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1], ma200.iloc[-22]
    lo52, hi52 = l.tail(252).min(), h.tail(252).max()

    r126 = c.iloc[-1] / c.iloc[-127] - 1
    r63 = c.iloc[-1] / c.iloc[-64] - 1
    rs = (0.6 * r126 + 0.4 * r63).rank(pct=True) * 100

    cond = pd.DataFrame({
        "c1": (px > m150) & (px > m200),          # 股價高於 150 / 200 日均
        "c2": m150 > m200,
        "c3": m200 > m200_1mo,                     # 200 日均上升中
        "c4": (m50 > m150) & (m150 > m200),        # 均線多頭排列
        "c5": px > m50,
        "c6": px >= lo52 * 1.30,                   # 高於 52 週低點 30%
        "c7": px >= hi52 * 0.75,                   # 距 52 週高點 25% 內
        "c8": rs >= 70,
    })
    v5, v10 = v.tail(5).mean(), v.tail(10).mean()

    return pd.DataFrame({
        "收盤": px.round(2),
        "RS": rs.round(0),
        "TT分": cond.sum(axis=1),
        "TT全符": cond.sum(axis=1) == 8,
        "距高點": ((px / hi52 - 1) * 100).round(1),
        "量比": (v5 / v10).round(2),
        "量增": v5 > v10,
        "均額億": (mo.tail(60).mean() / 1e8).round(2),
        "MA50": m50.round(2),
    })


def classify(row) -> str:
    """五色分流。觸發要求靠近 52 週高——書裡買點在新高附近，不是任何 20 日高。"""
    if row["TT分"] < 6 or row["階段"] != 2:
        return "淘汰"
    near_hi = row["距高點"] >= -10
    if row["突破"] and row["量增"] and row["TT分"] >= 7 and near_hi:
        return "觸發"
    if row["近樞紐"] and near_hi and (row["VCP"] or row["TT分"] >= 7):
        return "準備"
    return "觀察"


