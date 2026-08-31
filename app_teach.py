"""
週K教學圖 (Chart Reading Trainer)
台股 / parquet 資料層

用途不是選股，是訓練「用 Stan / Minervini / 扣抵值 三套方法讀同一張圖」。
輸出一張可存檔的加註解週K圖：階段分段、52週高低、Pivot、VCP 收縮、
扣抵值示意表，加上三欄規則化結論與三種可能方向。

結論全部由規則產生，刻意寫得死板 —— 這支 app 的價值在「圖上標了什麼」，
不在文字漂不漂亮。要口語判讀請把圖丟給 AI。

字型：Streamlit Cloud 沒有中文字型，必須把字型檔放進 repo 的 fonts/ 目錄
      （建議 NotoSansTC-Regular.otf 與 NotoSansTC-Bold.otf），
      否則 matplotlib 會出豆腐字。
"""
import io
import os
import glob
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.font_manager import FontProperties

st.set_page_config(page_title="週K教學圖", page_icon="📚", layout="wide")

# ---------------- config ----------------
REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
ADJ_URL = RAW + "/data/adj/prices_adj_{year}.parquet"
UNIVERSE_URL = RAW + "/data/universe.parquet"

MA_WEEKS = 30          # Weinstein 主判斷線
RANGE_WEEKS = 52       # 52 週高低區間
TOP_ZONE = 0.60        # 站上均線但均線未上彎：區間 60% 以上算頭部，以下算轉折
SLOPE_LAG = 4          # 斜率回看週數

STAGE_NAME = {1: "打底區", 2: "Stage 2 上升段", 3: "高檔整理區", 4: "第四階段下跌"}
STAGE_FACE = {1: "#e8f5e9", 2: "#e8eef8", 3: "#fdf1e3", 4: "#fdecea"}
STAGE_EDGE = {1: "#4caf50", 2: "#8fa8cf", 3: "#e69138", 4: "#c0392b"}
STAGE_TEXT = {1: "#2e7d32", 2: "#1f4e9c", 3: "#e07b1a", 4: "#c0392b"}

UP_C, DN_C = "#c0392b", "#1e8449"     # 台股慣例：紅漲綠跌


# ---------------- 字型 ----------------
def _font_file(bold: bool) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # Regular 的萬用比對必須排除 Bold：sorted(glob("*.otf")) 會把 Bold 排在前面，
    # 於是整張圖都用粗體畫。
    pats = (["NotoSansTC-Bold*", "NotoSansCJK*Bold*", "*Bold*.otf", "*Bold*.ttf"]
            if bold else
            ["NotoSansTC-Regular*", "NotoSansCJK*Regular*", "*Regular*.otf",
             "*Regular*.ttf", "*.otf", "*.ttf"])
    for pat in pats:
        hit = sorted(glob.glob(os.path.join(here, "fonts", pat)))
        if not bold:
            hit = [h for h in hit if "bold" not in os.path.basename(h).lower()]
        if hit:
            return hit[0]
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc" if bold else
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(p):
            return p
    return ""


_FP_PATH, _FPB_PATH = _font_file(False), _font_file(True)
FONT_OK = bool(_FP_PATH)
FP = FontProperties(fname=_FP_PATH) if _FP_PATH else FontProperties()
FPB = FontProperties(fname=_FPB_PATH) if _FPB_PATH else FP
plt.rcParams["axes.unicode_minus"] = False


def zh(size=10, bold=False):
    return {"fontproperties": (FPB if bold else FP).copy(), "fontsize": size}


# ---------------- 資料層 ----------------
@st.cache_data(ttl=60 * 60 * 4, show_spinner="讀取行情資料…")
def load_year(year: int) -> pd.DataFrame:
    r = requests.get(ADJ_URL.format(year=year), timeout=180)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_universe() -> pd.DataFrame:
    r = requests.get(UNIVERSE_URL, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(ttl=60 * 60 * 4, show_spinner="計算大盤等權指數與 RS 排名…")
def market_context(as_of=None, years: int = 2):
    """從已快取的年度 parquet 直接算等權大盤指數與 RS 百分位排名。
    load_year() 本來就把整年全母體抓下來了，這裡不會多一次下載。

    等權指數用母體每日報酬的算術平均（= 每日再平衡的等權組合），
    與 stages_core.market_index() 同一套：不用中位數（複利會產生假的負漂移），
    不用成交值加權（會過度加權當下最熱的股票，指數本身變成追高）。"""
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    parts = []
    for y in range(end.year - years + 1, end.year + 1):
        try:
            parts.append(load_year(y))
        except Exception:
            pass
    if not parts:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    if as_of is not None:
        d = d[d["date"] <= pd.Timestamp(as_of)]
    c = d.pivot_table(index="date", columns="stock_id", values="close",
                      aggfunc="last").sort_index()
    c = c.replace(0, np.nan)
    # 上市未滿一年的新股會讓等權指數在上市日跳動，先要求有足夠歷史
    keep = c.notna().sum() >= 120
    c = c.loc[:, keep[keep].index]
    idx = (1 + c.pct_change().mean(axis=1).fillna(0)).cumprod()

    # Minervini 第 8 條：RS 排名。0.6×半年 + 0.4×季報酬，取百分位。
    if len(c) > 127:
        r126 = c.iloc[-1] / c.iloc[-127] - 1
        r63 = c.iloc[-1] / c.iloc[-64] - 1
        rs_rank = ((0.6 * r126 + 0.4 * r63).rank(pct=True) * 100).round(0)
    else:
        rs_rank = pd.Series(dtype=float)
    return idx, rs_rank


def load_one(stock_id: str, as_of=None, years: int = 4) -> pd.DataFrame:
    """單檔取價。教學圖要畫兩年多的週線 + 30 週均線暖機，抓 4 年最保險。"""
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    parts = []
    for y in range(end.year - years + 1, end.year + 1):
        try:
            d = load_year(y)
            parts.append(d[d["stock_id"] == stock_id])
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0]
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    out = df[["date", "open", "max", "min", "close", "Trading_Volume"]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]
    return out.sort_values("date").reset_index(drop=True)


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """丟掉還沒收盤的當週：半根週線的量是半根的量，爆量判斷會失真。"""
    w = (df.set_index("date").resample("W-FRI")
         .agg({"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"})
         .dropna(subset=["close"]).reset_index())
    if len(w) > 1 and w["date"].iloc[-1] > df["date"].iloc[-1]:
        w = w.iloc[:-1]
    return w.reset_index(drop=True)


# ---------------- 分析層 ----------------
def stage_series(wk: pd.DataFrame) -> pd.Series:
    """逐週跑 Weinstein 階段判斷。與 Stan_stages 的 classify() 同一套規則，
    差別只在那邊算橫斷面（全母體今天），這邊算時間序列（單檔歷史）。"""
    c, h, l = wk["close"], wk["high"], wk["low"]
    ma = c.rolling(MA_WEEKS, min_periods=MA_WEEKS - 2).mean()
    slope = ma.diff(SLOPE_LAG)
    hi = h.rolling(RANGE_WEEKS, min_periods=20).max()
    lo = l.rolling(RANGE_WEEKS, min_periods=20).min()
    pos = ((c - lo) / (hi - lo)).where(hi > lo, 0.5)

    above, up = c > ma, slope > 0
    s = pd.Series(1, index=wk.index, dtype=float)
    s[above & up] = 2
    s[(~above) & (slope < 0)] = 4
    s[above & (~up) & (pos >= TOP_ZONE)] = 3
    s[above & (~up) & (pos < TOP_ZONE)] = 1
    s[ma.isna()] = np.nan
    return s


def smooth_segments(s: pd.Series, min_len: int = 3):
    """把逐週階段壓成連續區塊，並吃掉太短的雜訊段。
    不平滑的話一段上升段裡會被三、四根回檔週切成碎片，圖會花掉。"""
    v = s.dropna()
    if v.empty:
        return []
    segs, cur, start = [], v.iloc[0], v.index[0]
    prev = v.index[0]
    for i, x in v.items():
        if x != cur:
            segs.append([start, prev, int(cur)])
            cur, start = x, i
        prev = i
    segs.append([start, prev, int(cur)])

    # 吃掉太短的段：併進「較長的那個鄰居」，不是無腦併前一個。
    # 早期版本直接把 segs[k-1] 的尾巴接到 segs[k+1] 的尾巴，等於連後面那段
    # 一起吞掉——2330 整段第二階段因此被標成第四階段。
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k in range(len(segs)):
            a, b, _ = segs[k]
            if b - a + 1 >= min_len:
                continue
            prev_len = (segs[k - 1][1] - segs[k - 1][0] + 1) if k > 0 else -1
            next_len = (segs[k + 1][1] - segs[k + 1][0] + 1) if k < len(segs) - 1 else -1
            if prev_len >= next_len:
                segs[k - 1][1] = b
            else:
                segs[k + 1][0] = a
            segs.pop(k)
            changed = True
            break
        if not changed:
            break
        # 併完可能出現相鄰同階段，先收一次再檢查下一個短段
        tmp = [segs[0]]
        for a, b, st_ in segs[1:]:
            if st_ == tmp[-1][2]:
                tmp[-1][1] = b
            else:
                tmp.append([a, b, st_])
        segs = tmp

    # 相鄰同階段再併一次
    out = [segs[0]]
    for a, b, st_ in segs[1:]:
        if st_ == out[-1][2]:
            out[-1][1] = b
        else:
            out.append([a, b, st_])
    return out


def zigzag(high, low, pct=6.0):
    """轉折點偵測。與 Minervini app 的 zigzag 同一套，回傳 (idx, price, 'H'/'L')。"""
    piv, d, ei, ep = [], 0, 0, high[0]
    for i in range(1, len(high)):
        if d >= 0:
            if high[i] > ep:
                ei, ep = i, high[i]
            elif low[i] < ep * (1 - pct / 100):
                piv.append((ei, ep, "H")); d = -1; ei, ep = i, low[i]
        else:
            if low[i] < ep:
                ei, ep = i, low[i]
            elif high[i] > ep * (1 + pct / 100):
                piv.append((ei, ep, "L")); d = 1; ei, ep = i, high[i]
    piv.append((ei, ep, "H" if d >= 0 else "L"))
    return piv


def vcp_contractions(wk: pd.DataFrame, look: int = 26, pct: float = 6.0):
    """在最後 look 週上找收縮：抓 H→L 的回檔段，只保留深度遞減的那幾段。
    回傳 [(起週, 迄週, 深度%)]，用來畫圖上那幾條愈縮愈小的弧。"""
    b = wk.tail(look).reset_index(drop=True)
    if len(b) < 8:
        return [], False
    off = len(wk) - len(b)
    piv = zigzag(b["high"].values, b["low"].values, pct)
    legs = []
    for i in range(len(piv) - 1):
        if piv[i][2] == "H" and piv[i + 1][2] == "L":
            dep = (piv[i][1] - piv[i + 1][1]) / piv[i][1] * 100
            legs.append((piv[i][0] + off, piv[i + 1][0] + off, round(dep, 1)))
    legs = legs[-4:]
    ok = len(legs) >= 2 and legs[-1][2] <= legs[0][2] * 0.75
    return legs, ok


def deduct_table(wk: pd.DataFrame, period: int, steps: int = 6):
    """扣抵值：未來每一步將被移出均線窗口的舊價格。
    現價 > 扣抵值 → 均線容易上彎；反之支撐轉弱。"""
    c = wk["close"].to_numpy(float)
    n, price = len(c), float(c[-1])
    steps = min(steps, period - 1, n - period)
    if steps < 1:
        return pd.DataFrame(), price
    rows = []
    win = list(c[-period:])
    for k in range(1, steps + 1):
        idx = n - period + k - 1
        win.pop(0); win.append(price)
        rows.append({"步數": k, "扣抵日": wk["date"].iloc[idx].date(),
                     "扣抵值": round(float(c[idx]), 2),
                     "推估均線": round(float(np.mean(win)), 2),
                     "低於現價": bool(price > c[idx])})
    return pd.DataFrame(rows), price


def rs_line(wk: pd.DataFrame, idx: pd.Series):
    """相對強度線 = 個股 ÷ 等權大盤，對齊到週。
    這條線比價格本身更能分辨「真蓄勢」與「死魚窄箱」：
    價格橫盤但 RS 線創新高 = 相對大盤在變強；RS 線走平或緩降 = 只是不動。"""
    if idx.empty:
        return pd.Series(dtype=float), False
    wi = idx.resample("W-FRI").last().reindex(
        pd.DatetimeIndex(wk["date"])).ffill()
    r = (wk["close"].to_numpy() / wi.to_numpy())
    rs = pd.Series(r, index=wk.index).replace([np.inf, -np.inf], np.nan)
    rs = rs / rs.dropna().iloc[0] * 100 if rs.notna().any() else rs
    look = min(26, len(rs))
    new_high = bool(rs.notna().any() and rs.iloc[-1] >= rs.tail(look).max() * 0.999)
    return rs, new_high


def read_metrics(daily: pd.DataFrame, wk: pd.DataFrame, dtbl: pd.DataFrame,
                 box_look: int, rs_rank=None, rs_new_high=False) -> dict:
    """三欄結論要用到的所有數字，集中算一次。"""
    c = wk["close"]
    ma30 = c.rolling(MA_WEEKS, min_periods=MA_WEEKS - 2).mean()
    price = float(c.iloc[-1])
    ma_now = float(ma30.iloc[-1]) if pd.notna(ma30.iloc[-1]) else np.nan
    slope = float(ma30.iloc[-1] - ma30.iloc[-1 - SLOPE_LAG]) \
        if len(ma30.dropna()) > SLOPE_LAG else np.nan

    n52 = min(RANGE_WEEKS, len(wk))
    hi52 = float(wk["high"].tail(n52).max())
    lo52 = float(wk["low"].tail(n52).min())

    bl = min(box_look, len(wk))
    box_hi = float(wk["high"].tail(bl).max())
    box_lo = float(wk["low"].tail(bl).min())
    pivot = float(wk["high"].tail(min(6, len(wk))).max())

    # Minervini 趨勢模板（日線）——只算得出來的七條，RS 排名要全母體，這裡不算
    d = daily["close"]
    m50, m150, m200 = [float(d.rolling(k).mean().iloc[-1]) if len(d) >= k else np.nan
                       for k in (50, 150, 200)]
    m200_1mo = float(d.rolling(200).mean().iloc[-22]) if len(d) >= 222 else np.nan
    dpx = float(d.iloc[-1])
    dlo52 = float(daily["low"].tail(252).min())
    dhi52 = float(daily["high"].tail(252).max())
    tt = {
        "股價 > 150日及200日均線": dpx > m150 and dpx > m200,
        "150日均 > 200日均": m150 > m200,
        "200日均上升中": m200 > m200_1mo,
        "均線多頭排列 (50>150>200)": m50 > m150 > m200,
        "股價 > 50日均線": dpx > m50,
        "高於52週低點 30% 以上": dpx >= dlo52 * 1.30,
        "距52週高點 25% 以內": dpx >= dhi52 * 0.75,
    }
    if rs_rank is not None and np.isfinite(rs_rank):
        tt["RS 排名 ≥ 70"] = bool(rs_rank >= 70)
    tt_n = int(sum(1 for v in tt.values() if v is True))

    v5 = float(daily["volume"].tail(5).mean())
    v10 = float(daily["volume"].tail(10).mean())

    up_n = int(dtbl["低於現價"].sum()) if not dtbl.empty else 0
    return dict(price=price, ma_now=ma_now, slope=slope, hi52=hi52, lo52=lo52,
                box_hi=box_hi, box_lo=box_lo, pivot=pivot, tt=tt, tt_n=tt_n,
                dhi52=dhi52, dist_hi=(dpx / dhi52 - 1) * 100,
                volr=v5 / v10 if v10 else np.nan,
                rs_rank=rs_rank, rs_new_high=rs_new_high, tt_tot=len(tt),
                up_n=up_n, dtot=len(dtbl))


def verdicts(m: dict, stage_now: int, vcp_ok: bool, vcp_legs) -> dict:
    """規則化結論。刻意死板：門檻寫在這裡，看得到才學得會。"""
    price, ma, slope = m["price"], m["ma_now"], m["slope"]
    stan = [
        f"目前判定為 {STAGE_NAME.get(stage_now, '未定')}"
        f"（30 週線 {ma:,.0f}，{'上彎' if slope > 0 else '走平或下彎'}）",
        f"股價 {price:,.0f}，{'在' if price > ma else '跌破'} 30 週線"
        f"{'之上' if price > ma else ''}，乖離 {(price / ma - 1) * 100:+.1f}%",
        f"52 週區間 {m['lo52']:,.0f}–{m['hi52']:,.0f}，現價在區間 "
        f"{(price - m['lo52']) / (m['hi52'] - m['lo52']) * 100:.0f}% 位置",
        f"跌破 {m['box_lo']:,.0f} 且 30 週線轉平，就要提防 Stage 3 風險"
        if stage_now == 2 else
        f"站上 {m['box_hi']:,.0f} 且 30 週線上彎，才算進入 Stage 2",
    ]
    mini = [
        f"趨勢模板 {m['tt_n']}/{m['tt_tot']} 條符合"
        + (f"（RS 排名 {m['rs_rank']:.0f}）" if m["rs_rank"] is not None
           and np.isfinite(m["rs_rank"]) else "（RS 排名無法計算）"),
        f"距 52 週高點 {m['dist_hi']:+.1f}%（{m['dhi52']:,.0f}）",
        f"Pivot（近 6 週高）約 {m['pivot']:,.0f}，"
        f"現價為其 {price / m['pivot'] * 100:.1f}%",
        f"VCP 收縮 {'成立' if vcp_ok else '未成立'}"
        + (f"（{len(vcp_legs)} 段，深度 {vcp_legs[0][2]:.0f}% → {vcp_legs[-1][2]:.0f}%）"
           if vcp_legs else "（收縮段不足 2 段）"),
        f"5日均量 / 10日均量 = {m['volr']:.2f}"
        f"（{'量增' if m['volr'] > 1 else '量縮'}），"
        f"帶量突破 {m['pivot']:,.0f} 才算真正續強",
    ]
    mini.insert(1, f"RS 線（個股 ÷ 等權大盤）"
                   f"{'創 26 週新高' if m['rs_new_high'] else '未創新高'}"
                   f" → {'相對大盤轉強' if m['rs_new_high'] else '相對大盤未領先'}")
    ded = [
        f"未來 {m['dtot']} 期扣抵值中，{m['up_n']} 期低於現價 → "
        f"均線{'偏向續彎' if m['up_n'] >= m['dtot'] * 0.7 else '支撐可能轉弱'}",
        "現價 > 扣抵值 → 均線容易上彎；現價 < 扣抵值 → 支撐轉弱",
        f"若股價原地不動，{m['dtot']} 期後均線推估為右表最後一列",
        "扣抵值不預測股價，只看均線未來幾期的數學結構",
    ]
    scen = [
        ("▲", "#2e7d32", f"A. 向上：帶量突破 {m['pivot']:,.0f}",
         "→ 主升趨勢延續，Minervini 續強"),
        ("▶", "#e6a700", f"B. 橫向：在 {m['box_lo']:,.0f}–{m['pivot']:,.0f} 間整理、量縮",
         "→ 高檔消化 / 箱體延續"),
        ("▼", "#c0392b", f"C. 向下：跌破 {m['box_lo']:,.0f} 與 30 週線 {ma:,.0f}",
         "→ 結構轉弱，留意 Stage 3 風險"),
    ]
    return dict(Stan=stan, Minervini=mini, 扣抵值=ded, scen=scen)


# ---------------- 繪圖層 ----------------
def _wrap(t, w=17):
    import textwrap
    return "\n".join(textwrap.wrap(t, width=w, break_long_words=False,
                                   break_on_hyphens=False))


def build_figure(stock_id, name, daily, wk, segs, dtbl, m, vz,
                 vcp_legs, vcp_ok=False, rs=None, rs_new_high=False, show_verdict=True, show_deduct=True,
                 weeks=52, ded_period=5) -> plt.Figure:
    view = wk.tail(weeks).reset_index(drop=True)
    base = len(wk) - len(view)
    ma30 = wk["close"].rolling(MA_WEEKS, min_periods=MA_WEEKS - 2).mean()
    mav = ma30.iloc[base:].to_numpy(float)
    n = len(view)

    fig = plt.figure(figsize=(16, 12), dpi=110)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[.055, .62, .33],
                          width_ratios=[.70, .30], hspace=.16, wspace=.07,
                          left=.035, right=.972, top=.975, bottom=.028)

    # 標題
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    segs_t = [(f"{stock_id} {name}：用 ", "#111"), ("Stan", "#1f4e9c"), (" / ", "#111"),
              ("Minervini", "#1f7a3d"), (" / ", "#111"), ("扣抵值", "#d2691e"),
              (" 讀週K線", "#111")]
    fig.canvas.draw(); cx = 0.0
    for txt, col in segs_t:
        t = axt.text(cx, .3, txt, color=col, va="center",
                     transform=axt.transAxes, **zh(25, True))
        fig.canvas.draw()
        cx = t.get_window_extent().transformed(axt.transAxes.inverted()).x1

    # 主圖
    gsl = gs[1, 0].subgridspec(3, 1, height_ratios=[.66, .15, .19], hspace=.06)
    ax = fig.add_subplot(gsl[0])
    axr = fig.add_subplot(gsl[1], sharex=ax)
    axv = fig.add_subplot(gsl[2], sharex=ax)
    lo_all = float(view["low"].min()); hi_all = float(view["high"].max())
    pad = (hi_all - lo_all) * .18
    ylo, yhi = lo_all - pad, hi_all + pad * 1.25

    # 階段色塊（畫在 K 棒之前）
    for a, b, s_ in segs:
        if b < base or s_ not in STAGE_FACE:
            continue
        x0, x1 = max(a - base, 0) - .5, b - base + .5
        ax.add_patch(Rectangle((x0, ylo), x1 - x0, yhi - ylo,
                               facecolor=STAGE_FACE[s_], edgecolor="none",
                               alpha=.75, zorder=0))
        if a >= base:
            for _a in (ax, axr, axv):
                _a.axvline(a - base - .5, color=STAGE_EDGE[s_], ls="--",
                           lw=1.0, alpha=.65, zorder=1)
            ax.text(a - base - .5, ylo + (yhi - ylo) * .02,
                    f" {wk['date'].iloc[a]:%Y/%m/%d}", rotation=90, va="bottom",
                    color=STAGE_EDGE[s_], **zh(7.5))
        if x1 - x0 >= 5:
            _lv = .945 if (len([1 for aa, bb, _s in segs if bb >= base
                                and aa <= a]) % 2) else .885
            ax.text((x0 + x1) / 2, ylo + (yhi - ylo) * _lv, STAGE_NAME[s_],
                    ha="center", color=STAGE_TEXT[s_], **zh(13, True))

    for i in range(n):
        o, h, l, c = (float(view[k].iloc[i]) for k in ("open", "high", "low", "close"))
        col = UP_C if c >= o else DN_C
        ax.plot([i, i], [l, h], color=col, lw=.9, zorder=3)
        ax.add_patch(Rectangle((i - .32, min(o, c)), .64, abs(c - o) + (hi_all - lo_all) * 1e-3,
                               facecolor="white" if c >= o else col,
                               edgecolor=col, lw=.9, zorder=4))
        axv.bar(i, float(view["volume"].iloc[i]) / 1000, .64, color=col, alpha=.75)
    ax.plot(range(n), mav, color="#1f4e9c", lw=2.1, zorder=5)
    if np.isfinite(mav[-1]):
        ax.text(n - .5, mav[-1], f"  {MA_WEEKS}週線 {mav[-1]:,.0f}", va="center",
                color="#1f4e9c", zorder=7, **zh(9.5, True))

    # 52 週高低 / Pivot / 箱體
    ax.axhline(m["hi52"], color="#c0392b", ls=":", lw=1.1, zorder=2)
    _hx = int(np.argmax(view["high"].to_numpy()))
    ax.annotate(f"52 週高點 {m['hi52']:,.0f}", xy=(_hx, m["hi52"]),
                xytext=(max(_hx - n * .14, n * .04), m["hi52"] - pad * .75),
                ha="center", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"), **zh(10, True))
    ax.axhline(m["lo52"], color="#2e7d32", ls=":", lw=1.1, zorder=2)
    ax.text(n * .06, m["lo52"] - pad * .28, f"52 週低點 {m['lo52']:,.0f}",
            color="#2e7d32", **zh(9.5))
    ax.add_patch(Rectangle((n - min(12, n) - .5, m["pivot"] * .995),
                           min(12, n), m["pivot"] * .012,
                           facecolor="#f8d7ce", edgecolor="#c0392b", ls="--",
                           lw=1.2, alpha=.7, zorder=2))
    ax.text(n - min(12, n) / 2, m["pivot"] * 1.012,
            f"壓力 / Pivot 約 {m['pivot']:,.0f}", ha="center", color="#c0392b", **zh(10.5, True))
    ax.annotate(f"現價 {m['price']:,.0f}", xy=(n - 1, m["price"]),
                xytext=(n + n * .05, m["price"]), ha="left", color="#333",
                arrowprops=dict(arrowstyle="-", color="#555"), **zh(10))

    # VCP 收縮弧。不成立就整組不畫——沒有收縮還畫弧等於在圖上教錯東西。
    for a, b, dep in (vcp_legs if vcp_ok else []):
        if a < base:
            continue
        x0, x1 = a - base, b - base
        if x1 <= x0:
            continue
        t = np.linspace(0, np.pi, 50)
        top = float(wk["high"].iloc[a])
        ax.plot(x0 + (x1 - x0) * t / np.pi,
                top - top * dep / 100 * np.sin(t) * 1.0,
                color="#c0392b", ls="--", lw=1.3, zorder=6)
    if vcp_ok and vcp_legs:
        _mid = (vcp_legs[-1][0] + vcp_legs[-1][1]) / 2 - base
        ax.text(_mid, ylo + (yhi - ylo) * .10, "VCP / 波動收縮",
                ha="center", color="#c0392b", **zh(11.5, True))

    ax.set_ylim(ylo, yhi); ax.set_xlim(-1.5, n + n * .16)
    ax.set_ylabel("股價 (元)", **zh(10)); ax.grid(alpha=.22, ls=":")
    ax.tick_params(labelbottom=False, labelsize=8.5)
    ax.set_title(f"{stock_id} {name}（週線）   調整後股價  "
                 f"{view['date'].iloc[0]:%Y/%m}–{view['date'].iloc[-1]:%Y/%m}"
                 f"（近 {n} 週）", loc="left", pad=8, **zh(14, True))
    ax.text(.995, 1.012, "單位：元", transform=ax.transAxes, ha="right", **zh(9))

    # RS 線：價格橫盤但 RS 創新高 = 相對大盤在變強，這是死魚窄箱分不出來的
    rsv = rs.iloc[base:].to_numpy(float) if rs is not None and len(rs) else None
    if rsv is not None and np.isfinite(rsv).any():
        axr.plot(range(n), rsv, color="#6a1b9a", lw=1.5)
        rmax = pd.Series(rsv).cummax().to_numpy()
        axr.plot(range(n), rmax, color="#ce93d8", lw=1.0, ls="--")
        axr.text(n - .5, rsv[-1], "  RS", va="center", color="#6a1b9a", **zh(9, True))
        if rs_new_high:
            axr.scatter([n - 1], [rsv[-1]], s=45, color="#6a1b9a", zorder=5)
            axr.text(n - 1, rsv[-1], "創新高  ", ha="right", va="bottom",
                     color="#6a1b9a", **zh(8.5, True))
    else:
        axr.text(.5, .5, "RS 無法計算", transform=axr.transAxes, ha="center",
                 color="#999", **zh(9))
    axr.set_ylabel("RS", **zh(9)); axr.grid(alpha=.2, ls=":")
    axr.tick_params(labelbottom=False, labelsize=7.5)

    axv.set_ylabel("成交量 (張)", **zh(9)); axv.grid(alpha=.2, ls=":")
    axv.tick_params(labelsize=8.5)
    # 刻度密度隨顯示週數調整：52 週以內每月一格，再長就改季度。
    # 年份放第二行，不跟月份擠在同一個標籤裡（擠在一起就全糊了）。
    step = 1 if n <= 60 else (3 if n <= 130 else 6)
    ticks, labs, seen, last_year = [], [], set(), None
    for i, d0 in enumerate(view["date"]):
        key = (d0.year, d0.month)
        if key in seen or (d0.month - 1) % step:
            continue
        seen.add(key); ticks.append(i)
        if d0.year != last_year:
            labs.append(f"{d0:%m}月\n{d0.year}")
            last_year = d0.year
        else:
            labs.append(f"{d0:%m}月")
    axv.set_xticks(ticks); axv.set_xticklabels(labs, **zh(8))

    # 右欄三塊：日線放大 / 趨勢模板檢查 / 扣抵值明細
    gsr = gs[1, 1].subgridspec(3, 1, height_ratios=[.30, .30, .40], hspace=.22)
    axd = fig.add_subplot(gsr[0])
    dd = daily.tail(62).reset_index(drop=True)
    for i in range(len(dd)):
        o, h, l, c = (float(dd[k].iloc[i]) for k in ("open", "high", "low", "close"))
        col = UP_C if c >= o else DN_C
        axd.plot([i, i], [l, h], color=col, lw=.7)
        axd.add_patch(Rectangle((i - .3, min(o, c)), .6, abs(c - o) + 1e-6,
                                facecolor="white" if c >= o else col, edgecolor=col, lw=.7))
    axd.plot(dd["close"].rolling(20, min_periods=1).mean(), color="#1f4e9c", lw=1.6)
    axd.axhline(m["pivot"], color="#7e57c2", ls="--", lw=1.3)
    axd.text(len(dd) * .02, m["pivot"] * 1.004, f"Pivot {m['pivot']:,.0f}",
             color="#7e57c2", **zh(8, True))
    dlo, dhi = float(dd["low"].min()), float(dd["high"].max())
    dp = (dhi - dlo) * .12
    axd.set_ylim(dlo - dp, dhi + dp)
    axd.grid(alpha=.22, ls=":"); axd.tick_params(labelsize=7)
    dt_ticks = [i for i in range(0, len(dd), 21)]
    axd.set_xticks(dt_ticks)
    axd.set_xticklabels([f"{dd['date'].iloc[i]:%m}月" for i in dt_ticks], **zh(7.5))
    axd.set_title("日線放大圖（近 3 個月）", color="#1f4e9c", pad=5, **zh(11.5, True))
    for s_ in axd.spines.values():
        s_.set_color("#1f4e9c")

    # 趨勢模板逐條檢查
    axm = fig.add_subplot(gsr[1]); axm.axis("off")
    axm.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=.012",
                  transform=axm.transAxes, fc="#f2fbf5", ec="#1f7a3d", lw=1.6))
    axm.text(.5, .92, f"Minervini 趨勢模板　{m['tt_n']} / {m['tt_tot']} 條符合",
             ha="center", color="#1f7a3d", **zh(11.5, True))
    items = list(m["tt"].items())
    for j, (k, v) in enumerate(items):
        y = .80 - j * (.72 / max(len(items), 1))
        axm.text(.055, y, "✓" if v else "×", va="center",
                 color="#2e7d32" if v else "#c0392b", **zh(10, True))
        axm.text(.115, y, k, va="center",
                 color="#222" if v else "#999", **zh(8.6))

    # 扣抵值明細表
    axk = fig.add_subplot(gsr[2]); axk.axis("off")
    axk.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=.012",
                  transform=axk.transAxes, fc="#fffaf3", ec="#e07b1a", lw=1.6))
    axk.text(.5, .945, f"扣抵值明細（{ded_period} 期均線）",
             ha="center", color="#d2691e", **zh(11.5, True))
    if show_deduct and not dtbl.empty:
        cols = [("步數", .085), ("扣抵日", .245), ("扣抵值", .195),
                ("推估均線", .215), ("低於現價", .215)]
        xs, acc = [], .025
        for _, w_ in cols:
            xs.append(acc); acc += w_
        rows = len(dtbl)
        top, rh = .865, min(.075, .50 / max(rows + 1, 1))
        for j, (lab, w_) in enumerate(cols):
            axk.add_patch(Rectangle((xs[j], top - rh), w_, rh, fc="#fdf6e8",
                          ec="#c9a227", lw=.8, transform=axk.transAxes))
            axk.text(xs[j] + w_ / 2, top - rh / 2, lab, ha="center", va="center",
                     color="#8a4b08", **zh(7.6, True))
        for i in range(rows):
            r = dtbl.iloc[i]
            y = top - rh * (i + 2)
            vals = [str(int(r["步數"])), f"{r['扣抵日']:%Y/%m/%d}",
                    f"{r['扣抵值']:,.0f}", f"{r['推估均線']:,.0f}",
                    "✓ 是" if r["低於現價"] else "× 否"]
            for j, (lab, w_) in enumerate(cols):
                axk.add_patch(Rectangle((xs[j], y), w_, rh, fc="white",
                              ec="#e0cfa0", lw=.6, transform=axk.transAxes))
                cc = "#333"
                if j == 4:
                    cc = "#2e7d32" if r["低於現價"] else "#c0392b"
                axk.text(xs[j] + w_ / 2, y + rh / 2, vals[j], ha="center",
                         va="center", color=cc, **zh(7.4, j == 4))
        axk.text(.5, top - rh * (rows + 2) - .015,
                 f"現價 {m['price']:,.0f}　→　{m['up_n']}/{m['dtot']} 期扣抵值低於現價",
                 ha="center", va="top", color="#8a4b08", **zh(8.4, True))
    for i, (t1, c1) in enumerate([("現價 > 扣抵值 → 均線較容易 上彎", "#c0392b"),
                                  ("現價 ≈ 扣抵值 → 均線大致走平", "#555"),
                                  ("現價 < 扣抵值 → 均線支撐 轉弱", "#1f7a3d")]):
        axk.text(.06, .250 - i * .058, t1, color=c1, **zh(8.6, True))
    axk.add_patch(Rectangle((.04, .022), .92, .048, fc="#f6a623", alpha=.30,
                  ec="#e07b1a", lw=1, transform=axk.transAxes))
    axk.text(.5, .046, "扣抵值不是預測股價，是看均線未來幾期的數學結構。",
             ha="center", va="center", color="#8a4b08", **zh(8, True))

    # 底部四欄
    axb = fig.add_subplot(gs[2, :]); axb.axis("off")
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    palette = [("Stan", "#1f4e9c", "#eaf1fb"), ("Minervini", "#1f7a3d", "#eaf7ee"),
               ("扣抵值", "#d2691e", "#fdf3e7")]
    w, gap = .238, .0155
    for k, (nm, col, bg) in enumerate(palette):
        x0 = k * (w + gap)
        axb.add_patch(FancyBboxPatch((x0, .06), w, .88, boxstyle="round,pad=.008",
                      fc=bg, ec=col, lw=1.8))
        axb.text(x0 + w / 2, .845, nm, ha="center", color=col, **zh(17, True))
        axb.plot([x0 + .03, x0 + w - .03], [.795, .795], color=col, lw=1.1, alpha=.5)
        if not show_verdict:
            axb.text(x0 + w / 2, .45, "（先自己判讀）", ha="center", color="#999", **zh(12))
            continue
        yy = .735
        for it in vz[nm][:5]:
            wr = _wrap(it)
            axb.text(x0 + .022, yy, "●", color=col, va="top", **zh(7))
            axb.text(x0 + .045, yy + .006, wr, va="top", color="#222", **zh(8.4))
            yy -= .062 + .055 * wr.count("\n")

    x0 = 3 * (w + gap)
    axb.add_patch(FancyBboxPatch((x0, .06), w, .88, boxstyle="round,pad=.008",
                  fc="#f4f6f9", ec="#1b2a41", lw=1.8))
    axb.add_patch(Rectangle((x0 + .004, .795), w - .008, .145, fc="#1b2a41"))
    axb.text(x0 + w / 2, .866, "接下來可能的 3 種方向", ha="center",
             color="white", **zh(14.5, True))
    if show_verdict:
        for i, (arrow, ac, t1, t2) in enumerate(vz["scen"]):
            y = .68 - i * .215
            axb.text(x0 + .028, y, arrow, color=ac, va="center", **zh(15, True))
            axb.text(x0 + .062, y + .045, _wrap(t1, 22), color="#111", va="center", **zh(9.6, True))
            axb.text(x0 + .062, y - .030, _wrap(t2, 24), color="#444", va="center", **zh(9.0))
    else:
        axb.text(x0 + w / 2, .45, "（先自己判讀）", ha="center", color="#999", **zh(12))
    return fig


# ---------------- UI ----------------
st.title("📚 週K教學圖")
st.caption("同一張圖，用 Stan / Minervini / 扣抵值 三套方法各判一次。"
           "結論由規則產生，刻意死板——門檻看得到才學得會。")

if not FONT_OK:
    st.error("找不到中文字型。請把 NotoSansTC-Regular.otf 與 NotoSansTC-Bold.otf "
             "放進 repo 的 fonts/ 目錄，否則圖上會是豆腐字。")

with st.sidebar:
    st.header("設定")
    sid = st.text_input("股票代號", value="3008").strip()
    weeks = st.slider("顯示週數", 26, 156, 52, 2,
                      help="52 週 = 近一年。要看完整的打底→上升→整理，拉到 104 以上。")
    st.divider()
    hist = st.checkbox("回到某一天重算", value=False,
                       help="把資料截斷到指定日期，等同回到那天看這張圖。"
                            "教學用途最有價值的功能——先遮住後面，自己判一次。")
    as_of = None
    if hist:
        as_of = st.date_input("資料截止日", value=dt.date.today(),
                              min_value=dt.date(2016, 6, 1), max_value=dt.date.today())
        st.caption("還原股價，2015-06 起。")
    st.divider()
    st.subheader("偵測參數")
    vcp_look = st.slider("VCP 回看週數", 12, 52, 26, 2,
                         help="在最後幾週上找收縮段。太短抓不到完整的底部，"
                              "太長會把上一個底部也算進來。")
    vcp_pct = st.slider("轉折認定 %", 3.0, 12.0, 6.0, 0.5,
                        help="回檔超過這個幅度才算一個轉折點。台股週線 6% 起跳；"
                             "低波動股（金融）要調到 3-4% 才抓得到東西。")
    box_look = st.slider("箱體回看週數", 6, 26, 12, 1)
    ded_period = st.slider("扣抵值示意用均線期數", 4, 12, 5, 1,
                           help="右下角示意表用。5 期最好懂，數字多了看不出移出的是哪一格。")
    min_seg = st.slider("階段最短週數", 2, 12, 6, 1,
                        help="短於這個週數的階段會被併進鄰段。不平滑的話，"
                             "一段上升裡幾根回檔週就會把色塊切成碎片。")
    st.divider()
    show_verdict = st.checkbox("顯示結論", value=True,
                               help="取消勾選 → 只出圖，三欄留白。"
                                    "訓練讀圖時先自己判，判完再勾回來對答案。")
    run = st.button("產生教學圖", type="primary", use_container_width=True)

if not run:
    st.info("左側輸入代號，按「產生教學圖」。訓練讀圖建議：先勾「回到某一天重算」"
            "把後面遮起來，取消「顯示結論」自己判一次，再勾回來對答案。")
    st.stop()

with st.spinner("讀取資料…"):
    daily = load_one(sid, as_of)
if daily.empty:
    st.error(f"{sid} 沒有資料。確認代號正確，且在 data/adj/ 的涵蓋範圍內。")
    st.stop()

wk = to_weekly(daily)
if len(wk) < MA_WEEKS + SLOPE_LAG + 2:
    st.error(f"{sid} 週線只有 {len(wk)} 根，不足以算 {MA_WEEKS} 週均線。")
    st.stop()

try:
    name = dict(zip(load_universe()["stock_id"], load_universe()["stock_name"])).get(sid, "")
except Exception:
    name = ""

idx_mkt, rs_rank_all = market_context(as_of)
rs, rs_new_high = rs_line(wk, idx_mkt)
rs_rank = float(rs_rank_all.get(sid, np.nan)) if len(rs_rank_all) else np.nan

segs = smooth_segments(stage_series(wk), min_seg)
stage_now = segs[-1][2] if segs else 0
vcp_legs, vcp_ok = vcp_contractions(wk, vcp_look, vcp_pct)
dtbl, _ = deduct_table(wk, ded_period, steps=ded_period)
m = read_metrics(daily, wk, dtbl, box_look, rs_rank, rs_new_high)
vz = verdicts(m, stage_now, vcp_ok, vcp_legs)

fig = build_figure(sid, name, daily, wk, segs, dtbl, m, vz, vcp_legs, vcp_ok,
                   rs, rs_new_high, show_verdict, True, weeks, ded_period)
st.pyplot(fig, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
st.download_button("⬇️ 下載這張圖 (PNG)", data=buf.getvalue(),
                   file_name=f"教學圖_{sid}_{(as_of or dt.date.today()):%Y%m%d}.png",
                   mime="image/png", type="primary")

st.divider()
st.subheader("結論（純文字，可直接複製）")
_c = st.columns(3)
for _i, _k in enumerate(["Stan", "Minervini", "扣抵值"]):
    with _c[_i]:
        st.markdown(f"**{_k}**")
        for _it in vz[_k]:
            st.markdown(f"- {_it}")
st.markdown("**接下來可能的 3 種方向**")
for _a, _col, _t1, _t2 in vz["scen"]:
    st.markdown(f"- {_t1} {_t2}")

_txt = (f"{sid} {name}　資料截止 {wk['date'].iloc[-1]:%Y-%m-%d}\n\n"
        + "\n\n".join(f"【{k}】\n" + "\n".join(f"- {x}" for x in vz[k])
                        for k in ["Stan", "Minervini", "扣抵值"])
        + "\n\n【接下來可能的 3 種方向】\n"
        + "\n".join(f"- {t1} {t2}" for _, _, t1, t2 in vz["scen"]))
st.text_area("整段複製（貼給 AI 解讀用）", _txt, height=220)

st.divider()
with st.expander("趨勢模板逐條檢查（Minervini）"):
    st.dataframe(pd.DataFrame({"條件": list(m["tt"].keys()),
                               "符合": list(m["tt"].values())}),
                 use_container_width=True, hide_index=True)
    if np.isfinite(rs_rank):
        st.caption(f"RS 排名 {rs_rank:.0f}（全母體百分位，0.6×半年報酬 + 0.4×季報酬）。"
                   f"RS 線{'已' if rs_new_high else '未'}創 26 週新高。")
    else:
        st.caption("RS 排名無法計算（母體歷史不足 127 個交易日）。")

with st.expander("階段分段明細"):
    st.caption("「圖上可見」為否 = 該段落在顯示區間之外。圖上的虛線與日期標的是每段起點。")
    st.dataframe(pd.DataFrame(
        [{"起": wk['date'].iloc[a].date(), "迄": wk['date'].iloc[b].date(),
          "週數": b - a + 1, "階段": STAGE_NAME.get(s_, s_),
          "圖上可見": b >= len(wk) - weeks} for a, b, s_ in segs]),
        use_container_width=True, hide_index=True)

with st.expander("扣抵值明細"):
    st.dataframe(dtbl, use_container_width=True, hide_index=True)
