"""
週K教學圖的核心邏輯（無 UI 相依）。

從 app_teach.py 抽出來，讓 Streamlit app 和 GitHub Actions 的報表腳本
共用同一份程式碼。跟 stages_core.py 的做法一致——邏輯只有一份，
改一次兩邊都生效，不會出現「app 上是對的、報表是舊的」這種事。

快取用 lru_cache 而不是 st.cache_data：兩種執行環境都適用，
而且 Streamlit 重跑是同一個 process，效果一樣。
"""
import io
import os
import glob
import datetime as dt

from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.font_manager import FontProperties


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



# ---------------- 字型 ----------------
# ---------------- 主題與 K 棒樣式 ----------------
# 深色對照表：亮色 hex → 深色 hex。build_figure 裡所有顏色都經過 C() 查表，
# 所以切主題不用改繪圖邏輯，只要在這裡補對照。
DARK_MAP = {
    "white": "#12161c",          # 背景
    "#111": "#e8eaed", "#222": "#dfe3e8", "#333": "#d5dae0",
    "#444": "#c3c9d1", "#555": "#b3bac3", "#666": "#9aa2ad",
    "#999": "#7d8590",
    # 面板底色
    "#eaf1fb": "#151d2b", "#eaf7ee": "#132218", "#fdf3e7": "#241a10",
    "#f4f6f9": "#171b21", "#fffaf3": "#211a12", "#f2fbf5": "#111f16",
    "#fdf6e8": "#241f14", "#f8d7ce": "#3a1c17", "#f6a623": "#8a5a12",
    "#f3edfb": "#1f1830",
    # 階段色塊
    "#e8f5e9": "#12241a", "#e8eef8": "#141c2a", "#fdf1e3": "#241a11",
    "#fdecea": "#2a1614",
    # 線與框
    "#1f4e9c": "#6ea8fe", "#8fa8cf": "#3d5a86", "#c9a227": "#a98d3e",
    "#e0cfa0": "#5a4c2e", "#1b2a41": "#0d1620", "#e69138": "#c07a2a",
    # 文字強調
    "#2e7d32": "#5ec27a", "#1f7a3d": "#5ec27a", "#c0392b": "#ff6b5e",
    "#d2691e": "#ffa657", "#e07b1a": "#ffa657", "#8a4b08": "#e0a86a",
    "#6a1b9a": "#c58af9", "#ce93d8": "#7d4f96", "#7e57c2": "#b39ddb",
    "#00838f": "#4dd0e1", "#e6a700": "#ffd166", "#4caf50": "#4caf50",
    "#1e8449": "#26a269",
}

# 圖表區專用的深色對照。只套用在 K 線／RS／量／日線放大這四個座標軸內，
# 外圈面板與文字維持亮色——實測全黑底時面板和淺色文字大量看不清楚，
# 「只有圖表黑」才是實際好用的組合。
CHART_MAP = {
    "white": "#0f1319",
    "#111": "#e8eaed", "#222": "#dfe3e8", "#333": "#c8ced6",
    "#444": "#b8bfc8", "#555": "#a8b0ba", "#666": "#98a1ac", "#999": "#6b7480",
    "#1f4e9c": "#6ea8fe", "#8fa8cf": "#8fb8f0",
    "#2e7d32": "#5ec27a", "#c0392b": "#ff6b5e", "#e6a700": "#ffd166",
    "#00838f": "#4dd0e1", "#6a1b9a": "#c58af9", "#ce93d8": "#8f5faa",
    "#7e57c2": "#b39ddb", "#e07b1a": "#ffa657", "#1e8449": "#26a269",
    "#f8d7ce": "#3a1c17", "#e69138": "#e0a05a", "#4caf50": "#5ec27a",
    "#1f7a3d": "#5ec27a", "#d2691e": "#ffa657",
    # 階段色塊：深色底上要比亮色版本更收斂，不然一大片色塊會把 K 棒吃掉
    "#e8f5e9": "#16301d", "#e8eef8": "#141f33", "#fdf1e3": "#2e2110",
    "#fdecea": "#2e1714",
}

# 階段色塊透明度。太高會把 K 棒吃掉，尤其在深色圖表上。
STAGE_ALPHA = 0.40

THEME = "light"        # light / dark / chartdark
CANDLE = "紅漲綠跌"     # 紅漲綠跌 / 紅漲黑跌
SOLID_UP = True        # 台灣慣例是實心紅。False 則上漲畫空心

UP_C, DN_C = "#c0392b", "#1e8449"


def C(c: str) -> str:
    """外圈（面板、標題、底部四欄）的顏色查表。chartdark 時外圈維持亮色。"""
    return DARK_MAP.get(c, c) if THEME == "dark" else c


def CH(c: str) -> str:
    """圖表區內部的顏色查表。chartdark 與 dark 都套深色。"""
    if THEME in ("dark", "chartdark"):
        return CHART_MAP.get(c, DARK_MAP.get(c, c))
    return c


def set_style(theme: str = "light", candle: str = "紅漲綠跌",
              solid_up: bool = True, stage_alpha: float = 0.40) -> None:
    """切換主題與 K 棒配色。

    黑K在深色底上會看不見，所以深色主題選「紅漲黑跌」時改用淺灰——
    視覺上仍是「非紅即跌」的對比，只是把黑換成在深底上讀得到的顏色。
    """
    global THEME, CANDLE, SOLID_UP, UP_C, DN_C, STAGE_ALPHA
    THEME, CANDLE, SOLID_UP = theme, candle, solid_up
    STAGE_ALPHA = stage_alpha
    UP_C = CH("#c0392b")
    if candle == "紅漲黑跌":
        # 黑K在深色圖表上看不見，改用淺灰維持「非紅即跌」的對比
        DN_C = "#b6bcc6" if theme in ("dark", "chartdark") else "#1a1a1a"
    else:
        DN_C = CH("#1e8449")


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
def load_year(year: int) -> pd.DataFrame:
    return _read_parquet_local_first(
        f"data/adj/prices_adj_{year}.parquet", ADJ_URL.format(year=year))


@lru_cache(maxsize=None)
def load_universe() -> pd.DataFrame:
    return _read_parquet_local_first("data/universe.parquet", UNIVERSE_URL)


@lru_cache(maxsize=None)
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


def box_stats(wk: pd.DataFrame, look: int, ma_len: int = 20,
              pct_look: int = 104) -> dict:
    """箱體寬窄與現價位置。

    絕對門檻（30% / 60%）是實務常用切分，但它跟「箱寬 25% 當篩選門檻」是同一個病：
    金融股天生窄、飆股天生寬，絕對值分不出「在收縮」和「本來就不動」。
    所以同時給「自身百分位」——該股當前箱寬在自己過去 pct_look 週的分佈位置，
    這個才是判斷「有沒有在收縮」的依據。

    中線用 ma_len 週均線（大腸圈那條中線的概念），乖離率是相對中線不是相對箱底。"""
    b = wk.tail(look)
    hi, lo = float(b["high"].max()), float(b["low"].min())
    price = float(wk["close"].iloc[-1])
    width = (hi - lo) / lo * 100 if lo > 0 else np.nan
    pos = (price - lo) / (hi - lo) * 100 if hi > lo else 50.0

    grade = "窄箱" if width <= 30 else ("中箱" if width <= 60 else "寬箱")
    zone = "貼頂" if pos >= 70 else ("貼底" if pos <= 35 else "居中")

    ma = wk["close"].rolling(ma_len, min_periods=max(2, ma_len - 4)).mean()
    mid = float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else np.nan
    dev = (price / mid - 1) * 100 if mid == mid else np.nan

    # 歷史箱寬分佈：每一週往回看 look 週的箱寬
    rh = wk["high"].rolling(look).max()
    rl = wk["low"].rolling(look).min()
    hist = ((rh - rl) / rl * 100).tail(pct_look).dropna()
    pctile = float((hist <= width).mean() * 100) if len(hist) >= 20 else np.nan

    return dict(look=look, hi=hi, lo=lo, width=width, pos=pos, grade=grade,
                zone=zone, mid=mid, dev=dev, pctile=pctile, ma_len=ma_len,
                n_hist=len(hist))


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


def verdicts(m: dict, bx: dict, stage_now: int, vcp_ok: bool, vcp_legs) -> dict:
    """規則化結論。刻意死板：門檻寫在這裡，看得到才學得會。"""
    price, ma, slope = m["price"], m["ma_now"], m["slope"]
    stan = [
        f"目前判定為 {STAGE_NAME.get(stage_now, '未定')}"
        f"（30 週線 {ma:,.0f}，{'上彎' if slope > 0 else '走平或下彎'}）",
        f"股價 {price:,.0f}，{'在' if price > ma else '跌破'} 30 週線"
        f"{'之上' if price > ma else ''}，乖離 {(price / ma - 1) * 100:+.1f}%",
        f"箱體（近 {bx['look']} 週）{bx['lo']:,.0f}–{bx['hi']:,.0f}："
        f"箱寬 {bx['width']:.0f}% = {bx['grade']}，位置 {bx['pos']:.0f}% = {bx['zone']}",
        (f"箱寬在自身近 {bx['n_hist']} 週的第 {bx['pctile']:.0f} 百分位"
         f"（{'真的在收縮' if bx['pctile'] <= 20 else '不是收縮，是常態'}）"
         if bx["pctile"] == bx["pctile"] else "箱寬百分位：歷史不足")
        + (f"；{bx['ma_len']} 週中線乖離 {bx['dev']:+.1f}%"
           if bx["mid"] == bx["mid"] else ""),
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


def build_figure(stock_id, name, daily, wk, segs, dtbl, m, bx, vz,
                 vcp_legs, vcp_ok=False, rs=None, rs_new_high=False, show_verdict=True, show_deduct=True,
                 weeks=52, ded_period=5) -> plt.Figure:
    view = wk.tail(weeks).reset_index(drop=True)
    base = len(wk) - len(view)
    ma30 = wk["close"].rolling(MA_WEEKS, min_periods=MA_WEEKS - 2).mean()
    mav = ma30.iloc[base:].to_numpy(float)
    n = len(view)

    fig = plt.figure(figsize=(16, 12.8), dpi=110)
    fig.patch.set_facecolor(C("white"))
    def _style_axes(*axes):
        """圖表區的軸線、刻度、格線要自己設——matplotlib 預設是黑的，
        深色底上會整個看不見。逐軸設定而不是動全域 rcParams，
        這樣外圈面板不受影響。"""
        for a in axes:
            a.set_facecolor(CH("white"))
            a.tick_params(colors=CH("#555"))
            a.yaxis.label.set_color(CH("#333"))
            a.xaxis.label.set_color(CH("#333"))
            for sp in a.spines.values():
                sp.set_color(CH("#555"))
    gs = fig.add_gridspec(3, 2, height_ratios=[.05, .585, .365],
                          width_ratios=[.70, .30], hspace=.16, wspace=.07,
                          left=.035, right=.972, top=.975, bottom=.028)

    # 標題
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    segs_t = [(f"{stock_id} {name}：用 ", C("#111")), ("Stan", C("#1f4e9c")), (" / ", C("#111")),
              ("Minervini", C("#1f7a3d")), (" / ", C("#111")), ("扣抵值", C("#d2691e")),
              (" 讀週K線", C("#111"))]
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
                               facecolor=CH(STAGE_FACE[s_]), edgecolor="none",
                               alpha=STAGE_ALPHA, zorder=0))
        if a >= base:
            for _a in (ax, axr, axv):
                _a.axvline(a - base - .5, color=CH(STAGE_EDGE[s_]), ls="--",
                           lw=1.4, alpha=.85, zorder=2)
            ax.text(a - base - .5, ylo + (yhi - ylo) * .02,
                    f" {wk['date'].iloc[a]:%Y/%m/%d}", rotation=90, va="bottom",
                    color=CH(STAGE_EDGE[s_]), zorder=8,
                    bbox=dict(boxstyle="round,pad=.15", fc=CH("white"),
                              ec="none", alpha=.75), **zh(7.5))
        if x1 - x0 >= 5:
            _lv = .945 if (len([1 for aa, bb, _s in segs if bb >= base
                                and aa <= a]) % 2) else .885
            ax.text((x0 + x1) / 2, ylo + (yhi - ylo) * _lv, STAGE_NAME[s_],
                    ha="center", color=CH(STAGE_TEXT[s_]), **zh(13, True))

    for i in range(n):
        o, h, l, c = (float(view[k].iloc[i]) for k in ("open", "high", "low", "close"))
        col = UP_C if c >= o else DN_C
        ax.plot([i, i], [l, h], color=col, lw=.9, zorder=3)
        ax.add_patch(Rectangle((i - .32, min(o, c)), .64, abs(c - o) + (hi_all - lo_all) * 1e-3,
                               facecolor=(col if (c >= o and SOLID_UP)
                                          else (CH("white") if c >= o else col)),
                               edgecolor=col, lw=.9, zorder=4))
        axv.bar(i, float(view["volume"].iloc[i]) / 1000, .64, color=col, alpha=.75)
    ax.plot(range(n), mav, color=CH("#1f4e9c"), lw=2.1, zorder=5)
    if np.isfinite(mav[-1]):
        ax.text(n - .5, mav[-1], f"  {MA_WEEKS}週線 {mav[-1]:,.0f}", va="center",
                color=CH("#1f4e9c"), zorder=7, **zh(9.5, True))

    # 52 週高低 / Pivot / 箱體
    ax.axhline(m["hi52"], color=CH("#c0392b"), ls=":", lw=1.1, zorder=2)
    _hx = int(np.argmax(view["high"].to_numpy()))
    ax.annotate(f"52 週高點 {m['hi52']:,.0f}", xy=(_hx, m["hi52"]),
                xytext=(max(_hx - n * .14, n * .04), m["hi52"] - pad * .75),
                ha="center", color=CH("#c0392b"),
                arrowprops=dict(arrowstyle="->", color=CH("#c0392b")), **zh(10, True))
    ax.axhline(m["lo52"], color=CH("#2e7d32"), ls=":", lw=1.1, zorder=2)
    ax.text(n * .06, m["lo52"] - pad * .28, f"52 週低點 {m['lo52']:,.0f}",
            color=CH("#2e7d32"), **zh(9.5))
    # 箱體觀察窗：把「窄/中/寬」實際框出來，數字才有對應的東西可看
    _bl = min(bx["look"], n)
    ax.add_patch(Rectangle((n - _bl - .5, bx["lo"]), _bl, bx["hi"] - bx["lo"],
                           facecolor="none", edgecolor=CH("#00838f"), ls="--",
                           lw=1.5, zorder=6))
    _gc = {"窄箱": CH("#2e7d32"), "中箱": CH("#e6a700"), "寬箱": CH("#c0392b")}[bx["grade"]]
    # 標在箱底下方靠左，不要放箱頂——箱頂常常就是 Pivot，兩個標籤會疊在一起
    # 加底色框：這個位置會壓到中線，沒有底就讀不清楚
    ax.text(n - _bl - .3, bx["lo"] - pad * .12,
            f"{bx['grade']} {bx['width']:.0f}%｜位置 {bx['pos']:.0f}% {bx['zone']}",
            ha="left", va="top", color=_gc, zorder=8,
            bbox=dict(boxstyle="round,pad=.25", fc=CH("white"), ec=_gc,
                      lw=.8, alpha=.92), **zh(9.5, True))
    # 中線（大腸圈那條）
    _midv = wk["close"].rolling(bx["ma_len"],
                                min_periods=max(2, bx["ma_len"] - 4)).mean()
    ax.plot(range(n), _midv.iloc[base:].to_numpy(float), color=CH("#00838f"),
            lw=1.2, ls="-.", zorder=5)
    ax.text(n - .5, bx["mid"], f"  {bx['ma_len']}週中線 {bx['mid']:,.0f}",
            va="center", color=CH("#00838f"), zorder=7, **zh(8.5, True))

    ax.add_patch(Rectangle((n - min(12, n) - .5, m["pivot"] * .995),
                           min(12, n), m["pivot"] * .012,
                           facecolor=CH("#f8d7ce"), edgecolor=CH("#c0392b"), ls="--",
                           lw=1.2, alpha=.7, zorder=2))
    ax.text(n - min(12, n) / 2, m["pivot"] * 1.012,
            f"壓力 / Pivot 約 {m['pivot']:,.0f}", ha="center", color=CH("#c0392b"), **zh(10.5, True))
    ax.annotate(f"現價 {m['price']:,.0f}", xy=(n - 1, m["price"]),
                xytext=(n + n * .05, m["price"]), ha="left", color=CH("#333"),
                arrowprops=dict(arrowstyle="-", color=CH("#555")), **zh(10))

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
                color=CH("#c0392b"), ls="--", lw=1.3, zorder=6)
    if vcp_ok and vcp_legs:
        _mid = (vcp_legs[-1][0] + vcp_legs[-1][1]) / 2 - base
        ax.text(_mid, ylo + (yhi - ylo) * .10, "VCP / 波動收縮",
                ha="center", color=CH("#c0392b"), **zh(11.5, True))

    _style_axes(ax, axr, axv)
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
        axr.plot(range(n), rsv, color=CH("#6a1b9a"), lw=1.5)
        rmax = pd.Series(rsv).cummax().to_numpy()
        axr.plot(range(n), rmax, color=CH("#ce93d8"), lw=1.0, ls="--")
        axr.text(n - .5, rsv[-1], "  RS", va="center", color=CH("#6a1b9a"), **zh(9, True))
        if rs_new_high:
            axr.scatter([n - 1], [rsv[-1]], s=45, color=CH("#6a1b9a"), zorder=5)
            axr.text(n - 1, rsv[-1], "創新高  ", ha="right", va="bottom",
                     color=CH("#6a1b9a"), **zh(8.5, True))
    else:
        axr.text(.5, .5, "RS 無法計算", transform=axr.transAxes, ha="center",
                 color=CH("#999"), **zh(9))
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
                                facecolor=(col if (c >= o and SOLID_UP)
                                           else (CH("white") if c >= o else col)),
                                edgecolor=col, lw=.7))
    axd.plot(dd["close"].rolling(20, min_periods=1).mean(), color=CH("#1f4e9c"), lw=1.6)
    axd.axhline(m["pivot"], color=CH("#7e57c2"), ls="--", lw=1.3)
    axd.text(len(dd) * .02, m["pivot"] * 1.004, f"Pivot {m['pivot']:,.0f}",
             color=CH("#7e57c2"), **zh(8, True))
    dlo, dhi = float(dd["low"].min()), float(dd["high"].max())
    dp = (dhi - dlo) * .12
    axd.set_ylim(dlo - dp, dhi + dp)
    axd.grid(alpha=.22, ls=":"); axd.tick_params(labelsize=7)
    dt_ticks = [i for i in range(0, len(dd), 21)]
    axd.set_xticks(dt_ticks)
    axd.set_xticklabels([f"{dd['date'].iloc[i]:%m}月" for i in dt_ticks], **zh(7.5))
    _style_axes(axd)
    axd.set_title("日線放大圖（近 3 個月）", color=CH("#1f4e9c"), pad=5, **zh(11.5, True))
    for s_ in axd.spines.values():
        s_.set_color("#1f4e9c")

    # 趨勢模板逐條檢查
    axm = fig.add_subplot(gsr[1]); axm.axis("off")
    axm.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=.012",
                  transform=axm.transAxes, fc=C("#f2fbf5"), ec=C("#1f7a3d"), lw=1.6))
    axm.text(.5, .92, f"Minervini 趨勢模板　{m['tt_n']} / {m['tt_tot']} 條符合",
             ha="center", color=C("#1f7a3d"), **zh(11.5, True))
    items = list(m["tt"].items())
    for j, (k, v) in enumerate(items):
        y = .80 - j * (.72 / max(len(items), 1))
        axm.text(.055, y, "✓" if v else "×", va="center",
                 color=C("#2e7d32") if v else C("#c0392b"), **zh(10, True))
        axm.text(.115, y, k, va="center",
                 color=C("#222") if v else C("#999"), **zh(8.6))

    # 扣抵值明細表
    axk = fig.add_subplot(gsr[2]); axk.axis("off")
    axk.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=.012",
                  transform=axk.transAxes, fc=C("#fffaf3"), ec=C("#e07b1a"), lw=1.6))
    axk.text(.5, .945, f"扣抵值明細（{ded_period} 期均線）",
             ha="center", color=C("#d2691e"), **zh(11.5, True))
    if show_deduct and not dtbl.empty:
        cols = [("步數", .085), ("扣抵日", .245), ("扣抵值", .195),
                ("推估均線", .215), ("低於現價", .215)]
        xs, acc = [], .025
        for _, w_ in cols:
            xs.append(acc); acc += w_
        rows = len(dtbl)
        top, rh = .865, min(.075, .50 / max(rows + 1, 1))
        for j, (lab, w_) in enumerate(cols):
            axk.add_patch(Rectangle((xs[j], top - rh), w_, rh, fc=C("#fdf6e8"),
                          ec=C("#c9a227"), lw=.8, transform=axk.transAxes))
            axk.text(xs[j] + w_ / 2, top - rh / 2, lab, ha="center", va="center",
                     color=C("#8a4b08"), **zh(7.6, True))
        for i in range(rows):
            r = dtbl.iloc[i]
            y = top - rh * (i + 2)
            vals = [str(int(r["步數"])), f"{r['扣抵日']:%Y/%m/%d}",
                    f"{r['扣抵值']:,.0f}", f"{r['推估均線']:,.0f}",
                    "✓ 是" if r["低於現價"] else "× 否"]
            for j, (lab, w_) in enumerate(cols):
                axk.add_patch(Rectangle((xs[j], y), w_, rh, fc=C("white"),
                              ec=C("#e0cfa0"), lw=.6, transform=axk.transAxes))
                cc = C("#333")
                if j == 4:
                    cc = C("#2e7d32") if r["低於現價"] else C("#c0392b")
                axk.text(xs[j] + w_ / 2, y + rh / 2, vals[j], ha="center",
                         va="center", color=cc, **zh(7.4, j == 4))
        axk.text(.5, top - rh * (rows + 2) - .015,
                 f"現價 {m['price']:,.0f}　→　{m['up_n']}/{m['dtot']} 期扣抵值低於現價",
                 ha="center", va="top", color=C("#8a4b08"), **zh(8.4, True))
    for i, (t1, c1) in enumerate([("現價 > 扣抵值 → 均線較容易 上彎", C("#c0392b")),
                                  ("現價 ≈ 扣抵值 → 均線大致走平", C("#555")),
                                  ("現價 < 扣抵值 → 均線支撐 轉弱", C("#1f7a3d"))]):
        axk.text(.06, .250 - i * .058, t1, color=c1, **zh(8.6, True))
    axk.add_patch(Rectangle((.04, .022), .92, .048, fc=C("#f6a623"), alpha=.30,
                  ec=C("#e07b1a"), lw=1, transform=axk.transAxes))
    axk.text(.5, .046, "扣抵值不是預測股價，是看均線未來幾期的數學結構。",
             ha="center", va="center", color=C("#8a4b08"), **zh(8, True))

    # 底部四欄
    axb = fig.add_subplot(gs[2, :]); axb.axis("off")
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    palette = [("Stan", C("#1f4e9c"), C("#eaf1fb")), ("Minervini", C("#1f7a3d"), C("#eaf7ee")),
               ("扣抵值", C("#d2691e"), C("#fdf3e7"))]
    w, gap = .238, .0155
    for k, (nm, col, bg) in enumerate(palette):
        x0 = k * (w + gap)
        axb.add_patch(FancyBboxPatch((x0, .06), w, .88, boxstyle="round,pad=.008",
                      fc=bg, ec=col, lw=1.8))
        axb.text(x0 + w / 2, .845, nm, ha="center", color=col, **zh(17, True))
        axb.plot([x0 + .03, x0 + w - .03], [.795, .795], color=col, lw=1.1, alpha=.5)
        if not show_verdict:
            axb.text(x0 + w / 2, .45, "（先自己判讀）", ha="center", color=C("#999"), **zh(12))
            continue
        yy = .735
        for it in vz[nm][:5]:
            wr = _wrap(it, 19)
            axb.text(x0 + .022, yy, "●", color=col, va="top", **zh(7))
            axb.text(x0 + .045, yy + .006, wr, va="top", color=C("#222"), **zh(8.2))
            yy -= .056 + .050 * wr.count("\n")

    x0 = 3 * (w + gap)
    axb.add_patch(FancyBboxPatch((x0, .06), w, .88, boxstyle="round,pad=.008",
                  fc=C("#f4f6f9"), ec=C("#1b2a41"), lw=1.8))
    axb.add_patch(Rectangle((x0 + .004, .795), w - .008, .145, fc=C("#1b2a41")))
    # 這行的底是深藍色（兩種主題都是），文字要固定亮色。
    # 走 C() 查表的話深色主題會把 white 換成深色，字就隱形了。
    axb.text(x0 + w / 2, .866, "接下來可能的 3 種方向", ha="center",
             color="#f2f4f7", **zh(14.5, True))
    if show_verdict:
        for i, (arrow, ac, t1, t2) in enumerate(vz["scen"]):
            y = .68 - i * .215
            axb.text(x0 + .028, y, arrow, color=ac, va="center", **zh(15, True))
            axb.text(x0 + .062, y + .045, _wrap(t1, 22), color=C("#111"), va="center", **zh(9.6, True))
            axb.text(x0 + .062, y - .030, _wrap(t2, 24), color=C("#444"), va="center", **zh(9.0))
    else:
        axb.text(x0 + w / 2, .45, "（先自己判讀）", ha="center", color=C("#999"), **zh(12))
    return fig


