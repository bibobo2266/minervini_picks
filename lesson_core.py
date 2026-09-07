"""
lesson_core.py
老手看盤教學 app 的共用邏輯：資料載入、突破偵測、C01~C11 量測、出圖。

被 app_teach.py（Streamlit）與 scripts/build_lesson_assets.py（GitHub Actions）共用。
所有量測公式一律以「突破日 T 當天收盤」為截斷點，不使用未來資料。
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# 路徑設定
# ----------------------------------------------------------------------------

DATA_DIR = os.environ.get("ADJ_DATA_DIR", "data/adj")
ASSET_DIR = os.environ.get("LESSON_ASSET_DIR", "assets/lessons")
FONT_DIR = os.environ.get("FONT_DIR", "fonts")

# 母體條件（與 simple momentum 對齊）
MIN_PRICE = 10.0            # 不做 10 元以下
MIN_TURNOVER = 50_000_000   # 60 日均額 > 5000 萬
LOOKBACK_HIGH = 250         # 250 日新高

TAIEX_CANDIDATES = ["TAIEX", "Y9999", "^TWII", "TWII"]

# 教材母體過濾
MIN_HISTORY_DAYS = 400        # 突破日前至少要有這麼多交易日，否則型態判讀沒有依據
EXCLUDE_SUFFIX = ("B", "L", "R", "U")   # 債券 / 槓桿 / 反向 / 期信


def is_teachable_symbol(sid: str) -> bool:
    """能不能拿來當看盤教材。

    保留：普通股（2330、6415…）與股票型 ETF（0050、0056、006208…）
    排除：債券 ETF（00679B）、槓桿（00631L）、反向（00632R）、期信（00635U）、
          以及權證、TDR 等非四～六碼代號。
    債券 ETF 一天波動 0.2%，任何「振幅」類指標都會被它們洗版，但它們沒有籌碼結構，
    型態不可轉移到個股，所以不能當教材。
    """
    sid = str(sid).strip().upper()
    if sid in TAIEX_CANDIDATES:
        return False
    if not sid[:1].isdigit():
        return False
    if sid.endswith(EXCLUDE_SUFFIX):
        return False
    # 只留純數字代號：普通股與股票型 ETF。特別股（2891B）、DR、權證等一律排除。
    return sid.isdigit() and 4 <= len(sid) <= 6


# ----------------------------------------------------------------------------
# 中文字型
# ----------------------------------------------------------------------------

def setup_font() -> str:
    """註冊 repo 內的 NotoSansTC，避免 Streamlit Cloud 出豆腐字。"""
    found = []
    for ext in ("*.otf", "*.ttf", "*.ttc"):
        found += glob.glob(os.path.join(FONT_DIR, "**", ext), recursive=True)
    name = None
    for path in found:
        try:
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            if name is None or "Noto" in prop.get_name():
                name = prop.get_name()
        except Exception:
            continue
    if name:
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    return name or "(fallback)"


# ----------------------------------------------------------------------------
# 資料載入
# ----------------------------------------------------------------------------

def load_prices(years: list[int] | None = None) -> pd.DataFrame:
    """讀 data/adj/prices_adj_YYYY.parquet，回傳長表。"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))
    if years:
        keep = {str(y) for y in years}
        files = [f for f in files if any(y in os.path.basename(f) for y in keep)]
    if not files:
        raise FileNotFoundError(
            f"找不到還原股價檔案：{DATA_DIR}/prices_adj_*.parquet"
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _normalise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """容忍不同欄名（FinMind / 自建）。"""
    ren = {}
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    mapping = {
        "stock_id": pick("stock_id", "symbol", "code", "ticker"),
        "date": pick("date", "trade_date", "datetime"),
        "open": pick("open", "open_price", "o"),
        "high": pick("high", "max", "high_price", "h"),
        "low": pick("low", "min", "low_price", "l"),
        "close": pick("close", "close_price", "c"),
        "volume": pick("volume", "trading_volume", "vol"),
        "amount": pick("amount", "trading_money", "turnover", "value"),
    }
    for std, src in mapping.items():
        if src and src != std:
            ren[src] = std
    df = df.rename(columns=ren)
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    need = ["stock_id", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}；實際欄位={list(df.columns)}")
    return df


def load_taiex(df: pd.DataFrame) -> pd.Series | None:
    """從同一份資料裡找加權指數，用來算 rs_excess。找不到就回 None。"""
    for cand in TAIEX_CANDIDATES:
        sub = df[df["stock_id"].astype(str) == cand]
        if len(sub) > 200:
            s = sub.set_index("date")["close"].sort_index()
            return s
    return None


# ----------------------------------------------------------------------------
# 突破偵測
# ----------------------------------------------------------------------------

def find_breakouts(g: pd.DataFrame) -> pd.DataFrame:
    """單檔：找出創 250 日收盤新高的日子。g 需按日期排序。"""
    g = g.reset_index(drop=True)
    if len(g) < LOOKBACK_HIGH + 30:
        return pd.DataFrame()

    roll_max_prev = g["close"].rolling(LOOKBACK_HIGH, min_periods=LOOKBACK_HIGH).max().shift(1)
    turnover60 = g["amount"].rolling(60, min_periods=40).mean()

    is_bo = (
        (g["close"] > roll_max_prev)
        & (g["close"] >= MIN_PRICE)
        & (turnover60 >= MIN_TURNOVER)
    )
    idx = np.where(is_bo.fillna(False).values)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    # 同一波只取第一根（20 日內不重複計）
    kept, last = [], -999
    for i in idx:
        if i < MIN_HISTORY_DAYS:      # 前面歷史太短，型態沒有依據
            continue
        if i - last > 20:
            kept.append(i)
            last = i
    out = g.loc[kept, ["stock_id", "date", "close"]].copy()
    out["i"] = kept
    out["prior_high"] = roll_max_prev.values[kept]
    return out


# ----------------------------------------------------------------------------
# 量測（C01 ~ C11 的 entry-time primary + 少數 diagnostic）
# ----------------------------------------------------------------------------

def compute_features(g: pd.DataFrame, i: int, taiex: pd.Series | None = None) -> dict:
    """g 為單檔完整歷史（已排序），i 為突破日在 g 中的位置。只用 <= i 的資料。"""
    h, l, c, o = g["high"].values, g["low"].values, g["close"].values, g["open"].values
    v = g["volume"].values
    f: dict[str, float] = {}

    # --- C03 突破推力 ---
    rng = h[i] - l[i]
    f["close_quality"] = 0.5 if rng <= 0 else float((c[i] - l[i]) / rng)
    prior_high = float(np.max(c[max(0, i - LOOKBACK_HIGH):i])) if i > 0 else c[i]
    f["breakout_clearance_pct"] = float((c[i] - prior_high) / prior_high * 100) if prior_high else np.nan
    f["overnight_vs_intraday_ratio"] = (
        float((c[i] - o[i]) / (c[i] - c[i - 1])) if i > 0 and abs(c[i] - c[i - 1]) > 1e-9 else np.nan
    )

    # --- C01 壓縮 ---
    if i >= 15:
        w = slice(i - 15, i)
        f["tightness_15d_pct"] = float(np.mean((h[w] - l[w]) / c[w]) * 100)
        f["close_compression_pct"] = float(np.std(c[w]) / np.mean(c[w]) * 100)
    else:
        f["tightness_15d_pct"] = f["close_compression_pct"] = np.nan
    if i >= 30:
        mh, ml = float(np.max(h[i - 30:i])), float(np.min(l[i - 30:i]))
        f["base_depth_pct"] = float((mh - ml) / mh * 100) if mh else np.nan
        f["base_high"], f["base_low"] = mh, ml
    else:
        f["base_depth_pct"] = np.nan
        f["base_high"] = f["base_low"] = np.nan

    # 均線糾結
    if i >= 120:
        mas = [float(np.mean(c[i - n + 1:i + 1])) for n in (20, 60, 120)]
        f["ma_coiling_spread_pct"] = float(np.std(mas) / np.mean(mas) * 100)
    else:
        f["ma_coiling_spread_pct"] = np.nan

    # --- C02 阻力消耗 ---
    if i >= 60:
        band = prior_high * 0.98
        touch = h[i - 60:i] >= band
        # 波峰計數：連續觸及只算一次
        cnt, prev = 0, False
        for t in touch:
            if t and not prev:
                cnt += 1
            prev = t
        f["prior_high_tests_count"] = int(cnt)
        near = c[i - 20:i] >= prior_high * 0.98
        f["prior_high_proximity_ratio"] = float(near.mean())
    else:
        f["prior_high_tests_count"] = np.nan
        f["prior_high_proximity_ratio"] = np.nan

    # --- C04 結構防守 ---
    if i >= 33:
        box_low = float(np.min(l[i - 33:i - 3]))
        seg_l, seg_c = l[i - 30:i], c[i - 30:i]
        below = seg_l < box_low
        depth = 0.0
        if below.any():
            j = int(np.argmax(below))
            if (seg_c[j:j + 3] > box_low).any():
                depth = float((box_low - seg_l[j]) / box_low * 100)
        f["shakeout_depth_pct"] = depth
    else:
        f["shakeout_depth_pct"] = np.nan

    # --- C05 量價換手 ---
    if i >= 60:
        f["volume_dryness_ratio"] = float(
            np.median(v[i - 15:i]) / max(np.median(v[i - 60:i]), 1)
        )
    else:
        f["volume_dryness_ratio"] = np.nan
    if i >= 20:
        seg_c, seg_v = c[i - 20:i], v[i - 20:i]
        hi, lo = seg_c.max(), seg_c.min()
        thr = lo + (hi - lo) * 2 / 3
        f["high_zone_turnover_share_20d"] = (
            float(seg_v[seg_c >= thr].sum() / max(seg_v.sum(), 1) * 100) if hi > lo else np.nan
        )
        f["breakout_volume_ratio"] = float(v[i] / max(np.median(seg_v), 1))
    else:
        f["high_zone_turnover_share_20d"] = f["breakout_volume_ratio"] = np.nan

    # --- C06 相對強弱 ---
    f["rs_excess_taiex_pct"] = np.nan
    if taiex is not None and i > 0:
        d0, d1 = g["date"].iloc[i - 1], g["date"].iloc[i]
        if d0 in taiex.index and d1 in taiex.index and taiex[d0]:
            mkt = (taiex[d1] / taiex[d0] - 1) * 100
            stk = (c[i] / c[i - 1] - 1) * 100
            f["rs_excess_taiex_pct"] = float(stk - mkt)

    # --- C07 推進效率 ---
    if i >= 21:
        tr = np.maximum.reduce([
            h[i - 20:i] - l[i - 20:i],
            np.abs(h[i - 20:i] - c[i - 21:i - 1]),
            np.abs(l[i - 20:i] - c[i - 21:i - 1]),
        ])
        denom = tr.sum()
        f["price_progress_efficiency_20d"] = (
            float(abs(c[i - 1] - c[i - 21]) / denom) if denom > 0 else np.nan
        )
    else:
        f["price_progress_efficiency_20d"] = np.nan

    # --- C08 趨勢年齡 ---
    if i >= LOOKBACK_HIGH:
        seg = c[i - LOOKBACK_HIGH:i]
        rm = pd.Series(seg).rolling(60).max().shift(1).values
        with np.errstate(invalid="ignore"):
            nh = seg > rm
        cnt, prev = 0, False
        for t in np.nan_to_num(nh, nan=0).astype(bool):
            if t and not prev:
                cnt += 1
            prev = t
        f["major_breakout_count_250d"] = int(cnt)
    else:
        f["major_breakout_count_250d"] = np.nan
    if i >= 120:
        lo_i = int(np.argmin(l[i - 120:i])) + (i - 120)
        f["days_since_120d_low"] = int(i - lo_i)
        f["return_since_120d_low"] = float((c[i] / l[lo_i] - 1) * 100)
    else:
        f["days_since_120d_low"] = f["return_since_120d_low"] = np.nan

    # --- C11 壓力修復 ---
    if i >= 60:
        seg_c, seg_h = c[i - 60:i], h[i - 60:i]
        rets = np.diff(seg_c) / seg_c[:-1]
        if len(rets):
            j = int(np.argmin(rets)) + 1
            target = seg_h[j]
            after = seg_c[j + 1:]
            hit = np.where(after >= target)[0]
            f["bars_to_recover_largest_down_day"] = int(hit[0] + 1) if len(hit) else -1
        else:
            f["bars_to_recover_largest_down_day"] = np.nan
    else:
        f["bars_to_recover_largest_down_day"] = np.nan

    return f


def compute_outcomes(g: pd.DataFrame, i: int) -> dict:
    """揭曉用（作答後才給）。"""
    c, l = g["close"].values, g["low"].values
    entry = float(g["open"].values[i + 1]) if i + 1 < len(g) else float(c[i])
    out = {"entry_price": entry}
    for n in (20, 60):
        out[f"ret_{n}d_pct"] = (
            float((c[i + n] / entry - 1) * 100) if i + n < len(g) else np.nan
        )
    end = min(i + 60, len(g) - 1)
    if end > i:
        mae = float((np.min(l[i + 1:end + 1]) / entry - 1) * 100)
        mfe = float((np.max(c[i + 1:end + 1]) / entry - 1) * 100)
    else:
        mae = mfe = np.nan
    out["mae_60d_pct"], out["mfe_60d_pct"] = mae, mfe
    out["hit_stop_12pct"] = bool(mae <= -12) if not np.isnan(mae) else None
    return out


# ----------------------------------------------------------------------------
# 出圖
# ----------------------------------------------------------------------------

UP, DOWN = "#d94040", "#2ca05a"   # 台股：紅漲綠跌


def _candles(ax, d: pd.DataFrame):
    x = np.arange(len(d))
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    up = c >= o
    ax.vlines(x[up], l[up], h[up], color=UP, linewidth=0.8)
    ax.vlines(x[~up], l[~up], h[~up], color=DOWN, linewidth=0.8)
    for m, col in ((up, UP), (~up, DOWN)):
        for xi, oi, ci in zip(x[m], o[m], c[m]):
            bot, hgt = min(oi, ci), max(abs(ci - oi), 1e-6)
            ax.add_patch(Rectangle((xi - 0.35, bot), 0.7, hgt,
                                   facecolor=col, edgecolor=col, linewidth=0.5))
    return x


def _xticks(ax, d: pd.DataFrame, n: int = 8):
    step = max(len(d) // n, 1)
    pos = list(range(0, len(d), step))
    ax.set_xticks(pos)
    ax.set_xticklabels([d["date"].iloc[p].strftime("%Y-%m") for p in pos], fontsize=8)


def render_chart(
    g: pd.DataFrame,
    i: int,
    feats: dict,
    mode: str = "annotated",      # annotated | plain | reveal
    lookback: int = 160,
    forward: int = 0,
    title: str = "",
    subtitle: str = "",
    outpath: str | None = None,
):
    """
    mode:
      plain     — 券商 app 風格，零標註，右邊切在突破日（出題用）
      annotated — 全標註 + 右側數值表（教材 / 揭曉用）
      reveal    — annotated + 打開突破後的走勢
    """
    lo = max(0, i - lookback)
    hi = min(len(g) - 1, i + (forward if mode == "reveal" else 0))
    d = g.iloc[lo:hi + 1].reset_index(drop=True)
    k = i - lo  # 突破日在 d 中的位置

    show_panel = mode in ("annotated", "reveal")
    fig = plt.figure(figsize=(13.5, 7.6) if show_panel else (10.5, 7.0), dpi=120)
    if show_panel:
        gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1], height_ratios=[3, 1],
                              hspace=0.06, wspace=0.03)
        ax, axv, axp = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax, axv = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        axp = None

    x = _candles(ax, d)

    # 均線
    for n, col in ((20, "#1f77b4"), (60, "#ff7f0e"), (120, "#9467bd")):
        if len(d) >= n:
            ma = d["close"].rolling(n).mean()
            ax.plot(x, ma, color=col, linewidth=1.1, label=f"MA{n}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    right_pad = 9 if show_panel else 2
    ax.set_xlim(-1, len(d) + right_pad)
    span = d["high"].max() - d["low"].min()
    ax.set_ylim(d["low"].min() - span * 0.08, d["high"].max() + span * 0.20)
    ax.grid(alpha=0.18)
    ax.set_xticklabels([])
    ax.set_xticks([])

    # 成交量（漲紅跌綠 + 量均線）
    up = d["close"].values >= d["open"].values
    axv.bar(x[up], d["volume"].values[up], color=UP, width=0.7)
    axv.bar(x[~up], d["volume"].values[~up], color=DOWN, width=0.7)
    for n, col in ((5, "#1f77b4"), (20, "#ff7f0e")):
        if len(d) >= n:
            axv.plot(x, d["volume"].rolling(n).mean(), color=col, linewidth=1.0)
    axv.set_xlim(-1, len(d) + (9 if show_panel else 2))
    axv.grid(alpha=0.18)
    axv.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y/1000:,.0f} 張"))
    axv.tick_params(axis="y", labelsize=8)
    axv.set_ylabel("成交量", fontsize=9)
    _xticks(axv, d)

    if show_panel:
        _annotate(ax, axv, d, k, feats, x)

    ax.set_title(title or "", fontsize=14, loc="left", pad=12)
    if subtitle:
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=9.5, color="#555")

    if axp is not None:
        _panel(axp, feats)

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return outpath
    return fig


def _annotate(ax, axv, d, k, feats, x):
    """把老手在看的東西全部標出來。教材階段標到誇張為止。"""
    prior_high = feats.get("prior_high")
    base_hi, base_lo = feats.get("base_high"), feats.get("base_low")
    y0, y1 = ax.get_ylim()

    # 整理區藍框
    if base_hi and base_lo and not np.isnan(base_hi) and k >= 30:
        ax.add_patch(Rectangle((k - 30, base_lo), 30, base_hi - base_lo,
                               facecolor="#3b82f6", alpha=0.10, edgecolor="#3b82f6",
                               linewidth=1.0, linestyle="--", zorder=0))
        ax.text(k - 30 + 1, base_hi, "  整理區 Base", fontsize=9, color="#1d4ed8",
                va="bottom")

    # 前高壓力線
    if prior_high and not np.isnan(prior_high):
        ax.axhline(prior_high, color="#dc2626", linestyle="--", linewidth=1.2, zorder=1)
        ax.text(len(d) * 0.42, prior_high, f"250日前高 / 壓力線 {prior_high:.1f}",
                fontsize=9.5, color="#dc2626", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

        # 前高測試點
        band = prior_high * 0.98
        marked = 0
        prev = False
        for j in range(max(0, k - 60), k):
            t = d["high"].iloc[j] >= band
            if t and not prev and marked < 4:
                ax.annotate("前高測試", xy=(j, d["high"].iloc[j]),
                            xytext=(j, d["high"].iloc[j] + (y1 - y0) * 0.07),
                            ha="center", fontsize=8.5, color="#b45309",
                            arrowprops=dict(arrowstyle="->", color="#b45309", lw=1))
                marked += 1
            prev = t

    # 洗盤點
    if feats.get("shakeout_depth_pct", 0) and feats["shakeout_depth_pct"] > 0 and k >= 30:
        j = int(np.argmin(d["low"].iloc[k - 30:k].values)) + (k - 30)
        ax.annotate("假跌破洗盤 → 收回", xy=(j, d["low"].iloc[j]),
                    xytext=(j - 12, d["low"].iloc[j] - (y1 - y0) * 0.09),
                    fontsize=9, color="#047857",
                    arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2))
        ax.scatter([j], [d["low"].iloc[j]], s=70, facecolor="none",
                   edgecolor="#047857", linewidth=1.6, zorder=5)

    # 突破日
    ax.annotate("突破日 T", xy=(k, d["high"].iloc[k]),
                xytext=(k - 16, min(d["high"].iloc[k] + (y1 - y0) * 0.10, y1 - (y1 - y0) * 0.04)),
                fontsize=11, color="#dc2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8))
    ax.axvline(k, color="#dc2626", alpha=0.28, linewidth=1.0)
    axv.axvline(k, color="#dc2626", alpha=0.28, linewidth=1.0)

    # 突破當根 K 的高低收，直接標在旁邊
    row = d.iloc[k]
    ax.annotate(
        f"突破日\n開 {row['open']:.1f}\n高 {row['high']:.1f}\n低 {row['low']:.1f}\n收 {row['close']:.1f}",
        xy=(k + 2.0, row["close"]), fontsize=8.2, va="center", ha="left",
        color="#0f172a", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#dc2626", lw=1.0))

    if feats.get("breakout_volume_ratio") and not np.isnan(feats["breakout_volume_ratio"]):
        axv.annotate(f"突破量 = 20日中位數 × {feats['breakout_volume_ratio']:.1f}",
                     xy=(k, d["volume"].iloc[k]),
                     xytext=(max(k - 45, 1), d["volume"].max() * 0.92),
                     fontsize=8.5, color="#dc2626",
                     arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))


PANEL_ROWS = [
    ("close_quality", "收盤品質", "C03", "{:.2f}"),
    ("breakout_clearance_pct", "突破淨幅度", "C03", "{:+.1f}%"),
    ("tightness_15d_pct", "整理緊湊度", "C01", "{:.1f}%"),
    ("base_depth_pct", "Base 深度", "C01", "{:.1f}%"),
    ("ma_coiling_spread_pct", "均線糾結", "C01", "{:.1f}%"),
    ("prior_high_tests_count", "前高測試次數", "C02", "{:.0f}"),
    ("shakeout_depth_pct", "洗盤深度", "C04", "{:.1f}%"),
    ("volume_dryness_ratio", "量縮程度", "C05", "{:.2f}"),
    ("high_zone_turnover_share_20d", "高檔換手佔比", "C05", "{:.0f}%"),
    ("rs_excess_taiex_pct", "相對大盤強弱", "C06", "{:+.1f}%"),
    ("price_progress_efficiency_20d", "推進效率", "C07", "{:.2f}"),
    ("major_breakout_count_250d", "250日內突破次數", "C08", "{:.0f}"),
    ("days_since_120d_low", "距120日低點", "C08", "{:.0f} 天"),
    ("bars_to_recover_largest_down_day", "長黑收復天數", "C11", "{:.0f} 天"),
]


def _panel(axp, feats: dict):
    axp.axis("off")
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    axp.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=axp.transAxes,
                            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1))
    axp.text(0.5, 0.965, "老手觀察重點", ha="center", fontsize=12, fontweight="bold",
             color="#0f172a", transform=axp.transAxes)

    n = len(PANEL_ROWS)
    top, bottom = 0.925, 0.06
    step = (top - bottom) / n
    for idx, (key, label, cid, fmt) in enumerate(PANEL_ROWS):
        y = top - step * (idx + 0.55)
        val = feats.get(key)
        txt = "—" if val is None or (isinstance(val, float) and np.isnan(val)) else fmt.format(val)
        if idx % 2 == 0:
            axp.add_patch(Rectangle((0.04, y - step * 0.42), 0.92, step * 0.86,
                                    transform=axp.transAxes, facecolor="#eef2f7",
                                    edgecolor="none"))
        axp.text(0.07, y, f"{cid}  {label}", fontsize=9.2, va="center",
                 color="#334155", transform=axp.transAxes)
        axp.text(0.94, y, txt, fontsize=10.2, va="center", ha="right",
                 color="#0f172a", fontweight="bold", transform=axp.transAxes)


# ----------------------------------------------------------------------------
# 課綱
# ----------------------------------------------------------------------------

@dataclass
class Lesson:
    cid: str
    name: str
    order: int
    question: str          # 學會後你能回答什麼
    idea: str              # 心法（老手在問什麼）
    primary: str           # primary field
    formula: str
    good: str              # 什麼樣算「好」
    bad: str
    reasons: list[str]     # 作答時可選的理由標籤


LESSONS: list[Lesson] = [
    Lesson(
        "C03", "突破推力", 1,
        "這根 K 乾不乾淨？",
        "老手看突破那一根，第一眼不是看漲幾 %，是看「收在哪裡」。"
        "收在最高附近，代表買方從頭守到收盤，沒人在尾盤倒貨；"
        "收在中間或下緣、留一根長上影，代表衝上去被打回來，這根是假的。",
        "close_quality",
        "close_quality = (收盤 − 最低) ÷ (最高 − 最低)，值在 0 ~ 1 之間",
        "≥ 0.85：收盤幾乎貼著當日最高，上影線很短，K 棒實體長而紅",
        "≤ 0.40：長上影線，衝高被壓回，收盤靠近下緣",
        ["收盤夠強（貼最高）", "突破淨幅度夠", "帶量突破", "盤中推升非跳空",
         "上影線太長", "收盤偏弱", "量沒跟上", "靠跳空硬拉"],
    ),
    Lesson("C01", "壓縮整理", 2, "噴之前有沒有被壓成彈簧？",
           "老手看突破前那段：每天振幅是越來越小還是照樣三溫暖？"
           "振幅收斂代表買賣雙方分歧變小、浮額洗乾淨，一有題材就容易噴。",
           "tightness_15d_pct", "過去 15 日平均日振幅 (高−低)/收盤 的百分比",
           "≤ 3%：整理期像睡著一樣，均線糾結在一起", "≥ 6%：整理期天天大漲大跌，籌碼沒沉澱",
           ["整理夠緊", "均線糾結", "Base 夠淺", "整理期太震盪", "Base 太深"]),
    Lesson("C05", "量價換手", 3, "量縮得夠不夠？高檔換手夠不夠？",
           "老手看整理期的量有沒有縮乾淨（沒人想賣了），"
           "以及這段時間的成交量是不是集中在箱體上緣（願意用高價接的人變多）。",
           "high_zone_turnover_share_20d", "過去 20 日成交量落在箱體上 1/3 價格帶的佔比",
           "整理期量縮至 0.6 以下，且高檔換手佔比高", "整理期量沒縮、或量都堆在箱底",
           ["量縮得漂亮", "高檔換手足", "突破帶量", "量沒縮", "量堆在低檔", "突破無量"]),
    Lesson("C02", "阻力消耗", 4, "前高被撞幾次？賣壓吃乾淨沒？",
           "同一個價位被測試越多次，套牢的人越有機會在那裡出場走掉。"
           "撞第一次通常被打回來，撞第三、四次時上面的賣單已經稀薄了。",
           "prior_high_tests_count", "過去 60 日內，最高價觸及 250 日前高 2% 以內的波峰次數",
           "≥ 3 次，且每次回檔越來越淺", "第一次碰前高就想追",
           ["前高測試夠多次", "回檔一次比一次淺", "貼著前高不掉", "第一次碰前高", "回檔越來越深"]),
    Lesson("C08", "趨勢年齡", 5, "這是第一段還是末升段？",
           "同樣是創新高，剛從底部起來的第一次，跟已經漲了半年的第五次，右尾空間差很多。",
           "major_breakout_count_250d", "過去 250 日內突破 60 日以上新高的次數",
           "≤ 1 次，距 120 日低點不遠", "≥ 4 次，已經漲了一大段",
           ["還在初升段", "距底部不遠", "漲幅仍溫和", "已經漲太久", "突破次數太多", "離底部太遠"]),
    Lesson("C06", "相對強弱", 6, "是大盤帶它漲，還是它自己走？",
           "大盤漲 2% 它漲 2%，那是 beta，不是本事。要看大盤不好時它扛不扛得住。",
           "rs_excess_taiex_pct", "個股當日報酬 − 加權指數當日報酬",
           "顯著為正，且大盤重挫時相對抗跌", "只是跟著大盤動",
           ["明顯強於大盤", "大盤弱它照漲", "只是跟著大盤", "弱於大盤"]),
    Lesson("C04", "結構防守", 7, "被打下去有沒有立刻爬回來？",
           "突破前若出現一次淺淺的破底、隔兩三天就收回來，通常是最後一次洗盤，浮額被洗掉了。",
           "shakeout_depth_pct", "跌破 30 日箱底後 3 日內重新收回的破底幅度",
           "有淺洗盤（1~5%）並快速收回", "破底後回不來、或一路走低",
           ["有洗盤並收回", "回檔守住前低", "破底沒收回", "低點一路走低"]),
    Lesson("C07", "推進效率", 8, "走得順不順？",
           "同樣漲 20%，一路墊高走上去，跟上沖下洗震盪半年才到，是完全不同的股票。",
           "price_progress_efficiency_20d", "考夫曼效率比：20 日淨位移 ÷ 20 日真實區間總和",
           "≥ 0.4，走勢順暢", "≤ 0.15，無效震盪很多",
           ["走勢順暢", "陽線實體大", "上沖下洗", "推進沒效率"]),
    Lesson("C11", "壓力修復", 9, "這檔是不是慣性騙人？",
           "看它過去被大黑棒打下去之後，幾天能站回來。修復快的股票，抗壓性強。",
           "bars_to_recover_largest_down_day", "整理期最大跌幅日之後，收盤收復該日最高點所需天數",
           "≤ 5 天收復", "始終沒收復（記為 −1）",
           ["長黑很快收復", "沒有假突破前科", "修復很慢", "常常假突破"]),
    Lesson("C09", "資訊反應", 10, "利空打不打得下去？利多是不是出貨？",
           "壞消息出來不跌，是真的有人在收；好消息開高走低，通常是在出。",
           "resilience_excess_ret", "負面事件日或大盤重挫日的個股超額報酬",
           "利空日仍相對抗跌", "利多日開高走低",
           ["利空不跌", "利多後續強", "利多開高走低", "利空直接崩"]),
    Lesson("C10", "事後認可", 11, "進場後前三天該不該砍？",
           "突破隔天市場認不認這個新價格。前三天就被打很深，多半不對。（出場課，進場時不可用）",
           "mfe_mae_ratio_3d", "進場後 3 日最大浮盈 ÷ 最大回撤",
           "次日站穩突破線之上", "隔天直接跌回突破線下方",
           ["次日站穩", "缺口沒被補", "隔天就跌破", "前三天先挨打"]),
    Lesson("C12", "動能老化", 12, "什麼時候該走？",
           "創高間隔越拉越長、效率越來越差，代表這一段在老化。（出場課，進場時不可用）",
           "efficiency_decay", "突破後前 5 日效率比 − 之後 10 日效率比",
           "創高間隔穩定、高點持續擴張", "創高間隔拉長、效率明顯衰退",
           ["動能仍在延續", "高點持續擴張", "創高間隔拉長", "效率明顯衰退"]),
]

LESSON_BY_ID = {l.cid: l for l in LESSONS}

# 每課用哪個欄位挑正/反例與出題，方向 True = 值越大越符合這課的「好」
LESSON_SELECTOR: dict[str, tuple[str, bool]] = {
    "C03": ("close_quality", True),
    "C01": ("tightness_15d_pct", False),
    "C05": ("high_zone_turnover_share_20d", True),
    "C02": ("prior_high_tests_count", True),
    "C08": ("major_breakout_count_250d", False),
    "C06": ("rs_excess_taiex_pct", True),
    "C04": ("shakeout_depth_pct", True),
    "C07": ("price_progress_efficiency_20d", True),
    "C11": ("bars_to_recover_largest_down_day", False),
    "C09": ("rs_excess_taiex_pct", True),        # 事件資料未接，暫以 RS 代理
    "C10": ("close_quality", True),              # 出場課，教材沿用進場圖
    "C12": ("price_progress_efficiency_20d", True),
}


def lesson_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"順序": l.order, "代號": l.cid, "課名": l.name,
         "學會後你能回答": l.question, "Primary": l.primary}
        for l in sorted(LESSONS, key=lambda x: x.order)
    ])


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
