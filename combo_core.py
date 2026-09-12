"""
combo_core.py — 老手 60 組 Combo 引擎核心

只用日 OHLCV + 大盤指數 + (選配) event log。
所有門檻與定義來自 combo_registry.json，本檔不內建魔術數字。

紅線（照手冊）：
  1. 不計分、不投票、不推勝率。
  2. 缺值 = PARTIAL，不當 0、不當偏空、不算「幾項通過」。
  3. T 日判定只用截至 T 收盤已完成的資料；C10/C12 需要未來窗，未完成即 UNAVAILABLE。
"""
from __future__ import annotations

import ast
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- registry

def load_registry(path: str = "combo_registry.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _T:
    """thresholds 命名空間，讓規則字串可以寫 T.cq_high"""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


# ---------------------------------------------------------------- 資料載入

_COLMAP = {
    "max": "high", "min": "low", "High": "high", "Low": "low",
    "Open": "open", "Close": "close",
    "Trading_Volume": "vol", "Trading_money": "amount",
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _COLMAP.items() if k in df.columns}).copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df.sort_values("date").reset_index(drop=True)


def load_prices(adj_dir: str, stock_ids: list[str], years: list[int]) -> pd.DataFrame:
    """一次只載入需要的年份與代號，控制在 3GB 記憶體內。"""
    out = []
    for y in years:
        p = os.path.join(adj_dir, f"prices_adj_{y}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d = d[d["stock_id"].astype(str).isin([str(s) for s in stock_ids])]
        d = d[d["stock_id"].astype(str).str.len() == 4]   # 濾掉權證／六碼
        if len(d):
            out.append(d)
        del d
    if not out:
        return pd.DataFrame()
    return normalize_ohlcv(pd.concat(out, ignore_index=True))


def load_index(path: str) -> pd.DataFrame:
    d = normalize_ohlcv(pd.read_parquet(path))
    return d[["date", "close"]].rename(columns={"close": "idx_close"})


def load_events(path: str | None) -> pd.DataFrame:
    """append-only event log。缺檔就回空表，F/I 類 combo 自動 UNAVAILABLE。"""
    cols = ["event_id", "stock_id", "pub_date", "pub_time", "category",
            "polarity", "source_url", "excerpt", "verification_state"]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(path, dtype=str).fillna("")
    for c in cols:
        if c not in d.columns:
            d[c] = ""
    d["pub_date"] = pd.to_datetime(d["pub_date"], format="mixed", errors="coerce")
    return d


# ---------------------------------------------------------------- 結構工具

def _swing_lows(low: np.ndarray, w: int) -> np.ndarray:
    """局部低點旗標：前後各 w 根都不低於它。只用已完成資料，最後 w 根不判定。"""
    n = len(low)
    flag = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        seg = low[i - w:i + w + 1]
        if low[i] == seg.min() and not np.isnan(low[i]):
            flag[i] = True
    return flag


def _swing_highs(high: np.ndarray, w: int) -> np.ndarray:
    n = len(high)
    flag = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        seg = high[i - w:i + w + 1]
        if high[i] == seg.max() and not np.isnan(high[i]):
            flag[i] = True
    return flag


def _dedup_events(mask: np.ndarray, gap: int) -> np.ndarray:
    """事件去重：gap 根內只算一次。"""
    out = np.zeros(len(mask), dtype=bool)
    last = -10 ** 9
    for i, m in enumerate(mask):
        if m and i - last >= gap:
            out[i] = True
            last = i
    return out


# ---------------------------------------------------------------- 特徵計算

def compute_features(px: pd.DataFrame, idx: pd.DataFrame, reg: dict) -> pd.DataFrame:
    """回傳與 px 同長度的特徵表。NaN = UNAVAILABLE。"""
    D = reg["defs"]
    d = px.copy()
    d = d.merge(idx, on="date", how="left")

    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    v = d["vol"].astype(float)
    n = len(d)

    # --- 均線與波動
    d["ma20"] = c.rolling(20).mean()
    d["ma60"] = c.rolling(60).mean()
    d["ma120"] = c.rolling(120).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(D["atr_window"]).mean()
    d["low20"] = l.shift(1).rolling(20).min()

    # --- 突破線 B / 箱底 L（B 不含 T，避免自我比較）
    lb = D["breakout_lookback"]
    d["B"] = c.shift(1).rolling(lb).max()
    d["L"] = l.shift(1).rolling(D["box_window"]).min()
    d["above_B"] = c > d["B"]
    d["below_L"] = c < d["L"]

    # --- C01 壓縮 / base 深度
    tw = D["tight_window"]
    d["c01_tight"] = (h.rolling(tw).max() - l.rolling(tw).min()) / c * 100
    bw = D["base_depth_window"]
    top = h.rolling(bw).max()
    d["c01_base_depth"] = (top - l.rolling(bw).min()) / top * 100

    # --- C02 前高測試次數（2% 觸及帶 + 波峰去重）
    band = 1 - D["touch_band_pct"] / 100
    touch = (h >= d["B"] * band).fillna(False).to_numpy()
    touch = _dedup_events(touch, D["peak_dedup_gap"])
    d["c02_tests"] = pd.Series(touch, index=d.index).rolling(D["box_window"]).sum()

    # --- C03 收盤品質（高=低時分母為零 → NaN，不強填 1）
    rng = (h - l)
    d["c03_cq"] = np.where(rng > 0, (c - l) / rng.replace(0, np.nan), np.nan)

    # --- C05 量價（高檔換手 DISABLED）
    vma = v.rolling(D["vol_ma_window"]).mean().shift(1)
    d["c05_volratio"] = np.where(vma > 0, v / vma, np.nan)
    d["c05_contract"] = v.rolling(D["vol_contract_window"]).mean() / v.rolling(D["vol_ma_window"]).mean()
    d["c05_high_zone"] = np.nan  # 永遠 UNAVAILABLE

    # --- C06 相對強弱（同日、百分點）
    d["ret"] = c.pct_change(fill_method=None) * 100
    d["idx_ret"] = d["idx_close"].pct_change(fill_method=None) * 100
    d["c06_rs"] = d["ret"] - d["idx_ret"]
    r20 = (c / c.shift(20) - 1) * 100
    i20 = (d["idx_close"] / d["idx_close"].shift(20) - 1) * 100
    d["c06_rs20"] = r20 - i20
    d["c06_rs_ind"] = np.nan   # 需要產業基準，未提供 → PARTIAL
    d["ind_ret"] = np.nan

    # --- C07 標準 Kaufman ER + 帶號淨位移
    ew = D["er_window"]
    net = c - c.shift(ew)
    path = c.diff().abs().rolling(ew).sum()
    d["c07_er"] = np.where(path > 0, net.abs() / path, np.nan)
    d["c07_net"] = net

    # --- C08 趨勢年齡
    bo_raw = (c > d["B"]).fillna(False).to_numpy()
    bo = _dedup_events(bo_raw, D["breakout_dedup_gap"])
    d["_bo"] = bo
    d["c08_bo_count"] = pd.Series(bo, index=d.index).rolling(lb).sum()
    lowest = l.rolling(lb).min()
    d["c08_gain_from_low"] = (c / lowest - 1) * 100
    idx_of_low = l.rolling(lb).apply(lambda s: len(s) - 1 - int(np.argmin(s)), raw=True)
    d["c08_days_from_low"] = idx_of_low

    # --- C11 壓力修復（最大單日跌幅日的高點，是否已收復）
    sw = D["recover_scan_window"]
    chg = c.pct_change(fill_method=None)
    bars_rec, days_el = np.full(n, np.nan), np.full(n, np.nan)
    hi_arr, c_arr, chg_arr = h.to_numpy(), c.to_numpy(), chg.to_numpy()
    for i in range(sw, n):
        seg = chg_arr[i - sw + 1:i + 1]
        if np.all(np.isnan(seg)):
            continue
        j = i - sw + 1 + int(np.nanargmin(seg))
        target = hi_arr[j]
        days_el[i] = i - j
        rec = -1.0
        for k in range(j + 1, i + 1):
            if c_arr[k] > target:
                rec = k - j
                break
        bars_rec[i] = rec
    d["c11_bars_recover"] = bars_rec
    d["c11_days_elapsed"] = days_el

    # --- C12 動能老化
    run_max = c.rolling(lb, min_periods=20).max()
    is_high = (c >= run_max).to_numpy()
    since = np.full(n, np.nan)
    last_h = None
    for i in range(n):
        if is_high[i]:
            last_h = i
        if last_h is not None:
            since[i] = i - last_h
    d["c12_days_since_high"] = since

    er5 = np.where(c.diff().abs().rolling(5).sum() > 0,
                   (c - c.shift(5)).abs() / c.diff().abs().rolling(5).sum(), np.nan)
    er10 = np.where(c.diff().abs().rolling(10).sum() > 0,
                    (c - c.shift(10)).abs() / c.diff().abs().rolling(10).sum(), np.nan)
    d["c12_decay"] = pd.Series(er5, index=d.index).shift(10) - pd.Series(er10, index=d.index)

    # --- C10 事後三日窗（T 時禁用；只在歷史回看時才有值）
    fwd_low = l.shift(-1).rolling(3).min().shift(-2)
    fwd_high = h.shift(-1).rolling(3).max().shift(-2)
    d["c10_mae3"] = (c - fwd_low) / c * 100
    d["c10_mfe3"] = (fwd_high - c) / c * 100

    # --- 結構旗標
    sl = _swing_lows(l.to_numpy(), D["swing_window"])
    sh = _swing_highs(h.to_numpy(), D["swing_window"])
    lows_idx = np.where(sl)[0]
    highs_idx = np.where(sh)[0]
    hl, ll, hh = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    rsl = np.full(n, np.nan)
    psl = np.full(n, np.nan)
    for i in range(n):
        li = lows_idx[lows_idx <= i]
        if len(li) >= 1:
            rsl[i] = l.iloc[li[-1]]
        if len(li) >= 2:
            psl[i] = l.iloc[li[-2]]
        if len(li) >= 3:
            a, b, cc = l.iloc[li[-3]], l.iloc[li[-2]], l.iloc[li[-1]]
            hl[i] = cc > b > a
            ll[i] = cc < b < a
        hi = highs_idx[highs_idx <= i]
        if len(hi) >= 2:
            hh[i] = h.iloc[hi[-1]] > h.iloc[hi[-2]]
    d["higher_lows"], d["lower_lows"], d["higher_highs"] = hl, ll, hh
    d["recent_swing_low"], d["prior_swing_low"] = rsl, psl

    # --- C04 洗盤（近 shakeout_window 內是否曾收盤破 L，破多深，是否已收回）
    swin, rdays = D["shakeout_window"], D["shakeout_recover_days"]
    depth, recov, broken = np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool)
    Larr, carr = d["L"].to_numpy(), c.to_numpy()
    for i in range(swin, n):
        seg = slice(i - swin + 1, i + 1)
        br = np.where(carr[seg] < Larr[seg])[0]
        if len(br) == 0:
            continue
        j = i - swin + 1 + br[0]
        base = Larr[j]
        if not np.isfinite(base) or base <= 0:
            continue
        trough = np.nanmin(carr[j:i + 1])
        depth[i] = (base - trough) / base * 100
        broken[i] = carr[i] < Larr[i]
        end = min(j + rdays, n - 1)
        recov[i] = bool(np.any(carr[j:end + 1] >= base)) and not broken[i]
    d["shake_depth"], d["shake_recovered"], d["shake_broken"] = depth, recov, broken

    # --- G 類輔助
    had_bo = pd.Series(bo, index=d.index).rolling(20).sum() > 0
    d["had_breakout_recent"] = had_bo.fillna(False)
    d["had_failed_breakout"] = (had_bo & (~d["above_B"].shift(1).astype("boolean").fillna(False).astype(bool))).astype(bool)
    gap_up = (o > h.shift(1))
    d["big_gap"] = ((o - c.shift(1)).abs() > d["atr"] * D["gap_invalidate_atr"]).fillna(False)
    d["gap_filled"] = (gap_up.shift(1).astype("boolean").fillna(False).astype(bool)) & (l <= h.shift(2))

    # --- J 類資料品質旗標
    d["has_c03_c05"] = d["c03_cq"].notna() & pd.Series(d["c05_volratio"]).notna()
    d["lacks_background"] = d["c01_tight"].isna() | d["c02_tests"].isna() | d["c08_bo_count"].isna()
    d["same_day_evidence_only"] = d["lacks_background"] & d["has_c03_c05"]
    d["definition_dispute"] = False           # 由 registry 版本比對觸發，預設 False
    d["corporate_action_nearby"] = False      # 需除權息表；未接 → False
    d["future_data_present"] = False          # 由 runner 檢查，預設 False

    # 事件欄位預設 UNAVAILABLE
    for k in ["ev_neg", "ev_pos", "ev_after_close", "ev_contract_binding", "ev_mou",
              "ev_guidance", "ev_target_met", "ev_cashflow_flag", "ev_governance"]:
        d[k] = False
    d["ev_ret"] = np.nan
    d["c09_excess"] = np.nan

    return d


def attach_events(feat: pd.DataFrame, ev: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """把 event log 掛到特徵表。無事件 → 相關 combo 自動不成立（不是偏空）。"""
    if ev.empty:
        return feat
    e = ev[ev["stock_id"].astype(str) == str(stock_id)]
    if e.empty:
        return feat
    catmap = {
        "contract": "ev_contract_binding", "mou": "ev_mou", "guidance": "ev_guidance",
        "target": "ev_target_met", "cashflow": "ev_cashflow_flag", "governance": "ev_governance",
    }
    for _, r in e.iterrows():
        m = feat["date"] == r["pub_date"]
        if not m.any():
            continue
        pol = str(r.get("polarity", "")).lower()
        if pol.startswith("neg"):
            feat.loc[m, "ev_neg"] = True
        elif pol.startswith("pos"):
            feat.loc[m, "ev_pos"] = True
        t = str(r.get("pub_time", ""))
        if t and t >= "13:30":
            feat.loc[m, "ev_after_close"] = True
        cat = catmap.get(str(r.get("category", "")).lower())
        if cat:
            feat.loc[m, cat] = True
        feat.loc[m, "ev_ret"] = feat.loc[m, "ret"]
        feat.loc[m, "c09_excess"] = feat.loc[m, "c06_rs"]
    return feat


# ---------------------------------------------------------------- 規則評估

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Name,
    ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.USub, ast.Attribute,
    ast.Call, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
)


def _names(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"規則含不允許的語法：{expr}")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def eval_combo(combo: dict, ns: dict, T: _T) -> dict:
    """回傳 {status, conditions:[{expr, value, ok}]}。缺值 → PARTIAL，不算不符。"""
    env = dict(ns)
    env["T"] = T
    env["abs"] = abs
    conds, missing, failed = [], False, False
    for expr in combo["when"]:
        try:
            used = _names(expr) - {"T", "abs"}
            vals = {u: env.get(u, None) for u in used}
            if any(u not in env for u in used) or any(_is_missing(v) for v in vals.values()):
                conds.append({"expr": expr, "vals": vals, "ok": None})
                missing = True
                continue
            ok = bool(eval(compile(ast.parse(expr, mode="eval"), "<rule>", "eval"),
                           {"__builtins__": {}}, env))
            conds.append({"expr": expr, "vals": vals, "ok": ok})
            if not ok:
                failed = True
        except Exception as e:                       # noqa: BLE001
            conds.append({"expr": expr, "vals": {}, "ok": None, "err": str(e)})
            missing = True
    if failed:
        status = "NO_MATCH"
    elif missing:
        status = "PARTIAL"
    else:
        status = "MATCH"
    return {"status": status, "conditions": conds}


def evaluate_all(row: pd.Series, reg: dict) -> list[dict]:
    T = _T(reg["thresholds"])
    ns = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    out = []
    for combo in reg["combos"]:
        r = eval_combo(combo, ns, T)
        r.update({k: combo.get(k) for k in ("id", "cls", "name", "bias", "next", "fail", "gate")})
        out.append(r)
    order = {c: i for i, c in enumerate(reg["priority"])}
    out.sort(key=lambda x: (order.get(x["cls"], 99), x["id"]))
    return out


# ---------------------------------------------------------------- 明日分支

def branches(row: pd.Series, reg: dict) -> dict:
    """明日三分支的界線。up 取「收盤之上最近的一條線」，down 取「收盤之下最近的一條線」。"""
    D, close, atr = reg["defs"], float(row["close"]), float(row.get("atr", np.nan))
    B, L = row.get("B", np.nan), row.get("L", np.nan)
    rsl, ma20 = row.get("recent_swing_low", np.nan), row.get("ma20", np.nan)
    step = atr * 0.5 if np.isfinite(atr) else close * 0.01

    lo20 = row.get("low20", np.nan)
    ups = sorted(float(x) for x in [B, row.get("high", np.nan)] if np.isfinite(x) and x > close)
    dns = sorted((float(x) for x in [rsl, ma20, L, lo20, close - (atr if np.isfinite(atr) else 0)]
                  if np.isfinite(x) and x < close), reverse=True)
    up = ups[0] if ups else close + step
    down = dns[0] if dns else close - step

    nr = atr * D["normal_range_atr"] if np.isfinite(atr) else np.nan
    gi = atr * D["gap_invalidate_atr"] if np.isfinite(atr) else np.nan
    f = lambda x: None if not np.isfinite(x) else round(float(x), 2)  # noqa: E731
    return {
        "up_line": round(up, 2), "down_line": round(down, 2),
        "B": f(B), "L": f(L), "ma20": f(ma20), "atr": f(atr),
        "normal_lo": f(close - nr), "normal_hi": f(close + nr), "gap_invalidate": f(gi),
        "up_src": "B 突破線" if (np.isfinite(B) and abs(up - B) < 1e-9) else "前一日高" if ups else "收盤+0.5ATR",
        "down_src": ("回檔低點" if (np.isfinite(rsl) and abs(down - rsl) < 1e-9)
                     else "MA20" if (np.isfinite(ma20) and abs(down - ma20) < 1e-9)
                     else "20日低" if (np.isfinite(lo20) and abs(down - lo20) < 1e-9)
                     else "箱底 L" if (np.isfinite(L) and abs(down - L) < 1e-9)
                     else "收盤-1ATR"),
    }


def actions(br: dict, reg: dict, state: str = "空手") -> list[str]:
    """封閉動作清單，每個動作一定帶價位。"""
    up, dn = br["up_line"], br["down_line"]
    ma20 = br["ma20"] if br["ma20"] is not None else dn
    if state == "空手":
        return [f"甲 收>{up}：掛突破買 @{round(up * 1.001, 2)}",
                f"乙 {dn}~{up}：不動觀察",
                f"丙 收<{dn}：移出名單"]
    if state == "持有_獲利":
        return [f"甲 收>{up}：續抱，停損上移至 @{dn}",
                f"乙 {dn}~{up}：續抱，停損維持 @{ma20}",
                f"丙 收<{dn}：出場或減半 @{dn}"]
    if state == "持有_虧損":
        return [f"甲 收>{up}：續抱，停損 @{dn}",
                f"乙 {dn}~{up}：減碼，停損 @{dn}",
                f"丙 收<{dn}：出場 @{dn}"]
    return [f"甲 收>{up}：依原計畫執行 @{dn}",
            f"乙 {dn}~{up}：依原計畫執行 @{dn}",
            f"丙 收<{dn}：出場 @{dn}"]


# ---------------------------------------------------------------- 標註圖

def render_chart(feat: pd.DataFrame, stock_id: str, name: str, out_path: str,
                 reg: dict, bars: int = 280, matched: list[dict] | None = None) -> str:
    """沿用 lesson_core 的標註風格：K 棒 + B/L/MA20/回檔低點 + 明日三分支帶。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findSystemFonts

    fp = None
    for f in findSystemFonts():
        if any(k in os.path.basename(f) for k in ("NotoSansCJK", "NotoSansTC", "wqy", "PingFang")):
            fp = FontProperties(fname=f)
            break

    d = feat.tail(bars).reset_index(drop=True)
    x = np.arange(len(d))
    row = feat.iloc[-1]
    br = branches(row, reg)

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.05})

    for i in range(len(d)):
        o, h, l, c = d.loc[i, ["open", "high", "low", "close"]]
        # 台股慣例：紅漲黑跌（不是美股的綠漲紅跌）
        col = "#d32f2f" if c >= o else "#1c1c1c"
        ax.vlines(x[i], l, h, color=col, linewidth=0.6)
        ax.add_patch(plt.Rectangle((x[i] - 0.30, min(o, c)), 0.60,
                                   max(abs(c - o), 1e-9), facecolor=col, edgecolor=col))

    ax.plot(x, d["ma20"], color="#c78b2f", lw=1.0, label="MA20")
    ax.plot(x, d["ma60"], color="#7a7a7a", lw=1.0, label="MA60")

    def hline(val, color, label, ls="-"):
        if val is None or not np.isfinite(val):
            return
        ax.axhline(val, color=color, lw=1.2, ls=ls, alpha=0.9)
        ax.text(len(d) - 0.5, val, f" {label} {val}", color=color, va="center",
                fontsize=9, fontproperties=fp)

    hline(br["B"], "#1f5fbf", "B 突破線")
    hline(br["L"], "#8b5cf6", "L 箱底")
    hline(br["down_line"], "#b34700", "回檔低點", ls="--")

    rgt = len(d) - 1
    if br["normal_hi"] is not None:
        ax.axhspan(br["up_line"], max(br["normal_hi"], br["up_line"] * 1.005),
                   xmin=0.96, color="#d32f2f", alpha=0.20)
        ax.axhspan(br["down_line"], br["up_line"], xmin=0.96, color="#c78b2f", alpha=0.15)
        ax.axhspan(min(br["normal_lo"], br["down_line"] * 0.995), br["down_line"],
                   xmin=0.96, color="#1c1c1c", alpha=0.16)
        ax.text(rgt + 0.6, br["up_line"], "甲", fontsize=11, color="#d32f2f", fontproperties=fp)
        ax.text(rgt + 0.6, (br["up_line"] + br["down_line"]) / 2, "乙",
                fontsize=11, color="#c78b2f", fontproperties=fp)
        ax.text(rgt + 0.6, br["down_line"], "丙", fontsize=11, color="#1c1c1c", fontproperties=fp)

    axv.bar(x, d["vol"], color="#9aa0a6", width=0.60)
    axv.plot(x, d["vol"].rolling(20).mean(), color="#c78b2f", lw=1.0)

    ticks = list(range(0, len(d), max(1, len(d) // 8)))
    axv.set_xticks(ticks)
    axv.set_xticklabels([d.loc[i, "date"].strftime("%m/%d") for i in ticks], fontsize=8)

    title = f"{stock_id} {name}　{d.loc[len(d)-1,'date'].strftime('%Y-%m-%d')} 收盤"
    if matched:
        title += "　主 combo " + "／".join(m["id"] for m in matched[:3])
    ax.set_title(title, fontproperties=fp, fontsize=12)
    ax.legend(prop=fp, fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)
    axv.grid(alpha=0.15)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- 向量化評估

_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
        ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}


def _vec(node, env):
    if isinstance(node, ast.Expression):
        return _vec(node.body, env)
    if isinstance(node, ast.BoolOp):
        vals = [_vec(v, env) for v in node.values]
        op = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        r = vals[0]
        for v in vals[1:]:
            r = op(r, v)
        return r
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return np.logical_not(_vec(node.operand, env))
        return -_vec(node.operand, env)
    if isinstance(node, ast.Compare):
        left, res = _vec(node.left, env), None
        for op, cmp in zip(node.ops, node.comparators):
            right = _vec(cmp, env)
            with np.errstate(invalid="ignore"):
                m = _CMP[type(op)](left, right)
            res = m if res is None else np.logical_and(res, m)
            left = right
        return res
    if isinstance(node, ast.BinOp):
        with np.errstate(invalid="ignore", divide="ignore"):
            return _BIN[type(node.op)](_vec(node.left, env), _vec(node.right, env))
    if isinstance(node, ast.Call):
        return np.abs(_vec(node.args[0], env))
    if isinstance(node, ast.Attribute):
        return getattr(env["T"], node.attr)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"不支援的語法節點：{type(node)}")


def evaluate_vectorized(feat: pd.DataFrame, reg: dict) -> dict[str, dict[str, np.ndarray]]:
    """整表一次算完。回傳 {combo_id: {'match': bool array, 'partial': bool array}}。

    語意與 eval_combo 相同：缺值 → PARTIAL，不算不符。NaN 比較一律 False。
    """
    T = _T(reg["thresholds"])
    env = {"T": T}
    valid = {}
    for col in feat.columns:
        a = feat[col].to_numpy()
        env[col] = a
        valid[col] = np.ones(len(a), bool) if a.dtype == bool else np.asarray(pd.notna(a))

    out = {}
    n = len(feat)
    for combo in reg["combos"]:
        ok_all = np.ones(n, bool)
        any_missing = np.zeros(n, bool)
        broken = False
        for expr in combo["when"]:
            try:
                tree = ast.parse(expr, mode="eval")
                used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)} - {"T", "abs"}
                if any(u not in env for u in used):
                    broken = True
                    break
                vmask = np.ones(n, bool)
                for u in used:
                    vmask &= valid[u]
                r = _vec(tree, env)
                r = np.asarray(r, dtype=bool) if np.ndim(r) else np.full(n, bool(r))
                ok_all &= (r | ~vmask)
                any_missing |= ~vmask
            except Exception:                                  # noqa: BLE001
                broken = True
                break
        if broken:
            out[combo["id"]] = {"match": np.zeros(n, bool), "partial": np.ones(n, bool)}
            continue
        out[combo["id"]] = {"match": ok_all & ~any_missing, "partial": ok_all & any_missing}
    return out
