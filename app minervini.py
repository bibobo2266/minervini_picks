"""
Minervini SEPA Scanner — Taiwan stocks (parquet edition)

資料來源改為 GitHub Actions 每日更新的 data/prices.parquet（全市場 ~2000 檔），
不再逐檔打 FinMind。掃描從 1-2 分鐘降到 2-3 秒，且不吃 API 額度。

三個分頁：
1. 選股 — 流動性母體 → Stage 2 → 趨勢模板 → VCP → 五色狀態分流
2. 持股監控 — 貼代號，回報階段 + 出場旗標（含爆量最大跌幅）
3. 進場計算 — 樞紐 / 停損 / 部位大小

FinMind token 只有在勾選「月營收」或「法人買賣超」時才需要，且只對最終名單抓。
"""
import datetime as dt
import io
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Minervini SEPA · 選股", page_icon="◈", layout="wide")

REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
PRICES_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/prices.parquet"
UNIVERSE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/universe.parquet"
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

@st.cache_data(ttl=60 * 60 * 4, show_spinner="讀取行情資料…")
def load_parquet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def build_matrices():
    """把長表轉成 date × stock_id 的寬矩陣，之後所有計算都向量化。"""
    df = load_parquet(PRICES_URL)
    df["date"] = pd.to_datetime(df["date"])
    m = {}
    for k, col in [("c", "close"), ("h", "max"), ("l", "min"),
                   ("v", "Trading_Volume"), ("mo", "Trading_money")]:
        m[k] = df.pivot(index="date", columns="stock_id", values=col).sort_index()
    return m, df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_universe() -> pd.DataFrame:
    return load_parquet(UNIVERSE_URL).set_index("stock_id")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
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


# ------------------------------------------------------------------ 主掃描

def scan(m, liq_wan: float, min_days: int = 250):
    """向量化計算全母體的趨勢模板、RS、量比。回傳 DataFrame。"""
    c, h, l, v, mo = m["c"], m["h"], m["l"], m["v"], m["mo"]
    keep = (mo.tail(60).mean() > liq_wan * 1e4) & (c.notna().sum() >= min_days)
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


# ------------------------------------------------------------------ 樣式

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {font-weight:700; letter-spacing:-.5px;}
.muted {color:#8a8f98;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("Minervini SEPA 選股器")
st.markdown('<span class="muted">全市場流動性母體 · 趨勢模板 + VCP 足跡 + 底部計數 · '
            '五色狀態分流</span>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("設定")
    token = st.text_input("FinMind Token（選填）", type="password",
                          value=st.secrets.get("FINMIND_TOKEN", ""),
                          help="只有勾選月營收或法人買賣超時才需要")
    st.caption("行情資料由 GitHub Actions 每日 15:30 更新，不吃 API 額度。")

tab1, tab2, tab3 = st.tabs(["① 選股", "② 持股監控", "③ 進場計算"])

# ================================================================== TAB 1
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        liq = st.number_input("60日均額門檻（萬元）", 500, 100000, DEFAULT_LIQ, 500,
                              key="t1_liq", help="流動性過濾，取代舊版的成交值前 N 名")
    with c2:
        min_tt = st.slider("最低 TT 分", 4, 8, 6, 1, key="t1_tt",
                           help="低於此分直接歸為淘汰，不顯示")
    with c3:
        want_fund = st.checkbox("加月營收 / 法人（需 token）", value=False, key="t1_fund",
                                help="只對非淘汰名單抓，每檔 2 次 API")

    if st.button("開始掃描", type="primary", key="t1_go"):
        m, _ = build_matrices()
        uni = load_universe()
        base = scan(m, liq)
        if base.empty:
            st.error("母體為空，檢查門檻或資料檔。"); st.stop()

        st.caption(f"母體 {len(base)} 檔 · 資料日 {m['c'].index[-1]:%Y-%m-%d}")
        cand = base[base["TT分"] >= min_tt].index

        rows, prog = [], st.progress(0.0, text="型態分析中…")
        for i, sid in enumerate(cand, 1):
            prog.progress(i / len(cand), text=f"型態分析中… {sid}")
            px = pd.DataFrame({k: m[kk][sid] for k, kk in
                               [("close", "c"), ("max", "h"), ("min", "l"),
                                ("Trading_Volume", "v")]}).dropna()
            if len(px) < 60:
                continue
            stg, ma30w, below = stage_of(px)
            f = vcp_foot(px)
            b = base.loc[sid]
            rows.append(dict(
                代號=sid, 名稱=uni.loc[sid, "stock_name"] if sid in uni.index else "?",
                產業=uni.loc[sid, "industry_category"] if sid in uni.index else "",
                收盤=b["收盤"], RS=b["RS"], TT分=int(b["TT分"]), 階段=stg or 0,
                距高點=b["距高點"], 量比=b["量比"], 量增=bool(b["量增"]),
                VCP=f["ok"], 足跡=f["foot"], 樞紐價=f["pivot"],
                近樞紐=f["near"], 突破=bool(b["收盤"] >= (f["pivot"] or 1e9)),
                底部序=base_count(px), 均額億=b["均額億"],
                新股=len(px) < 300,
            ))
        prog.empty()

        df = pd.DataFrame(rows)
        if df.empty:
            st.info("沒有標的通過門檻。降低 TT 分或均額門檻再試。"); st.stop()
        df["狀態"] = df.apply(classify, axis=1)
        df["晚期"] = df["底部序"] >= 4
        df = df[df["狀態"] != "淘汰"]

        order = {"觸發": 0, "準備": 1, "觀察": 2}
        df = df.assign(_o=df["狀態"].map(order)).sort_values(
            ["_o", "TT分", "RS"], ascending=[True, False, False]).drop(columns="_o")

        # 基本面（只對留下來的名單抓）
        if want_fund and token:
            st.caption(f"抓取 {len(df)} 檔基本面…")
            yoys, insts = [], []
            fprog = st.progress(0.0)
            since = (dt.date.today() - dt.timedelta(days=500)).isoformat()
            since30 = (dt.date.today() - dt.timedelta(days=30)).isoformat()
            for i, sid in enumerate(df["代號"], 1):
                fprog.progress(i / len(df))
                rv = finmind("TaiwanStockMonthRevenue", token, data_id=sid, start_date=since)
                y = None
                if not rv.empty and "revenue" in rv.columns:
                    rv["ym"] = rv["revenue_year"].astype(int) * 100 + rv["revenue_month"].astype(int)
                    s = rv.set_index("ym")["revenue"].astype(float).sort_index()
                    last = s.index[-1]
                    if last - 100 in s.index and s[last - 100] > 0:
                        y = round((s[last] / s[last - 100] - 1) * 100, 1)
                yoys.append(y)
                iv = finmind("TaiwanStockInstitutionalInvestors", token,
                             data_id=sid, start_date=since30)
                r = None
                if not iv.empty and "buy" in iv.columns:
                    iv["net"] = iv["buy"].astype(float) - iv["sell"].astype(float)
                    d = iv.groupby("date")["net"].sum().sort_index() / 1000
                    if len(d) >= 6:
                        r = bool(d.iloc[-1] > d.iloc[-6:-1].mean())
                insts.append(r)
            fprog.empty()
            df["營收YoY%"] = yoys
            df["法人轉強"] = insts

        # 狀態統計
        cnt = df["狀態"].value_counts()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("🟢 觸發", int(cnt.get("觸發", 0)))
        s2.metric("🟠 準備", int(cnt.get("準備", 0)))
        s3.metric("🟡 觀察", int(cnt.get("觀察", 0)))
        s4.metric("⚠ 第4底以上", int(df["晚期"].sum()))

        colcfg = {
            "RS": st.column_config.NumberColumn("RS", format="%d", help="全母體百分位"),
            "TT分": st.column_config.NumberColumn("TT分", format="%d", help="/8"),
            "距高點": st.column_config.NumberColumn("距高%", format="%.1f",
                                                  help="距 52 週高點，越接近 0 越好"),
            "量比": st.column_config.NumberColumn("V5/V10", format="%.2f"),
            "量增": st.column_config.CheckboxColumn("量增"),
            "VCP": st.column_config.CheckboxColumn("VCP", help="收縮遞減且末段 ≤ 首段 60%"),
            "足跡": st.column_config.TextColumn("VCP足跡", help="週數-首次/末次跌幅-收縮次數"),
            "樞紐價": st.column_config.NumberColumn("樞紐", format="%.2f"),
            "近樞紐": st.column_config.CheckboxColumn("近樞紐", help="距樞紐 ≤3%"),
            "突破": st.column_config.CheckboxColumn("突破"),
            "底部序": st.column_config.NumberColumn("底部#", format="%d",
                                                 help="1-2 最佳，3 尚可，4 以上偏晚"),
            "晚期": st.column_config.CheckboxColumn("⚠晚期", help="第 4 底以上"),
            "均額億": st.column_config.NumberColumn("均額(億)", format="%.2f"),
            "新股": st.column_config.CheckboxColumn("新股", help="上市未滿約 15 個月"),
        }
        if want_fund and token:
            colcfg["營收YoY%"] = st.column_config.NumberColumn("營收YoY%", format="%.1f")
            colcfg["法人轉強"] = st.column_config.CheckboxColumn("法人轉強",
                                                             help="當日三大法人淨買 > 近5日均")

        show = ["狀態", "代號", "名稱", "產業", "收盤", "RS", "TT分", "距高點",
                "量比", "量增", "VCP", "足跡", "樞紐價", "近樞紐", "突破",
                "底部序", "晚期", "均額億", "新股"]
        show += [c for c in ("營收YoY%", "法人轉強") if c in df.columns]
        st.dataframe(df[show], hide_index=True, use_container_width=True,
                     column_config=colcfg, height=560)

        with st.expander("產業分佈（多頭初期領導群通常集中在少數產業）"):
            st.dataframe(df.groupby("產業").size().sort_values(ascending=False)
                         .rename("檔數").reset_index(), hide_index=True)

        st.text_area("複製代號清單（貼進扣抵值 app）",
                     " ".join(df["代號"].tolist()), height=68, key="t1_copy")
        st.download_button("下載 CSV", df[show].to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"minervini_{dt.date.today().isoformat()}.csv",
                           mime="text/csv")
        st.caption("狀態定義：🟢突破樞紐+量增 · 🟠VCP完成且近樞紐 · 🟡條件夠但型態未成。"
                   "僅供研究，非投資建議。")

# ================================================================== TAB 2
with tab2:
    st.caption("貼上持股代號，回報階段與出場旗標。含書裡 CROX 案例的爆量最大跌勢警訊。")
    holdings = st.text_area("持股代號（逗號、空白或換行分隔）",
                            placeholder="2330 2454 6488", height=80, key="t2_in")
    drop_pct = st.slider("回檔警示門檻 %", 5, 30, 15, 1, key="t2_drop")

    if st.button("檢查持股", type="primary", key="t2_go") and holdings.strip():
        m, _ = build_matrices()
        uni = load_universe()
        ids = [s for s in re.split(r"[,\s]+", holdings.strip()) if s]
        rows = []
        for sid in ids:
            if sid not in m["c"].columns:
                rows.append(dict(代號=sid, 名稱="?", 階段="無資料", 現價=None,
                                 距高點=None, 底部序=None, 旗標="?")); continue
            px = pd.DataFrame({k: m[kk][sid] for k, kk in
                               [("close", "c"), ("max", "h"), ("min", "l"),
                                ("Trading_Volume", "v")]}).dropna()
            stg, ma30w, below = stage_of(px)
            price = float(px["close"].iloc[-1])
            hi = float(px["close"].tail(120).max())
            pull = (price / hi - 1) * 100
            worst, today, volx, alarm = worst_drop(px)
            bc = base_count(px)

            flags = []
            if alarm:
                flags.append("🔥爆量最大跌勢")
            if stg == 4:
                flags.append("⛔出場")
            elif stg == 3:
                flags.append("⚠減碼")
            if below:
                flags.append("跌破30週")
            if pull <= -drop_pct:
                flags.append(f"回檔{pull:.0f}%")
            if bc >= 4:
                flags.append(f"晚期底部#{bc}")

            rows.append(dict(
                代號=sid,
                名稱=uni.loc[sid, "stock_name"] if sid in uni.index else "?",
                階段={2: "上升", 1: "打底", 3: "頭部", 4: "下跌"}.get(stg, "?"),
                現價=round(price, 2), 距高點=round(pull, 1),
                今日=today, 量倍=volx, 最大跌=worst, 底部序=bc,
                旗標=" ".join(flags) if flags else "✓持有"))

        mon = pd.DataFrame(rows)
        st.dataframe(mon, hide_index=True, use_container_width=True, column_config={
            "距高點": st.column_config.NumberColumn("距高%", format="%.1f"),
            "今日": st.column_config.NumberColumn("今日%", format="%.1f"),
            "量倍": st.column_config.NumberColumn("量倍", format="%.2f",
                                                help="今日量 / 近50日均量"),
            "最大跌": st.column_config.NumberColumn("最大跌%", format="%.1f",
                                                 help="近一年單日最大跌幅"),
            "底部序": st.column_config.NumberColumn("底部#", format="%d"),
        })
        st.download_button("下載 CSV", mon.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"holdings_{dt.date.today().isoformat()}.csv",
                           mime="text/csv")
        st.caption("🔥爆量最大跌勢＝第2階段以來單日最大跌幅且量放大 1.8 倍以上。"
                   "書裡明講這通常是賣訊，即使盈餘仍然很好。")

# ================================================================== TAB 3
with tab3:
    st.caption("輸入樞紐/買價與帳戶規模，算停損位與部位大小。純風險數學，不需資料。")
    a, b = st.columns(2)
    with a:
        acct = st.number_input("帳戶總額 (TWD)", 0, value=1_000_000, step=50_000, key="t3_acct")
        buy = st.number_input("預計買價 (樞紐突破價)", 0.0, value=100.0, step=0.5, key="t3_buy")
    with b:
        risk_pct = st.number_input("每筆帳戶風險 %", 0.1, 5.0, DEFAULT_RISK_PCT, 0.05, key="t3_risk")
        stop_pct = st.slider("停損 % (樞紐下方)", 3, 12, 8, 1, key="t3_stop")

    if buy > 0 and acct > 0:
        stop_price = buy * (1 - stop_pct / 100)
        rps = buy - stop_price
        risk_amount = acct * risk_pct / 100
        shares = int(risk_amount // rps) if rps > 0 else 0
        position = shares * buy
        pos_pct = position / acct * 100 if acct else 0

        st.markdown("### 結果")
        m1, m2, m3 = st.columns(3)
        m1.metric("停損價", f"{stop_price:,.2f}", f"-{stop_pct}%")
        m2.metric("每股風險", f"{rps:,.2f}")
        m3.metric("可承受虧損", f"{risk_amount:,.0f}", f"{risk_pct}% 帳戶")
        m4, m5, m6 = st.columns(3)
        m4.metric("建議股數", f"{shares:,}", f"{shares/1000:.1f} 張")
        m5.metric("部位金額", f"{position:,.0f}")
        m6.metric("佔帳戶", f"{pos_pct:.1f}%")
        if pos_pct > 100:
            st.warning("部位超過帳戶總額（需融資）。調高停損% 或降低風險%。")
