"""
Minervini SEPA Scanner — Taiwan stocks
Separate app from Stan_stages. Three tabs:
  1. 選股  — Stage 2 universe + Trend Template + VCP + revenue YoY (buy candidates)
  2. 持股監控 — paste your tickers, get stage + 3/4 exit flags (sell side)
  3. 進場計算 — pivot / 7-8% stop / position size (risk math)

Setup:
  1. Free token at https://finmindtrade.com  (Login > API Token)
  2. Streamlit secret FINMIND_TOKEN (or paste in sidebar)
  3. requirements.txt -> streamlit\nFinMind\npandas\nnumpy
Note: FinMind free tier = 600 req/hour. Caching keeps re-runs cheap.
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

st.set_page_config(page_title="Minervini SEPA · 選股", page_icon="◈", layout="wide")

MA_WEEKS = 30
VOL_MULT = 1.5
LOOKBACK_DAYS = 430          # ~260 trading bars: 200MA slope + 52w range need this
DEFAULT_RISK_PCT = 1.25      # account risk per trade (Minervini standard)

STAGE_META = {
    2: ("上升 Advancing", "買 / 觀察", "#1f9d55"),
    1: ("打底 Basing",    "觀望",     "#8a8f98"),
    3: ("頭部 Topping",   "減碼 / 收緊停損", "#d9a441"),
    4: ("下跌 Declining", "出場",     "#d9534f"),
}

BIGCAP = [
    "2330","2317","2454","2308","2382","2891","2412","2303","2881","3711",
    "2882","2886","1216","2884","2357","3034","2892","2885","2890","3231",
    "2345","2379","2603","3037","2880","5880","2887","2883","1303","2002",
    "1301","2327","3008","2395","3045","4938","2409","2301","2408","6505",
    "5871","2207","1101","2618","2610","2615","9910","2801","2823","2474",
    "6669","3661","3017","2376","2356","2360","3702","2385","6415","3005",
    "2377","4904","2337","6446","1476","2049","1590","9945","2542","2455",
    "2353","3443","2451","8046","2324","6488","3533","5269","2368","3653",
    "2347","2344","2371","3529","2492","2312","1102","2404","1802","9917",
    "2809","2812","6412","3406","2439","2201","1326","2105","2633","2354",
    "2338","3260","2504","4958","3019","8210","2481","6213","1229","2231",
    "1503","3044",
]

# ------------------------------------------------------------------ data loaders
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_universe(token: str) -> pd.DataFrame:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_info()
    return df[["stock_id", "stock_name", "type", "industry_category"]].drop_duplicates("stock_id")

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_prices(token: str, sid: str, start: str) -> pd.DataFrame:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_daily(stock_id=sid, start_date=start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_revenue_yoy(token: str, sid: str, start: str):
    """Month revenue YoY% (latest) + accelerating flag (last 3 YoY rising).
    Returns (yoy_latest, accel_bool) or (None, False). Cached 12h — revenue is monthly."""
    api = DataLoader(); api.login_by_token(api_token=token)
    try:
        df = api.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
    except Exception:
        return None, False
    if df.empty or "revenue" not in df.columns:
        return None, False
    df = df.sort_values(["revenue_year", "revenue_month"])
    df["ym"] = df["revenue_year"].astype(int) * 100 + df["revenue_month"].astype(int)
    rev = df.set_index("ym")["revenue"].astype(float)
    # YoY = this month vs same month last year (offset 100 in ym key)
    yoys = []
    for ym in rev.index:
        prev = ym - 100
        if prev in rev.index and rev[prev] > 0:
            yoys.append((ym, (rev[ym] / rev[prev] - 1) * 100))
    if not yoys:
        return None, False
    yoys.sort()
    latest = yoys[-1][1]
    accel = False
    if len(yoys) >= 3:
        last3 = [y for _, y in yoys[-3:]]
        accel = last3[0] < last3[1] < last3[2]        # YoY strictly rising 3 months
    return round(latest, 1), bool(accel)

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_benchmark(token: str, start: str) -> pd.Series:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_total_return_index(index_id="TAIEX", start_date=start)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["price"].resample("W-FRI").last()

# ------------------------------------------------------------------ core logic
def weekly(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].resample("W-FRI").first()
    h = df["max"].resample("W-FRI").max()
    l = df["min"].resample("W-FRI").min()
    c = df["close"].resample("W-FRI").last()
    v = df["Trading_Volume"].resample("W-FRI").sum()
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}).dropna()

def stage_of(w: pd.DataFrame, bench: pd.Series):
    """Weinstein stage + weekly detail. Returns (stage, dict) or (None,None)."""
    if len(w) < MA_WEEKS + 5:
        return None, None
    ma = w["c"].rolling(MA_WEEKS).mean()
    price = w["c"].iloc[-1]; ma_now = ma.iloc[-1]; slope = ma.diff(4).iloc[-1]
    above = price > ma_now
    prior_high = w["h"].iloc[-7:-1].max()
    vol_avg = w["v"].iloc[-11:-1].mean()
    breakout = price > prior_high
    vol_surge = w["v"].iloc[-1] > VOL_MULT * vol_avg
    rs = None
    if len(bench) > 14:
        j = w.join(bench.rename("bm"), how="inner")
        if len(j) > 14:
            rs = (j["c"].iloc[-1]/j["c"].iloc[-14]-1) - (j["bm"].iloc[-1]/j["bm"].iloc[-14]-1)
    up = slope > 0; dn = slope < 0
    stage = 2 if (above and up) else 4 if (not above and dn) else 3 if (above and not up) else 1
    return stage, dict(price=round(price,2), ma=round(ma_now,2),
                       breakout=breakout, vol_surge=vol_surge,
                       rs=None if rs is None else round(rs*100,1),
                       below_ma=not above)

def trend_template(px: pd.DataFrame, rs_ok):
    """Minervini Trend Template on daily data. rs_ok = bool (RS-rank>=70 proxy, passed in).
    Returns (pass_all, hits/8)."""
    c = px["close"]
    if len(c) < 210:
        return False, 0
    ma50 = c.rolling(50).mean().iloc[-1]
    ma150 = c.rolling(150).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]
    ma200_1mo = c.rolling(200).mean().iloc[-22]
    price = c.iloc[-1]
    lo52 = c.iloc[-252:].min() if len(c) >= 252 else c.min()
    hi52 = c.iloc[-252:].max() if len(c) >= 252 else c.max()
    cond = [
        price > ma150 and price > ma200,
        ma150 > ma200,
        ma200 > ma200_1mo,
        ma50 > ma150 > ma200,
        price > ma50,
        price >= lo52 * 1.30,
        price >= hi52 * 0.75,
        bool(rs_ok),
    ]
    hits = int(sum(bool(x) for x in cond))
    return hits == 8, hits

def vcp_scan(px: pd.DataFrame):
    """Volatility Contraction Pattern via AMPLITUDE contraction (robust — no
    swing-point counting). Split the base into 3 equal segments and measure each
    segment's normalized daily volatility (mean of (high-low)/close). A base is
    contracting when volatility steps down segment to segment and the latest
    segment is tight. Pairs with volume dry-up. Pivot = 20d high."""
    out = dict(vcp=False, contr="", pivot=None, near=False)
    if len(px) < 120:
        return out
    base = px.iloc[-60:]                     # ~12 weeks base
    hi = base["max"].values
    lo = base["min"].values
    cl = base["close"].values
    vol = base["Trading_Volume"].values
    amp = (hi - lo) / cl * 100               # daily amplitude % of close

    seg = len(amp) // 3
    s1 = amp[:seg].mean()                    # earliest third
    s2 = amp[seg:2*seg].mean()               # middle third
    s3 = amp[2*seg:].mean()                  # most recent third

    # Contraction: volatility stepping down. Require the recent third clearly
    # below the earliest third, and not expanding vs the middle third. Absolute
    # tightness guard on the latest segment.
    contracting = (
        s3 <= s1 * 0.75                      # recent vol clearly below early vol
        and s3 <= s2 * 1.05                  # not expanding into the latest leg
        and s3 <= 6.0                        # latest segment tight in absolute terms
    )
    # Volume dry-up: recent 10d avg at/below the base's earlier avg (softened)
    vol_dry = vol[-10:].mean() <= vol[:-10].mean() * 1.05

    pivot = float(base["max"].iloc[-20:].max())
    price = float(cl[-1])
    out.update(vcp=bool(contracting and vol_dry),
               contr=f"{s1:.0f}%→{s2:.0f}%→{s3:.0f}%",     # amplitude trend, not pullback depths
               pivot=round(pivot, 2), near=bool(price >= pivot * 0.97))
    return out

# ------------------------------------------------------------------ style
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  h1 {font-weight:700; letter-spacing:-.5px;}
  .pill {display:inline-block;padding:2px 10px;border-radius:999px;
         font-size:.78rem;font-weight:600;color:#fff;}
  .muted {color:#8a8f98;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("Minervini SEPA 選股器")
st.markdown('<span class="muted">趨勢模板 + VCP + 月營收 YoY · 台股 · 與 Weinstein 版分開</span>',
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("設定")
    token = st.text_input("FinMind Token", type="password",
                          value=st.secrets.get("FINMIND_TOKEN", ""),
                          help="免費申請：finmindtrade.com")

tab1, tab2, tab3 = st.tabs(["① 選股", "② 持股監控", "③ 進場計算"])

# ================================================================== TAB 1
with tab1:
    st.caption("只掃 Stage 2，疊加 Minervini 趨勢模板、VCP、月營收 YoY。全部當欄位，不硬篩。")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_scan = st.slider("掃描檔數上限", 30, 400, 120, 10, key="t1_scan")
    with c2:
        want_fund = st.checkbox("加月營收 YoY 欄", value=True, key="t1_fund",
                                help="每檔多 1 次 API。免費但吃額度，已快取 12h。")
    with c3:
        relaxed = st.checkbox("寬鬆模式 (RS 前 40%)", value=False, key="t1_relax",
                              help="嚴格=RS 前 30%(Minervini 標準)。寬鬆看更多近似標的。")
    go1 = st.button("開始掃描", type="primary", key="t1_go")

    if go1 and token:
        start = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
        try:
            uni = load_universe(token)
        except Exception as e:
            st.error(f"讀取清單失敗：{e}"); st.stop()
        pool = uni[~(uni["type"].str.contains("etf|ETF", case=False, na=False) |
                     uni["stock_id"].str.match(r"^00\d{2,4}"))]
        pool = pool[pool["stock_id"].str.match(r"^\d{4,6}$")]
        order = {s: i for i, s in enumerate(BIGCAP)}
        pool = pool.assign(_r=pool["stock_id"].map(order).fillna(9999)).sort_values("_r").head(max_scan)
        bench = load_benchmark(token, start)

        # pass 1: collect prices + stage + raw RS for percentile ranking
        recs, prog = [], st.progress(0.0, text="掃描中…")
        for i, (_, r) in enumerate(pool.iterrows(), 1):
            prog.progress(i/len(pool), text=f"掃描中… {r.stock_id} {r.stock_name}")
            try:
                px = load_prices(token, r.stock_id, start)
                if px.empty: continue
                stg, d = stage_of(weekly(px), bench)
                if stg != 2:            # Minervini: Stage 2 only
                    continue
                recs.append(dict(sid=r.stock_id, name=r.stock_name, px=px, d=d))
            except Exception:
                continue
        prog.empty()

        if not recs:
            st.info("這批沒有 Stage 2 標的。提高掃描檔數再試。"); st.stop()

        # percentile RS across the Stage-2 pool -> condition #8 (RS>=70 => top 30%)
        rs_vals = [x["d"]["rs"] for x in recs if x["d"]["rs"] is not None]
        cutoff_pct = 60 if relaxed else 70          # top 40% vs top 30%
        rs_cut = np.percentile(rs_vals, cutoff_pct) if rs_vals else -1

        rows = []
        for x in recs:
            d = x["d"]
            rs_ok = (d["rs"] is not None) and (d["rs"] >= rs_cut)
            tt_pass, tt_hits = trend_template(x["px"], rs_ok)
            v = vcp_scan(x["px"])
            yoy, accel = (None, False)
            if want_fund:
                yoy, accel = load_revenue_yoy(token, x["sid"],
                                              (dt.date.today()-dt.timedelta(days=500)).isoformat())
            rows.append(dict(代號=x["sid"], 名稱=x["name"], 收盤=d["price"], MA30W=d["ma"],
                             RS=d["rs"], TT符合=tt_pass, TT分=tt_hits,
                             VCP=v["vcp"], 收縮=v["contr"], 樞紐價=v["pivot"], 近樞紐=v["near"],
                             **({"營收YoY%": yoy, "營收加速": accel} if want_fund else {})))
        df = pd.DataFrame(rows).sort_values(["TT分", "RS"], ascending=[False, False])

        st.caption(f"掃描 {len(pool)} 檔 · Stage 2 命中 {len(df)} 檔 · "
                   f"RS 門檻(前{100-cutoff_pct}%)={rs_cut:.1f} · {dt.date.today():%Y-%m-%d}")
        colcfg = {
            "RS": st.column_config.NumberColumn("RS%", format="%.1f"),
            "TT符合": st.column_config.CheckboxColumn("TT全符", help="趨勢模板 8 條全過"),
            "TT分": st.column_config.NumberColumn("TT分", format="%d", help="/8"),
            "VCP": st.column_config.CheckboxColumn("VCP"),
            "收縮": st.column_config.TextColumn("收縮序列"),
            "樞紐價": st.column_config.NumberColumn("樞紐價", format="%.2f"),
            "近樞紐": st.column_config.CheckboxColumn("近樞紐", help="距樞紐≤3%"),
        }
        if want_fund:
            colcfg["營收YoY%"] = st.column_config.NumberColumn("營收YoY%", format="%.1f",
                                    help="最新月營收年增率")
            colcfg["營收加速"] = st.column_config.CheckboxColumn("營收加速", help="近3月 YoY 遞增")
        st.dataframe(df, hide_index=True, use_container_width=True, column_config=colcfg)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載選股 CSV", csv,
                           file_name=f"minervini_pick_{dt.date.today().isoformat()}.csv",
                           mime="text/csv")
        st.caption("進場前自行覆核基本面與題材。僅供研究，非投資建議。")
    elif go1:
        st.warning("請先在左側填 FinMind Token。")

# ================================================================== TAB 2
with tab2:
    st.caption("貼上你的持股代號，回報階段 + 出場旗標。只看你手上的，不掃全市場。")
    holdings = st.text_area("持股代號（逗號、空白或換行分隔）",
                            placeholder="2330 2454 6488", height=80, key="t2_in")
    drop_pct = st.slider("回檔警示門檻 %", 5, 30, 15, 1, key="t2_drop",
                         help="距近期高點回檔超過此值 → 警示")
    go2 = st.button("檢查持股", type="primary", key="t2_go")

    if go2 and token and holdings.strip():
        ids = [s for s in __import__("re").split(r"[,\s]+", holdings.strip()) if s]
        start = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
        bench = load_benchmark(token, start)
        try:
            uni = load_universe(token).set_index("stock_id")["stock_name"].to_dict()
        except Exception:
            uni = {}
        rows, prog = [], st.progress(0.0, text="檢查中…")
        for i, sid in enumerate(ids, 1):
            prog.progress(i/len(ids), text=f"檢查中… {sid}")
            try:
                px = load_prices(token, sid, start)
                if px.empty:
                    rows.append(dict(代號=sid, 名稱=uni.get(sid,"?"), 階段="無資料",
                                     現價=None, 距高點=None, 旗標="?")); continue
                stg, d = stage_of(weekly(px), bench)
                hi = float(px["close"].iloc[-120:].max()) if len(px)>=120 else float(px["close"].max())
                price = float(px["close"].iloc[-1])
                pull = (price/hi - 1) * 100
                flags = []
                if stg == 4: flags.append("⛔出場")
                if stg == 3: flags.append("⚠減碼")
                if d and d["below_ma"]: flags.append("跌破30週")
                if pull <= -drop_pct: flags.append(f"回檔{pull:.0f}%")
                rows.append(dict(代號=sid, 名稱=uni.get(sid, "?"),
                                 階段=STAGE_META[stg][0].split()[0] if stg else "?",
                                 現價=round(price,2), 距高點=round(pull,1),
                                 旗標="  ".join(flags) if flags else "✓持有"))
            except Exception:
                rows.append(dict(代號=sid, 名稱=uni.get(sid,"?"), 階段="錯誤",
                                 現價=None, 距高點=None, 旗標="?"))
        prog.empty()
        mon = pd.DataFrame(rows)
        st.dataframe(mon, hide_index=True, use_container_width=True, column_config={
            "距高點": st.column_config.NumberColumn("距高點%", format="%.1f",
                        help="距近120日高點；負值=已回檔"),
        })
        csv = mon.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載持股監控 CSV", csv,
                           file_name=f"holdings_{dt.date.today().isoformat()}.csv",
                           mime="text/csv")
        st.caption("週線階段會落後盤中；停損仍以進場時設的 7-8% 為準，別只等階段翻。")
    elif go2 and not token:
        st.warning("請先在左側填 FinMind Token。")
    elif go2:
        st.warning("請先貼上持股代號。")

# ================================================================== TAB 3
with tab3:
    st.caption("輸入樞紐/買價與帳戶規模，算停損位與部位大小。純風險數學，不需 API。")
    a, b = st.columns(2)
    with a:
        acct = st.number_input("帳戶總額 (TWD)", min_value=0, value=1_000_000, step=50_000, key="t3_acct")
        buy = st.number_input("預計買價 (樞紐突破價)", min_value=0.0, value=100.0, step=0.5, key="t3_buy")
    with b:
        risk_pct = st.number_input("每筆帳戶風險 %", min_value=0.1, max_value=5.0,
                                   value=DEFAULT_RISK_PCT, step=0.05, key="t3_risk")
        stop_pct = st.slider("停損 % (樞紐下方)", 3, 12, 8, 1, key="t3_stop")

    if buy > 0 and acct > 0:
        stop_price = buy * (1 - stop_pct/100)
        risk_per_share = buy - stop_price
        risk_amount = acct * risk_pct/100
        shares = int(risk_amount // risk_per_share) if risk_per_share > 0 else 0
        position = shares * buy
        pos_pct = position/acct*100 if acct else 0
        lots = shares/1000

        st.markdown("### 結果")
        m1, m2, m3 = st.columns(3)
        m1.metric("停損價", f"{stop_price:,.2f}", f"-{stop_pct}%")
        m2.metric("每股風險", f"{risk_per_share:,.2f}")
        m3.metric("可承受虧損", f"{risk_amount:,.0f}", f"{risk_pct}% 帳戶")
        m4, m5, m6 = st.columns(3)
        m4.metric("建議股數", f"{shares:,}", f"{lots:.1f} 張")
        m5.metric("部位金額", f"{position:,.0f}")
        m6.metric("佔帳戶", f"{pos_pct:.1f}%")
        st.caption(f"邏輯：帳戶風險 {risk_pct}% = {risk_amount:,.0f} 元；"
                   f"停損 {stop_pct}% → 每股賠 {risk_per_share:.2f}；"
                   f"股數 = 可虧損 ÷ 每股風險。跌到 {stop_price:.2f} 出場即約賠 {risk_pct}% 帳戶。")
        if pos_pct > 100:
            st.warning("部位超過帳戶總額（需融資）。調高停損%或降低風險%。")
