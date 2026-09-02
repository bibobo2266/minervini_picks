"""
Minervini SEPA Scanner — Streamlit 介面

掃描邏輯全在 minervini_core.py，這支只負責 UI。
每日報表 scripts/report_minervini.py 走同一份 core。
"""
import datetime as dt
import io
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

from minervini_core import *      # noqa: F401,F403

st.set_page_config(page_title="Minervini SEPA · 選股", page_icon="◈", layout="wide")

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
    try:
        _tok = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        _tok = ""          # 沒設 secrets 也不要讓整個 app 掛掉
    token = st.text_input("FinMind Token（選填）", type="password",
                          value=_tok,
                          help="只有勾選月營收或法人買賣超時才需要")
    st.caption("還原股價，2015-06 起（漲跌幅放寬到 10% 的起點）。"
               "每交易日 17:00 自動更新。")
    st.divider()
    rewind = st.checkbox("回到某一天重算", value=False,
                         help="把資料截斷到指定日期，等同回到那天跑一次掃描。")
    as_of = None
    yrs = YEARS_DEFAULT
    if rewind:
        as_of = st.date_input("資料截止日", value=dt.date.today(),
                              min_value=dt.date(2016, 6, 1),
                              max_value=dt.date.today())
        yrs = st.slider("往前載入幾年", 2, 5, 3,
                        help="趨勢模板最多用到一年多的歷史。載越多年越慢。")

tab1, tab2, tab3, tab4 = st.tabs(["① 選股", "② 持股監控", "③ 進場計算",
                                  "④ 準備名單追蹤"])

# ================================================================== TAB 1
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        liq = st.number_input("60日均額門檻（萬元）", 500, 100000, DEFAULT_LIQ, 500,
                              key="t1_liq", help="流動性過濾，取代舊版的成交值前 N 名")
    with c2:
        min_tt = st.slider("最低 TT 分", 4, 8, 8, 1, key="t1_tt",
                           help="低於此分直接歸為淘汰，不顯示")
    with c3:
        want_fund = st.checkbox("加月營收 / 法人（需 token）", value=False, key="t1_fund",
                                help="只對非淘汰名單抓，每檔 2 次 API")

    if st.button("開始掃描", type="primary", key="t1_go"):
        st.session_state.pop("scan_df", None)
        m, _ = build_matrices(yrs, as_of)
        uni = load_universe()
        base = scan(m, liq)
        if base.empty:
            st.error("母體為空，檢查門檻或資料檔。"); st.stop()

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

        st.session_state["scan_df"] = df
        st.session_state["scan_day"] = f"{m['c'].index[-1]:%Y-%m-%d}"
        st.session_state["scan_pool"] = len(base)

    # ---- 渲染（放在按鈕外，這樣按「帶到持股監控」不會清空結果）----
    if "scan_df" in st.session_state:
        df = st.session_state["scan_df"]
        want_fund = "營收YoY%" in df.columns
        st.caption(f"母體 {st.session_state['scan_pool']} 檔 · "
                   f"資料日 {st.session_state['scan_day']}"
                   + (f" · ⏪ 回溯至 {as_of}" if as_of else ""))

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

        hot = df[df["狀態"].isin(["觸發", "準備"])]["代號"].astype(str).tolist()
        watch = df[df["狀態"] == "觀察"]["代號"].astype(str).tolist()

        if st.button("→ 帶到 ② 持股監控", key="t1_send"):
            st.session_state["t2_in"] = " ".join(hot)
            st.toast(f"已帶入 {len(hot)} 檔到 ② 持股監控")

        k1, k2 = st.columns(2)
        with k1:
            st.text_area(f"🟢觸發 + 🟠準備（{len(hot)} 檔）",
                         " ".join(hot), height=88, key="t1_copy_hot",
                         help="今晚的工作區：抄樞紐價、Tab③ 算部位、設券商到價通知。"
                              "不要丟進扣抵值——這批依定義就在高點附近，"
                              "扣抵值只會回答「剛噴出，不追」。")
        with k2:
            st.text_area(f"🟡觀察 → 扣抵值 app（{len(watch)} 檔）",
                         " ".join(watch), height=88, key="t1_copy_watch",
                         help="這批還在整理，才是扣抵值該吃的。"
                              "出現「窄箱貼頂」的，通常是下週會進準備區的預告。")
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
    cA, cB = st.columns(2)
    with cA:
        drop_pct = st.slider("回檔警示門檻 %", 5, 30, 15, 1, key="t2_drop")
    with cB:
        buy_dates = st.text_input("買進日（選填，對應上面順序，YYYY-MM-DD 空白分隔）",
                                  key="t2_bd",
                                  help="留白則自動偵測最近一次突破日")

    if st.button("檢查持股", type="primary", key="t2_go") and holdings.strip():
        m, _ = build_matrices(yrs, as_of)
        uni = load_universe()
        ids = [s for s in re.split(r"[,\s]+", holdings.strip()) if s]
        bds = [s for s in re.split(r"[,\s]+", buy_dates.strip()) if s]
        rows = []
        for k, sid in enumerate(ids):
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

            bo_pos = None
            if k < len(bds):
                try:
                    d0 = pd.Timestamp(bds[k])
                    pos = px.index.searchsorted(d0)
                    if 0 <= pos < len(px):
                        bo_pos = int(pos)
                except Exception:
                    bo_pos = None
            tb = tennis(px, bo_pos)

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
            if tb["judge"].startswith("🥚"):
                flags.append("🥚雞蛋")

            rows.append(dict(
                代號=sid,
                名稱=uni.loc[sid, "stock_name"] if sid in uni.index else "?",
                階段={2: "上升", 1: "打底", 3: "頭部", 4: "下跌"}.get(stg, "?"),
                現價=round(price, 2), 距高點=round(pull, 1),
                今日=today, 量倍=volx, 最大跌=worst, 底部序=bc,
                網球=tb["judge"], 突破後=tb["days"], 峰漲幅=tb["peak"],
                距峰=tb["pull"], 回檔天=tb["pull_days"], 回檔量比=tb["volr"],
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
            "網球": st.column_config.TextColumn("網球/雞蛋",
                help="🎾回檔淺且短且量縮 · 🥚回檔深或拖太久或量放大"),
            "突破後": st.column_config.NumberColumn("突破後天", format="%d"),
            "峰漲幅": st.column_config.NumberColumn("峰漲幅%", format="%.1f",
                help="突破日到最高點的漲幅"),
            "距峰": st.column_config.NumberColumn("距峰%", format="%.1f"),
            "回檔天": st.column_config.NumberColumn("回檔天", format="%d"),
            "回檔量比": st.column_config.NumberColumn("回檔量比", format="%.2f",
                help="回檔期均量 / 上漲期均量。<1 是好事"),
        })
        st.download_button("下載 CSV", mon.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"holdings_{dt.date.today().isoformat()}.csv",
                           mime="text/csv")
        st.caption("🔥爆量最大跌勢＝第2階段以來單日最大跌幅且量放大 1.8 倍以上，"
                   "書裡明講這通常是賣訊，即使盈餘仍然很好。　"
                   "🎾網球＝突破後拉回淺(≤8%)、短(≤10根)、量縮，或已再創新高；"
                   "🥚雞蛋＝回檔>12% 或拖過 15 根或量在放大。"
                   "突破超過 25 天顯示「突破已久」，網球行為只描述突破後幾天到一兩週。")

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

    # ---------------- 批次：把準備區整批算成一張下單紙 ----------------
    st.divider()
    st.subheader("批次計算：準備區下單紙")

    if "scan_df" not in st.session_state:
        st.info("先到 ① 選股 按「開始掃描」，這裡才會有名單。")
    else:
        sdf = st.session_state["scan_df"]
        pick = st.multiselect("要算哪幾區", ["觸發", "準備", "觀察"],
                              default=["準備"], key="t3_zones")
        skip_late = st.checkbox("排除第 4 底以上", value=True, key="t3_late")
        alert_off = st.number_input("警示價設在樞紐下方 %", 0.0, 3.0, 0.5, 0.1,
                                    key="t3_off",
                                    help="券商智慧單有幾秒延遲，早一點準備")

        sel = sdf[sdf["狀態"].isin(pick)].copy()
        if skip_late:
            sel = sel[~sel["晚期"]]

        if sel.empty:
            st.info("這個條件下沒有標的。")
        else:
            out = []
            for _, r in sel.iterrows():
                pv = r["樞紐價"]
                if not pv or pv <= 0:
                    continue
                stop = pv * (1 - stop_pct / 100)
                rps_ = pv - stop
                sh = int((acct * risk_pct / 100) // rps_) if rps_ > 0 else 0
                lots = sh / 1000
                pos = sh * pv
                note = []
                # 零股可以買，但盤中零股撮合間隔長、掛單簿薄，價差會侵蝕停損空間
                if lots < 0.3:
                    note.append("零股風險")
                elif lots < 1:
                    note.append("零股")
                if acct and pos / acct * 100 > 40:
                    note.append("部位過大")
                if r["均額億"] < 3:
                    note.append("量小滑價")
                out.append(dict(
                    狀態=r["狀態"], 代號=r["代號"], 名稱=r["名稱"],
                    樞紐價=round(float(pv), 2),
                    警示價=round(float(pv) * (1 - alert_off / 100), 2),
                    停損價=round(float(stop), 2),
                    張數=round(lots, 2), 股數=sh,
                    部位金額=int(pos),
                    佔帳戶=round(pos / acct * 100, 1) if acct else 0,
                    底部序=int(r["底部序"]), RS=int(r["RS"]),
                    均額億=r["均額億"], 產業=r["產業"],
                    備註=" ".join(note)))
            sheet = pd.DataFrame(out).sort_values(["狀態", "RS"],
                                                  ascending=[True, False])
            st.dataframe(sheet, hide_index=True, use_container_width=True,
                         column_config={
                "樞紐價": st.column_config.NumberColumn(format="%.2f"),
                "警示價": st.column_config.NumberColumn(format="%.2f",
                    help="設進券商 app 的到價通知"),
                "停損價": st.column_config.NumberColumn(format="%.2f"),
                "張數": st.column_config.NumberColumn(format="%.2f"),
                "部位金額": st.column_config.NumberColumn(format="%d"),
                "佔帳戶": st.column_config.NumberColumn("佔帳戶%", format="%.1f"),
                "均額億": st.column_config.NumberColumn(format="%.2f"),
                "備註": st.column_config.TextColumn(
                    help="零股<1張(撮合較差) · 零股風險<0.3張(滑價恐吃掉停損) · 部位過大>40% · 量小滑價<3億"),
            }, height=420)
            st.caption(f"帳戶 {acct:,.0f} · 每筆風險 {risk_pct}% · 停損 {stop_pct}%"
                       f" · 共 {len(sheet)} 檔")
            st.download_button("下載下單紙 CSV",
                               sheet.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"orders_{dt.date.today().isoformat()}.csv",
                               mime="text/csv")


# ================================================================== TAB 4
with tab4:
    st.subheader("準備名單追蹤")
    st.caption("名單由 scripts/watchlist_update.py 每日機械更新。"
               "這裡只讀不寫——判斷留給人，執行留給腳本。")

    def _load_csv(path: str, url_path: str) -> pd.DataFrame:
        """本機優先。Streamlit Cloud 會 checkout repo，所以通常讀得到本機檔。"""
        import os
        if os.path.exists(path):
            return pd.read_csv(path, dtype={"代號": str})
        r = requests.get(RAW + url_path, timeout=60)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(r.content), dtype={"代號": str})

    wl = _load_csv("data/watchlist.csv", "/data/watchlist.csv")
    hist = _load_csv("data/watchlist_history.csv", "/data/watchlist_history.csv")

    if wl.empty:
        st.info("還沒有名單。先讓 watchlist_update.py 跑過一次。")
    else:
        wl["在榜天數"] = pd.to_numeric(wl["在榜天數"], errors="coerce")
        n_trig = int(wl["觸發日"].notna().sum())
        soon = wl[(wl["觸發日"].isna()) & (wl["在榜天數"] >= 8 * 7 - 7)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("在榜", len(wl))
        c2.metric("已觸發", n_trig)
        c3.metric("一週內逾期", len(soon))
        c4.metric("平均在榜天數", f"{wl['在榜天數'].mean():.0f}")

        if len(soon):
            st.warning("一週內逾期：" + "、".join(
                f"{r['代號']} {r['名稱']}（第 {r['在榜天數']:.0f} 天）"
                for _, r in soon.iterrows()))

        only_trig = st.checkbox("只看已觸發", value=False, key="t4_trig")
        view = wl[wl["觸發日"].notna()] if only_trig else wl
        st.dataframe(
            view.drop(columns=["移出日", "移出原因"], errors="ignore"),
            use_container_width=True, hide_index=True, height=420,
            column_config={
                "進榜價": st.column_config.NumberColumn(format="%.1f"),
                "最新價": st.column_config.NumberColumn(format="%.1f"),
                "進榜樞紐": st.column_config.NumberColumn("樞紐", format="%.1f"),
                "停損價": st.column_config.NumberColumn(format="%.1f"),
                "進榜RS": st.column_config.NumberColumn("進榜RS", format="%.0f"),
                "最新RS": st.column_config.NumberColumn(format="%.0f"),
                "在榜天數": st.column_config.NumberColumn(format="%d"),
                "進榜VCP": st.column_config.CheckboxColumn("VCP"),
            })

        d1, d2 = st.columns(2)
        d1.download_button("下載名單 CSV",
                           wl.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"watchlist_{dt.date.today().isoformat()}.csv",
                           mime="text/csv", use_container_width=True)
        if not hist.empty:
            d2.download_button("下載移出紀錄 CSV",
                               hist.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"watchlist_history_{dt.date.today().isoformat()}.csv",
                               mime="text/csv", use_container_width=True)

        # ---------------- 教學圖 ----------------
        st.divider()
        st.subheader("產生教學圖")
        st.caption("用的是教學圖 app 同一份 teach_core，圖不會兩邊長不一樣。"
                   "第一次要載大盤資料算 RS，會慢一點。")

        labels = {f"{r['代號']} {r['名稱']}": r["代號"] for _, r in wl.iterrows()}
        default = [k for k, v in labels.items()
                   if wl.loc[wl["代號"] == v, "觸發日"].notna().any()][:5]
        picked = st.multiselect("選要出圖的股票", list(labels), default=default,
                                key="t4_pick")
        cc1, cc2, cc3, cc4 = st.columns(4)
        weeks = cc1.slider("顯示週數", 26, 156, 52, 2, key="t4_weeks")
        cap = cc2.number_input("一次最多幾張", 1, 20, 5, 1, key="t4_cap")
        t4_theme = cc3.selectbox("背景", ["白底", "圖表黑底", "全黑底"],
                                 key="t4_theme")
        t4_candle = cc4.selectbox("K 棒", ["紅漲綠跌", "紅漲黑跌"], key="t4_candle")

        if st.button("產生教學圖", type="primary", key="t4_go"):
            if not picked:
                st.warning("先選至少一檔。")
            else:
                import zipfile
                import matplotlib.pyplot as plt
                import teach_core as T
                T.set_style({"白底": "light", "圖表黑底": "chartdark",
                             "全黑底": "dark"}[t4_theme], t4_candle, True, 0.30)
                ids = [labels[p] for p in picked][:int(cap)]
                names = dict(zip(wl["代號"], wl["名稱"]))
                idx_mkt, rank_all = T.market_context()
                prog = st.progress(0.0, text="出圖中…")
                made = []
                for i, sid in enumerate(ids, 1):
                    prog.progress(i / len(ids), text=f"出圖中… {sid}")
                    try:
                        daily = T.load_one(sid)
                        wkk = T.to_weekly(daily)
                        if len(wkk) < T.MA_WEEKS + T.SLOPE_LAG + 2:
                            st.caption(f"跳過 {sid}：週線僅 {len(wkk)} 根")
                            continue
                        rs, nh = T.rs_line(wkk, idx_mkt)
                        rr = float(rank_all.get(sid, np.nan)) if len(rank_all) else np.nan
                        segs = T.smooth_segments(T.stage_series(wkk), 6)
                        legs, ok = T.vcp_contractions(wkk, 26, 6.0)
                        dtbl, _ = T.deduct_table(wkk, 5, 5)
                        mm = T.read_metrics(daily, wkk, dtbl, 12, rr, nh)
                        bx = T.box_stats(wkk, 12, 20)
                        vz = T.verdicts(mm, bx, segs[-1][2] if segs else 0, ok, legs)
                        fig = T.build_figure(sid, names.get(sid, ""), daily, wkk,
                                             segs, dtbl, mm, bx, vz, legs, ok,
                                             rs, nh, True, True, weeks, 5)
                        st.pyplot(fig, use_container_width=True)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=140,
                                    bbox_inches="tight",
                                    facecolor=fig.get_facecolor())
                        plt.close(fig)
                        made.append((f"{sid}_{names.get(sid, '')}.png", buf.getvalue()))
                    except Exception as e:
                        st.caption(f"跳過 {sid}：{type(e).__name__} {e}")
                prog.empty()
                if made:
                    zb = io.BytesIO()
                    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
                        for fn, data in made:
                            z.writestr(fn, data)
                    st.download_button(
                        f"下載全部 {len(made)} 張圖 (ZIP)", zb.getvalue(),
                        file_name=f"charts_{dt.date.today().isoformat()}.zip",
                        mime="application/zip", type="primary")

        # ---------------- 移出原因統計 ----------------
        if not hist.empty:
            st.divider()
            st.subheader("移出原因統計")
            vc = hist["移出原因"].value_counts()
            st.bar_chart(vc)
            hist["在榜天數"] = pd.to_numeric(hist["在榜天數"], errors="coerce")
            t = int(hist["觸發日"].notna().sum())
            st.caption(
                f"累計移出 {len(hist)} 筆 · 平均在榜 {hist['在榜天數'].mean():.0f} 天 · "
                f"曾觸發 {t}/{len(hist)} = {t / len(hist):.0%}")
            st.caption("讀法：逾期佔多數 → 進榜門檻太鬆；突破失敗佔多數 → "
                       "觸發判定有問題；階段破壞佔多數 → 進場時機太早。")
