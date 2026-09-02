"""
週K教學圖 (Chart Reading Trainer) — Streamlit 介面

分析與繪圖的邏輯全在 teach_core.py，這支只負責 UI。
排程報表 scripts/report_*.py 走同一份 core，所以圖不會兩邊長不一樣。
"""
import datetime as dt
import io

import numpy as np
import pandas as pd
import streamlit as st

from teach_core import *          # noqa: F401,F403
from teach_core import (MA_WEEKS, SLOPE_LAG, STAGE_NAME, FONT_OK,
                        load_one, load_universe, market_context, to_weekly,
                        stage_series, smooth_segments, vcp_contractions,
                        deduct_table, read_metrics, box_stats, rs_line,
                        verdicts, build_figure, set_style)

st.set_page_config(page_title="週K教學圖", page_icon="📚", layout="wide")

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
    box_look = st.slider("箱體觀察窗（週）", 6, 26, 12, 1,
                         help="窄/中/寬與貼頂/貼底都用這個窗口算。"
                              "週線 12 根 ≈ 一季。")
    mid_len = st.slider("中線均線期數（週）", 5, 40, 20, 1,
                        help="乖離率的基準線，大腸圈中線的概念。")
    ded_period = st.slider("扣抵值示意用均線期數", 4, 12, 5, 1,
                           help="右下角示意表用。5 期最好懂，數字多了看不出移出的是哪一格。")
    min_seg = st.slider("階段最短週數", 2, 12, 6, 1,
                        help="短於這個週數的階段會被併進鄰段。不平滑的話，"
                             "一段上升裡幾根回檔週就會把色塊切成碎片。")
    st.divider()
    st.subheader("外觀")
    theme = st.selectbox("背景", ["白底", "圖表黑底", "全黑底"], index=0,
                         help="「圖表黑底」只有 K 線／RS／量／日線放大四個圖是黑的，"
                              "外圈面板與文字維持白底。全黑底很多地方讀不清楚，"
                              "留著當實驗性選項。")
    candle = st.selectbox("K 棒配色", ["紅漲綠跌", "紅漲黑跌"], index=0,
                          help="黑K在黑底上看不見，選黑底時會自動改用淺灰，"
                               "維持「非紅即跌」的對比。")
    solid_up = st.checkbox("上漲畫實心", value=True,
                           help="台灣慣例是實心紅。取消則畫空心，密集區比較好讀。")
    stage_alpha = st.slider("階段色塊透明度", 0.10, 0.70, 0.30, 0.05,
                            help="太高會把 K 棒吃掉。深色圖表建議 0.25–0.35。")
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

set_style({"白底": "light", "圖表黑底": "chartdark", "全黑底": "dark"}[theme],
          candle, solid_up, stage_alpha)

segs = smooth_segments(stage_series(wk), min_seg)
stage_now = segs[-1][2] if segs else 0
vcp_legs, vcp_ok = vcp_contractions(wk, vcp_look, vcp_pct)
dtbl, _ = deduct_table(wk, ded_period, steps=ded_period)
m = read_metrics(daily, wk, dtbl, box_look, rs_rank, rs_new_high)
bx = box_stats(wk, box_look, mid_len)
vz = verdicts(m, bx, stage_now, vcp_ok, vcp_legs)

fig = build_figure(sid, name, daily, wk, segs, dtbl, m, bx, vz, vcp_legs, vcp_ok,
                   rs, rs_new_high, show_verdict, True, weeks, ded_period)
st.pyplot(fig, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
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

with st.expander("箱體判定明細"):
    _g = pd.DataFrame([
        {"項目": "觀察窗", "值": f"{bx['look']} 週"},
        {"項目": "區間高點", "值": f"{bx['hi']:,.2f}"},
        {"項目": "區間低點", "值": f"{bx['lo']:,.2f}"},
        {"項目": "箱體寬度", "值": f"{bx['width']:.1f}%　→　{bx['grade']}"},
        {"項目": "現價位置", "值": f"{bx['pos']:.1f}%　→　{bx['zone']}"},
        {"項目": f"{bx['ma_len']} 週中線", "值": f"{bx['mid']:,.2f}"},
        {"項目": "中線乖離率", "值": f"{bx['dev']:+.2f}%"},
        {"項目": "箱寬自身百分位",
         "值": (f"第 {bx['pctile']:.0f} 百分位（近 {bx['n_hist']} 週）"
                if bx["pctile"] == bx["pctile"] else "歷史不足，無法計算")},
    ])
    st.dataframe(_g, use_container_width=True, hide_index=True)
    st.caption("寬度切分 ≤30% 窄 / ≤60% 中 / >60% 寬，位置切分 ≥70% 貼頂 / ≤35% 貼底。"
               "這是絕對門檻，金融股天生窄、飆股天生寬——"
               "要判斷「有沒有在收縮」請看自身百分位，低於 20 才是真的縮。")

with st.expander("扣抵值明細"):
    st.dataframe(dtbl, use_container_width=True, hide_index=True)
