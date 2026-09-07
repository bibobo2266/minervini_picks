"""
app_dojo.py — 老手看盤道場

三個分頁：
  教材庫   隨時可回去看的全標註圖 + 速查卡（不記分）
  練習     步驟三（半標註）、步驟四（零標註），會記分
  我的紀錄 練習紀錄表 + CSV 匯出／匯入（Streamlit Cloud 重開會清空，所以要自己存）

資料來源：data/adj/prices_adj_*.parquet（沿用 minervini_picks 既有資料層）
教材圖：assets/lessons/<CID>/，由 scripts/build_lesson_assets.py 事先產生
"""

from __future__ import annotations

import io
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import lesson_core as lc

st.set_page_config(page_title="老手看盤道場", page_icon="🥋", layout="wide")

RECORD_COLUMNS = [
    "answered_at", "lesson_cid", "lesson_name", "step",
    "stock_id", "signal_date",
    "answer", "confidence", "reasons", "stop_loss_pct",
    "correct", "used_reference", "seconds",
    "close_quality", "tightness_15d_pct", "high_zone_turnover_share_20d",
    "prior_high_tests_count", "major_breakout_count_250d", "rs_excess_taiex_pct",
    "shakeout_depth_pct", "price_progress_efficiency_20d",
    "volume_dryness_ratio", "bars_to_recover_largest_down_day",
    "ret_20d_pct", "ret_60d_pct", "mae_60d_pct", "hit_stop_12pct",
]


# ---------------------------------------------------------------- 資料載入

@st.cache_resource
def _font():
    return lc.setup_font()


@st.cache_data(show_spinner="載入還原股價…")
def load_prices():
    return lc.load_prices()


@st.cache_data(show_spinner=False)
def load_pool():
    p = os.path.join(lc.ASSET_DIR, "pool.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_index(cid: str):
    p = os.path.join(lc.ASSET_DIR, cid, "index.json")
    return lc.load_json(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def stock_history(sid: str):
    df = load_prices()
    return (df[df["stock_id"].astype(str) == str(sid)]
            .sort_values("date").reset_index(drop=True))


def init_state():
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("stars", set())
    st.session_state.setdefault("question", None)
    st.session_state.setdefault("revealed", False)
    st.session_state.setdefault("used_ref", False)
    st.session_state.setdefault("t_start", None)


init_state()
_font()


# ---------------------------------------------------------------- 側欄

lessons_sorted = sorted(lc.LESSONS, key=lambda x: x.order)
labels = [f"第 {l.order} 課　{l.cid} {l.name}" for l in lessons_sorted]

with st.sidebar:
    st.title("🥋 老手看盤道場")
    pick = st.radio("課程", labels, index=0, label_visibility="collapsed")
    lesson = lessons_sorted[labels.index(pick)]
    st.caption(lesson.question)
    st.divider()
    n_done = sum(1 for r in st.session_state.records if r["lesson_cid"] == lesson.cid)
    st.metric("本課已作答", f"{n_done} 題")
    st.metric("總作答", f"{len(st.session_state.records)} 題")
    st.divider()
    st.caption("練習紀錄只存在瀏覽器工作階段，離開前記得到「我的紀錄」匯出 CSV。")


tab_lib, tab_drill, tab_log = st.tabs(["📚 教材庫", "✍️ 練習", "📊 我的紀錄"])


# ---------------------------------------------------------------- 教材庫

with tab_lib:
    st.subheader(f"{lesson.cid} {lesson.name}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**心法**　{lesson.idea}")
        st.markdown(f"**量測**　`{lesson.primary}`　—　{lesson.formula}")
        st.success(f"**這樣算好**　{lesson.good}")
        st.error(f"**這樣算不好**　{lesson.bad}")
    with c2:
        card = os.path.join(lc.ASSET_DIR, lesson.cid, "cheatsheet.png")
        if os.path.exists(card):
            st.image(card, caption="速查卡（長按可存到手機相簿）")
        else:
            st.info("速查卡尚未產生，請先跑 build_lesson_assets.py")

    st.divider()
    idx = load_index(lesson.cid)
    if not idx:
        st.warning(
            f"還沒有 {lesson.cid} 的教材圖。請先在 GitHub Actions 執行：\n\n"
            f"`python scripts/build_lesson_assets.py --lessons {lesson.cid}`"
        )
    else:
        mode = st.radio("看哪一組", ["正例（這樣算好）", "反例（這樣算不好）", "正反並排"],
                        horizontal=True, key="lib_mode")
        only_star = st.checkbox("只看我收藏的", value=False)

        def show(items, kind):
            for it in items:
                key = f"{lesson.cid}/{kind}/{it['file']}"
                if only_star and key not in st.session_state.stars:
                    continue
                path = os.path.join(lc.ASSET_DIR, lesson.cid, it["file"])
                if not os.path.exists(path):
                    continue
                st.image(path, width="stretch")
                cols = st.columns([1, 3])
                starred = key in st.session_state.stars
                if cols[0].button("⭐ 已收藏" if starred else "☆ 收藏", key="s_" + key):
                    (st.session_state.stars.discard if starred
                     else st.session_state.stars.add)(key)
                    st.rerun()
                cols[1].caption(
                    f"{it['stock_id']}　{it['date']}　"
                    f"{idx['selector_field']} = {it['value']:.2f}　｜　"
                    f"後20日 {it['ret_20d_pct']:+.1f}%　後60日 {it['ret_60d_pct']:+.1f}%"
                )
                st.divider()

        if mode.startswith("正例"):
            show(idx["positive"], "pos")
        elif mode.startswith("反例"):
            show(idx["negative"], "neg")
        else:
            a, b = st.columns(2)
            with a:
                st.markdown("#### 正例")
                show(idx["positive"], "pos")
            with b:
                st.markdown("#### 反例")
                show(idx["negative"], "neg")


# ---------------------------------------------------------------- 練習

def new_question(cid: str, step: int):
    pool = load_pool()
    if pool.empty:
        return None
    field, hib = _selector(cid)
    sub = pool.dropna(subset=[field, "ret_20d_pct"])
    if sub.empty:
        return None
    # 步驟三刻意抽極端值（好壞分明），步驟四抽全分佈（含模糊地帶）
    if step == 3:
        lo, hi = sub[field].quantile(0.2), sub[field].quantile(0.8)
        sub = sub[(sub[field] <= lo) | (sub[field] >= hi)]
    r = sub.sample(1).iloc[0]
    return {"row": r.to_dict(), "field": field, "higher_is_better": hib, "step": step}


def _selector(cid: str):
    return lc.LESSON_SELECTOR.get(cid, ("close_quality", True))


with tab_drill:
    pool = load_pool()
    if pool.empty:
        st.warning("找不到 assets/lessons/pool.parquet，請先跑 build_lesson_assets.py")
        st.stop()

    step = st.radio(
        "練習階段",
        ["步驟三 · 半標註（把眼睛跟數字校準）", "步驟四 · 零標註（券商 app 模式）"],
        key="drill_step",
    )
    step_no = 3 if step.startswith("步驟三") else 4

    cnew, cref = st.columns([1, 1])
    if cnew.button("🎲 出新題", type="primary", width="stretch"):
        st.session_state.question = new_question(lesson.cid, step_no)
        st.session_state.revealed = False
        st.session_state.used_ref = False
        st.session_state.t_start = datetime.now()
        st.rerun()

    if cref.button("📖 看教材（不扣分，但會記錄）", width="stretch"):
        st.session_state.used_ref = True
        idx = load_index(lesson.cid)
        if idx and idx["positive"]:
            st.image(os.path.join(lc.ASSET_DIR, lesson.cid, idx["positive"][0]["file"]),
                     caption="正例參考")
        card = os.path.join(lc.ASSET_DIR, lesson.cid, "cheatsheet.png")
        if os.path.exists(card):
            st.image(card, width=380)

    q = st.session_state.question
    if not q:
        st.info("按「出新題」開始。")
        st.stop()

    row = q["row"]
    sid, i = str(row["stock_id"]), int(row["i"])
    g = stock_history(sid)
    feats = {k: v for k, v in row.items() if k not in ("stock_id", "date", "i")}

    # ---- 題目圖
    if step_no == 3:
        # 半標註：只標「這是突破日」，數值不給
        fig = lc.render_chart(g, i, {"prior_high": row.get("prior_high")},
                              mode="plain", title=f"{sid}",
                              subtitle="步驟三：只告訴你突破日在最後一根，數值自己判斷")
        st.pyplot(fig, width="stretch")
        st.caption("圖的最後一根 K 就是突破日 T。")
    else:
        fig = lc.render_chart(g, i, {}, mode="plain", title=f"{sid}",
                              subtitle="步驟四：券商 app 模式，什麼都不標")
        st.pyplot(fig, width="stretch")

    st.divider()

    # ---- 作答
    if not st.session_state.revealed:
        if step_no == 3:
            st.markdown(f"**問題：這一題的 `{lesson.primary}` 是高還是低？**")
            ans = st.radio("你的判斷", ["高（符合這課的好條件）", "低（不符合）"],
                           horizontal=True, key="ans3")
        else:
            st.markdown("**問題：這個突破你會怎麼處理？**")
            ans = st.radio("你的判斷", ["進場", "觀察", "不進"], horizontal=True, key="ans4")

        conf = st.slider("信心度", 1, 5, 3,
                         help="1 = 完全用猜的，5 = 很確定。這欄是用來檢查你的信心校準。")
        reasons = st.multiselect("理由（可複選，之後用來查哪條心法對你是負分）",
                                 lesson.reasons)
        stop = st.number_input("你會把停損放在幾 %？", -30.0, -1.0, -12.0, 0.5)

        if st.button("✅ 送出，看答案", type="primary"):
            secs = (datetime.now() - (st.session_state.t_start or datetime.now())).total_seconds()
            field, hib = q["field"], q["higher_is_better"]
            val = row[field]
            med = load_pool()[field].median()
            is_high = (val >= med) if hib else (val <= med)

            if step_no == 3:
                correct = bool(is_high == ans.startswith("高"))
            else:
                good = (row.get("ret_60d_pct", 0) or 0) > 0 and not row.get("hit_stop_12pct", False)
                correct = bool((ans == "進場" and good) or (ans == "不進" and not good))

            rec = {c: None for c in RECORD_COLUMNS}
            rec.update({
                "answered_at": datetime.now().isoformat(timespec="seconds"),
                "lesson_cid": lesson.cid, "lesson_name": lesson.name, "step": step_no,
                "stock_id": sid, "signal_date": str(pd.Timestamp(row["date"]).date()),
                "answer": ans, "confidence": conf, "reasons": "|".join(reasons),
                "stop_loss_pct": stop, "correct": correct,
                "used_reference": st.session_state.used_ref, "seconds": round(secs, 1),
            })
            for c in RECORD_COLUMNS:
                if rec[c] is None and c in row:
                    rec[c] = row[c]
            st.session_state.records.append(rec)
            st.session_state.revealed = True
            st.rerun()

    # ---- 揭曉
    else:
        rec = st.session_state.records[-1]
        (st.success if rec["correct"] else st.error)(
            "答對了" if rec["correct"] else "這題錯了 — 看下面的標註圖，找出你漏掉什麼"
        )
        fig = lc.render_chart(
            g, i, feats, mode="reveal", forward=60,
            title=f"{sid}　{rec['signal_date']}　揭曉",
            subtitle=f"{lesson.cid} {lesson.name}　{q['field']} = {row[q['field']]:.2f}",
        )
        st.pyplot(fig, width="stretch")

        m = st.columns(4)
        m[0].metric("後 20 日", f"{row.get('ret_20d_pct', float('nan')):+.1f}%")
        m[1].metric("後 60 日", f"{row.get('ret_60d_pct', float('nan')):+.1f}%")
        m[2].metric("最大回撤(60日)", f"{row.get('mae_60d_pct', float('nan')):+.1f}%")
        m[3].metric("有沒有觸及 -12%", "有" if row.get("hit_stop_12pct") else "沒有")

        with st.expander("這一題的全部量測值"):
            show = {k: v for k, v in feats.items()
                    if isinstance(v, (int, float, np.floating)) and not pd.isna(v)}
            st.dataframe(pd.Series(show, name="值").to_frame(), width="stretch")


# ---------------------------------------------------------------- 我的紀錄

with tab_log:
    st.subheader("練習紀錄")

    up = st.file_uploader("匯入之前存的 CSV（會接在現有紀錄前面）", type="csv")
    if up is not None and st.button("匯入"):
        old = pd.read_csv(up).to_dict("records")
        st.session_state.records = old + st.session_state.records
        st.success(f"匯入 {len(old)} 筆")
        st.rerun()

    if not st.session_state.records:
        st.info("還沒有紀錄。")
        st.stop()

    df = pd.DataFrame(st.session_state.records)
    st.download_button(
        "⬇️ 匯出 CSV（離開前務必按這個）",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"dojo_log_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv", type="primary",
    )

    st.divider()
    c = st.columns(4)
    c[0].metric("總題數", len(df))
    c[1].metric("正確率", f"{df['correct'].mean() * 100:.0f}%")
    c[2].metric("查教材比例", f"{df['used_reference'].mean() * 100:.0f}%")
    c[3].metric("平均作答秒數", f"{df['seconds'].mean():.0f}s")

    st.caption("以下是粗略切面。真正的偏誤診斷要等題數夠多（建議 100 題以上）"
               "再把 CSV 拿去做完整分析。")

    if df["confidence"].notna().any():
        st.markdown("**信心度 × 正確率**（如果 5 跟 2 差不多，代表還在背不是在看）")
        st.dataframe(
            df.groupby("confidence")["correct"].agg(["count", "mean"])
              .rename(columns={"count": "題數", "mean": "正確率"}),
            width="stretch",
        )

    rows = []
    for _, r in df.iterrows():
        for tag in str(r.get("reasons") or "").split("|"):
            if tag:
                rows.append({"理由": tag, "correct": r["correct"]})
    if rows:
        st.markdown("**理由標籤 × 正確率**（正確率明顯偏低的那條，就是在騙你的心法）")
        st.dataframe(
            pd.DataFrame(rows).groupby("理由")["correct"]
              .agg(["count", "mean"]).rename(columns={"count": "用過幾次", "mean": "正確率"})
              .sort_values("正確率"),
            width="stretch",
        )

    st.divider()
    st.dataframe(df.iloc[::-1], width="stretch", height=420)
