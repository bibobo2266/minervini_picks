#!/usr/bin/env python3
"""
scripts/oldhand_cards.py — 每日盤後看盤卡

python scripts/oldhand_cards.py --universe universe_defense.csv --date 2026-09-10
輸出：cards/YYYY-MM-DD/  (index.md + 每檔 .md + 每檔 .png)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combo_core as cc  # noqa: E402

MARK = {"MATCH": "✓", "NO_MATCH": "✗", "PARTIAL": "—"}

# 固定 12 行，順序永不變動。沒有值就印缺值，不跳過 —— 一致性比簡潔重要，
# 版面固定才建立得起直覺。(編號, panel 欄位, 顯示名, calibration key)
TWELVE = [
    ("C01", "壓縮整理",   "tightness_15d_pct",               "c01_tight",           "c01_tight"),
    ("C02", "阻力消耗",   "prior_high_tests_count",          "c02_tests",           "c02_tests"),
    ("C03", "突破推力",   "close_quality",                   "c03_cq",              "c03_cq"),
    ("C04", "結構防守",   "shakeout_depth_pct",              "shake_depth",         None),
    ("C05", "量價換手",   "high_zone_turnover_share_20d",    "c05_high_zone",       None),
    ("C06", "相對強弱",   "rs_excess_taiex_pct",             "c06_rs",              "c06_rs"),
    ("C07", "推進效率",   "price_progress_efficiency_20d",   "c07_er",              "c07_er"),
    ("C08", "趨勢年齡",   "major_breakout_count_250d",       "c08_bo_count",        "c08_bo_count"),
    ("C09", "資訊反應",   "resilience_excess_ret",           "c09_excess",          None),
    ("C10", "事後認可",   "mfe_mae_ratio_3d",                "c10_ratio",           None),
    ("C11", "壓力修復",   "bars_to_recover_largest_down_day", "c11_bars_recover",   None),
    ("C12", "動能老化",   "efficiency_decay",                "c12_decay",           None),
]

# 手冊把這些列為「配套欄位」，不是 C 欄本身，所以分開列，避免跟主欄混淆。
COMPANION = [
    ("C01 配套", "base_depth_pct",        "c01_base_depth",       "c01_base_depth"),
    ("C05 配套", "突破量比",               "c05_volratio",         "c05_volratio"),
    ("C05 配套", "量縮比",                 "c05_contract",         "c05_contract"),
    ("C06 配套", "20 日 RS 超額",          "c06_rs20",             "c06_rs20"),
    ("C08 配套", "自底漲幅 %",             "c08_gain_from_low",    "c08_gain_from_low"),
    ("C12 配套", "距最近新高日數",          "c12_days_since_high",  "c12_days_since_high"),
]

reg_g = None
trans_g = None
transdir_g = None
pct_g = {}   # stock_id -> {欄位_pct: 值}，as-of 當日全市場橫斷面


def fmt(v):
    if v is None:
        return "缺值"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return "缺值" if not np.isfinite(v) else f"{v:.2f}"
    return str(v)


def verdict(matched, br, row):
    """把命中的組寫成一句話。方向衝突要並列保留，絕不做多數決、絕不換算成分數。
    手冊第 26 頁：不要平均成一個 78 分，也不要用 4 項偏多 2 項偏空推出勝率。"""
    if not matched:
        return ("目前沒有任何 combo 完全命中——這是有效結果，代表 65 組沒有描述到現在的狀態，"
                "不是偏空也不是偏多。照關鍵價位與三分支操作即可。")
    bull = [m for m in matched if m.get("dir") == "多"]
    bear = [m for m in matched if m.get("dir") == "空"]
    neut = [m for m in matched if m.get("dir") == "中性"]
    head = matched[0]
    parts = [f"主狀態是 **{head['id']} {head['name']}**（{head['bias']}）。"]

    if bull and bear:
        b = "、".join(f"{m['id']} {m['name']}" for m in bull[:2])
        s_ = "、".join(f"{m['id']} {m['name']}" for m in bear[:2])
        parts.append(f"同時出現偏多的 {b} 與偏空的 {s_}——**兩邊並列，不相抵**。"
                     f"寫法是「向上的部分（{bull[0]['name']}）仍在，"
                     f"但{bear[0]['name']}，延續待確認」，"
                     f"不是「幾項偏多幾項偏空所以幾分」。")
    elif bull and not bear:
        parts.append(f"命中的 {len(bull)} 組方向一致偏多，沒有偏空組同時成立。"
                     f"方向一致不等於機率高於五成，只代表目前沒有互相矛盾的證據。")
    elif bear and not bull:
        parts.append(f"命中的 {len(bear)} 組方向一致偏空，沒有偏多組同時成立。"
                     f"同樣不代表下跌機率，只代表證據方向一致。")
    elif neut:
        parts.append("命中的組都是中性或狀態描述，沒有方向主張。")

    if len(matched) >= 3:
        parts.append(f"另有 {len(matched) - 1} 組同時成立——注意手冊 combo 56 的提醒："
                     f"收高、量大、RS 正這類欄位可能只是在重複描述同一天，不是幾張獨立選票。")
    parts.append(f"下一步確認：{head['next']}　／　失效條件：{head['fail']}")
    return "".join(f"{x}\n\n" for x in parts).strip()


def card_md(sid, name, row, results, br, state_actions, chart_rel, ev_note,
            stale_days=0, asof_str=""):
    global reg_g, trans_g, transdir_g
    matched = [r for r in results if r["status"] == "MATCH"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    L = []
    L.append(f"# {sid} {name}　{row['date'].strftime('%Y-%m-%d')} 收盤後\n")
    if stale_days:
        L.append(f"> ⚠️ **資料過期**：這是 {row['date'].strftime('%Y-%m-%d')} 的狀態，"
                 f"落後 as-of {asof_str} 共 {stale_days} 個交易日。"
                 f"分支價位不可當作明日依據，先查停牌／下市／資料缺漏。\n")
    L.append(f"![chart]({chart_rel})\n")

    L.append("## 12 項指標（欄名同速查手冊）")
    cal = (reg_g or {}).get("calibration", {}).get("fields", {})

    def refstr(ck):
        if not ck or ck not in cal:
            return ""
        return "；".join(f"{t}→P{d['pct']:.0f}"
                         for t, d in list(cal[ck]["thresholds"].items())[:3])

    def pctstr(col):
        pv = pct_g.get(sid, {}).get(col + "_pct", np.nan)
        try:
            return f"P{float(pv):.0f}" if np.isfinite(float(pv)) else "—"
        except (TypeError, ValueError):
            return "—"

    L.append("| # | 概念 | 手冊欄位 | 原值 | 同日分位 | 手冊線→實際百分位 |")
    L.append("|---|---|---|---|---|---|")
    for no, concept, field, col, ck in TWELVE:
        v = row.get(col, np.nan)
        note = ""
        if col == "c05_high_zone":
            note = " ⛔停用"
        if col in ("c10_ratio", "c09_excess"):
            note = " ⛔需事件/T+3"
        L.append(f"| {no} | {concept} | `{field}`{note} | {fmt(v)} | "
                 f"{pctstr(col)} | {refstr(ck)} |")
    L.append("")
    L.append("| 配套欄位 | 名稱 | 原值 | 同日分位 |\n|---|---|---|---|")
    for tag, nm, col, ck in COMPANION:
        L.append(f"| {tag} | {nm} | {fmt(row.get(col, np.nan))} | {pctstr(col)} |")
    L.append("")

    L.append("## 關鍵價位")
    L.append("| 項目 | 值 |\n|---|---|")
    L.append(f"| 收盤 | {fmt(float(row['close']))} |")
    L.append(f"| 突破線 B | {fmt(br['B'])} |")
    L.append(f"| 箱底 L | {fmt(br['L'])} |")
    L.append(f"| 最近回檔低點 | {fmt(br['down_line'])} |")
    L.append(f"| MA20 | {fmt(br['ma20'])} |")
    L.append(f"| ATR({14}) | {fmt(br['atr'])} → 明日常態區 {fmt(br['normal_lo'])}~{fmt(br['normal_hi'])} |\n")

    L.append("## 明日三分支")
    L.append("| 分支 | 條件 | 空手 | 持有 |\n|---|---|---|---|")
    sp, hd = state_actions["空手"], state_actions["持有_獲利"]
    for i, (tag, cond) in enumerate([("甲", f"收 > {br['up_line']}（{br['up_src']}）"),
                                     ("乙", f"收 {br['down_line']}~{br['up_line']}"),
                                     ("丙", f"收 < {br['down_line']}（{br['down_src']}）")]):
        L.append(f"| {tag} | {cond} | {sp[i].split('：',1)[1]} | {hd[i].split('：',1)[1]} |")
    L.append(f"\n**作廢條件**：開盤跳空超出 ±{fmt(br['gap_invalidate'])} → 全部分支失效，當日重算\n")

    if matched and trans_g is not None:
        pid = matched[0]["id"]
        t3 = trans_g[trans_g["primary"] == pid].nlargest(3, "n")
        dsum = transdir_g[transdir_g["primary"] == pid] if transdir_g is not None else None
        if len(t3):
            L.append(f"## 這個狀態之後，歷史上最常變成什麼")
            L.append(f"_主狀態 {pid} 的 episode 結束後，下一段是什麼。"
                     f"2021 年起全市場統計，不是預測。_\n")
            L.append("| 下一個狀態 | 占比 |\n|---|---|")
            for _, r_ in t3.iterrows():
                L.append(f"| {r_['next']} {r_.get('next_name','')} | {r_['占比%']}% |")
            if dsum is not None and len(dsum):
                dd = "、".join(f"{r_['dir']} {r_['占比%']}%"
                              for _, r_ in dsum.sort_values("占比%", ascending=False).iterrows())
                L.append(f"\n依方向彙總：{dd}\n")
            else:
                L.append("")

    L.append("## 綜合判讀")
    L.append(verdict(matched, br, row) + "\n")

    L.append("## 命中的 combo")
    if not matched:
        L.append("_無完全命中。以下 PARTIAL 供參考。_\n")
    for r in matched:
        L.append(f"### {r['id']} {r['name']}　（{r['cls']} 類）— {r['bias']}")
        L.append("| 條件 | 實際值 | |\n|---|---|---|")
        for c in r["conditions"]:
            vals = "、".join(f"{k}={fmt(v)}" for k, v in c["vals"].items()) or "—"
            L.append(f"| `{c['expr']}` | {vals} | {MARK[ 'MATCH' if c['ok'] else 'NO_MATCH' ] if c['ok'] is not None else '—'} |")
        L.append(f"\n- 接著確認｜{r['next']}")
        L.append(f"- 改判／失效｜{r['fail']}\n")

    if partial:
        L.append("## PARTIAL（有資料缺漏，不算不符）")
        L.append("| combo | 缺的條件 |\n|---|---|")
        for r in partial[:12]:
            miss = [c["expr"] for c in r["conditions"] if c["ok"] is None]
            L.append(f"| {r['id']} {r['name']} | {'；'.join(f'`{m}`' for m in miss[:3])} |")
        L.append("")

    L.append("## 資料狀態")
    stale = [k for k in ("c06_rs", "c05_volratio", "c01_tight", "c02_tests") if not np.isfinite(row.get(k, np.nan))]
    L.append(("COMPLETE — 全部欄位可用" if not stale
              else "PARTIAL — 缺值：" + "、".join(stale)
              + "。缺值不當偏空，相關 combo 已降級不參與判讀。") + "\n")
    L.append("## 事件")
    L.append(ev_note + "\n")
    L.append("---\n_本卡為狀態描述，無勝率估計。動作皆附價位；未附價位者代表定義未凍結，已略去。_")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_defense.csv")
    ap.add_argument("--ids", default="",
                    help="直接給代號，逗號分隔，例如 --ids 2330,2454,3008。給了就不讀 universe")
    ap.add_argument("--from-signals", default="",
                    help="從 simple_momentum 的 SIGNALS.md 或任何含代號的文字檔抓四碼代號")
    ap.add_argument("--adj-dir", default="data/adj")
    ap.add_argument("--index", default="data/futures/index_taiex.parquet")
    ap.add_argument("--events", default="data/events/events.csv")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，預設用資料最後一天")
    ap.add_argument("--outdir", default="cards")
    ap.add_argument("--positions", default="data/simple_positions.csv",
                    help="持倉檔。自動把持有中的標成對應 state，動作表才會給對的價位")
    ap.add_argument("--limit", type=int, default=40,
                    help="最多產幾張完整卡片（含圖）。超過的只進總表，不出圖")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--bars", type=int, default=280,
                    help="標註圖回看根數。預設 280（要看得到 250 日前高 B 的來源）")
    ap.add_argument("--years", default=None, help="逗號分隔；預設近三年")
    args = ap.parse_args()

    global reg_g, trans_g, transdir_g
    reg = cc.load_registry(args.registry)
    reg_g = reg
    meta_all = {c["id"]: c for c in reg["combos"]}
    tp = "data/states/transitions.parquet"
    if os.path.exists(tp):
        trans_g = pd.read_parquet(tp)
        trans_g["next_name"] = [meta_all.get(c, {}).get("name", "") for c in trans_g["next"]]
        dp = "data/states/transitions_dir.parquet"
        if os.path.exists(dp):
            transdir_g = pd.read_parquet(dp)
    else:
        print("[WARN] 找不到 transitions.parquet，卡片不會有「之後最常變成什麼」。"
              "先跑 build_states.py --report")
    names = {}
    if os.path.exists("data/universe.parquet"):
        _u = pd.read_parquet("data/universe.parquet")
        names = dict(zip(_u["stock_id"].astype(str), _u["stock_name"].astype(str)))

    # 持倉狀態：多頭時原始訊號可能 40~60 檔，其中有些是你已經持有的。
    # 不接這個檔，卡片全部標「空手」，動作表的「持有」三格就永遠用不到。
    held = {}
    if args.positions and os.path.exists(args.positions):
        try:
            pdf = pd.read_csv(args.positions, dtype={"代號": str},
                              engine="python", on_bad_lines="warn")
            for _, r in pdf.iterrows():
                if str(r.get("狀態", "")) not in ("持有", "待進場"):
                    continue
                sid = str(r["代號"]).strip()
                ent = pd.to_numeric(r.get("進場價"), errors="coerce")
                stp = pd.to_numeric(r.get("停損價"), errors="coerce")
                held[sid] = {"entry": ent, "stop": stp,
                             "status": str(r.get("狀態", ""))}
            print(f"持倉檔：{len(held)} 檔（持有或待進場）")
        except Exception as e:                                   # noqa: BLE001
            print(f"[WARN] 讀不到持倉檔 {args.positions}：{e}")

    if args.ids or args.from_signals:
        ids = []
        if args.ids:
            ids += [x.strip() for x in args.ids.replace("，", ",").split(",") if x.strip()]
        if args.from_signals:
            import re
            txt = open(args.from_signals, encoding="utf-8").read()
            # 先抓「每行第一個 token 是四碼數字」的情況 —— 這是訊號檔的標準長相。
            # ⚠️ 不能直接用寬鬆正則掃全文：那會把價格「2465.00」的 2465
            #    當成股票代號撈進來（實測踩過，憑空多出麗臺與騰雲兩檔）。
            strict = re.findall(r"(?m)^\s*([0-9]{4})(?![0-9A-Za-z.])", txt)
            if strict:
                ids += strict
            else:
                # 退路：整段文字裡撈四碼，但排除小數點前後的數字
                ids += re.findall(
                    r"(?<![0-9A-Za-z.])([0-9]{4})(?![0-9A-Za-z.])", txt)
        seen, clean = set(), []
        for i in ids:
            if len(i) == 4 and i not in seen:
                seen.add(i)
                clean.append(i)
        if not clean:
            sys.exit("沒抓到任何四碼代號")
        uni = pd.DataFrame({"stock_id": clean,
                            "name": [names.get(i, "") for i in clean],
                            "state": "空手"})
        src_label = ("手動指定" if args.ids
                     else os.path.splitext(os.path.basename(args.from_signals))[0])
        print(f"母體來自 {'--ids' if args.ids else args.from_signals}：{len(clean)} 檔")
    else:
        src_label = os.path.splitext(os.path.basename(args.universe))[0]
        uni = pd.read_csv(args.universe, dtype=str)
        if "stock_id" not in uni.columns:
            sys.exit("universe 檔需要 stock_id 欄（可另含 name、thesis_date、state）")
        if "name" not in uni.columns:
            uni["name"] = [names.get(str(s), "") for s in uni["stock_id"]]

    years = ([int(y) for y in args.years.split(",")] if args.years
             else list(range(pd.Timestamp.today().year - 2, pd.Timestamp.today().year + 1)))
    idx = cc.load_index(args.index)
    ev = cc.load_events(args.events)

    # 全域 as-of：預設用資料最後一天，所有卡片寫在同一個資料夾
    asof = pd.Timestamp(args.date) if args.date else None
    if asof is None:
        last = []
        for y in years[-1:]:
            p = os.path.join(args.adj_dir, f"prices_adj_{y}.parquet")
            if os.path.exists(p):
                dd = pd.read_parquet(p, columns=["date"])
                last.append(pd.to_datetime(dd["date"], format="mixed").max())
                del dd
        asof = max(last) if last else pd.Timestamp.today().normalize()
    out_root = os.path.join(args.outdir, asof.strftime("%Y-%m-%d"))
    os.makedirs(out_root, exist_ok=True)
    print(f"as-of {asof.date()}")

    # 同日全市場橫斷面分位：直接讀 panel 當天那一橫切，不自己重算
    global pct_g
    fp = os.path.join(args.features, f"features_{asof.year}.parquet")
    if os.path.exists(fp):
        fd = pd.read_parquet(fp)
        fd["date"] = pd.to_datetime(fd["date"])
        fd = fd[fd["date"] == asof]
        pc = [c for c in fd.columns if c.endswith("_pct")]
        if len(fd) and pc:
            pct_g = {str(r["stock_id"]): {c: r[c] for c in pc}
                     for _, r in fd[["stock_id"] + pc].iterrows()}
            print(f"分位基準：{asof.date()} 全市場 {len(fd)} 檔")
        del fd
    else:
        print(f"[WARN] 找不到 {fp}，12 項指標的分位欄會是空的。先跑 build_features.py")
    index_rows = []

    for _, u in uni.iterrows():
        sid, name = str(u["stock_id"]), str(u.get("name", ""))
        state = str(u.get("state", "空手")) or "空手"
        thesis = pd.to_datetime(u.get("thesis_date"), format="mixed", errors="coerce")

        px = cc.load_prices(args.adj_dir, [sid], years)
        if px.empty or len(px) < 260:
            print(f"[SKIP] {sid} 資料不足（{len(px)} 列）")
            continue

        feat = cc.compute_features(px, idx, reg)
        feat = cc.attach_events(feat, ev, sid)

        feat = feat[feat["date"] <= asof]
        if pd.notna(thesis):
            # 論點成立日之前的價格不進入判讀（但保留在時間序列裡供指標計算）
            feat.loc[feat["date"] < thesis, "_before_thesis"] = True
        if feat.empty:
            continue

        row = feat.iloc[-1]
        if not np.isfinite(row.get("idx_close", np.nan)):
            print(f"[WARN] {sid} {row['date'].date()} 大盤指數缺值 → D 類（相對強弱）整類降級")
        if pd.notna(thesis) and row["date"] < thesis:
            print(f"[SKIP] {sid} 尚未到論點成立日 {thesis.date()}")
            continue

        # C10/C12 未完成窗一律清空，防前視
        for k in ("c10_mae3", "c10_mfe3"):
            row[k] = np.nan

        # 身份由「現價 vs 進場價／停損價」決定，不由使用者填。
        # 接近停損 = 距停損 3% 以內，那時只剩「依原計畫執行」與「出場」兩個選項。
        hp = held.get(sid)
        if hp and hp["status"] == "持有" and np.isfinite(hp.get("entry", np.nan)):
            cl, ent, stp = float(row["close"]), float(hp["entry"]), hp.get("stop")
            if np.isfinite(stp) and stp > 0 and cl <= stp * 1.03:
                state = "持有_近停損"
            elif cl >= ent:
                state = "持有_獲利"
            else:
                state = "持有_虧損"
            print(f"      持倉 {sid}：進場 {ent:.2f} 現價 {cl:.2f} → {state}")
        elif hp and hp["status"] == "待進場":
            state = "空手"

        results = cc.evaluate_all(row, reg)
        br = cc.branches(row, reg)
        acts = {s: cc.actions(br, reg, s) for s in ["空手", "持有_獲利", "持有_虧損", "持有_近停損"]}

        stale_days = int(np.busday_count(row["date"].date(), asof.date()))
        if stale_days > 0:
            print(f"[STALE] {sid} {name} 最後一根 {row['date'].date()}，"
                  f"落後 as-of {stale_days} 個交易日 → 卡片標記為過期，不可當今日狀態")

        matched = [r for r in results if r["status"] == "MATCH"]
        full = len(index_rows) < args.limit
        if full:
            png = os.path.join(out_root, f"{sid}.png")
            cc.render_chart(feat, sid, name, png, reg, bars=args.bars,
                            matched=matched)

        e = ev[(ev["stock_id"].astype(str) == sid)] if not ev.empty else ev
        if e.empty:
            ev_note = "無 event log 紀錄。F／I 類 combo 全部 UNAVAILABLE。"
        else:
            last = e.sort_values("pub_date").iloc[-1]
            ev_note = (f"最近事件 {last['pub_date'].date()} {last.get('category','')} "
                       f"（{last.get('verification_state','')}）")
            recent = e[e["pub_date"] >= row["date"] - pd.Timedelta(days=3)]
            if len(recent):
                ev_note += "　⚠️ 三日內有新事件，分支價位可信度下降"

        if full:
            md = card_md(sid, name, row, results, br, acts, f"{sid}.png", ev_note,
                         stale_days, asof.strftime("%Y-%m-%d"))
            with open(os.path.join(out_root, f"{sid}.md"), "w", encoding="utf-8") as f:
                f.write(md)

        index_rows.append({
            "代號": sid, "名稱": name, "收盤": round(float(row["close"]), 2),
            "主 combo": "／".join(m["id"] for m in matched[:3]) or "—",
            "傾向": matched[0]["bias"] if matched else "—",
            "甲 >": br["up_line"], "丙 <": br["down_line"], "部位": state,
            "資料": "OK" if not stale_days else f"過期 {stale_days} 日（{row['date'].date()}）",
        })
        print(f"[OK] {sid} {name} 命中 {len(matched)} 組"
              + ("" if full else "　（超過 --limit，只進總表不出卡）"))
        del px, feat

    if out_root and index_rows:
        t = pd.DataFrame(index_rows)
        # 每份名單各寫一份總表，檔名帶名單來源。
        # 原本全部寫成 index.md，一天跑四份名單就只剩最後一份的總表 ——
        # 那會讓「天天跑、事後回頭對照」變成不可能，等於毀掉觀察紀錄本身。
        idx_name = f"index_{src_label}.md"
        with open(os.path.join(out_root, idx_name), "w", encoding="utf-8") as f:
            f.write(f"# 盤後看盤卡 {os.path.basename(out_root)}"
                    f"　｜　名單：{src_label}（{len(index_rows)} 檔）\n\n")
            f.write(t.to_markdown(index=False))
            n_stale = int((t["資料"] != "OK").sum())
            if n_stale:
                f.write(f"\n\n⚠️ {n_stale} 檔資料過期，其分支價位不可用。\n")
            f.write("\n_狀態描述，非買賣指令，無勝率估計。_\n")
        # 把名單來源寫進檔案，workflow 拿它當信件主旨
        with open(os.path.join(out_root, "_source.txt"), "w", encoding="utf-8") as f:
            f.write(src_label)

        # 當日所有名單的目錄，後跑的會把先跑的一起列進來，不互相覆蓋
        import glob as _glob
        allidx = sorted(_glob.glob(os.path.join(out_root, "index_*.md")))
        with open(os.path.join(out_root, "index.md"), "w", encoding="utf-8") as f:
            f.write(f"# 盤後看盤卡 {os.path.basename(out_root)}\n\n")
            f.write(f"今天共 {len(allidx)} 份名單：\n\n")
            for a in allidx:
                nm = os.path.basename(a)[6:-3]
                first = ""
                for line in open(a, encoding="utf-8"):
                    if line.startswith("# "):
                        first = line[2:].strip()
                        break
                f.write(f"- [{nm}]({os.path.basename(a)})　{first}\n")
        print(f"\n寫入 {out_root}/{idx_name}　（名單：{src_label}）")


if __name__ == "__main__":
    main()
