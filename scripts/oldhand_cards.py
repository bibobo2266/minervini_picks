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
    ("C01", "c01_tight", "壓縮 15d %", "c01_tight"),
    ("C02", "c02_tests", "前高測試次數", "c02_tests"),
    ("C03", "c03_cq", "收盤品質", "c03_cq"),
    ("C04", "shake_depth", "洗盤破底深度 %", None),
    ("C05", "c05_volratio", "突破量比", "c05_volratio"),
    ("C06", "c06_rs", "RS 超額（百分點）", "c06_rs"),
    ("C07", "c07_er", "推進效率 ER", "c07_er"),
    ("C08", "c08_bo_count", "250d 突破次數", "c08_bo_count"),
    ("C09", "c09_excess", "事件反應超額", None),
    ("C10", "c10_mae3", "三日 MAE %（T 時禁用）", None),
    ("C11", "c11_bars_recover", "壓力修復根數（-1=未收復）", None),
    ("C12", "c12_days_since_high", "距最近新高日數", "c12_days_since_high"),
]

reg_g = None
pct_g = {}   # stock_id -> {欄位_pct: 值}，as-of 當日全市場橫斷面


def fmt(v):
    if v is None:
        return "缺值"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return "缺值" if not np.isfinite(v) else f"{v:.2f}"
    return str(v)


def card_md(sid, name, row, results, br, state_actions, chart_rel, ev_note,
            stale_days=0, asof_str=""):
    global reg_g
    matched = [r for r in results if r["status"] == "MATCH"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    L = []
    L.append(f"# {sid} {name}　{row['date'].strftime('%Y-%m-%d')} 收盤後\n")
    if stale_days:
        L.append(f"> ⚠️ **資料過期**：這是 {row['date'].strftime('%Y-%m-%d')} 的狀態，"
                 f"落後 as-of {asof_str} 共 {stale_days} 個交易日。"
                 f"分支價位不可當作明日依據，先查停牌／下市／資料缺漏。\n")
    L.append(f"![chart]({chart_rel})\n")

    L.append("## 12 項指標")
    cal = (reg_g or {}).get("calibration", {}).get("fields", {})
    L.append("| # | 欄位 | 原值 | 同日分位 | 手冊線 |\n|---|---|---|---|---|")
    for no, key, label, refkey in TWELVE:
        v = row.get(key, np.nan) if key else np.nan
        pv = pct_g.get(sid, {}).get((key or "") + "_pct", np.nan)
        ref = ""
        if refkey and refkey in cal:
            ts = cal[refkey]["thresholds"]
            ref = "；".join(f"{t}→P{d['pct']:.0f}" for t, d in list(ts.items())[:3])
        L.append(f"| {no} | {label} | {fmt(v)} | "
                 f"{'—' if not np.isfinite(pv if isinstance(pv,float) else np.nan) else f'P{pv:.0f}'} | {ref} |")
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
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--bars", type=int, default=280,
                    help="標註圖回看根數。預設 280（要看得到 250 日前高 B 的來源）")
    ap.add_argument("--years", default=None, help="逗號分隔；預設近三年")
    args = ap.parse_args()

    global reg_g
    reg = cc.load_registry(args.registry)
    reg_g = reg
    names = {}
    if os.path.exists("data/universe.parquet"):
        _u = pd.read_parquet("data/universe.parquet")
        names = dict(zip(_u["stock_id"].astype(str), _u["stock_name"].astype(str)))

    if args.ids or args.from_signals:
        ids = []
        if args.ids:
            ids += [x.strip() for x in args.ids.replace("，", ",").split(",") if x.strip()]
        if args.from_signals:
            import re
            txt = open(args.from_signals, encoding="utf-8").read()
            # 抓行首或空白後的四碼數字，去重保序
            ids += re.findall(r"(?<![0-9A-Za-z])([0-9]{4})(?![0-9A-Za-z])", txt)
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
        print(f"母體來自 {'--ids' if args.ids else args.from_signals}：{len(clean)} 檔")
    else:
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

        results = cc.evaluate_all(row, reg)
        br = cc.branches(row, reg)
        acts = {s: cc.actions(br, reg, s) for s in ["空手", "持有_獲利", "持有_虧損", "持有_近停損"]}

        stale_days = int(np.busday_count(row["date"].date(), asof.date()))
        if stale_days > 0:
            print(f"[STALE] {sid} {name} 最後一根 {row['date'].date()}，"
                  f"落後 as-of {stale_days} 個交易日 → 卡片標記為過期，不可當今日狀態")

        matched = [r for r in results if r["status"] == "MATCH"]
        png = os.path.join(out_root, f"{sid}.png")
        cc.render_chart(feat, sid, name, png, reg, bars=args.bars, matched=matched)

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
        print(f"[OK] {sid} {name} 命中 {len(matched)} 組")
        del px, feat

    if out_root and index_rows:
        t = pd.DataFrame(index_rows)
        with open(os.path.join(out_root, "index.md"), "w", encoding="utf-8") as f:
            f.write(f"# 盤後看盤卡 {os.path.basename(out_root)}\n\n")
            f.write(t.to_markdown(index=False))
            n_stale = int((t["資料"] != "OK").sum())
            if n_stale:
                f.write(f"\n\n⚠️ {n_stale} 檔資料過期，其分支價位不可用。\n")
            f.write("\n_狀態描述，非買賣指令，無勝率估計。_\n")
        print(f"\n寫入 {out_root}/index.md")


if __name__ == "__main__":
    main()
