r"""
精髓版動能掃描器（Simple Momentum）

十一年回測證明：複雜的選股模板輸給一條規則。這支就是那一條規則。

    進場：收盤創 250 個交易日新高（且前一日不是新高）→ 隔日開盤買
          同一檔 20 個交易日內不重複進場
    母體：當日成交值排名前 25% 的普通股
    出場：進場價 -8% 停損（盤中觸價；開盤已跌破則用開盤價）
          沒觸及停損就一直抱，上限 250 個交易日
    規模：每筆等權重

沒有均線排列、沒有 RS 門檻、沒有 VCP、沒有底部序。刻意的——那八個條件在
2016-2026 的實測是負貢獻（笨基準 +3.89 vs SEPA +2.40，11/11 年為正 vs 7/11）。

這支同時做兩件事：
  1. 掃出今天的新訊號
  2. 用 paper trade 追蹤已發出的訊號，讓你 forward test 幾週再決定要不要下真錢

狀態存在 data/simple_positions.csv，每天累積。刪掉它就從頭開始。

用法：
    python scripts/simple_momentum.py                 # 每日跑
    python scripts/simple_momentum.py --max-new 2     # 每天最多收 2 筆新訊號
    python scripts/simple_momentum.py --stats         # 只看累計統計
"""
import argparse
import datetime as dt
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minervini_core as M  # noqa: E402

HIGH_WIN = 250         # 創幾日新高才算訊號（250 個交易日 ≈ 52 週）
COOLDOWN = 20          # 同一檔幾日內不重複收
LIQ_PCT = 25.0         # 母體：成交值前幾 %
STOP = 0.08            # 停損幅度
MAX_HOLD = 250         # 持有上限（交易日）
MIN_HIST = 250         # 至少要有幾天歷史才納入

POS = "data/simple_positions.csv"
REPORT = "SIGNALS.md"

COLS = ["代號", "名稱", "產業", "訊號日", "進場日", "進場價", "停損價",
        "狀態", "出場日", "出場價", "出場原因", "報酬%", "持有日"]


def load_pos() -> pd.DataFrame:
    if not os.path.exists(POS):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(POS, dtype={"代號": str})
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan
    return d[COLS]


def matrices(years: int):
    m, _ = M.build_matrices(years=years)
    keep = [s for s in m["c"].columns
            if str(s).isdigit() and len(str(s)) == 4 and str(s)[0] != "0"]
    return {k: v[keep] for k, v in m.items() if hasattr(v, "columns")}


def todays_signals(m, today):
    """回傳今天創新高的股票清單。

    「前一日不是新高」這個條件很重要——沒有它，一檔股票連漲十天會產生
    十個訊號，而那十個其實是同一段行情。
    """
    c, mo = m["c"], m["mo"]
    if today not in c.index:
        return []
    i = list(c.index).index(today)
    if i < MIN_HIST:
        return []
    hi = c.iloc[max(0, i - HIGH_WIN + 1):i + 1].max()
    hi_prev = c.iloc[max(0, i - HIGH_WIN):i].max()
    px, prev = c.iloc[i], c.iloc[i - 1]

    avg = mo.iloc[max(0, i - 59):i + 1].mean()
    liq = avg >= avg.quantile(1 - LIQ_PCT / 100)
    hist = c.iloc[:i + 1].notna().sum() >= MIN_HIST

    new = (px >= hi - 1e-9) & (prev < hi_prev - 1e-9) & liq & hist & px.notna()
    return sorted(new[new].index.tolist())


def step(pos, m, today, max_new=0):
    """推進一天：先結算已有部位，再收今天的新訊號。

    順序不能反——先結算再收新單，否則同一天出場又進場的股票會亂掉。
    """
    c, o, l = m["c"], m["o"], m["l"]
    idx = list(c.index)
    i_today = idx.index(today)
    filled, exited, added = [], [], []

    # 1) 待進場 -> 用今天開盤成交
    for k, r in pos[pos["狀態"] == "待進場"].iterrows():
        sid = r["代號"]
        if sid not in c.columns:
            continue
        px = o.at[today, sid] if today in o.index else np.nan
        if not np.isfinite(px) or px <= 0:
            continue
        pos.at[k, "進場日"] = today.date().isoformat()
        pos.at[k, "進場價"] = round(float(px), 2)
        pos.at[k, "停損價"] = round(float(px) * (1 - STOP), 2)
        pos.at[k, "狀態"] = "持有"
        filled.append(f"{sid} {r['名稱']} @ {px:.2f}（停損 {px * (1 - STOP):.2f}）")

    # 2) 持有中 -> 檢查停損與持有上限
    for k, r in pos[pos["狀態"] == "持有"].iterrows():
        sid = r["代號"]
        if sid not in c.columns or not np.isfinite(r["停損價"]):
            continue
        i_in = idx.index(pd.Timestamp(r["進場日"]))
        held = i_today - i_in
        lo = l.at[today, sid] if today in l.index else np.nan
        op = o.at[today, sid] if today in o.index else np.nan
        sp = float(r["停損價"])
        out = None
        if np.isfinite(lo) and lo <= sp:
            # 跳空穿價時成交在開盤，不是停損價。假設停在停損價會高估績效。
            out = (min(sp, op) if np.isfinite(op) and op < sp else sp, "停損")
        elif held >= MAX_HOLD:
            cl = c.at[today, sid]
            if np.isfinite(cl):
                out = (float(cl), "持有上限")
        if out:
            px, why = out
            pos.at[k, "狀態"] = "已出場"
            pos.at[k, "出場日"] = today.date().isoformat()
            pos.at[k, "出場價"] = round(px, 2)
            pos.at[k, "出場原因"] = why
            pos.at[k, "報酬%"] = round((px / float(r["進場價"]) - 1) * 100, 1)
            pos.at[k, "持有日"] = held
            exited.append(f"{sid} {r['名稱']} {why} "
                          f"{(px / float(r['進場價']) - 1) * 100:+.1f}%（{held} 日）")

    # 3) 今天的新訊號
    sigs = todays_signals(m, today)
    live = set(pos[pos["狀態"].isin(["待進場", "持有"])]["代號"])
    recent = set()
    for _, r in pos.iterrows():
        if pd.notna(r["訊號日"]):
            d0 = pd.Timestamp(r["訊號日"])
            if d0 in idx and i_today - idx.index(d0) < COOLDOWN:
                recent.add(r["代號"])
    cands = [s for s in sigs if s not in live and s not in recent]

    if max_new and len(cands) > max_new:
        # 用日期當種子隨機取樣。刻意不排序後取前 N——任何排序都是在挑單，
        # 而回測顯示 edge 集中在極少數交易，主觀挑單會隨機丟掉獲利來源。
        seed = int(hashlib.md5(today.date().isoformat().encode()).hexdigest()[:8], 16)
        cands = sorted(np.random.default_rng(seed).choice(
            cands, size=max_new, replace=False).tolist())

    uni = M.load_universe()
    rows = []
    for sid in cands:
        nm = uni.loc[sid, "stock_name"] if sid in uni.index else sid
        ind = uni.loc[sid, "industry_category"] if sid in uni.index else ""
        rows.append({"代號": sid, "名稱": nm, "產業": ind,
                     "訊號日": today.date().isoformat(), "狀態": "待進場"})
        added.append(f"{sid} {nm}（{ind}）收 {c.at[today, sid]:.2f}")
    if rows:
        pos = pd.concat([pos, pd.DataFrame(rows)], ignore_index=True)
    return pos, filled, exited, added, len(sigs)


def summary(pos):
    done = pos[pos["狀態"] == "已出場"].copy()
    hold = pos[pos["狀態"] == "持有"]
    if done.empty:
        return (f"進行中 {len(hold)} 檔，尚無已結束的交易。"
                "回測顯示勝率只有 22%，前十筆全虧是完全正常的。")
    r = pd.to_numeric(done["報酬%"], errors="coerce").dropna()
    return (f"已結束 {len(r)} 筆　平均 {r.mean():+.1f}%　中位 {r.median():+.1f}%　"
            f"勝率 {(r > 0).mean() * 100:.0f}%　"
            f"最佳 {r.max():+.0f}%　最差 {r.min():+.0f}%　持有中 {len(hold)} 檔")


def report(pos, m, today, filled, exited, added, n_sig, max_new):
    c = m["c"]
    L = [f"# 動能訊號 {today.date()}", "",
         f"規則：創 {HIGH_WIN} 日新高 → 隔日開盤買 → -{STOP:.0%} 停損 → "
         f"抱到出場（上限 {MAX_HOLD} 日）｜母體：成交值前 {LIQ_PCT:.0f}%", ""]
    L += [f"今日全市場新高訊號 **{n_sig}** 檔"
          + (f"，隨機取 {max_new} 檔" if max_new else "，全數收錄"), ""]

    L += ["## 明天開盤要買", ""]
    L += ([f"- {a}" for a in added] if added else ["_無_"]) + [""]

    if filled:
        L += ["## 今天已成交（進場）", ""] + [f"- {f}" for f in filled] + [""]
    if exited:
        L += ["## 今天出場", ""] + [f"- {e}" for e in exited] + [""]

    hold = pos[pos["狀態"] == "持有"].copy()
    L += [f"## 持有中（{len(hold)} 檔）", ""]
    if hold.empty:
        L += ["_無_", ""]
    else:
        L += ["| 代號 | 名稱 | 進場日 | 進場價 | 停損價 | 現價 | 報酬 | 距停損 |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in hold.iterrows():
            sid = r["代號"]
            px = c.at[today, sid] if (today in c.index
                                      and sid in c.columns) else np.nan
            ent, sp = float(r["進場價"]), float(r["停損價"])
            if np.isfinite(px):
                L.append(f"| {sid} | {r['名稱']} | {r['進場日']} | {ent:.2f} | "
                         f"{sp:.2f} | {px:.2f} | {(px / ent - 1) * 100:+.1f}% | "
                         f"{(sp / px - 1) * 100:+.1f}% |")
        L.append("")

    L += ["## 累計", "", summary(pos), "",
          "---", "",
          "**執行規則（不要改）**：每筆等權重；不提前停利；不因為看起來不好就跳過。",
          "回測 4,086 筆中最好的 20 筆承載全部獲利，任何主觀篩選都在隨機丟掉獲利來源。",
          f"勝率約 22%，連續 5-8 次停損是常態。", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=0,
                    help="每天最多收幾筆新訊號（隨機取樣）。0 = 全收。"
                         "回測顯示每年 100 筆與全做的超額相同（+3.95），"
                         "所以限量不會損失 edge，只會降低部位數")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    pos = load_pos()
    if args.stats:
        print(summary(pos))
        return

    m = matrices(args.years)
    today = m["c"].index[-1]
    print(f"資料最新日 {today.date()}　母體 {m['c'].shape[1]} 檔")

    pos, filled, exited, added, n_sig = step(pos, m, today, args.max_new)

    os.makedirs("data", exist_ok=True)
    pos.to_csv(POS, index=False, encoding="utf-8-sig")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report(pos, m, today, filled, exited, added, n_sig, args.max_new))

    print(f"新訊號 {n_sig} 檔，收錄 {len(added)}；成交 {len(filled)}；"
          f"出場 {len(exited)}")
    print(summary(pos))
    print(f"已寫入 {POS} 與 {REPORT}")


if __name__ == "__main__":
    main()
