r"""
準備名單追蹤（Watchlist Tracker）

解決的問題：Minervini 掃描每天產出名單，但沒有東西決定「什麼時候該把一檔
從名單上拿掉」。不處理的話名單只會單向成長，最後變成看都不想看的長清單。

核心原則來自 thesis-tracker：**論點必須可證偽**。
一檔進榜時就把作廢條件寫死，之後每天只做機械檢查，不需要重新判斷。
判斷留給人，執行留給腳本。

只追蹤「準備」與「觸發」。「觀察」不進表——每天 80+ 檔，收進來三天就爆。

用法：
    python scripts/watchlist_update.py              # 每日更新
    python scripts/watchlist_update.py --dry-run    # 只看不寫
    python scripts/watchlist_update.py --stats      # 印移出原因統計
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_minervini import scan_today   # noqa: E402

WATCHLIST = "data/watchlist.csv"
HISTORY = "data/watchlist_history.csv"

# ---- 移出規則的參數。全部集中在這裡，改參數不用動邏輯 ----
STALE_WEEKS = 8        # 準備滿幾週仍未觸發就移出（逾期）
COOLDOWN_DAYS = 10     # 逾期移出後，幾個交易日內不得重新進榜
FAR_FROM_HIGH = -15.0  # 距 52 週高點低於這個百分比就移出
MIN_TT = 6             # TT 分低於此就移出（與 classify 的淘汰門檻一致）
MIN_RS = 70            # RS 低於此就移出（趨勢模板第 8 條）
LATE_BASE = 4          # 底部序達到此值且未觸發就移出
BREAK_FAIL = -8.0      # 觸發後跌破樞紐價這個百分比就移出

COLS = ["代號", "名稱", "產業", "進榜日", "進榜狀態", "進榜價", "進榜樞紐",
        "進榜底部序", "進榜RS", "進榜VCP", "停損價",
        "目前狀態", "最新價", "最新RS", "在榜天數", "觸發日",
        "移出日", "移出原因"]


def empty_wl() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def load_wl() -> pd.DataFrame:
    if os.path.exists(WATCHLIST):
        df = pd.read_csv(WATCHLIST, dtype={"代號": str})
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    return empty_wl()


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY):
        return pd.read_csv(HISTORY, dtype={"代號": str})
    return empty_wl()


def check_exit(row, cur, today: dt.date):
    """回傳 (要不要移出, 原因)。順序即優先序——先命中的先報。

    前六條都要「變壞」才移出。第七條逾期是唯一一條「沒變壞也移出」的，
    也是唯一能阻止名單單向成長的一條：一檔溫溫的股票可以在準備區躺半年
    不觸發也不變壞，那就是純佔名額。
    """
    if cur is None:
        return True, "掉出掃描母體"

    if int(cur["階段"]) != 2:
        return True, "階段破壞"
    if cur["MA30W"] == cur["MA30W"] and float(cur["收盤"]) < float(cur["MA30W"]):
        return True, "跌破30週線"
    if int(cur["TT分"]) < MIN_TT:
        return True, "模板失效"
    if float(cur["RS"]) < MIN_RS:
        return True, "相對轉弱"
    if float(cur["距高點"]) < FAR_FROM_HIGH:
        return True, "離高點太遠"

    triggered = bool(pd.notna(row["觸發日"])) or cur["狀態"] == "觸發"
    if int(cur["底部序"]) >= LATE_BASE and not triggered:
        return True, "晚期未觸發"

    # 觸發後跌破樞紐，型態失敗
    if triggered and row["進榜樞紐"] == row["進榜樞紐"]:
        piv = float(row["進榜樞紐"])
        if piv > 0 and (float(cur["收盤"]) / piv - 1) * 100 < BREAK_FAIL:
            return True, "突破失敗"

    # 逾期只適用於「從來沒觸發過」的
    if not triggered:
        age = (today - pd.to_datetime(row["進榜日"]).date()).days
        if age >= STALE_WEEKS * 7:
            return True, "逾期"

    return False, ""


def step(wl, hist, scan, today, verbose=True):
    """一天的名單推進。回傳 (新名單, 移出, 新增, 升級為觸發)。

    抽出來讓 watchlist_replay.py 逐日重放時呼叫同一份邏輯——回測與實跑
    共用一份判定，不然兩邊會慢慢分岔。
    """
    scan_idx = scan.set_index("代號")
    kept, removed, promoted = [], [], []

    # ---- 1. 對在榜的逐檔檢查 ----
    for _, row in wl.iterrows():
        sid = row["代號"]
        cur = scan_idx.loc[sid] if sid in scan_idx.index else None
        out, why = check_exit(row, cur, today)
        age = (today - pd.to_datetime(row["進榜日"]).date()).days
        r = row.copy()
        r["在榜天數"] = age
        if cur is not None:
            r["最新價"], r["最新RS"] = cur["收盤"], cur["RS"]
            if cur["狀態"] == "觸發" and pd.isna(r["觸發日"]):
                r["觸發日"] = today.isoformat()
                promoted.append(f"{sid} {r['名稱']}")
            r["目前狀態"] = cur["狀態"]
        if out:
            r["移出日"], r["移出原因"] = today.isoformat(), why
            removed.append(r.to_dict())
        else:
            kept.append(r.to_dict())

    kept_ids = {r["代號"] for r in kept}

    # ---- 2. 冷卻期：逾期移出的不能馬上回來 ----
    cool = {}
    past = [hist] if len(hist) else []
    if removed:
        past.append(pd.DataFrame(removed))
    if past:
        allh = pd.concat(past, ignore_index=True)
        for _, h in allh[allh["移出原因"] == "逾期"].iterrows():
            if pd.notna(h.get("移出日")):
                d = pd.to_datetime(h["移出日"]).date()
                cool[h["代號"]] = max(cool.get(h["代號"], d), d)

    # ---- 3. 新進榜 ----
    # 進榜必須通過跟移出同一套規則。不然會出現「今天移出、今天又收回來」的
    # 無限循環——例如底部序 4 的股票被判晚期未觸發移出，隔一行又被當新進榜收進來。
    removed_today = {r["代號"] for r in removed}
    added = []
    for _, c in scan[scan["狀態"].isin(["準備", "觸發"])].iterrows():
        sid = c["代號"]
        if sid in kept_ids or sid in removed_today:
            continue
        if sid in cool and (today - cool[sid]).days < COOLDOWN_DAYS * 7 / 5:
            if verbose:
                print(f"  冷卻中，不收：{sid} {c['名稱']}")
            continue
        probe = {"進榜日": today.isoformat(), "進榜樞紐": c["樞紐價"],
                 "觸發日": today.isoformat() if c["狀態"] == "觸發" else np.nan,
                 "名稱": c["名稱"]}
        bad, why = check_exit(pd.Series(probe), c, today)
        if bad:
            if verbose:
                print(f"  不符進榜條件（{why}）：{sid} {c['名稱']}")
            continue
        piv = c["樞紐價"]
        # 停損：樞紐下方 8%，沒有樞紐就用 30 週線
        stop = (round(float(piv) * (1 + BREAK_FAIL / 100), 2)
                if piv == piv and piv else c["MA30W"])
        added.append({
            "代號": sid, "名稱": c["名稱"], "產業": c["產業"],
            "進榜日": today.isoformat(), "進榜狀態": c["狀態"],
            "進榜價": c["收盤"], "進榜樞紐": piv, "進榜底部序": c["底部序"],
            "進榜RS": c["RS"], "進榜VCP": c["VCP"], "停損價": stop,
            "目前狀態": c["狀態"], "最新價": c["收盤"], "最新RS": c["RS"],
            "在榜天數": 0,
            "觸發日": today.isoformat() if c["狀態"] == "觸發" else np.nan,
            "移出日": np.nan, "移出原因": np.nan,
        })

    new_wl = pd.DataFrame(kept + added, columns=COLS) if (kept or added) \
        else empty_wl()
    new_wl = new_wl.sort_values(
        ["目前狀態", "在榜天數"], ascending=[True, False]).reset_index(drop=True)
    return new_wl, removed, added, promoted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liq", type=float, default=5000)
    ap.add_argument("--min-tt", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        h = load_history()
        if h.empty:
            print("還沒有移出紀錄")
            return
        print(f"累計移出 {len(h)} 筆\n")
        print("移出原因分佈：")
        print(h["移出原因"].value_counts().to_string())
        h["在榜天數"] = pd.to_numeric(h["在榜天數"], errors="coerce")
        print(f"\n平均在榜 {h['在榜天數'].mean():.0f} 天"
              f"（中位數 {h['在榜天數'].median():.0f}）")
        trig = h["觸發日"].notna().sum()
        print(f"曾經觸發 {trig} 檔 / {len(h)} 檔 = {trig / len(h):.0%}")
        print("\n讀法：逾期佔多數 → 進榜門檻太鬆；突破失敗佔多數 → 觸發判定有問題；"
              "階段破壞佔多數 → 進場時機太早。")
        return

    today = dt.date.today()
    os.makedirs("data", exist_ok=True)

    scan = scan_today(args.liq, args.min_tt)
    if scan.empty:
        print("掃描結果為空，不更新")
        return
    print(f"今日掃描 {len(scan)} 檔："
          + "、".join(f"{k} {v}" for k, v in scan["狀態"].value_counts().items()))

    wl = load_wl()
    hist = load_history()
    new_wl, removed, added, promoted = step(wl, hist, scan, today)

    # ---- 4. 報告 ----
    print(f"\n在榜 {len(new_wl)} 檔（保留 {len(new_wl) - len(added)}、新增 {len(added)}）")
    if added:
        print("新進榜：" + "、".join(f"{a['代號']} {a['名稱']}" for a in added))
    if promoted:
        print("升級為觸發：" + "、".join(promoted))
    if removed:
        rd = pd.DataFrame(removed)
        print(f"移出 {len(rd)} 檔：")
        for why, g in rd.groupby("移出原因"):
            print(f"  {why}：" + "、".join(f"{r['代號']} {r['名稱']}"
                                          for _, r in g.iterrows()))

    # 逾期預警：三天內會到期的，先講
    if not new_wl.empty:
        soon = new_wl[(new_wl["觸發日"].isna()) &
                      (new_wl["在榜天數"] >= STALE_WEEKS * 7 - 3)]
        if not soon.empty:
            print(f"\n三天內逾期：" + "、".join(
                f"{r['代號']} {r['名稱']}（第 {r['在榜天數']} 天）"
                for _, r in soon.iterrows()))

    if args.dry_run:
        print("\n--dry-run，未寫入")
        return

    new_wl.to_csv(WATCHLIST, index=False, encoding="utf-8-sig")
    if removed:
        pd.concat([hist, pd.DataFrame(removed, columns=COLS)],
                  ignore_index=True)[COLS] \
            .to_csv(HISTORY, index=False, encoding="utf-8-sig")
    print(f"\n已寫入 {WATCHLIST}"
          + (f" 與 {HISTORY}" if removed else ""))


if __name__ == "__main__":
    main()
