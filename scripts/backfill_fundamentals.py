r"""
一次性回補：月營收與季報全歷史 → data/fundamentals/

為什麼現在要做：
    TaiwanStockFinancialStatements 需要訂閱等級才撈得動全歷史，而訂閱快到期。
    跟上次搶官方還原股價一樣，這是時間視窗問題，跟「這個因子有沒有用」無關。
    先把資料存進 repo，之後要不要測都還來得及。

⚠️ 前視偏誤：FinMind 的 date 是「期別」，不是「公告日」
    這是這份資料最容易搞爛結論的地方，所以本程式在回補時就把 available_date
    算好寫進去。之後任何回測，一律用 available_date 過濾，不要用 date。

    月營收：證交法第 36 條，每月 10 日以前公告上月營運情形。
            → available_date = 次月 15 日（多留幾天緩衝）

    季報：Q1 → 5/15、Q2 → 8/14、Q3 → 11/14、年報 → 次年 3/31。
          → available_date 用上述法定期限
          ⚠️ 金控可延到 5/30、8/31、11/29，銀行票券業另有規定。本程式一律用
             一般公司的期限，所以金融股會有大約兩週的前視偏誤。要嚴謹的話，
             測因子時把金融股排除，或自己再往後推兩週。

⚠️ 台灣的 Q2 是半年報，不是單季
    第二季財報涵蓋 1–6 月整體，跟美股 10-Q 只涵蓋單季不同。
    所以「單季 EPS」必須自己相減算出來（Q2單季 = 半年報 − Q1），
    直接拿 FinMind 的數字當單季會錯得離譜。本程式只存原始值，不做這個換算，
    換算留給因子腳本，但這件事寫在這裡免得忘記。

⚠️ 財報重編
    FinMind 給的是「現在的」數字，不是當年公告的原始數字。被重編過的財報
    無法還原成當時所見。這個偏誤沒辦法從這個資料源修掉，只能在結論註明。

可中斷續跑：已完成的代號記在 _done_<dataset>.txt，失敗再按一次 Run 即可。

用法：
    FINMIND_TOKEN=xxx python scripts/backfill_fundamentals.py
    FINMIND_TOKEN=xxx python scripts/backfill_fundamentals.py --dataset month
    FINMIND_TOKEN=xxx python scripts/backfill_fundamentals.py --limit 20 --sleep 2
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
OUT_DIR = "data/fundamentals"
UNI = "data/universe.parquet"
START = "2014-01-01"          # 早於價格資料起點，讓 YoY 從第一年就算得出來
CHECKPOINT = 40

SETS = {
    "month": ("TaiwanStockMonthRevenue", "month_revenue.parquet"),
    "financial": ("TaiwanStockFinancialStatements", "financials.parquet"),
}


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def call(dataset: str, **params) -> pd.DataFrame:
    """呼叫 FinMind，遇到額度用盡就等待重試。"""
    payload = {"dataset": dataset, "token": token(), **params}
    for _ in range(5):
        try:
            r = requests.get(API, params=payload, timeout=90)
        except requests.RequestException as e:
            print(f"    連線錯誤 {e}，30 秒後重試")
            time.sleep(30)
            continue
        if r.status_code == 402:
            print("    額度用盡，等待 10 分鐘")
            time.sleep(600)
            continue
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}，20 秒後重試")
            time.sleep(20)
            continue
        js = r.json()
        if js.get("status") not in (200, None):
            msg = js.get("msg", "")
            if "requests" in msg.lower() or "limit" in msg.lower():
                print(f"    {msg}，等待 10 分鐘")
                time.sleep(600)
                continue
            print(f"    API 回應 {msg}")
            return pd.DataFrame()
        return pd.DataFrame(js.get("data", []))
    return pd.DataFrame()


def month_available(d: pd.Series) -> pd.Series:
    """月營收：所屬月份 → 次月 15 日。法定是次月 10 日前，留幾天緩衝。"""
    d = pd.to_datetime(d)
    nxt = (d.dt.to_period("M") + 1).dt.to_timestamp()
    return nxt + pd.Timedelta(days=14)


def quarter_available(d: pd.Series) -> pd.Series:
    """季報：法定公告期限。⚠️ 金控更晚，這裡一律用一般公司的期限。"""
    d = pd.to_datetime(d)
    out = []
    for x in d:
        y, m = x.year, x.month
        if m <= 3:
            out.append(pd.Timestamp(y, 5, 15))      # Q1
        elif m <= 6:
            out.append(pd.Timestamp(y, 8, 14))      # Q2（半年報）
        elif m <= 9:
            out.append(pd.Timestamp(y, 11, 14))     # Q3
        else:
            out.append(pd.Timestamp(y + 1, 3, 31))  # Q4 / 年報
    return pd.Series(out, index=d.index)


def universe():
    if os.path.exists(UNI):
        u = pd.read_parquet(UNI)
    else:
        u = call("TaiwanStockInfo")
        if u.empty:
            sys.exit("取不到 TaiwanStockInfo，請檢查 token")
    u = u[u["type"].isin(["twse", "tpex"])]
    u = u[u["stock_id"].str.match(r"^[1-9]\d{3}$")]
    return sorted(u["stock_id"].drop_duplicates())


def save(frames, path, key):
    if not frames:
        return
    new = pd.concat(frames, ignore_index=True)
    if os.path.exists(path):
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=key, keep="last")
    new = new.sort_values(key).reset_index(drop=True)
    new.to_parquet(path, index=False, compression="zstd")


def run(kind, sids, sleep, done_path):
    dataset, fname = SETS[kind]
    path = os.path.join(OUT_DIR, fname)
    done = set()
    if os.path.exists(done_path):
        done = set(open(done_path).read().split())
    todo = [s for s in sids if s not in done]
    print(f"\n=== {dataset} ===")
    print(f"已完成 {len(done)} 檔，本次要抓 {len(todo)} 檔，"
          f"預估 {len(todo) * sleep / 60:.0f} 分鐘")

    key = ["stock_id", "date"] if kind == "month" else ["stock_id", "date", "type"]
    buf, ok, fail = [], 0, 0
    for i, sid in enumerate(todo, 1):
        df = call(dataset, data_id=sid, start_date=START)
        if df.empty or "date" not in df.columns:
            fail += 1
        else:
            df["stock_id"] = sid
            df["available_date"] = (month_available(df["date"]) if kind == "month"
                                    else quarter_available(df["date"]))
            df["available_date"] = df["available_date"].dt.strftime("%Y-%m-%d")
            buf.append(df)
            ok += 1
        with open(done_path, "a") as f:
            f.write(sid + "\n")
        if i % 20 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {sid} ok={ok} fail={fail}")
        if len(buf) >= CHECKPOINT:
            save(buf, path, key)
            buf = []
            print("    -- 存檔")
        if i < len(todo):
            time.sleep(sleep)

    save(buf, path, key)
    if os.path.exists(path):
        d = pd.read_parquet(path)
        print(f"完成：{d['stock_id'].nunique()} 檔 / {len(d):,} 列 → {path}")
        print(f"  期別範圍 {d['date'].min()} → {d['date'].max()}")
        if kind == "financial" and "type" in d.columns:
            top = d["type"].value_counts().head(8)
            print("  最常見的會計科目：" + "、".join(top.index.astype(str)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both",
                    choices=["both", "month", "financial"])
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="每檔間隔秒數。免費層 600 req/hr 要設 6.5")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sids = universe()
    if args.limit:
        sids = sids[: args.limit]
    print(f"母體 {len(sids)} 檔")

    kinds = ["month", "financial"] if args.dataset == "both" else [args.dataset]
    for k in kinds:
        run(k, sids, args.sleep, os.path.join(OUT_DIR, f"_done_{k}.txt"))

    print("\n⚠️ 之後做因子時：一律用 available_date 過濾，不要用 date。")
    print("⚠️ 台灣 Q2 是半年報，單季 EPS 要自己相減。")


if __name__ == "__main__":
    main()
