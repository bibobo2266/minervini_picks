"""
還原股價回補（Backer 版）：建立 data/adj/prices_adj_YYYY.parquet

需要 FinMind Backer 以上。跑完可退訂——這是一次性回補，之後的每日增量
繼續走 TWSE/TPEx OpenAPI（免費）。

為什麼按「日期」抓而不是按「代號」抓：
  按代號抓要先有一份代號清單，而 TaiwanStockInfo 給的是「今天還活著」的股票。
  拿今天的名單回頭抓 2015 年，等於預先知道哪些公司不會倒——倖存者偏差，
  而且偏多少你不知道。按日期抓拿的是「那天實際在交易的所有股票」，
  自然包含後來下市的。這個偏差只有 Backer 的批次端點解得掉。

為什麼從 2015-06-01 起：
  台股 2015 年 6 月把漲跌幅從 7% 放寬到 10%，這對本系統是實質斷點——
  「爆量最大跌勢」在 7% 時代上限就是 -7%，底部計數的 12% 回檔門檻要跌兩天
  才觸發，VCP 的振幅天花板也不同。跨過這條線的資料不可直接比較。
  真要往前拿更早的資料，統計時必須把 2015-06 前後分開。

分年存檔：單檔十年約 70MB，GitHub 超過 50MB 會警告、100MB 直接擋。
可中斷續跑：已完成的日期記在 data/adj/_done_dates.txt。

用法：
    FINMIND_TOKEN=xxx python scripts/backfill_adj.py
    FINMIND_TOKEN=xxx python scripts/backfill_adj.py --start 2015-06-01 --limit 5
"""
import argparse
import datetime as dt
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR = "data/adj"
DONE = os.path.join(DATA_DIR, "_done_dates.txt")
SLEEP = 2.3          # Backer 1600/hr → 2.25 秒/次，留一點餘裕
FLUSH = 120          # 每 N 天寫一次盤
COLS = ["date", "stock_id", "open", "max", "min", "close",
        "Trading_Volume", "Trading_money"]


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def call(dataset: str, **params) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token()}"}
    payload = {"dataset": dataset, **params}
    for _ in range(4):
        try:
            r = requests.get(API, params=payload, headers=headers, timeout=120)
        except requests.RequestException as e:
            print(f"    連線錯誤 {e}，30 秒後重試")
            time.sleep(30)
            continue
        if r.status_code == 402:
            print("    額度用盡，等待 10 分鐘")
            time.sleep(600)
            continue
        if r.status_code == 400:
            print(f"    HTTP 400：{r.text[:200]}")
            return pd.DataFrame()
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}，20 秒後重試")
            time.sleep(20)
            continue
        js = r.json()
        msg = str(js.get("msg", ""))
        if js.get("status") not in (200, None):
            if "requests" in msg.lower() or "limit" in msg.lower():
                print(f"    {msg}，等待 10 分鐘")
                time.sleep(600)
                continue
            print(f"    FinMind 拒絕：{msg}")
            return pd.DataFrame()
        return pd.DataFrame(js.get("data", []))
    return pd.DataFrame()


def trading_days(start: str) -> list:
    """交易日曆。抓不到就退回用工作日——多打的日子會回空資料，
    成本只是多幾次呼叫。"""
    df = call("TaiwanStockTradingDate", start_date=start)
    if not df.empty and "date" in df.columns:
        today = dt.date.today().isoformat()
        return sorted(d for d in df["date"].astype(str) if start <= d <= today)
    print("  取不到交易日曆，改用工作日")
    return [d.date().isoformat()
            for d in pd.bdate_range(start, dt.date.today())]


def load_done() -> set:
    if not os.path.exists(DONE):
        return set()
    with open(DONE, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def flush(buf, done_new):
    if not buf and not done_new:
        return
    if buf:
        df = pd.concat(buf, ignore_index=True)
        df["year"] = df["date"].astype(str).str.slice(0, 4)
        for yr, g in df.groupby("year"):
            path = os.path.join(DATA_DIR, f"prices_adj_{yr}.parquet")
            g = g.drop(columns="year")
            if os.path.exists(path):
                g = pd.concat([pd.read_parquet(path), g], ignore_index=True)
            g = g.drop_duplicates(subset=["stock_id", "date"], keep="last")
            g = g.sort_values(["stock_id", "date"]).reset_index(drop=True)
            g.to_parquet(path, index=False, compression="zstd")
    with open(DONE, "a", encoding="utf-8") as f:
        for d in done_new:
            f.write(d + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-06-01",
                    help="預設 2015-06-01：台股漲跌幅放寬到 10%% 的起點")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 天（測試用）")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # 探測：Backer 才有「不帶 data_id、只給日期」的批次端點
    probe = call("TaiwanStockPriceAdj",
                 start_date="2026-08-28", end_date="2026-08-28")
    if probe.empty:
        sys.exit("\n>>> 批次端點取不到資料。確認訂閱是否生效（Backer 以上），"
                 "以及 token 是否為訂閱帳號的 token。")
    print(f"探測 2026-08-28：{len(probe)} 檔 · 欄位 {list(probe.columns)}")
    time.sleep(SLEEP)

    days = trading_days(args.start)
    done = load_done()
    todo = [d for d in days if d not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"交易日 {len(days)} 天，已完成 {len(done)}，本次 {len(todo)} 天"
          f"（預估 {len(todo) * SLEEP / 3600:.1f} 小時）")

    buf, done_new, ok, empty = [], [], 0, 0
    for i, d in enumerate(todo, 1):
        df = call("TaiwanStockPriceAdj", start_date=d, end_date=d)
        if df.empty or "close" not in df.columns:
            empty += 1
        else:
            df = df.reindex(columns=COLS)
            buf.append(df)
            ok += 1
        done_new.append(d)
        if i % 20 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {d} 有資料 {ok} 天 / 空 {empty} 天")

        if len(done_new) >= FLUSH:
            flush(buf, done_new)
            buf, done_new = [], []
            files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
            size = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in files)
            print(f"    -- 存檔，{len(files)} 個年份檔 / {size / 1e6:.1f} MB")

        if i < len(todo):
            time.sleep(SLEEP)

    flush(buf, done_new)
    print(f"\n完成：有資料 {ok} 天 / 空 {empty} 天")
    for f in sorted(x for x in os.listdir(DATA_DIR) if x.endswith(".parquet")):
        p = os.path.join(DATA_DIR, f)
        g = pd.read_parquet(p, columns=["stock_id"])
        print(f"  {f}  {os.path.getsize(p) / 1e6:5.1f} MB  "
              f"{len(g):>9,} 列  {g['stock_id'].nunique():>5} 檔")


if __name__ == "__main__":
    main()
