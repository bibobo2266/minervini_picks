"""
一次性回補：用 FinMind 逐檔抓 2 年日線，建立 data/prices.parquet

免費版限制 600 req/hr，所以每檔間隔 6.5 秒。約 2150 檔 → 約 4 小時。
可中斷續跑：已經在 parquet 裡的代號會自動跳過，失敗再按一次 Run workflow 即可。

用法：
    FINMIND_TOKEN=xxx python scripts/backfill.py
    FINMIND_TOKEN=xxx python scripts/backfill.py --years 3 --limit 50   # 小量試跑
"""
import argparse
import datetime as dt
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "prices.parquet")
UNI = os.path.join(DATA_DIR, "universe.parquet")
SLEEP = 6.5          # 600/hr 的安全間隔
CHECKPOINT = 50      # 每 N 檔存一次檔
COLS = ["date", "stock_id", "open", "max", "min", "close",
        "Trading_Volume", "Trading_money"]


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def call(dataset: str, **params) -> pd.DataFrame:
    """呼叫 FinMind，遇到額度用盡就等待重試。"""
    payload = {"dataset": dataset, "token": token(), **params}
    for attempt in range(5):
        try:
            r = requests.get(API, params=payload, timeout=60)
        except requests.RequestException as e:
            print(f"    連線錯誤 {e}，30 秒後重試")
            time.sleep(30)
            continue
        if r.status_code == 402:                      # 額度用盡
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
            print(f"    FinMind 回傳：{msg}")
            return pd.DataFrame()
        return pd.DataFrame(js.get("data", []))
    return pd.DataFrame()


def universe() -> pd.DataFrame:
    """上市 + 上櫃普通股（四碼、開頭非 0，排除權證/ETF/存託憑證）。"""
    info = call("TaiwanStockInfo")
    if info.empty:
        sys.exit("取不到 TaiwanStockInfo，請檢查 token")
    info = info[info["type"].isin(["twse", "tpex"])]
    info = info[info["stock_id"].str.match(r"^[1-9]\d{3}$")]
    info = info.drop_duplicates(subset=["stock_id"]).sort_values("stock_id")
    return info[["stock_id", "stock_name", "industry_category", "type"]]


def save(frames, existing):
    if not frames:
        return existing
    df = pd.concat([existing] + frames, ignore_index=True) if len(existing) \
        else pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    args = ap.parse_args()

    # 一切寫檔之前先確保目錄存在
    os.makedirs(DATA_DIR, exist_ok=True)

    start = (dt.date.today() - dt.timedelta(days=365 * args.years + 30)).isoformat()

    uni = universe()
    print(f"母體 {len(uni)} 檔")
    uni.to_parquet(UNI, index=False)

    existing = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    done = set(existing["stock_id"]) if len(existing) else set()
    if done:
        print(f"已有 {len(done)} 檔，續跑剩下的")

    todo = [s for s in uni["stock_id"] if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"本次要抓 {len(todo)} 檔，預估 {len(todo) * SLEEP / 60:.0f} 分鐘")

    buf, ok, fail = [], 0, 0
    for i, sid in enumerate(todo, 1):
        df = call("TaiwanStockPrice", data_id=sid, start_date=start)
        if df.empty or "close" not in df.columns:
            fail += 1
            print(f"[{i}/{len(todo)}] {sid} 無資料")
        else:
            df = df.reindex(columns=COLS)
            df["stock_id"] = sid
            buf.append(df)
            ok += 1
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {sid} ok={ok} fail={fail}")

        if len(buf) >= CHECKPOINT:
            existing = save(buf, existing)
            buf = []
            print(f"    -- 存檔，累計 {existing['stock_id'].nunique()} 檔 "
                  f"/ {len(existing):,} 列")

        if i < len(todo):
            time.sleep(SLEEP)

    existing = save(buf, existing)
    if len(existing):
        print(f"\n完成：{existing['stock_id'].nunique()} 檔 / {len(existing):,} 列")
        print(f"檔案大小 {os.path.getsize(OUT) / 1e6:.1f} MB")
    else:
        print("\n沒有抓到任何資料")


if __name__ == "__main__":
    main()
