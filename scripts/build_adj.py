"""
用除權息表自己還原：產生 data/adj/prices_adj_YYYY.parquet

為什麼不用 FinMind 的 TaiwanStockPriceAdj：那個 dataset 免費層不給
（回 400 "Your level is register"），要 Backer $699/月。
但 TaiwanStockDividendResult 在免費層拿得到，而且它直接給
before_price / after_price，調整因子 = after/before，
不用自己從現金股利、股票股利、增資配股三種情況拆開算。

做法：
  1. 抓全市場除權息事件（先試不帶 data_id 一次拿完，失敗才逐檔）
  2. 每檔算累積調整因子：某日的因子 = 該日之後所有除權息的 after/before 連乘
  3. 把因子套到現有 prices.parquet 的 OHLC 上（量不調整）
  4. 分年存檔

用法：
    FINMIND_TOKEN=xxx python scripts/build_adj.py
    FINMIND_TOKEN=xxx python scripts/build_adj.py --start 2017-01-01
    FINMIND_TOKEN=xxx python scripts/build_adj.py --dividends-only   # 只抓除權息表
"""
import argparse
import datetime as dt
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
PRICES_URL = ("https://raw.githubusercontent.com/bibobo2266/minervini_picks"
              "/main/data/prices.parquet")
DATA_DIR = "data/adj"
DIV_PATH = os.path.join(DATA_DIR, "dividend_events.parquet")
SLEEP = 1.2
FLUSH = 200


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
            r = requests.get(API, params=payload, headers=headers, timeout=90)
        except requests.RequestException as e:
            print(f"    連線錯誤 {e}，30 秒後重試")
            time.sleep(30)
            continue
        if r.status_code == 402:
            print("    額度用盡，等待 10 分鐘")
            time.sleep(600)
            continue
        if r.status_code == 400:
            return pd.DataFrame()          # 權限或參數問題，重試沒意義
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
            return pd.DataFrame()
        return pd.DataFrame(js.get("data", []))
    return pd.DataFrame()


def fetch_dividends(start: str) -> pd.DataFrame:
    """先試不帶 data_id 一次拿全市場；不行才逐檔。
    除權息表資料量遠小於股價表，全市場限制可能跟股價不同——值得先試，
    成功的話 4 小時變 1 分鐘。"""
    print("試探：不帶 data_id 一次抓全市場除權息…")
    bulk = call("TaiwanStockDividendResult", start_date=start)
    if not bulk.empty and bulk["stock_id"].nunique() > 50:
        print(f"  成功！{len(bulk):,} 列 / {bulk['stock_id'].nunique()} 檔")
        return bulk
    print("  不支援，改逐檔抓")

    uni = call("TaiwanStockInfo")
    uni = uni[uni["type"].isin(["twse", "tpex"])]
    uni = uni[uni["stock_id"].str.match(r"^[1-9]\d{3}$")]
    ids = sorted(uni["stock_id"].unique())
    print(f"  母體 {len(ids)} 檔，預估 {len(ids) * SLEEP / 3600:.1f} 小時")

    parts, ok = [], 0
    for i, sid in enumerate(ids, 1):
        g = call("TaiwanStockDividendResult", data_id=sid, start_date=start)
        if not g.empty:
            parts.append(g)
            ok += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(ids)}] 有除權息紀錄 {ok} 檔")
        time.sleep(SLEEP)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_prices() -> pd.DataFrame:
    if os.path.exists("data/prices.parquet"):
        df = pd.read_parquet("data/prices.parquet")
    else:
        r = requests.get(PRICES_URL, timeout=180)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
    df["date"] = pd.to_datetime(df["date"])
    return df[df["close"] > 0]


def build_factors(div: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """每檔每日的累積調整因子。

    因子定義：某日的因子 = 該日「之後」所有除權息的 (after/before) 連乘。
    也就是把歷史價格往下調，最新價格因子為 1——這樣今天的收盤價不變，
    圖看起來跟券商軟體一致。
    """
    div = div.copy()
    div["date"] = pd.to_datetime(div["date"])
    div = div[(div["before_price"] > 0) & (div["after_price"] > 0)]
    div["ratio"] = div["after_price"] / div["before_price"]
    # 防呆：比值離 1 太遠多半是資料錯誤（正常除權息很少超過 ±40%）
    bad = (div["ratio"] < 0.5) | (div["ratio"] > 1.2)
    if bad.any():
        print(f"  跳過 {bad.sum()} 筆異常比值（<0.5 或 >1.2）")
        div = div[~bad]

    out = []
    for sid, g in prices.groupby("stock_id", sort=False):
        d = div[div["stock_id"] == sid]
        f = pd.Series(1.0, index=g["date"].values)
        if not d.empty:
            # 由近而遠累乘：除息日「當天及之後」不受影響，之前的要乘上比值
            for _, ev in d.sort_values("date", ascending=False).iterrows():
                f[f.index < ev["date"]] *= ev["ratio"]
        out.append(pd.DataFrame({"stock_id": sid, "date": f.index,
                                 "factor": f.values}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--dividends-only", action="store_true",
                    help="只抓除權息表，不做還原")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DIV_PATH):
        div = pd.read_parquet(DIV_PATH)
        print(f"沿用既有除權息表：{len(div):,} 列 / {div['stock_id'].nunique()} 檔")
    else:
        div = fetch_dividends(args.start)
        if div.empty:
            sys.exit("抓不到任何除權息資料")
        div.to_parquet(DIV_PATH, index=False, compression="zstd")
        print(f"除權息表已存：{len(div):,} 列 / {div['stock_id'].nunique()} 檔")

    if args.dividends_only:
        return

    prices = load_prices()
    print(f"行情：{len(prices):,} 列 / {prices['stock_id'].nunique()} 檔 · "
          f"{prices['date'].min():%Y-%m-%d} ~ {prices['date'].max():%Y-%m-%d}")

    fac = build_factors(div, prices)
    adj = prices.merge(fac, on=["stock_id", "date"], how="left")
    adj["factor"] = adj["factor"].fillna(1.0)
    for c in ["open", "max", "min", "close"]:
        adj[c] = (adj[c] * adj["factor"]).round(4)
    adj = adj.drop(columns="factor")

    n_adj = (fac["factor"] != 1.0).sum()
    print(f"實際被調整的列數：{n_adj:,} / {len(fac):,} "
          f"({n_adj / len(fac) * 100:.1f}%)")

    adj["year"] = adj["date"].dt.year.astype(str)
    for yr, g in adj.groupby("year"):
        p = os.path.join(DATA_DIR, f"prices_adj_{yr}.parquet")
        g.drop(columns="year").to_parquet(p, index=False, compression="zstd")
        print(f"  {os.path.basename(p)}  "
              f"{os.path.getsize(p) / 1e6:5.1f} MB  {len(g):>9,} 列")


if __name__ == "__main__":
    main()
