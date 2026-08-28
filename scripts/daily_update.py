"""
每日增量：用證交所 + 櫃買中心 OpenAPI 抓當日全市場，append 進 data/prices.parquet

兩支 API 都不用金鑰，各一次呼叫。都只有「最新一個交易日」，沒有日期參數。
所以歷史靠 backfill.py 建立，之後每天靠這支累積。

用法：
    python scripts/daily_update.py
    python scripts/daily_update.py --dry-run   # 只印欄位，不寫檔
"""
import argparse
import datetime as dt
import os
import sys

import pandas as pd
import requests

TWSE = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
OUT = "data/prices.parquet"
KEEP_DAYS = 800          # 只留最近 800 天，控制檔案大小
COLS = ["date", "stock_id", "open", "max", "min", "close",
        "Trading_Volume", "Trading_money"]


def num(s):
    """'1,234.5' / '--' / '' → float 或 NaN"""
    return pd.to_numeric(
        pd.Series(s).astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"--": None, "---": None, "": None, "null": None}),
        errors="coerce",
    )


def pick(df: pd.DataFrame, *names):
    """從多個可能的欄位名挑第一個存在的。API 改欄位名時只要在這裡加。"""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df))


def roc_to_iso(s: str) -> str:
    """民國日期 '1150730' → '2026-07-30'"""
    s = str(s).strip().replace("/", "")
    return f"{int(s[:3]) + 1911:04d}-{s[3:5]}-{s[5:7]}"


def fetch_twse() -> pd.DataFrame:
    r = requests.get(TWSE, timeout=60, headers={"accept": "application/json"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    out = pd.DataFrame({
        "stock_id": pick(df, "Code").astype(str).str.strip(),
        "open": num(pick(df, "OpeningPrice")),
        "max": num(pick(df, "HighestPrice")),
        "min": num(pick(df, "LowestPrice")),
        "close": num(pick(df, "ClosingPrice")),
        "Trading_Volume": num(pick(df, "TradeVolume")),
        "Trading_money": num(pick(df, "TradeValue")),
    })
    return out


def fetch_tpex() -> tuple:
    r = requests.get(TPEX, timeout=60, headers={"accept": "application/json"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df, None
    date = None
    dcol = pick(df, "Date", "date")
    if dcol.notna().any():
        try:
            date = roc_to_iso(dcol.dropna().iloc[0])
        except Exception:
            date = None
    out = pd.DataFrame({
        "stock_id": pick(df, "SecuritiesCompanyCode", "Code",
                         "CompanyCode").astype(str).str.strip(),
        "open": num(pick(df, "Open", "OpeningPrice")),
        "max": num(pick(df, "High", "HighestPrice")),
        "min": num(pick(df, "Low", "LowestPrice")),
        "close": num(pick(df, "Close", "ClosingPrice")),
        "Trading_Volume": num(pick(df, "TradingShares", "TradeVolume")),
        "Trading_money": num(pick(df, "TransactionAmount", "TradeValue")),
    })
    return out, date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tw = fetch_twse()
    tp, session_date = fetch_tpex()
    print(f"TWSE {len(tw)} 筆 / TPEx {len(tp)} 筆 / 交易日 {session_date}")

    if args.dry_run:
        print("\nTWSE 欄位：", list(pd.DataFrame(
            requests.get(TWSE, timeout=60).json()).columns))
        print("\nTPEx 欄位：", list(pd.DataFrame(
            requests.get(TPEX, timeout=60).json()).columns))
        return

    # TWSE 那支沒有日期欄，借用 TPEx 的交易日（兩市同時開收盤）
    if not session_date:
        session_date = dt.date.today().isoformat()
        print(f"警告：TPEx 沒給日期，改用今天 {session_date}")

    df = pd.concat([tw, tp], ignore_index=True)
    df["date"] = session_date
    df = df[df["stock_id"].str.match(r"^[1-9]\d{3}$")]      # 只留普通股
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.reindex(columns=COLS)
    print(f"清理後 {len(df)} 檔")

    if len(df) < 800:
        sys.exit(f"筆數異常偏少（{len(df)}），可能是休市或 API 改版，不寫入")

    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        if session_date in set(old["date"].astype(str)):
            print(f"{session_date} 已存在，跳過")
            return
        df = pd.concat([old, df], ignore_index=True)
    else:
        print("警告：找不到既有 parquet，本次會建立新檔（歷史從今天開始）")

    df["date"] = df["date"].astype(str)
    df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")
    cutoff = (dt.date.today() - dt.timedelta(days=KEEP_DAYS)).isoformat()
    df = df[df["date"] >= cutoff]
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"已寫入 {df['stock_id'].nunique()} 檔 / {len(df):,} 列 / "
          f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
