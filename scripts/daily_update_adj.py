r"""
每日增量（還原股價版）：更新 data/adj/prices_adj_YYYY.parquet

為什麼不能直接把交易所的收盤價 append 進去：
  TWSE/TPEx OpenAPI 給的是「未還原」價格。除息當天股價會跳空下跌一個
  股利的幅度——那不是真的跌，但 append 進還原序列後會製造假跌幅，
  污染 30 週均線、52 週低點、RS、以及「爆量最大跌勢」這個賣訊。

做法：
  1. 抓當日除權息（FinMind TaiwanStockDividendResult，免費層）
  2. 有除息的股票：把它在 data/adj/ 裡的「歷史」價格乘上 after/before，
     讓舊價格往下對齊新的價格水準
  3. 再 append 當日的未還原收盤價（它本身就是新水準的價格，不用調）

這樣做的等價說法：因子錨定在「最新價格」，歷史價格隨每次除息往下調。
跟券商軟體的還原線圖一致，今天的收盤數字永遠等於實際成交價。

母體用 data/universe.parquet 當白名單，不用 regex：
  舊版用 r"^[1-9]\d{3}$"（首位 1-9 且剛好四位），這條把 487 檔全部濾掉——
  所有 00 開頭的 ETF（0050、00878）、帶字母的（00981T、00679B）、
  六位數的（006208）。回補用的是 FinMind 批次端點沒有這條，所以歷史是完整的，
  是每日增量把它們切掉，導致那 487 檔從 2026-08-31 起靜默停止更新。
  universe 有 3,141 檔且不含權證（六位數只有 400 檔），拿來當白名單剛好。

當日資料已存在時會「補缺」而不是整批跳過：
  這樣上面那種漏檔可以靠重跑修復。補缺時不會重新套除權息，
  因為歷史價格在同一天的第一次執行就已經調過了，再調一次會變成雙重調整。

用法：
    FINMIND_TOKEN=xxx python scripts/daily_update_adj.py
    FINMIND_TOKEN=xxx python scripts/daily_update_adj.py --dry-run
"""
import argparse
import datetime as dt
import glob
import os
import sys

import pandas as pd
import requests

TWSE = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
FINMIND = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR = "data/adj"
UNIVERSE = "data/universe.parquet"
# universe.parquet 讀不到時的退路。刻意放寬到「4-6 位數字 + 可選一個大寫字母」，
# 寧可多收也不要再靜默漏掉 ETF；真有雜訊會在下面的母體檢查印出來。
FALLBACK_RE = r"^\d{4,6}[A-Z]?$"
COLS = ["date", "stock_id", "open", "max", "min", "close",
        "Trading_Volume", "Trading_money"]

# 除權息比值的合理範圍。超出就是資料異常（或減資之類的特殊事件），
# 寧可不調也不要把整條歷史乘壞。
RATIO_LO, RATIO_HI = 0.5, 1.2


def num(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", "", regex=False).str.strip()
        .replace({"--": None, "---": None, "": None, "null": None}),
        errors="coerce")


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df))


def roc_to_iso(s: str) -> str:
    s = str(s).strip().replace("/", "")
    return f"{int(s[:3]) + 1911:04d}-{s[3:5]}-{s[5:7]}"


def fetch_twse() -> pd.DataFrame:
    r = requests.get(TWSE, timeout=60, headers={"accept": "application/json"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    return pd.DataFrame({
        "stock_id": pick(df, "Code").astype(str).str.strip(),
        "open": num(pick(df, "OpeningPrice")),
        "max": num(pick(df, "HighestPrice")),
        "min": num(pick(df, "LowestPrice")),
        "close": num(pick(df, "ClosingPrice")),
        "Trading_Volume": num(pick(df, "TradeVolume")),
        "Trading_money": num(pick(df, "TradeValue")),
    })


def fetch_tpex():
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


def fetch_dividends(day: str) -> pd.DataFrame:
    """當日除權息。免費層拿得到，而且直接給 before_price / after_price，
    不用自己從現金股利、股票股利、增資配股三種情況拆開算。"""
    tok = os.environ.get("FINMIND_TOKEN", "").strip()
    if not tok:
        print("警告：沒有 FINMIND_TOKEN，跳過除權息檢查。"
              "若當日有個股除息，還原序列會出現假跌幅。")
        return pd.DataFrame()
    try:
        r = requests.get(FINMIND,
                         params={"dataset": "TaiwanStockDividendResult",
                                 "start_date": day, "end_date": day},
                         headers={"Authorization": f"Bearer {tok}"}, timeout=60)
        if r.status_code != 200:
            print(f"警告：除權息查詢 HTTP {r.status_code}，跳過")
            return pd.DataFrame()
        return pd.DataFrame(r.json().get("data", []))
    except Exception as e:
        print(f"警告：除權息查詢失敗 {e}，跳過")
        return pd.DataFrame()


def allowlist() -> set:
    """有效代號白名單。universe.parquet 由 update_universe.yaml 每月更新。"""
    if os.path.exists(UNIVERSE):
        try:
            u = pd.read_parquet(UNIVERSE, columns=["stock_id"])
            ids = set(u["stock_id"].astype(str).str.strip())
            if len(ids) >= 1000:
                return ids
            print(f"警告：universe 只有 {len(ids)} 檔，看起來不完整，改用 regex")
        except Exception as e:
            print(f"警告：讀 universe 失敗 {e}，改用 regex")
    else:
        print(f"警告：找不到 {UNIVERSE}，改用 regex。"
              f"先跑 update_universe.yaml 會比較準。")
    return set()


def year_files() -> dict:
    return {os.path.basename(p)[11:15]: p
            for p in sorted(glob.glob(os.path.join(DATA_DIR,
                                                   "prices_adj_*.parquet")))}


def apply_dividends(div: pd.DataFrame) -> int:
    """把除息股票在所有年份檔裡的歷史價格往下調。"""
    if div.empty or "before_price" not in div.columns:
        return 0
    d = div.copy()
    d["before_price"] = pd.to_numeric(d["before_price"], errors="coerce")
    d["after_price"] = pd.to_numeric(d["after_price"], errors="coerce")
    d = d[(d["before_price"] > 0) & (d["after_price"] > 0)]
    d["ratio"] = d["after_price"] / d["before_price"]
    bad = (d["ratio"] < RATIO_LO) | (d["ratio"] > RATIO_HI)
    if bad.any():
        print(f"  跳過 {int(bad.sum())} 筆異常比值："
              f"{d[bad]['stock_id'].tolist()[:10]}")
        d = d[~bad]
    if d.empty:
        return 0

    ratios = d.groupby("stock_id")["ratio"].prod()   # 同日多筆就連乘
    print(f"  當日除權息 {len(ratios)} 檔，"
          f"比值 {ratios.min():.4f} ~ {ratios.max():.4f}")

    files = year_files()
    touched = 0
    for _, path in files.items():
        g = pd.read_parquet(path)
        m = g["stock_id"].isin(ratios.index)
        if not m.any():
            continue
        f = g.loc[m, "stock_id"].map(ratios).astype(float)
        for c in ["open", "max", "min", "close"]:
            g.loc[m, c] = (g.loc[m, c] * f).round(4)
        g.to_parquet(path, index=False, compression="zstd")
        touched += int(m.sum())
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(DATA_DIR) or not year_files():
        sys.exit(f"找不到 {DATA_DIR} 下的年份檔。先跑 backfill_adj.py。")

    tw = fetch_twse()
    tp, session_date = fetch_tpex()
    print(f"TWSE {len(tw)} 筆 / TPEx {len(tp)} 筆 / 交易日 {session_date}")

    if not session_date:
        session_date = dt.date.today().isoformat()
        print(f"警告：TPEx 沒給日期，改用今天 {session_date}")

    df = pd.concat([tw, tp], ignore_index=True)
    df["date"] = session_date
    raw_n = len(df)

    ids = allowlist()
    if ids:
        df = df[df["stock_id"].isin(ids)]
        print(f"白名單過濾：{raw_n} → {len(df)}（universe {len(ids)} 檔）")
    else:
        df = df[df["stock_id"].str.match(FALLBACK_RE)]
        print(f"regex 過濾：{raw_n} → {len(df)}")

    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.drop_duplicates(subset=["stock_id"], keep="first")
    df = df.reindex(columns=COLS)
    print(f"清理後 {len(df)} 檔")

    if len(df) < 800:
        sys.exit(f"筆數異常偏少（{len(df)}），可能是休市或 API 改版，不寫入")

    yr = session_date[:4]
    path = os.path.join(DATA_DIR, f"prices_adj_{yr}.parquet")

    # 已有多少當日資料？決定是「新的一天」還是「補缺」。
    # 補缺時不重新套除權息——歷史在第一次執行就調過了。
    existing_ids, is_repair = set(), False
    if os.path.exists(path):
        old = pd.read_parquet(path, columns=["date", "stock_id"])
        existing_ids = set(old.loc[old["date"].astype(str) == session_date,
                                   "stock_id"])
        if existing_ids:
            is_repair = True
            miss = df[~df["stock_id"].isin(existing_ids)]
            print(f"{session_date} 已有 {len(existing_ids)} 檔，"
                  f"本次可補 {len(miss)} 檔")
            if miss.empty:
                print("沒有缺漏，結束")
                return
            df = miss

    if args.dry_run:
        div = fetch_dividends(session_date)
        print(f"[dry-run] 當日除權息 {len(div)} 筆")
        if not div.empty:
            print(div.head(10).to_string(index=False))
        return

    # 順序很重要：先調歷史，再 append 今天。
    # 反過來的話今天的價格也會被乘一次，變成雙重調整。
    if is_repair:
        print("補缺模式：跳過除權息調整（同日第一次執行已經調過）")
    else:
        div = fetch_dividends(session_date)
        n = apply_dividends(div)
        if n:
            print(f"  已調整 {n:,} 列歷史價格")

    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df["date"] = df["date"].astype(str)
    df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df.to_parquet(path, index=False, compression="zstd")
    today_n = int((df["date"] == session_date).sum())
    print(f"已寫入 {os.path.basename(path)}：{df['stock_id'].nunique()} 檔 / "
          f"{len(df):,} 列 / {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"{session_date} 當日 {today_n} 檔")

    # 跟前一個交易日比對。少 10% 以上通常代表來源改版或又被濾掉一批，
    # 這種事不會報錯只會靜默累積，所以一定要印出來。
    days = sorted(set(df["date"].astype(str)))
    if len(days) >= 2:
        prev = days[-2]
        prev_n = int((df["date"] == prev).sum())
        if prev_n and today_n < prev_n * 0.9:
            print(f"警告：當日 {today_n} 檔，前一交易日（{prev}）{prev_n} 檔，"
                  f"少了 {(1 - today_n / prev_n):.0%}。檢查來源或白名單。")


if __name__ == "__main__":
    main()
