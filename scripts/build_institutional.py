#!/usr/bin/env python3
"""三大法人資料回補（個股買賣超 / 大盤合計 / 期貨未平倉）

輸出：
  data/inst/stock_inst_YYYY.parquet   個股，寬表，單位=股
  data/inst/total_inst.parquet        大盤合計，原始欄位
  data/inst/futures_inst.parquet      期貨三大法人，原始欄位

用法：
  python scripts/build_institutional.py --parts stock,total,futures --start 2015-06-01
可重跑：個股部分會跳過已存在的日期，中斷後直接再跑同一個指令即可續抓。
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "")
ROOT = Path(__file__).resolve().parents[1]
ADJ_DIR = ROOT / "data" / "adj"
OUT_DIR = ROOT / "data" / "inst"

NAME_MAP = {
    "Foreign_Investor": "foreign",
    "Foreign_Dealer_Self": "foreign_dealer",
    "Investment_Trust": "trust",
    "Dealer_self": "dealer_self",
    "Dealer_Hedging": "dealer_hedge",
}
NET_COLS = list(NAME_MAP.values())


# ---------- API ----------

def api_get(dataset, start_date, end_date=None, data_id=None, sleep=0.6):
    params = {"dataset": dataset, "start_date": start_date, "token": TOKEN}
    if end_date:
        params["end_date"] = end_date
    if data_id:
        params["data_id"] = data_id

    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=90)
        except requests.RequestException as e:
            print(f"    連線失敗 {e}，等 {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code in (402, 429):
            print("    觸發流量限制，等 60s")
            time.sleep(60)
            continue

        if r.status_code != 200:
            print(f"    HTTP {r.status_code}，等 {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
            continue

        js = r.json()
        msg = str(js.get("msg", ""))
        if js.get("status") == 200:
            time.sleep(sleep)
            return pd.DataFrame(js.get("data") or [])
        if "level" in msg.lower():
            sys.exit(f"[權限不足] {dataset}：{msg}（這支要更高訂閱層級）")
        print(f"    API msg={msg}，等 10s")
        time.sleep(10)

    raise RuntimeError(f"{dataset} {start_date} 連續失敗")


# ---------- 交易日 ----------

def trading_days(start, end):
    files = sorted(ADJ_DIR.glob("prices_adj_*.parquet"))
    if not files:
        sys.exit(f"找不到 {ADJ_DIR}/prices_adj_*.parquet，交易日清單來源缺失")
    days = set()
    for f in files:
        d = pd.read_parquet(f, columns=["date"])["date"]
        days.update(pd.to_datetime(d).dt.strftime("%Y-%m-%d").unique())
    days = sorted(d for d in days if start <= d <= end)
    return days


# ---------- 個股 ----------

def shape_stock(df):
    df = df.copy()
    df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["net"] = df["buy"] - df["sell"]
    df["col"] = df["name"].map(NAME_MAP).fillna(df["name"])
    w = (
        df.pivot_table(index=["date", "stock_id"], columns="col",
                       values="net", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for c in NET_COLS:
        if c not in w.columns:
            w[c] = 0
    w[NET_COLS] = w[NET_COLS].fillna(0).astype("int64")
    w["total_net"] = w[NET_COLS].sum(axis=1)
    return w[["date", "stock_id"] + NET_COLS + ["total_net"]]


def load_year(year):
    p = OUT_DIR / f"stock_inst_{year}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def save_year(year, df):
    df = df.drop_duplicates(subset=["date", "stock_id"], keep="last")
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)
    df.to_parquet(OUT_DIR / f"stock_inst_{year}.parquet", index=False)


def run_stock(start, end, sleep, flush_every=20):
    days = trading_days(start, end)
    print(f"[個股] 交易日 {len(days)} 天：{days[0]} ~ {days[-1]}")

    by_year = {}
    done = set()
    for y in sorted({d[:4] for d in days}):
        df = load_year(y)
        by_year[y] = df
        if not df.empty:
            done |= set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))

    todo = [d for d in days if d not in done]
    print(f"[個股] 已有 {len(days) - len(todo)} 天，待抓 {len(todo)} 天")

    pending = {}
    for i, day in enumerate(todo, 1):
        raw = api_get("TaiwanStockInstitutionalInvestorsBuySell", day, sleep=sleep)
        if raw.empty:
            print(f"  {day} 無資料（休市或延遲）")
            continue
        w = shape_stock(raw)
        y = day[:4]
        pending.setdefault(y, []).append(w)
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} {day} {len(w)} 檔")
        if i % flush_every == 0 or i == len(todo):
            for yy, chunks in pending.items():
                by_year[yy] = pd.concat([by_year.get(yy, pd.DataFrame())] + chunks,
                                        ignore_index=True)
                save_year(yy, by_year[yy])
            pending = {}
            print(f"  -- 已寫檔 ({i}/{len(todo)})")


# ---------- 大盤合計 ----------

def run_total(start, end, sleep):
    p = OUT_DIR / "total_inst.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    frames = [old] if not old.empty else []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        s = max(f"{y}-01-01", start)
        e = min(f"{y}-12-31", end)
        df = api_get("TaiwanStockTotalInstitutionalInvestors", s, e, sleep=sleep)
        print(f"[大盤] {y} {len(df)} 列")
        if not df.empty:
            frames.append(df)
    if not frames:
        print("[大盤] 無資料")
        return
    out = pd.concat(frames, ignore_index=True)
    key = [c for c in ["date", "name"] if c in out.columns]
    out = out.drop_duplicates(subset=key, keep="last").sort_values(key)
    out.to_parquet(p, index=False)
    print(f"[大盤] 完成 {len(out)} 列 → {p.name}")


# ---------- 期貨 ----------

def run_futures(start, end, ids, sleep):
    p = OUT_DIR / "futures_inst.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    frames = [old] if not old.empty else []
    for fid in ids:
        for y in range(int(start[:4]), int(end[:4]) + 1):
            s = max(f"{y}-01-01", start)
            e = min(f"{y}-12-31", end)
            df = api_get("TaiwanFuturesInstitutionalInvestors", s, e,
                         data_id=fid, sleep=sleep)
            print(f"[期貨] {fid} {y} {len(df)} 列")
            if not df.empty:
                if "futures_id" not in df.columns:
                    df["futures_id"] = fid
                frames.append(df)
    if not frames:
        print("[期貨] 無資料")
        return
    out = pd.concat(frames, ignore_index=True)
    key = [c for c in ["date", "futures_id", "institutional_investors", "name"]
           if c in out.columns]
    out = out.drop_duplicates(subset=key, keep="last").sort_values(key)
    out.to_parquet(p, index=False)
    print(f"[期貨] 完成 {len(out)} 列 → {p.name}，欄位：{list(out.columns)}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="stock,total,futures")
    ap.add_argument("--start", default="2015-06-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--futures-ids", default="TX,MTX")
    args = ap.parse_args()

    if not TOKEN:
        sys.exit("缺少環境變數 FINMIND_TOKEN")

    end = args.end or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    if "total" in parts:
        run_total(args.start, end, args.sleep)
    if "futures" in parts:
        run_futures(args.start, end,
                    [x.strip() for x in args.futures_ids.split(",") if x.strip()],
                    args.sleep)
    if "stock" in parts:
        run_stock(args.start, end, args.sleep)

    print("完成")


if __name__ == "__main__":
    main()
