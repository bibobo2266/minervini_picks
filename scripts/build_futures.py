#!/usr/bin/env python3
"""台指期 basis —— A 段原料回補（期貨日成交 + 兩種現貨指數）

輸出：
  data/futures/futures_tx.parquet    TX 大台日成交，各契約月份分開
  data/futures/index_taiex.parquet   加權指數（價格指數）
  data/futures/index_tri.parquet     發行量加權報酬指數（含息）

這支只負責把原料抓回來，不算 basis、不做任何績效評估。
handover 的規定：A 段（建 PIT fair basis）沒完成前，禁止看 B 段（均值回歸）的績效。

為什麼只抓 TX：
  MTX（小台）是同一個標的、同一條 basis，抓了只是複製一份重複資料。
  電子期／金融期是不同標的，各自需要配一條現貨腿，屬於另一個題目。

為什麼兩個指數都抓：
  fair basis 要扣股利率。用價格指數（TAIEX）算出來的 basis 會內含整段
  除息的預期缺口；用報酬指數（含息）算則不會。兩個都留著才有辦法交叉驗證
  fair value 模型有沒有算錯——這正是 A 段要驗的東西。

用法：
  python scripts/build_futures.py --start 2015-06-01
  python scripts/build_futures.py --probe-only     # 只探測，不抓資料，30 秒
可重跑：已經抓到的年份會跳過，中斷後直接再跑同一個指令即可續抓。
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
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "futures"

# (輸出檔名, dataset, data_id) —— data_id 為 None 表示該 dataset 不吃 data_id
TARGETS = [
    ("futures_tx",  "TaiwanFuturesDaily",            "TX"),
    ("index_taiex", "TaiwanStockPrice",              "TAIEX"),
    ("index_tri",   "TaiwanStockTotalReturnIndex",   "TAIEX"),
]


# ---------- API ----------

def api_get(dataset, start_date, end_date=None, data_id=None, sleep=0.6,
            fatal_on_level=True):
    """照 build_institutional.py 的慣例：402/429 等 60 秒，其他錯誤遞增退避。"""
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
            try:
                msg = str(r.json().get("msg", ""))
            except Exception:
                msg = r.text[:150]
            if "level" in msg.lower():
                if fatal_on_level:
                    sys.exit(f"[權限不足] {dataset}：{msg}（這支要更高訂閱層級）")
                return None
            print(f"    HTTP {r.status_code} {msg}，等 {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
            continue

        js = r.json()
        msg = str(js.get("msg", ""))
        if js.get("status") == 200:
            time.sleep(sleep)
            return pd.DataFrame(js.get("data") or [])
        if "level" in msg.lower():
            if fatal_on_level:
                sys.exit(f"[權限不足] {dataset}：{msg}（這支要更高訂閱層級）")
            return None
        print(f"    API msg={msg}，等 10s")
        time.sleep(10)

    raise RuntimeError(f"{dataset} {start_date} 連續失敗")


# ---------- 探測 ----------

def probe():
    """先確認免費層拿不拿得到，再決定要不要花一小時抓。

    上次的教訓是跑了一小時才發現問題。這裡每個 dataset 只打一次、
    抓一個月，30 秒內就有答案。
    """
    print("=== 探測（每個 dataset 抓一個月）===")
    ok = {}
    for name, dataset, data_id in TARGETS:
        df = api_get(dataset, "2026-07-01", "2026-07-31", data_id,
                     fatal_on_level=False)
        if df is None:
            print(f"  ✗ {name:<12} {dataset:<30} 權限不足")
            ok[name] = False
            continue
        if df.empty:
            print(f"  ✗ {name:<12} {dataset:<30} 回空（data_id 可能不對）")
            ok[name] = False
            continue
        print(f"  ✓ {name:<12} {dataset:<30} {len(df)} 列")
        print(f"      欄位：{list(df.columns)}")
        print(f"      首列：{df.iloc[0].to_dict()}")
        ok[name] = True
    return ok


# ---------- 抓取 ----------

def load_existing(name):
    p = OUT_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save(name, df):
    df = df.copy()
    # 日期一律 'YYYY-MM-DD' 字串，不做格式推斷。
    # 這是 2026-09-07 事故的教訓：同一欄混了 '2026-04-10 00:00:00' 和
    # '2026-09-04' 兩種格式，下游 pd.to_datetime 用第一列推格式就爆。
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    keys = ["date"]
    for c in ("contract_date", "futures_id", "stock_id"):
        if c in df.columns:
            keys.append(c)
    df = df.drop_duplicates(subset=keys, keep="last")
    df = df.sort_values(keys).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    print(f"  -- 已寫檔 {name}.parquet：{len(df)} 列 "
          f"（{df['date'].min()} ~ {df['date'].max()}）")


def run_one(name, dataset, data_id, start, end, sleep):
    print(f"\n[{name}] {dataset}")
    old = load_existing(name)
    done_years = set()
    if not old.empty:
        d = old["date"].astype(str).str.slice(0, 10)
        counts = d.str.slice(0, 4).value_counts()
        # 只把「有像樣天數」的年份當作抓完了，避免半途中斷的年份被跳過
        done_years = set(counts[counts >= 200].index)
        print(f"  已有 {len(old)} 列，完整年份 {len(done_years)} 個")

    years = sorted({str(y) for y in range(int(start[:4]), int(end[:4]) + 1)})
    parts = [old] if not old.empty else []
    got = 0
    for y in years:
        if y in done_years:
            continue
        s = max(f"{y}-01-01", start)
        e = min(f"{y}-12-31", end)
        if s > e:
            continue
        df = api_get(dataset, s, e, data_id, sleep=sleep)
        if df is None or df.empty:
            print(f"  {y} 無資料")
            continue
        print(f"  {y} 取得 {len(df)} 列")
        parts.append(df)
        got += len(df)
        # 每年寫一次檔。上次的教訓：不要等全部跑完才落地。
        save(name, pd.concat(parts, ignore_index=True))

    if got == 0 and not old.empty:
        print("  無新增")
    return got


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-06-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    if not TOKEN:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    end = args.end.strip() or date.today().isoformat()

    ok = probe()
    if args.probe_only:
        print("\n--probe-only，不抓資料")
        return

    usable = [t for t in TARGETS if ok.get(t[0])]
    if not usable:
        sys.exit("三個 dataset 都拿不到，沒有原料可抓")
    if len(usable) < len(TARGETS):
        missing = [t[0] for t in TARGETS if not ok.get(t[0])]
        print(f"\n⚠️ 拿不到：{missing}，只抓得到的部分")

    print(f"\n=== 抓取 {args.start} ~ {end} ===")
    for name, dataset, data_id in usable:
        run_one(name, dataset, data_id, args.start, end, args.sleep)

    print("\n完成")
    print("⚠️ 這只是 A 段的原料。fair basis 還沒建，"
          "在 A 段驗過之前不要看 B 段（均值回歸）的績效。")


if __name__ == "__main__":
    main()
