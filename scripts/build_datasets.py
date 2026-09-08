#!/usr/bin/env python3
"""FinMind 原料回補 —— 母體建置（市值／PBR）＋ basis A 段（期貨／指數）

輸出：
  data/fundamentals/market_value_YYYY.parquet   個股市值，逐年分檔
  data/fundamentals/stock_per_YYYY.parquet      個股 PER / PBR / 殖利率，逐年分檔
  data/fundamentals/balance_sheet.parquet       資產負債表（選配，非必要）
  data/futures/futures_tx.parquet               TX 大台日成交，各契約月份分開
  data/futures/index_taiex.parquet              加權指數（價格指數）
  data/futures/index_tri.parquet                發行量加權報酬指數（含息）

=== 2026-09-08 修正：三種抓取模式，不要再搞混 ===

  bulk_year  帶固定 data_id（TX / TAIEX），日期區間正常運作，一次拉一整年。
             futures_tx / index_taiex / index_tri 走這條。

  bulk_day   不帶 data_id 的全市場日資料。FinMind 對這類 dataset 只回
             start_date 當天，end_date 直接被忽略。所以必須逐交易日打。
             market_value / stock_per 走這條。

  per_stock  帶 data_id 且日期區間有效，一檔一檔打全歷史。
             balance_sheet 走這條。

⚠️ 上一版把 bulk_day 誤判成 bulk_year，症狀是：
     2015 取得 1551 列（2015-06-01 ~ 2015-06-01）
     2016 無資料 … 2026 無資料
   因為逐年迴圈的 start_date 是 1/1，元旦休市所以每年都回空，
   只有 2015 那次剛好用了 --start 的 2015-06-01（有開市）才拿到一天。
   現在 probe 會檢查回傳的「不同日期數」而不只是「不同股票數」。

--- 優先序：market_value 第一 ---
TaiwanStockMarketValue 是 Backer/Sponsor 層，隨訂閱到期就會斷，先抓它。
    股數 = market_value / 收盤價
    股本 ≈ 股數 × 10（台股面額）
市值分層與「投信淨買超 / 股本」的分母都直接解決，不用碰 EPS 反推。
((稅後淨利 − 少數股權) / EPS 在 EPS 接近 0 或為負時分母炸開，會在景氣低點
把虧損股從母體剔除 = 自己造出景氣偏誤。有現成市值表就不必走這條。)

--- balance_sheet 為什麼降級 ---
本來要拿 OrdinaryShare（普通股股本）反推股數，市值表能用之後就不必要了。
留著是因為 Equity（權益總計）以後做基本面因子可能用到，但它 per_stock、
約 3,100 次呼叫，不要跟其他幾支綁在一起跑。

--- 期貨為什麼只抓 TX、指數為什麼抓兩個 ---
MTX 是同一標的同一條 basis，抓了只是複製。電子期／金融期是不同標的，
各自要配一條現貨腿，屬於另一個題目。
fair basis 要扣股利率：用價格指數算的 basis 內含整段除息預期缺口，用報酬
指數（含息）算則不會。兩個都留著才能交叉驗證 fair value 模型——這正是
A 段要驗的東西。

⚠️ 下游注意（probe 實測，寫在這裡免得忘記）：
  1. futures_tx 的 trading_session 有 'position'（日盤）與 'after_market'
     （夜盤）兩種，2026-07 樣本是 264 / 277。算 basis 只能取日盤，
     否則跟現貨收盤對不上時間。
  2. stock_per 的 PER 在虧損股是 0.0，分層一律用 PBR，不要用 PER。
  3. market_value / stock_per 都含 ETF 與六碼標的（首列是 00400A），
     下游一律先 len(stock_id) == 4。

用法：
  python scripts/build_datasets.py --probe-only
  python scripts/build_datasets.py --only market_value --sleep 2
  python scripts/build_datasets.py --only stock_per --sleep 2
  python scripts/build_datasets.py --only balance_sheet --sleep 2
可重跑：bulk_year 跳過已完整年份，bulk_day 跳過已抓到的日期，
per_stock 跳過已抓到的 stock_id。中斷後跑同一個指令即可續抓。
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data" / "fundamentals"
FUT_DIR = ROOT / "data" / "futures"

# shard=True 表示逐年分檔（out 當作 prefix），False 表示單一檔案
# 順序即抓取順序：市值表是付費層、隨時可能斷，排第一。
TARGETS = {
    "market_value": dict(
        dataset="TaiwanStockMarketValue", dir=FUND_DIR, stem="market_value",
        data_id=None, mode="auto", sample_id="2330", shard=True,
    ),
    "stock_per": dict(
        dataset="TaiwanStockPER", dir=FUND_DIR, stem="stock_per",
        data_id=None, mode="auto", sample_id="2330", shard=True,
    ),
    "futures_tx": dict(
        dataset="TaiwanFuturesDaily", dir=FUT_DIR, stem="futures_tx",
        data_id="TX", mode="bulk_year", sample_id="TX", shard=False,
    ),
    "index_taiex": dict(
        dataset="TaiwanStockPrice", dir=FUT_DIR, stem="index_taiex",
        data_id="TAIEX", mode="bulk_year", sample_id="TAIEX", shard=False,
    ),
    "index_tri": dict(
        dataset="TaiwanStockTotalReturnIndex", dir=FUT_DIR, stem="index_tri",
        data_id="TAIEX", mode="bulk_year", sample_id="TAIEX", shard=False,
    ),
    # 選配，不要跟上面幾支一起跑
    "balance_sheet": dict(
        dataset="TaiwanStockBalanceSheet", dir=FUND_DIR, stem="balance_sheet",
        data_id=None, mode="auto", sample_id="2330", shard=False,
    ),
}

TPEX_SAMPLES = ["6488", "3105"]   # 上櫃：環球晶、穩懋


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


# ---------- 交易日與母體 ----------

def trading_days(start, end):
    """交易日清單。從 repo 現成的還原股價讀，不花 API 額度。"""
    files = sorted((ROOT / "data" / "adj").glob("prices_adj_*.parquet"))
    days = set()
    if files:
        for f in files:
            d = pd.read_parquet(f, columns=["date"])["date"]
            days |= set(d.astype(str).str.slice(0, 10))
        print(f"  交易日來自 data/adj/（{len(files)} 個檔）")
    else:
        p = FUT_DIR / "index_taiex.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["date"])["date"]
            days |= set(d.astype(str).str.slice(0, 10))
            print("  交易日來自 index_taiex.parquet")
        else:
            print("  本地找不到交易日來源，改打 TaiwanStockTradingDate")
            df = api_get("TaiwanStockTradingDate", start, end)
            if df is None or df.empty:
                sys.exit("拿不到交易日清單")
            days |= set(df["date"].astype(str).str.slice(0, 10))
    out = sorted(d for d in days if start <= d <= end)
    print(f"  {start} ~ {end} 共 {len(out)} 個交易日")
    return out


def load_universe():
    """四碼普通股清單。優先讀 repo 現成的 universe.parquet，沒有才打 API。"""
    p = ROOT / "data" / "universe.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        print("  data/universe.parquet 不存在，改打 TaiwanStockInfo")
        df = api_get("TaiwanStockInfo", "2015-06-01")
        if df is None or df.empty:
            sys.exit("拿不到 TaiwanStockInfo，無法建母體")
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.len() == 4]
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().isin(["twse", "tpex"])]
    ids = sorted(df["stock_id"].unique())
    print(f"  母體 {len(ids)} 檔四碼普通股（含上市＋上櫃）")
    return ids


# ---------- 探測 ----------

def probe_target(name, cfg):
    """判斷 dataset 走哪一種抓法。每個 dataset 最多打兩次，一分鐘有答案。"""
    ds = cfg["dataset"]
    print(f"\n  [{name}] {ds}")

    if cfg["mode"] == "bulk_year":
        df = api_get(ds, "2026-07-01", "2026-07-31", cfg["data_id"],
                     fatal_on_level=False)
        if df is None:
            print("    ✗ 權限不足")
            return None
        if df.empty:
            print("    ✗ 回空（data_id 可能不對）")
            return None
        n_days = df["date"].astype(str).str.slice(0, 10).nunique()
        print(f"    ✓ bulk_year，{len(df)} 列 / {n_days} 個日期")
        print(f"      欄位：{list(df.columns)}")
        print(f"      首列：{df.iloc[0].to_dict()}")
        _probe_extras(name, df)
        return "bulk_year"

    # auto：先試不帶 data_id。關鍵是看回傳「幾個不同日期」——
    # 只回一天就代表 end_date 被忽略，必須逐日打（上一版就是漏了這個檢查）。
    df = api_get(ds, "2026-07-01", "2026-07-31", None, fatal_on_level=False)
    if df is not None and not df.empty and "stock_id" in df.columns:
        n_ids = df["stock_id"].nunique()
        n_days = df["date"].astype(str).str.slice(0, 10).nunique()
        print(f"      不帶 data_id：{len(df)} 列 / {n_ids} 檔 / {n_days} 個日期")
        if n_ids > 50 and n_days > 5:
            print("    ✓ bulk_year（日期區間有效）")
            print(f"      欄位：{list(df.columns)}")
            _probe_extras(name, df)
            return "bulk_year"
        if n_ids > 50:
            print("    ✓ bulk_day（end_date 被忽略，只回 start_date 當天，"
                  "必須逐交易日打）")
            print(f"      欄位：{list(df.columns)}")
            print(f"      首列：{df.iloc[0].to_dict()}")
            _probe_extras(name, df)
            return "bulk_day"
        print(f"    不帶 data_id 只回 {n_ids} 檔，改走 per_stock")

    df = api_get(ds, "2024-01-01", "2024-12-31", cfg["sample_id"],
                 fatal_on_level=False)
    if df is None:
        print("    ✗ 權限不足")
        return None
    if df.empty:
        print("    ✗ 單檔也回空")
        return None
    print(f"    ✓ per_stock（{cfg['sample_id']} 一年 {len(df)} 列）")
    print(f"      欄位：{list(df.columns)}")
    print(f"      首列：{df.iloc[0].to_dict()}")
    _probe_extras(name, df)
    return "per_stock"


def _probe_extras(name, df):
    """把下游會踩到的東西直接印出來，省一輪來回。"""
    if name == "balance_sheet" and "origin_name" in df.columns:
        cap = df[df["origin_name"].astype(str).str.contains("股本", na=False)]
        if not cap.empty:
            print("      --- 名稱含「股本」的列 ---")
            print(cap[["type", "origin_name", "value"]]
                  .drop_duplicates().to_string(index=False))

    if name == "stock_per":
        print("      --- 上櫃覆蓋檢查 ---")
        for sid in TPEX_SAMPLES:
            d = api_get("TaiwanStockPER", "2024-01-01", "2024-01-31", sid,
                        fatal_on_level=False)
            got = 0 if d is None or d.empty else len(d)
            flag = "✓" if got else "✗ 沒有（可能只含上市）"
            print(f"        {sid}: {got} 列 {flag}")

    if name == "futures_tx" and "trading_session" in df.columns:
        print("      --- trading_session 分佈（算 basis 只能取日盤）---")
        print(df["trading_session"].value_counts().to_string())


def probe():
    print("=== 探測 ===")
    modes = {}
    for name, cfg in TARGETS.items():
        modes[name] = probe_target(name, cfg)
    print("\n=== 探測結果 ===")
    for name, m in modes.items():
        print(f"  {name:<14} {m or '不可用'}")
    return modes


# ---------- 落地 ----------

def path_for(cfg, year=None):
    if cfg["shard"]:
        return cfg["dir"] / f"{cfg['stem']}_{year}.parquet"
    return cfg["dir"] / f"{cfg['stem']}.parquet"


def dedup_keys(df):
    keys = ["date"]
    # balance_sheet 是 long format，同一天同一檔有幾十個科目；
    # futures 同一天同一契約有日盤／夜盤兩列。少一個鍵就會被砍掉大半。
    for c in ("contract_date", "trading_session", "futures_id",
              "stock_id", "type"):
        if c in df.columns:
            keys.append(c)
    return keys


def save(out_path, df, quiet=False):
    df = df.copy()
    # 日期一律 'YYYY-MM-DD' 字串，不做格式推斷。
    # 這是 2026-09-07 事故的教訓：同一欄混了 '2026-04-10 00:00:00' 和
    # '2026-09-04' 兩種格式，下游 pd.to_datetime 用第一列推格式就爆。
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    keys = dedup_keys(df)
    df = df.drop_duplicates(subset=keys, keep="last")
    df = df.sort_values(keys).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if not quiet:
        print(f"  -- 已寫檔 {out_path.name}：{len(df)} 列 "
              f"（{df['date'].min()} ~ {df['date'].max()}）")
    return len(df)


def load_existing(p):
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ---------- 抓取：bulk_year ----------

def run_bulk_year(name, cfg, start, end, sleep):
    out = path_for(cfg)
    old = load_existing(out)
    done_years = set()
    if not old.empty:
        d = old["date"].astype(str).str.slice(0, 10)
        counts = d.str.slice(0, 4).value_counts()
        # 只把「有像樣天數」的年份當作抓完，避免半途中斷的年份被跳過
        done_years = set(counts[counts >= 200].index)
        print(f"  已有 {len(old)} 列，完整年份 {len(done_years)} 個")

    parts = [old] if not old.empty else []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        y = str(y)
        if y in done_years:
            continue
        s = max(f"{y}-01-01", start)
        e = min(f"{y}-12-31", end)
        if s > e:
            continue
        df = api_get(cfg["dataset"], s, e, cfg["data_id"], sleep=sleep)
        if df is None or df.empty:
            print(f"  {y} 無資料")
            continue
        print(f"  {y} 取得 {len(df)} 列")
        parts.append(df)
        # 每年寫一次檔。上次的教訓：不要等全部跑完才落地。
        save(out, pd.concat(parts, ignore_index=True))


# ---------- 抓取：bulk_day ----------

def run_bulk_day(name, cfg, start, end, sleep, batch=50):
    days = trading_days(start, end)

    done = set()
    for y in range(int(start[:4]), int(end[:4]) + 1):
        p = path_for(cfg, y)
        if p.exists():
            d = pd.read_parquet(p, columns=["date"])["date"]
            done |= set(d.astype(str).str.slice(0, 10))
    if done:
        print(f"  已有 {len(done)} 個交易日")

    todo = [d for d in days if d not in done]
    if not todo:
        print("  全部已抓完")
        return
    print(f"  待抓 {len(todo)} 天，sleep={sleep}s，"
          f"預估 {len(todo) * (sleep + 0.4) / 3600:.1f} 小時")

    buf = defaultdict(list)

    def flush():
        for y, frames in list(buf.items()):
            if not frames:
                continue
            p = path_for(cfg, y)
            old = load_existing(p)
            allf = ([old] if not old.empty else []) + frames
            n = save(p, pd.concat(allf, ignore_index=True), quiet=True)
            print(f"    {p.name}: {n} 列")
        buf.clear()

    got = 0
    for i, day in enumerate(todo, 1):
        df = api_get(cfg["dataset"], day, None, None, sleep=sleep)
        if df is not None and not df.empty:
            buf[day[:4]].append(df)
            got += len(df)
        if i % batch == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {day} 累計 {got} 列")
            # 每 batch 天落地一次，中斷不會全丟
            flush()


# ---------- 抓取：per_stock ----------

def run_per_stock(name, cfg, start, end, sleep, batch=50):
    out = path_for(cfg)
    old = load_existing(out)
    done = set()
    if not old.empty:
        done = set(old["stock_id"].astype(str).unique())
        print(f"  已有 {len(old)} 列 / {len(done)} 檔")

    ids = [i for i in load_universe() if i not in done]
    if not ids:
        print("  全部已抓完")
        return
    print(f"  待抓 {len(ids)} 檔，sleep={sleep}s，"
          f"預估 {len(ids) * (sleep + 0.4) / 3600:.1f} 小時")

    parts = [old] if not old.empty else []
    got = 0
    for n, sid in enumerate(ids, 1):
        df = api_get(cfg["dataset"], start, end, sid, sleep=sleep)
        if df is not None and not df.empty:
            parts.append(df)
            got += len(df)
        if n % batch == 0 or n == len(ids):
            print(f"  [{n}/{len(ids)}] {sid} 累計 {got} 列")
            save(out, pd.concat(parts, ignore_index=True))
            parts = [pd.read_parquet(out)]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-06-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--only", default="",
                    help="逗號分隔，例：market_value,stock_per")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    if not TOKEN:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    end = args.end.strip() or date.today().isoformat()

    wanted = list(TARGETS)
    if args.only.strip():
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        bad = [x for x in wanted if x not in TARGETS]
        if bad:
            sys.exit(f"未知的 target：{bad}，可用：{list(TARGETS)}")

    modes = probe()
    if args.probe_only:
        print("\n--probe-only，不抓資料")
        return

    usable = [n for n in wanted if modes.get(n)]
    if not usable:
        sys.exit("指定的 dataset 都拿不到，沒有原料可抓")
    skipped = [n for n in wanted if not modes.get(n)]
    if skipped:
        print(f"\n⚠️ 拿不到，跳過：{skipped}")

    runners = {"bulk_year": run_bulk_year,
               "bulk_day": run_bulk_day,
               "per_stock": run_per_stock}

    print(f"\n=== 抓取 {args.start} ~ {end} ===")
    for name in usable:
        cfg = TARGETS[name]
        m = modes[name]
        print(f"\n[{name}] {cfg['dataset']}  mode={m}")
        runners[m](name, cfg, args.start, end, args.sleep)

    print("\n完成")
    if "market_value" in usable:
        print("⚠️ 下一步：股數 = market_value / 收盤價，切 LARGE(前33%) / "
              "MID(中33%)，逐年檢查各格檔數。先過濾 len(stock_id)==4。")
    if any(n.startswith(("futures", "index")) for n in usable):
        print("⚠️ futures_tx 含夜盤（trading_session='after_market'），"
              "算 basis 前只留 'position'。fair basis 建好前不要看 B 段績效。")


if __name__ == "__main__":
    main()
