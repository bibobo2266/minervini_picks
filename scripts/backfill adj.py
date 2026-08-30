"""
還原股價回補：建立 data/adj/prices_adj_YYYY.parquet

為什麼要另外一份：現有的 prices.parquet 是未還原收盤價（TWSE/TPEx OpenAPI
與 FinMind 原始價）。除權息會在圖上留缺口，跨 9 年累積後：
  - 52 週低點被低估 → 趨勢模板第 6 條過關率虛高
  - 30 週均線在除息後被拉低 → 階段判斷在 7-9 月偏樂觀
  - 台股平均殖利率 3~4%，九年下來報酬被系統性低估 30% 以上
歷史回測沒有還原股價，出來的名單是錯的。

分年存檔：單檔 9 年約 65MB，GitHub 超過 50MB 會警告、100MB 直接擋，
而且每天 commit 會讓 git 歷史膨脹。分年後每個 7~8MB，app 只載需要的年份。

免費版 600 req/hr，每檔間隔 6.5 秒。約 2150 檔 → 約 4 小時。
可中斷續跑：已完成的代號記在 data/adj/_done.txt。

用法：
    FINMIND_TOKEN=xxx python scripts/backfill_adj.py
    FINMIND_TOKEN=xxx python scripts/backfill_adj.py --start 2017-01-01 --limit 30
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
DONE = os.path.join(DATA_DIR, "_done.txt")
SLEEP = 1.0      # 逐年分段後每檔已打 ~10 次，不需要再等 6.5 秒
FLUSH = 40          # 每 N 檔寫一次盤
COLS = ["date", "stock_id", "open", "max", "min", "close",
        "Trading_Volume", "Trading_money"]


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def call(dataset: str, **params) -> pd.DataFrame:
    """官方文件的範例用 Authorization header，不是 token 查詢參數。
    TaiwanStockPrice 兩種都吃，但 TaiwanStockPriceAdj 用 params 會回 400。"""
    headers = {"Authorization": f"Bearer {token()}"}
    payload = {"dataset": dataset, **params}
    for _ in range(5):
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
            print(f"    HTTP 400（請求被拒，不重試）：{r.text[:200]}")
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
            # 權限不足 / dataset 不存在之類：重試沒有意義，直接放棄這一檔
            print(f"    FinMind 拒絕：status={js.get('status')} msg={msg}")
            return pd.DataFrame()
        data = js.get("data", [])
        if not data:
            # 空資料也不重試——之前每檔空轉 5 輪 × 20 秒，30 檔就燒掉 50 分鐘
            print(f"    回傳空資料：status={js.get('status')} msg={msg}")
        return pd.DataFrame(data)
    return pd.DataFrame()


def universe() -> pd.DataFrame:
    """上市 + 上櫃普通股。TaiwanStockInfo 保留已下市代號，
    這些抓不到價格會被跳過——已知的倖存者偏差來源，未修。"""
    info = call("TaiwanStockInfo")
    if info.empty:
        sys.exit("取不到 TaiwanStockInfo，請檢查 token")
    info = info[info["type"].isin(["twse", "tpex"])]
    info = info[info["stock_id"].str.match(r"^[1-9]\d{3}$")]
    return info.drop_duplicates("stock_id").sort_values("stock_id")


def load_done() -> set:
    if not os.path.exists(DONE):
        return set()
    with open(DONE, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def flush(buf, done_new):
    """把緩衝區按年份 append 進各年檔案。"""
    if not buf:
        return
    df = pd.concat(buf, ignore_index=True)
    df["year"] = df["date"].str.slice(0, 4)
    for yr, g in df.groupby("year"):
        path = os.path.join(DATA_DIR, f"prices_adj_{yr}.parquet")
        g = g.drop(columns="year")
        if os.path.exists(path):
            g = pd.concat([pd.read_parquet(path), g], ignore_index=True)
        g = g.drop_duplicates(subset=["stock_id", "date"], keep="last")
        g = g.sort_values(["stock_id", "date"]).reset_index(drop=True)
        g.to_parquet(path, index=False, compression="zstd")
    with open(DONE, "a", encoding="utf-8") as f:
        for s in done_new:
            f.write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # 先探測一次：免費版可能不支援 TaiwanStockPriceAdj，早點知道比跑 30 檔快
    probe = call("TaiwanStockPriceAdj", data_id="2330", start_date=args.start)
    if probe.empty:
        print("\n>>> TaiwanStockPriceAdj 取不到資料（連 2330 都空）。")
        print(">>> 多半是免費版不支援還原股價。看上面那行 FinMind 的回應訊息。")
        print(">>> 若確認如此，改用除權息資料自行還原，或放棄還原股價。")
        sys.exit(1)
    print(f"探測 2330：{len(probe)} 列，"
          f"{probe['date'].min()} ~ {probe['date'].max()}")
    time.sleep(SLEEP)

    uni = universe()
    uni.to_parquet("data/universe.parquet", index=False)
    print(f"母體 {len(uni)} 檔 · 起始 {args.start}")

    done = load_done()
    if done:
        print(f"已完成 {len(done)} 檔，續跑剩下的")
    todo = [s for s in uni["stock_id"] if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"本次 {len(todo)} 檔，預估 {len(todo) * SLEEP / 3600:.1f} 小時")

    y0 = int(args.start[:4])
    y1 = dt.date.today().year
    years = list(range(y0, y1 + 1))

    buf, done_new, ok, fail = [], [], 0, 0
    for i, sid in enumerate(todo, 1):
        # 逐年分段：一次要 9 年的資料可能超過單次筆數上限
        parts = []
        for y in years:
            s_ = args.start if y == y0 else f"{y}-01-01"
            g = call("TaiwanStockPriceAdj", data_id=sid,
                     start_date=s_, end_date=f"{y}-12-31")
            if not g.empty:
                parts.append(g)
            time.sleep(0.6)          # 年段之間的小間隔
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty or "close" not in df.columns:
            fail += 1
            done_new.append(sid)          # 抓不到也標記，避免續跑時重試
            print(f"[{i}/{len(todo)}] {sid} 無資料")
        else:
            df = df.reindex(columns=COLS)
            df["stock_id"] = sid
            buf.append(df)
            done_new.append(sid)
            ok += 1
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {sid} ok={ok} fail={fail}")

        if len(done_new) >= FLUSH:
            flush(buf, done_new)
            buf, done_new = [], []
            files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
            size = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in files)
            print(f"    -- 存檔，{len(files)} 個年份檔 / {size / 1e6:.1f} MB 總計")

        if i < len(todo):
            time.sleep(SLEEP)

    flush(buf, done_new)
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    print(f"\n完成：ok={ok} fail={fail}")
    for f in files:
        p = os.path.join(DATA_DIR, f)
        n = len(pd.read_parquet(p, columns=["stock_id"]))
        print(f"  {f}  {os.path.getsize(p) / 1e6:5.1f} MB  {n:>9,} 列")


if __name__ == "__main__":
    main()
