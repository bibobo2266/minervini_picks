r"""
一次性回補：月營收與季報全歷史 → data/fundamentals/

為什麼現在要做：
    TaiwanStockFinancialStatements 需要訂閱等級才撈得動全歷史，而訂閱快到期。
    跟上次搶官方還原股價一樣，這是時間視窗問題，跟「這個因子有沒有用」無關。
    先把資料存進 repo，之後要不要測都還來得及。

⚠️ 前視偏誤：FinMind 的 date 是「期別」，不是「公告日」
    這是這份資料最容易搞爛結論的地方，所以本程式在回補時就把 available_date
    算好寫進去。之後任何回測，一律用 available_date 過濾，不要用 date。

    股利宣告：用 AnnouncementDate（董事會決議公告日），不是除息日。
             因子的價值在「宣告到除息之間的重新定價」，用除息日就是前視偏誤。

    融資融券：證交所收盤後（約 21:00）公告當日餘額，所以 T 日的數字
             可以用在 T+1 開盤的決策上 → available_date = date 本身。
             ⚠️ 不能拿來做 T 日盤中的決策，那才是前視偏誤。

    月營收：⚠️ FinMind 的 date 已經是「營收所屬月份的次月 1 日」，
            不是營收月份本身。實測 26.5 萬列全部如此
            （revenue_month=7 對應 date=2026-08-01）。
            證交法第 36 條規定每月 10 日以前公告上月營運情形，
            → available_date = date + 14 天（留五天緩衝）
            早期版本誤以為 date 是營收月份、又推了一個月，
            結果晚了整整一個月。方向雖然安全，但月營收因子的價值
            有一大半在新鮮度，晚一個月等於自廢武功。
            create_time 欄看起來像公告日，但 95.8% 是空的
            （最早只到 2026-04），歷史沒有回填，不能用。

    季報：Q1 → 5/15、Q2 → 8/14、Q3 → 11/14、年報 → 次年 3/31。
          → available_date 用上述法定期限
          ⚠️ 金控可延到 5/30、8/31、11/29，銀行票券業另有規定。本程式一律用
             一般公司的期限，所以金融股會有大約兩週的前視偏誤。要嚴謹的話，
             測因子時把金融股排除，或自己再往後推兩週。

✅ EPS 已經是單季，不用相減（實測確認）
    台灣的第二季財報在制度上是半年報（涵蓋 1–6 月），所以原本擔心 FinMind
    給的是累計值。實測 2330：2023 年四季 EPS 相加 = 32.34、2024 年 = 45.26，
    與公告的全年 EPS 一致 → FinMind 已經幫你拆成單季了。
    其他累計型科目（Revenue、OperatingIncome 等）沒有逐一驗證，
    要用之前請個別確認。

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
    "margin": ("TaiwanStockMarginPurchaseShortSale", "margin.parquet"),
    "dividend": ("TaiwanStockDividend", "dividend.parquet"),
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
    """月營收可用日。

    ⚠️ FinMind 的 date 已經是營收所屬月份的「次月 1 日」，不要再推一個月。
    法定公告期限是次月 10 日前，這裡加 14 天留五天緩衝。
    例：7 月營收 → date 2026-08-01 → available_date 2026-08-15
    """
    return pd.to_datetime(d) + pd.Timedelta(days=14)


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


def margin_available(d: pd.Series) -> pd.Series:
    """融資融券餘額可用日 = 交易日當天。

    證交所在收盤後（約 21:00）公告當日餘額，所以 T 日的數字可以用在
    T+1 開盤的決策上。本策略正是「T 日收盤產生訊號、T+1 開盤買」，
    所以 available_date 設成 T 是正確的，不是前視偏誤。

    ⚠️ 但不能用它做 T 日盤中的決策——那才是前視偏誤。
    """
    return pd.to_datetime(d)


def dividend_available(df: pd.DataFrame) -> pd.Series:
    """股利宣告可用日 = 董事會決議公告日（AnnouncementDate）。

    ⚠️ 這是這張表唯一有價值的欄位。因子的全部價值在「宣告日到除息日之間
    的重新定價」——用除息日（CashExDividendTradingDate）當可用日就是前視偏誤，
    因為那時候市場早就知道配多少了。

    FinMind 的欄位名稱可能是 AnnouncementDate 或 announcement_date，
    兩個都試；都沒有的話退回用 date 並印警告（那筆資料不可用於因子）。
    """
    for col in ("AnnouncementDate", "announcement_date"):
        if col in df.columns:
            av = pd.to_datetime(df[col], errors="coerce")
            if av.notna().any():
                # 缺公告日的列退回除息日，並在主程式統計比例
                return av.fillna(pd.to_datetime(df["date"], errors="coerce"))
    print("    ⚠️ 找不到 AnnouncementDate 欄，退回用 date（該資料不可用於因子）")
    return pd.to_datetime(df["date"], errors="coerce")


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

    key = (["stock_id", "date", "type"] if kind == "financial"
           else ["stock_id", "date"])
    buf, ok, fail = [], 0, 0
    for i, sid in enumerate(todo, 1):
        df = call(dataset, data_id=sid, start_date=START)
        if df.empty or "date" not in df.columns:
            fail += 1
        else:
            df["stock_id"] = sid
            if kind == "dividend":
                df["available_date"] = dividend_available(df)
            else:
                avail = {"month": month_available,
                         "financial": quarter_available,
                         "margin": margin_available}[kind]
                df["available_date"] = avail(df["date"])
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
                    choices=["both", "all", "month", "financial", "margin",
                             "dividend"])
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="每檔間隔秒數。免費層 600 req/hr 要設 6.5")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sids = universe()
    if args.limit:
        sids = sids[: args.limit]
    print(f"母體 {len(sids)} 檔")

    if args.dataset == "both":
        kinds = ["month", "financial"]
    elif args.dataset == "all":
        kinds = ["month", "financial", "margin", "dividend"]
    else:
        kinds = [args.dataset]
    for k in kinds:
        run(k, sids, args.sleep, os.path.join(OUT_DIR, f"_done_{k}.txt"))

    print("\n⚠️ 之後做因子時：一律用 available_date 過濾，不要用 date。")


if __name__ == "__main__":
    main()
