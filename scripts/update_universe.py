"""
更新 data/universe.parquet（代號 → 名稱 / 產業 / 市場別 / 上市日）

為什麼要獨立一支：新版的 backfill_adj.py 是按「日期」抓價格，不再走
TaiwanStockInfo，所以 universe 不會被順帶更新。舊的 universe 停在 2148 檔，
而還原股價的母體有 2841 檔——新上市的股票在掃描結果裡會顯示成「?」。

⚠️ 為什麼一定要留 listing_date：
    `type` 欄是「現在的市場別」，不是「當時的」。一檔 2016 年在興櫃、2019 年
    轉上市的股票，現在 type 是 twse，但它 2016 那段期間沒有漲跌幅限制。
    fix_corporate_actions.py 靠「單日逾 11% 不可能」偵測公司行動，如果只看
    type，這批股票的興櫃期間會整段被誤判——實測命中 2,078 筆，其中 6550 一檔
    就佔 58 筆。一檔股票不可能有 58 次公司行動。

    TaiwanStockInfo 的 date 欄就是上市（櫃）日期，免費層就有，之前被 keep
    清單丟掉了。有了它才能把興櫃期間整段排除。

每月跑一次就夠（新上市不會太頻繁），或發現名單出現「?」時手動跑。

用法：
    FINMIND_TOKEN=xxx python scripts/update_universe.py
"""
import os
import sys

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
OUT = "data/universe.parquet"


def main():
    tok = os.environ.get("FINMIND_TOKEN", "").strip()
    if not tok:
        sys.exit("缺少環境變數 FINMIND_TOKEN")

    r = requests.get(API, params={"dataset": "TaiwanStockInfo"},
                     headers={"Authorization": f"Bearer {tok}"}, timeout=90)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("data", []))
    if df.empty:
        sys.exit("TaiwanStockInfo 回傳空資料")

    if "date" in df.columns:
        df = df.rename(columns={"date": "listing_date"})
    else:
        print("⚠️ TaiwanStockInfo 沒有 date 欄，興櫃期間將無法排除")
        df["listing_date"] = pd.NaT

    keep = ["stock_id", "stock_name", "industry_category", "type", "listing_date"]
    df = df.reindex(columns=keep)
    # 不在這裡過濾四碼普通股：三個 app 各自會濾，universe 保留完整對照表，
    # 這樣 ETF、興櫃出現在名單裡時至少查得到名字，不會變成「?」
    df = df.dropna(subset=["stock_id"])
    # 同一代號若有多列（轉板會留下歷史列），保留最晚的上市日——那是現在這個
    # 市場別的掛牌日，也就是漲跌幅限制真正開始適用的那天
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df = (df.sort_values(["stock_id", "listing_date"])
            .drop_duplicates("stock_id", keep="last"))
    df["listing_date"] = df["listing_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("stock_id").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    before = 0
    if os.path.exists(OUT):
        before = len(pd.read_parquet(OUT))
    df.to_parquet(OUT, index=False, compression="zstd")

    print(f"universe：{before} → {len(df)} 檔（{len(df) - before:+d}）")
    print(df["type"].value_counts().to_dict())
    common = df[df["stock_id"].str.match(r"^[1-9]\d{3}$")]
    print(f"四碼普通股 {len(common)} 檔")

    # 稽核：上市日拿到幾檔、是不是真的長得像上市日
    have = df["listing_date"].notna().sum()
    print(f"有上市日的 {have} 檔（{have / len(df):.0%}）")
    if have:
        ld = pd.to_datetime(df["listing_date"], errors="coerce").dropna()
        print(f"上市日範圍 {ld.min().date()} → {ld.max().date()}")
        print(f"2015-06 之後掛牌的 {int((ld >= '2015-06-01').sum())} 檔"
              "（這批的早期資料多半是興櫃）")


if __name__ == "__main__":
    main()
