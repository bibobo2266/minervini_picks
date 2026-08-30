"""
更新 data/universe.parquet（代號 → 名稱 / 產業 / 市場別）

為什麼要獨立一支：新版的 backfill_adj.py 是按「日期」抓價格，不再走
TaiwanStockInfo，所以 universe 不會被順帶更新。舊的 universe 停在 2148 檔，
而還原股價的母體有 2841 檔——新上市的股票在掃描結果裡會顯示成「?」。

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

    keep = ["stock_id", "stock_name", "industry_category", "type"]
    df = df.reindex(columns=keep)
    # 不在這裡過濾四碼普通股：三個 app 各自會濾，universe 保留完整對照表，
    # 這樣 ETF、興櫃出現在名單裡時至少查得到名字，不會變成「?」
    df = df.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
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


if __name__ == "__main__":
    main()
