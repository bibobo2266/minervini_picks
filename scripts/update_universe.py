"""
更新 data/universe.parquet（代號 → 名稱 / 產業 / 市場別）

為什麼要獨立一支：新版的 backfill_adj.py 是按「日期」抓價格，不再走
TaiwanStockInfo，所以 universe 不會被順帶更新。舊的 universe 停在 2148 檔，
而還原股價的母體有 2841 檔——新上市的股票在掃描結果裡會顯示成「?」。

⚠️ 別再試圖從這裡拿上市日期（已經試過，失敗）：
    TaiwanStockInfo 有一個 date 欄，看起來像上市日，實際上是「資料快照日」。
    實測 3,147 檔裡有 2,740 檔的值都是同一天（抓取當天），台積電那一列寫的是
    2026-09-04。拿它當上市日的話，每個歷史事件算出來的掛牌天數都是負的。

    fix_corporate_actions.py 曾經靠它排除轉板前的興櫃期間，結果 3,460 筆命中
    全部被判成「新上市」、建議修正 0 筆。那支現在改用價格自己的密集度推斷
    興櫃期間，不需要上市日，也不需要訂閱。

⚠️ type 是「現在的市場別」，不是「當時的」：
    一檔 2016 年在興櫃、2019 年轉上市的股票，這裡的 type 是 twse。任何回貼
    歷史的判斷都要先想過這件事——跟「產業分類是現在的分類回貼歷史」同一個坑。

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
