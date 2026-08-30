"""
探測 FinMind 免費層能拿到哪些資料集

背景：TaiwanStockPriceAdj（還原股價）回 400「Your level is register」，
確認免費層不給。備案是抓除權息資料自己算調整因子——但要先確認拿不拿得到。

這支只打幾次 API，30 秒就有答案，不寫任何檔案。

用法：
    FINMIND_TOKEN=xxx python scripts/probe_datasets.py
"""
import os
import sys

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"

# 候選：台股除權息相關的資料集。名稱不確定的都試，反正一次呼叫很便宜。
CANDIDATES = [
    ("TaiwanStockDividend", dict(data_id="2330", start_date="2017-01-01")),
    ("TaiwanStockDividendResult", dict(data_id="2330", start_date="2017-01-01")),
    ("TaiwanStockCapitalReductionReferencePrice",
     dict(data_id="2330", start_date="2017-01-01")),
    ("TaiwanStockPrice", dict(data_id="2330", start_date="2026-08-01")),   # 對照組，應該要成功
    ("TaiwanStockPriceAdj", dict(data_id="2330", start_date="2026-08-01")),  # 對照組，應該要失敗
]


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def probe(dataset: str, params: dict):
    headers = {"Authorization": f"Bearer {token()}"}
    try:
        r = requests.get(API, params={"dataset": dataset, **params},
                         headers=headers, timeout=60)
    except requests.RequestException as e:
        return "連線錯誤", str(e), None

    if r.status_code != 200:
        try:
            msg = r.json().get("msg", "")
        except Exception:
            msg = r.text[:150]
        return f"HTTP {r.status_code}", msg, None

    js = r.json()
    data = js.get("data", [])
    if not data:
        return "空資料", str(js.get("msg", "")), None
    df = pd.DataFrame(data)
    return "OK", f"{len(df)} 列", df


def main():
    print("=" * 70)
    for ds, params in CANDIDATES:
        status, msg, df = probe(ds, params)
        mark = "✅" if status == "OK" else "❌"
        print(f"{mark} {ds:48s} {status}")
        if msg:
            print(f"      {msg[:120]}")
        if df is not None:
            print(f"      欄位：{list(df.columns)}")
            print(f"      範例：{df.head(2).to_dict('records')}")
        print("-" * 70)

    print("\n判讀：")
    print("  除權息類的資料集若有 ✅ → 可以自己算調整因子，不用付費")
    print("  全部 ❌ → 要嘛付費贊助，要嘛放棄還原股價（把限制寫進手冊）")


if __name__ == "__main__":
    main()
