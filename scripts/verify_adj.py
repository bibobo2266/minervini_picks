r"""
驗證 TaiwanStockPriceAdj 是不是真的還原股價

背景：probe_datasets.py 顯示 TaiwanStockPriceAdj 回 200 有資料，但它對 2330
回傳的數字跟未還原的 TaiwanStockPrice 完全相同。這有兩種可能：

  (a) 2330 最近一次除息之後本來就不需要調整（還原因子 = 1），數字相同是對的
  (b) 那個端點其實只是把未還原資料回傳給你

差別很大。用「剛做過分割或減資的股票」就能一次分辨——那種股票的歷史價格
在還原後一定會被除以一個明顯的倍數。

測試標的取自現有 parquet 裡偵測到的 107 筆異常（單日跌逾 30%，比值接近
1.5 / 2 / 3 / 10 的整數倍，是未還原公司行動的特徵）。

結果寫進 VERIFY.md 並印在畫面上，不用複製 Actions log。

用法：
    FINMIND_TOKEN=xxx python scripts/verify_adj.py
"""
import datetime as dt
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
OUT = "VERIFY.md"
SLEEP = 2.3

# (代號, 事件日, 現有 parquet 觀察到的比值)
# 事件日前後各抓一段，看還原版的事件日「之前」有沒有被除以比值。
CASES = [
    ("6669", "2026-09-02", 2.99),    # 前收 7800 → 收 2610，疑似 1:3
    ("6696", "2026-08-31", 9.82),    # 前收  939 → 收 95.6，疑似 1:10
    ("4546", "2026-07-20", 2.07),
    ("7855", "2026-08-11", 1.87),
    ("6428", "2026-06-22", 1.50),
    ("2330", "2026-06-25", 1.00),    # 對照組：沒有分割，兩邊應該一致
]


def token() -> str:
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    if not t:
        sys.exit("缺少環境變數 FINMIND_TOKEN")
    return t


def call(dataset: str, **params) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token()}"}
    for _ in range(3):
        try:
            r = requests.get(API, params={"dataset": dataset, **params},
                             headers=headers, timeout=90)
        except requests.RequestException as e:
            print(f"    連線錯誤 {e}，20 秒後重試")
            time.sleep(20)
            continue
        if r.status_code == 402:
            print("    額度用盡，等 10 分鐘")
            time.sleep(600)
            continue
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        if js.get("status") not in (200, None):
            return pd.DataFrame()
        return pd.DataFrame(js.get("data", []))
    return pd.DataFrame()


def window(sid: str, day: str, back: int = 12, fwd: int = 3):
    d = dt.date.fromisoformat(day)
    s = (d - dt.timedelta(days=back * 2)).isoformat()
    e = (d + dt.timedelta(days=fwd * 2)).isoformat()
    raw = call("TaiwanStockPrice", data_id=sid, start_date=s, end_date=e)
    time.sleep(SLEEP)
    adj = call("TaiwanStockPriceAdj", data_id=sid, start_date=s, end_date=e)
    time.sleep(SLEEP)
    return raw, adj


def analyse(sid, day, expect):
    raw, adj = window(sid, day)
    if raw.empty or adj.empty:
        return None, [f"### {sid} @ {day}", "",
                      f"抓不到資料（raw {len(raw)} 列 / adj {len(adj)} 列）。"
                      "可能是訂閱到期或該檔沒有資料。", ""]
    for df in (raw, adj):
        df["date"] = df["date"].astype(str)
    m = raw[["date", "close"]].merge(adj[["date", "close"]], on="date",
                                     suffixes=("_原始", "_還原"))
    if m.empty:
        return None, [f"### {sid} @ {day}", "", "兩個端點的日期對不起來。", ""]
    m["倍數"] = (m["close_原始"] / m["close_還原"]).round(3)
    before = m[m["date"] < day]["倍數"]
    after = m[m["date"] >= day]["倍數"]

    b = float(before.median()) if len(before) else float("nan")
    a = float(after.median()) if len(after) else float("nan")
    # 事件日之前若被還原，原始/還原 應該 ≈ 預期比值；事件日之後應該 ≈ 1
    adjusted = (b == b) and abs(b - expect) < max(0.15, expect * 0.12) \
        and (a != a or abs(a - 1) < 0.05)
    flat = (b == b) and abs(b - 1) < 0.02

    lines = [f"### {sid} @ {day}（現有 parquet 觀察到的比值 {expect}）", "",
             f"- 事件日**之前** 原始/還原 的中位倍數：**{b:.3f}**",
             f"- 事件日**之後** 原始/還原 的中位倍數：**{a:.3f}**", ""]
    if expect == 1.0:
        lines.append("對照組：沒有公司行動，兩邊本來就該一致。"
                     + ("✅ 符合預期。" if flat else
                        f"⚠️ 倍數是 {b:.3f}，跟預期的 1 不同，值得看一下。"))
    elif adjusted:
        lines.append(f"✅ **有還原**。事件日前的原始價是還原價的 {b:.2f} 倍，"
                     f"跟預期的 {expect} 吻合。")
    elif flat:
        lines.append("❌ **沒有還原**。兩個端點回傳同樣的數字，"
                     "`TaiwanStockPriceAdj` 對這檔沒有做任何調整。")
    else:
        lines.append(f"⚠️ **判讀不明**。倍數 {b:.3f} 既不是 1 也不是 {expect}。"
                     "可能是事件日抓錯，或有多重公司行動疊加。")
    lines += ["", "```", m.tail(10).to_string(index=False), "```", ""]
    return ("有還原" if adjusted else "沒還原" if flat else "不明"), lines


def main():
    print("驗證 TaiwanStockPriceAdj 是否真的還原…\n")
    body, verdicts = [], []
    for sid, day, exp in CASES:
        print(f"  {sid} @ {day} …")
        v, lines = analyse(sid, day, exp)
        if exp != 1.0 and v:
            verdicts.append(v)
        body += lines

    n_adj = verdicts.count("有還原")
    n_raw = verdicts.count("沒還原")
    if n_adj and not n_raw:
        head = ("✅ **確認 `TaiwanStockPriceAdj` 是真的還原股價。**\n\n"
                "應該立刻改用官方還原價，停用 `build_adj.py` 的自算因子邏輯——"
                "自算版漏掉減資與分割，現有 parquet 有 107 筆未還原的公司行動。")
    elif n_raw and not n_adj:
        head = ("❌ **`TaiwanStockPriceAdj` 沒有還原**，它回的跟未還原資料相同。\n\n"
                "維持自算因子的路線，但要修 `build_adj.py`：加減資與分割、"
                "`--start` 改 2015-06-01、放寬 ratio 防呆下界到 0.05。")
    else:
        head = ("⚠️ **結果不一致**，有些檔有還原、有些沒有。"
                "在釐清之前不要改資料管線。")

    md = ["# TaiwanStockPriceAdj 驗證", "",
          f"執行時間：{dt.datetime.now():%Y-%m-%d %H:%M}", "",
          "## 結論", "", head, "",
          f"判定：有還原 {n_adj} 檔／沒還原 {n_raw} 檔／不明 "
          f"{len(verdicts) - n_adj - n_raw} 檔（不含對照組）", "",
          "## 逐檔明細", ""] + body
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + head)
    print(f"\n已寫入 {OUT}")


if __name__ == "__main__":
    main()
