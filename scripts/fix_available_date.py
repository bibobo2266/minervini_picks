r"""
就地修正 data/fundamentals/ 的 available_date 欄位（不用重抓）

為什麼需要：
    第一版 backfill_fundamentals.py 誤以為月營收的 date 是「營收所屬月份」，
    又往後推了一個月，結果 available_date 晚了整整一個月。
    實際上 FinMind 的 date 已經是營收所屬月份的「次月 1 日」——
    實測 26.5 萬列全部如此（revenue_month=7 對應 date=2026-08-01）。

    方向是安全的（過度保守不會造成前視偏誤），但月營收因子的價值有一大半
    在新鮮度，晚一個月等於自廢武功。

    原始資料都在，只有 available_date 那一欄算錯，所以就地重算即可。

修正後：
    月營收  date + 14 天（法定次月 10 日前，留五天緩衝）
            例：7 月營收 → date 2026-08-01 → available_date 2026-08-15
    季報    不變（Q1 5/15、Q2 8/14、Q3 11/14、年報次年 3/31）

用法：
    python scripts/fix_available_date.py --dry-run
    python scripts/fix_available_date.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backfill_fundamentals import month_available, quarter_available

DIR = "data/fundamentals"
FILES = {"month_revenue.parquet": month_available,
         "financials.parquet": quarter_available}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for fname, fn in FILES.items():
        path = os.path.join(DIR, fname)
        if not os.path.exists(path):
            print(f"{fname}：不存在，跳過")
            continue
        d = pd.read_parquet(path)
        old = d["available_date"].copy() if "available_date" in d.columns else None
        new = fn(d["date"]).dt.strftime("%Y-%m-%d")

        if old is not None:
            diff = (old != new)
            shift = (pd.to_datetime(new) - pd.to_datetime(old)).dt.days
            print(f"\n{fname}：{len(d):,} 列，其中 {int(diff.sum()):,} 列會變動")
            if diff.any():
                print(f"  平移天數分佈：{shift[diff].value_counts().head(3).to_dict()}")
                sample = d.loc[diff, ["stock_id", "date"]].head(3)
                for i, row in sample.iterrows():
                    print(f"  例：{row['stock_id']} 期別 {row['date']}　"
                          f"{old[i]} → {new[i]}")
        else:
            print(f"\n{fname}：原本沒有 available_date 欄，新增")

        if args.dry_run:
            print("  （dry-run，未寫入）")
            continue
        d["available_date"] = new
        d.to_parquet(path, index=False, compression="zstd")
        print("  已寫入")

    print("\n⚠️ 做因子時一律用 available_date 過濾，不要用 date。")


if __name__ == "__main__":
    main()
