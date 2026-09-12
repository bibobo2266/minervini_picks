#!/usr/bin/env python3
"""
scripts/build_features.py — 特徵 panel

把 C01–C12 與結構旗標寫成 (date, stock_id) 的長表，之後 combo 只是對它的查詢，
改規則不用重算特徵。

兩趟：
  Pass 1  逐 chunk 算特徵 → data/features/part_XXX.parquet
  Pass 2  逐年讀回，算「同日全市場橫斷面分位」→ data/features/features_YYYY.parquet
          橫斷面分位在當日收盤就完整，沒有前視。

台股特有狀態（手冊沒有，這裡補上）：
  漲跌停偵測。還原股價不會破壞 ±10%：實測 2025 年 |ret|>9.9% 有 6,459 日，
  >10.0% 只剩 757 日，斷崖乾淨。興櫃無漲跌幅限制，一律排除。

  python scripts/build_features.py                 # 全量重建
  python scripts/build_features.py --years 2026    # 只重建某幾年
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combo_core as cc  # noqa: E402

OUT = "data/features"

# panel 要留下的欄位。C01–C12 一旦凍結就不再改定義，只准新增。
KEEP = [
    "date", "stock_id",
    "open", "high", "low", "close", "vol", "amount",
    "ma20", "ma60", "ma120", "atr", "B", "L", "low20",
    "c01_tight", "c01_base_depth", "c02_tests", "c03_cq",
    "c05_volratio", "c05_contract",
    "c06_rs", "c06_rs20", "c07_er", "c07_net",
    "c08_bo_count", "c08_days_from_low", "c08_gain_from_low",
    "c11_bars_recover", "c11_days_elapsed",
    "c12_days_since_high", "c12_decay",
    "ret", "idx_ret",
    "above_B", "below_L", "higher_lows", "lower_lows", "higher_highs",
    "recent_swing_low", "prior_swing_low",
    "shake_depth", "shake_recovered", "shake_broken",
    "had_breakout_recent", "had_failed_breakout", "big_gap", "gap_filled",
]

# 同日全市場橫斷面分位要算的欄位。只挑「跨股可比」的，價位類不算。
PCT_COLS = [
    "c01_tight", "c01_base_depth", "c02_tests", "c03_cq",
    "c05_volratio", "c05_contract", "c06_rs", "c06_rs20",
    "c07_er", "c08_bo_count", "c08_gain_from_low", "c12_days_since_high",
]

TW_COLS = ["limit_up", "limit_down", "limit_locked", "limit_opened", "limit_streak"]


def tw_states(d: pd.DataFrame) -> pd.DataFrame:
    """台股漲跌停狀態。只對 twse/tpex 有效，興櫃由呼叫端排除。

    不用「等於漲停價」判定，因為還原股價不落在 tick 上。改用報酬帶 + 日內結構：
      漲停     ret 落在 +9.2~10.3%
      鎖到收盤 收盤 = 當日最高
      打開     曾漲停但收盤低於最高
      連鎖天數 連續漲停的第幾根
    """
    r = d["ret"].to_numpy()
    hi, lo, cl = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    up = (r > 9.2) & (r < 10.3)
    dn = (r < -9.2) & (r > -10.3)
    locked = up & (cl >= hi - 1e-9)
    opened = up & (cl < hi - 1e-9)
    streak = np.zeros(len(d), dtype=np.int16)
    n = 0
    for i, u in enumerate(up):
        n = n + 1 if u else 0
        streak[i] = n
    d["limit_up"], d["limit_down"] = up, dn
    d["limit_locked"], d["limit_opened"] = locked, opened
    d["limit_streak"] = streak
    # 跌停也用同一套結構，鎖死 = 收盤等於當日最低
    d["limit_down_locked"] = dn & (cl <= lo + 1e-9)
    return d


def pass1(args, reg, idx, ids):
    os.makedirs(OUT, exist_ok=True)
    if args.fresh:
        for f in glob.glob(os.path.join(OUT, "part_*.parquet")):
            os.remove(f)
    years = args.all_years          # 特徵一律用完整歷史算（B 是 250 日窗）
    done = 0
    for ci in range(0, len(ids), args.chunk):
        # 斷點續跑：這個 chunk 已經有 part 檔就跳過（容器 OOM 後可直接重跑）
        if glob.glob(os.path.join(OUT, f"part_*_{ci:05d}.parquet")):
            print(f"  skip chunk {ci:05d}（已存在）", flush=True)
            continue
        chunk = set(ids[ci:ci + args.chunk])
        parts = []
        for y in years:
            p = f"{args.adj_dir}/prices_adj_{y}.parquet"
            if not os.path.exists(p):
                continue
            d = pd.read_parquet(p)
            d = d[d["stock_id"].astype(str).isin(chunk)]
            if len(d):
                parts.append(d)
            del d
        if not parts:
            continue
        px = cc.normalize_ohlcv(pd.concat(parts, ignore_index=True))
        del parts
        rows = []
        for sid, g in px.groupby("stock_id", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            if len(g) < args.min_bars:
                continue
            f = cc.compute_features(g, idx, reg)
            f = tw_states(f)
            f["stock_id"] = str(sid)
            keep = [c for c in KEEP if c in f.columns] + TW_COLS + ["limit_down_locked"]
            f = f[keep].iloc[args.warmup:]
            if len(f):
                rows.append(f)
            del f
            done += 1
        if rows:
            out = pd.concat(rows, ignore_index=True)
            for c in out.columns:
                if out[c].dtype == "float64":
                    out[c] = out[c].astype("float32")
            out["_y"] = pd.to_datetime(out["date"]).dt.year
            # 先按年份切檔，pass2 只讀自己那年的 part，不用重複掃全部
            for y, gy in out.groupby("_y"):
                if y not in args.year_list:
                    continue
                gy.drop(columns=["_y"]).to_parquet(
                    os.path.join(OUT, f"part_{y}_{ci:05d}.parquet"), index=False)
            del out
        del px, rows
        gc.collect()
        print(f"  pass1 {done} 檔 ({ci + args.chunk}/{len(ids)})", flush=True)
    return done


def pass2(args):
    """逐年合併並加上同日橫斷面分位。"""
    for y in args.year_list:
        files = sorted(glob.glob(os.path.join(OUT, f"part_{y}_*.parquet")))
        if not files:
            continue
        p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for c in PCT_COLS:
            if c not in p.columns:
                continue
            p[c + "_pct"] = (p.groupby("date")[c]
                             .rank(pct=True, na_option="keep") * 100).astype("float32")
        p = p.sort_values(["stock_id", "date"]).reset_index(drop=True)
        dst = os.path.join(OUT, f"features_{y}.parquet")
        p.to_parquet(dst, index=False)
        print(f"  pass2 {y}: {len(p):,} 列 → {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")
        del p
        gc.collect()
    for f in glob.glob(os.path.join(OUT, "part_*.parquet")):
        os.remove(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj-dir", default="data/adj")
    ap.add_argument("--index", default="data/futures/index_taiex.parquet")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--years", default="", help="逗號分隔；空白 = 全部")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--fresh", action="store_true", help="清掉既有 part 檔重跑")
    ap.add_argument("--min-bars", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=300,
                    help="前 N 根不輸出。B 是 250 日前高，不足就不該判。")
    ap.add_argument("--min-amt", type=float, default=0,
                    help=">0 時只保留 20 日均額達標的股票（0 = 全收，篩選留給查詢端）")
    args = ap.parse_args()

    all_years = sorted(int(os.path.basename(p)[11:15])
                       for p in glob.glob(f"{args.adj_dir}/prices_adj_*.parquet"))
    args.all_years = all_years
    args.year_list = ([int(y) for y in args.years.split(",")] if args.years else all_years)

    reg = cc.load_registry(args.registry)
    idx = cc.load_index(args.index)
    u = pd.read_parquet("data/universe.parquet")
    u["stock_id"] = u["stock_id"].astype(str)
    # 興櫃無漲跌幅限制，漲跌停欄位會失真 → 整個排除
    u = u[(u["stock_id"].str.len() == 4) & (u["type"].isin(["twse", "tpex"]))]
    ids = sorted(u["stock_id"].unique())
    print(f"母體 {len(ids)} 檔（四碼上市櫃，已排除興櫃）　年份 {args.year_list}")

    n = pass1(args, reg, idx, ids)
    print(f"pass1 完成 {n} 檔")
    pass2(args)

    px_last = None
    ip = args.index
    if os.path.exists(ip):
        di = pd.read_parquet(ip, columns=["date"])
        ix = pd.to_datetime(di["date"], format="mixed").max()
        fs = sorted(glob.glob(os.path.join(OUT, "features_*.parquet")))
        if fs:
            dd = pd.read_parquet(fs[-1], columns=["date"])
            px_last = pd.to_datetime(dd["date"]).max()
            print(f"panel 最新日 {px_last.date()}　指數最新日 {ix.date()}")
            if ix < px_last:
                print(f"::warning::指數落後 panel，c06_rs 在該區間為缺值")


if __name__ == "__main__":
    main()
