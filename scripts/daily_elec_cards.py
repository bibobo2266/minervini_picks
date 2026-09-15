#!/usr/bin/env python3
"""
scripts/daily_elec_cards.py — ELEC_LARGE 每日訊號（三張卡）

規格來源：ELEC_LARGE_FINDINGS.md 第十一節「每日執行規格」，逐條對齊。
定義來源：scripts/probe_vcp.py 的 sector_ids / add_size / add_features / add_revenue，
         **不自己重寫**。自己重寫過一次，市值分層就從「該季第一個交易日排名、
         季內固定」變成「取最近一筆」，rv 的 min_periods 也掉了。

三張卡（文件的定位，不要搞反）
  卡片二  主卡    T 日收盤創 250 日新高，且 T−1 未創          n=3,150
  卡片一  加碼    創 60 日新高 ∧ rv20/rv60 < 0.60             n=259
  卡片三  濾網    創 60 日新高 ∧ 最新月營收 YoY > 20%         n=1,934
          與卡二重疊 45%、相關 0.881 —— 是卡二的品質濾網，不是獨立卡

母體（每日重算）
  七個電子產業；**排除半導體業**（SEMI 測過淘汰，裸突破超額等於零）
  族群內該季市值前 33%、收盤 ≥ 10 元、20 日均額 ≥ 3,000 萬

去重（回測有，實盤常漏）
  同一檔 20 個交易日內只進場一次
  同一天被多張卡觸發只算一筆，否則會買成兩倍部位

出場（本腳本只提供規格，不執行）
  停損 = T+1 實際開盤成交價 × 0.88，進場後立即掛出，當日盤中觸價即成交
  時間出場 = 進場日起算滿 60 個交易日，開盤市價平倉
  ⚠️ 停損不可改成「今日觸及、明日開盤清倉」，那會讓虧損分佈系統性劣於回測

  python scripts/daily_elec_cards.py
  python scripts/daily_elec_cards.py --date 2026-09-09 --no-revenue
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEDUP_BARS = 20          # 同一檔 N 個交易日內只進場一次
HOLD_BARS = 60           # 時間出場
STOP_MULT = 0.88         # 停損 = T+1 開盤 × 0.88


def load_probe():
    path = os.path.join(ROOT, "scripts", "probe_vcp.py")
    if not os.path.exists(path):
        sys.exit("找不到 scripts/probe_vcp.py —— 這支的定義來源，不能沒有它")
    spec = importlib.util.spec_from_file_location("probe_vcp", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="as-of，預設用資料最後一天")
    ap.add_argument("--out-dir", default="out/signals")
    ap.add_argument("--no-revenue", action="store_true", help="跳過卡三（月營收較慢）")
    args = ap.parse_args()

    pv = load_probe()
    pv.END = "2099-12-31"          # 研究用的固定截止日，日更要看到最新資料
    ids = pv.sector_ids("ELEC")
    print(f"ELEC 母體 {len(ids)} 檔（七個電子產業，已排除半導體業）")

    p = pv.load_sector(ids)
    p = pv.add_size(p, ids)
    keep = p[["date", "stock_id", "close", "Trading_money"]].copy()
    p = pv.add_features(p)
    if not args.no_revenue:
        try:
            p = pv.add_revenue(p, th=0.20)
        except Exception as e:                                    # noqa: BLE001
            print(f"[WARN] 月營收載入失敗，卡三停用：{e}")
            p["rev_ok"] = False
    else:
        p["rev_ok"] = False
    p = p.merge(keep, on=["date", "stock_id"], how="left")
    p["amt20"] = (p.groupby("stock_id", sort=False)["Trading_money"]
                  .transform(lambda s: s.rolling(20, min_periods=10).mean()))

    cal = sorted(p["date"].unique())
    asof = str(pd.Timestamp(args.date).date()) if args.date else cal[-1]
    if asof not in cal:
        sys.exit(f"{asof} 沒有資料")
    di = {d: i for i, d in enumerate(cal)}
    print(f"as-of {asof}")

    u = pd.read_parquet(f"{ROOT}/data/universe.parquet")
    u["stock_id"] = u["stock_id"].astype(str)
    names = dict(zip(u["stock_id"], u["stock_name"].astype(str)))

    gate = p["px_ok"] & p["liq_ok"] & (p["size"] == "LARGE")
    revcol = "rev_ok" if "rev_ok" in p.columns else None
    CARDS = [
        ("二", "主卡 裸 250 日新高", gate & p["brk250"]),
        ("一", "加碼 VCP 壓縮突破", gate & p["brk60"] & (p["vc"] < 0.60)),
    ]
    if revcol is not None and p[revcol].any():
        CARDS.append(("三", "濾網 創60高 ∧ 月營收YoY>20%",
                      gate & p["brk60"] & p[revcol]))

    # 20 日去重：要看整段歷史，不能只看今天
    today, skipped = {}, []
    for tag, desc, mask in CARDS:
        s = p[mask].sort_values(["stock_id", "date"])
        last = {}
        for sid, d in zip(s["stock_id"].to_numpy(), s["date"].to_numpy()):
            i = di[d]
            if sid in last and i - last[sid] < DEDUP_BARS:
                if d == asof:
                    skipped.append((sid, tag, DEDUP_BARS - (i - last[sid])))
                continue
            last[sid] = i
            if d == asof:
                today.setdefault(sid, []).append(tag)     # 跨卡合併成一筆

    t = p[p["date"] == asof].set_index("stock_id")
    n_gate = int(gate[p["date"] == asof].sum())
    exit_i = di[asof] + 1 + HOLD_BARS
    exit_note = (cal[exit_i] if exit_i < len(cal)
                 else f"約 {pd.Timestamp(asof) + pd.Timedelta(days=int(HOLD_BARS * 1.45)):%Y-%m-%d}（估）")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"elec_{asof}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# ELEC_LARGE 每日訊號　{asof}\n")
        f.write(f"# 規格 ELEC_LARGE_FINDINGS.md 第十一節｜定義 scripts/probe_vcp.py（未修改）\n")
        f.write(f"# 母體 {len(ids)} 檔 → 過門檻 {n_gate} 檔"
                f"（七個電子產業、排除半導體、≥10元、20日均額≥3000萬、族群內每季市值前33%）\n")
        f.write(f"#\n# === 明日買進清單　{len(today)} 檔（已跨卡合併、已做 20 日去重） ===\n")
        f.write(f"# 停損 = T+1 實際開盤成交價 × {STOP_MULT}（開盤後回填，立即掛出，盤中觸價成交）\n")
        f.write(f"# 時間出場 = 進場日起算滿 {HOLD_BARS} 個交易日 → {exit_note}\n")
        f.write(f"# 部位比例不指定，由你自己決定\n")
        for sid in sorted(today):
            r = t.loc[sid]
            f.write(f"{sid} {names.get(sid, '')} 收{r['close']:.2f}"
                    f" 均額{r['amt20']/1e8:.2f}億 rv比{r['vc']:.2f}"
                    f" 卡{'+'.join(today[sid])}\n")
        f.write("#\n# === 因 20 日去重被略過（供對帳） ===\n")
        for sid, tag, left in sorted(set(skipped)):
            f.write(f"# {sid} {names.get(sid, '')} 卡{tag}　還要等 {left} 個交易日\n")
        if not skipped:
            f.write("# 無\n")
    print(f"明日買進 {len(today)} 檔　略過 {len(set(skipped))} 檔　→ {out}")
    for sid in sorted(today):
        print(f"  {sid} {names.get(sid, '')}　卡{'+'.join(today[sid])}")
    if skipped:
        print("  略過：" + "、".join(f"{s}卡{tg}(還{l}日)" for s, tg, l in sorted(set(skipped))))


if __name__ == "__main__":
    main()
