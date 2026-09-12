#!/usr/bin/env python3
"""
scripts/build_states.py — 狀態表 + 殘差桶 + 查詢報表

讀 data/features/ 的 panel，對每一個 (date, stock_id) 判定 combo，輸出：

  data/states/states_YYYY.parquet    每日主狀態、次要標籤、episode 編號
  data/states/residual_YYYY.parquet  一組都沒命中的日子（含特徵向量）→ 之後分群補組
  reports/STATES.md                  episode 統計、產業分佈、狀態轉移矩陣

三個設計決定，都跟「能不能計數」有關：

1. 主狀態唯一。60 組會同時亮好幾個，要能 groupby 就必須挑一個。
   用手冊第 26 頁的五層排序（可用性 → 結構位置 → 當日與背景 → 相對與防守 →
   事後更新與基本面），不投票、不計分、不平均。其餘命中的存成 secondary。

2. 計數單位是 episode 不是 day。40/60 組的回看窗是 250 根，一亮會連亮好幾週。
   用「日」計數的話，長窗組會被自己的自相關灌成 80% 轉移到自己，那是廢話。
   episode = 同一檔同一組連續命中的一段（中斷 gap 天以內視為同一段）。

3. 驗證器（J 類 56–60）不進轉移統計。它們描述資料品質，不描述價格方向。

  python scripts/build_states.py                    # 全部年份
  python scripts/build_states.py --years 2026 --report
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combo_core as cc  # noqa: E402

OUT = "data/states"

# panel 沒有、但 combo 規則會引用的欄位。補成缺值，語意才會是 PARTIAL 而不是「規則壞掉」。
MISSING_FLOAT = ["c05_high_zone", "c06_rs_ind", "ind_ret", "c09_excess",
                 "c10_mae3", "c10_mfe3", "ev_ret"]
MISSING_BOOL = ["ev_neg", "ev_pos", "ev_after_close", "ev_contract_binding",
                "ev_mou", "ev_guidance", "ev_target_met", "ev_cashflow_flag",
                "ev_governance", "definition_dispute", "corporate_action_nearby",
                "future_data_present"]

# 殘差桶要保留的特徵欄位（之後拿去分群）
RESID_COLS = ["c01_tight", "c01_base_depth", "c02_tests", "c03_cq",
              "c05_volratio", "c05_contract", "c06_rs", "c07_er",
              "c08_bo_count", "c08_gain_from_low", "c11_bars_recover",
              "c12_days_since_high", "shake_depth",
              "above_B", "below_L", "higher_lows", "lower_lows", "higher_highs"]

VALIDATORS = {"56", "57", "58", "59", "60"}


def prep(p: pd.DataFrame) -> pd.DataFrame:
    for c in MISSING_FLOAT:
        if c not in p.columns:
            p[c] = np.float32(np.nan)
    for c in MISSING_BOOL:
        if c not in p.columns:
            p[c] = False
    # J 類要的資料品質旗標，從 panel 現有欄位推
    if "has_c03_c05" not in p.columns:
        p["has_c03_c05"] = p["c03_cq"].notna() & p["c05_volratio"].notna()
    if "lacks_background" not in p.columns:
        p["lacks_background"] = (p["c01_tight"].isna() | p["c02_tests"].isna()
                                 | p["c08_bo_count"].isna())
    if "same_day_evidence_only" not in p.columns:
        p["same_day_evidence_only"] = p["lacks_background"] & p["has_c03_c05"]
    if "c11_days_elapsed" not in p.columns:
        p["c11_days_elapsed"] = np.float32(np.nan)
    return p


def build_year(y: int, args, reg, order, meta):
    src = f"{args.features}/features_{y}.parquet"
    if not os.path.exists(src):
        return None, None
    p = pd.read_parquet(src)
    p["date"] = pd.to_datetime(p["date"])
    p["stock_id"] = p["stock_id"].astype(str)
    p = prep(p).sort_values(["stock_id", "date"]).reset_index(drop=True)

    res = cc.evaluate_vectorized(p, reg)
    n = len(p)

    # --- 主狀態：照優先序挑第一個命中的
    primary = np.full(n, "", dtype=object)
    primary_cls = np.full(n, "", dtype=object)
    n_match = np.zeros(n, dtype=np.int16)
    sec = [[] for _ in range(n)]
    for cid in order:                       # order 已經照 priority 排好
        m = res[cid]["match"]
        if not m.any():
            continue
        n_match += m.astype(np.int16)
        first = m & (primary == "")
        primary[first] = cid
        primary_cls[first] = meta[cid]["cls"]
        for i in np.where(m & ~first)[0]:
            sec[i].append(cid)

    out = pd.DataFrame({
        "date": p["date"], "stock_id": p["stock_id"],
        "primary": primary, "primary_cls": primary_cls,
        "primary_dir": [meta[c]["dir"] if c else "" for c in primary],
        "n_match": n_match,
        "secondary": [",".join(s) for s in sec],
        "close": p["close"].astype("float32"),
        "limit_up": p.get("limit_up", pd.Series(False, index=p.index)),
        "limit_down": p.get("limit_down", pd.Series(False, index=p.index)),
    })

    # --- episode：同檔同組連續命中視為一段，中斷 <= gap 天仍算同段
    out["episode_id"] = ""
    has = out["primary"] != ""
    sub = out[has]
    if len(sub):
        key = sub["stock_id"] + "|" + sub["primary"]
        newseg = (key != key.shift(1))
        gapdays = sub.groupby(key.values).cumcount()  # 佔位，實際用日期間隔判斷
        d = sub["date"].values
        brk = np.ones(len(sub), bool)
        brk[1:] = (key.values[1:] != key.values[:-1]) | (
            (d[1:] - d[:-1]).astype("timedelta64[D]").astype(int) > args.gap)
        eid = np.cumsum(brk)
        out.loc[has, "episode_id"] = [f"{y}-{e:06d}" for e in eid]
        del gapdays, newseg

    # --- 殘差桶
    resid = p.loc[~has.values, ["date", "stock_id"] +
                  [c for c in RESID_COLS if c in p.columns]].copy()

    os.makedirs(OUT, exist_ok=True)
    out.to_parquet(f"{OUT}/states_{y}.parquet", index=False)
    if len(resid):
        resid.to_parquet(f"{OUT}/residual_{y}.parquet", index=False)
    rate = 100 * (~has).mean()
    print(f"  {y}: {len(out):,} 檔-日　有主狀態 {int(has.sum()):,}　"
          f"殘差 {int((~has).sum()):,}（{rate:.1f}%）")
    del p, res
    return out, rate


def report(args, meta):
    files = sorted(glob.glob(f"{OUT}/states_*.parquet"))
    if not files:
        return
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s["date"] = pd.to_datetime(s["date"])
    if args.report_start:
        s = s[s["date"] >= pd.Timestamp(args.report_start)]
    u = pd.read_parquet("data/universe.parquet")
    u["stock_id"] = u["stock_id"].astype(str)
    s = s.merge(u[["stock_id", "stock_name", "industry_category"]],
                on="stock_id", how="left")

    L = [f"# 狀態表報告　{args.report_start or '全期間'} 起",
         f"\n樣本 {len(s):,} 檔-日、{s['stock_id'].nunique()} 檔。"
         f"殘差率（一組都沒命中）**{100 * (s['primary'] == '').mean():.1f}%**。\n"]

    hit = s[s["primary"] != ""]

    # episode 統計
    ep = hit.groupby(["primary", "episode_id"]).agg(
        天數=("date", "size")).reset_index()
    t = ep.groupby("primary").agg(episode數=("episode_id", "size"),
                                  平均天數=("天數", "mean")).reset_index()
    t["天數"] = hit.groupby("primary").size().values
    t["類"] = [meta[c]["cls"] for c in t["primary"]]
    t["名稱"] = [meta[c]["name"] for c in t["primary"]]
    t["平均天數"] = t["平均天數"].round(1)
    t = t.sort_values("episode數", ascending=False)
    L.append("## 每組的 episode 數與持續長度\n")
    L.append("_episode = 同一檔連續命中的一段。天數/episode 越大代表狀態越黏，"
             "長窗組天生就黏。_\n")
    L.append(t[["primary", "類", "名稱", "episode數", "天數", "平均天數"]]
             .to_markdown(index=False))

    # 產業分佈
    L.append("\n\n## 產業 × 主狀態（各產業最常見的三組）\n")
    g = (hit.groupby(["industry_category", "primary"]).size()
         .reset_index(name="n"))
    tot = g.groupby("industry_category")["n"].transform("sum")
    g["占比%"] = (100 * g["n"] / tot).round(1)
    rows = []
    for ind, gg in g.groupby("industry_category"):
        if gg["n"].sum() < 500:
            continue
        for _, r in gg.nlargest(3, "n").iterrows():
            rows.append({"產業": ind, "combo": r["primary"],
                         "名稱": meta[r["primary"]]["name"],
                         "檔-日": int(r["n"]), "占比%": r["占比%"]})
    if rows:
        L.append(pd.DataFrame(rows).to_markdown(index=False))

    # 轉移矩陣（episode 層級，排除驗證器）
    L.append("\n\n## 狀態轉移（episode 層級）\n")
    L.append("_這一段結束後，下一段是什麼。驗證器 56–60 已排除。_\n")
    h = hit[~hit["primary"].isin(VALIDATORS)].sort_values(["stock_id", "date"])
    last = h.drop_duplicates(["episode_id"], keep="last")[
        ["stock_id", "date", "primary", "episode_id"]]
    last = last.sort_values(["stock_id", "date"])
    last["next"] = last.groupby("stock_id")["primary"].shift(-1)
    last["next_gap"] = (last.groupby("stock_id")["date"].shift(-1)
                        - last["date"]).dt.days
    tr = last[(last["next"].notna()) & (last["next_gap"] <= args.trans_gap)]
    if len(tr):
        m = (tr.groupby(["primary", "next"]).size().reset_index(name="n"))
        m["占比%"] = (100 * m["n"] / m.groupby("primary")["n"]
                     .transform("sum")).round(1)
        rows = []
        for cid, gg in m.groupby("primary"):
            if gg["n"].sum() < 50:
                continue
            for _, r in gg.nlargest(3, "n").iterrows():
                rows.append({"從": f"{cid} {meta[cid]['name']}",
                             "到": f"{r['next']} {meta[r['next']]['name']}",
                             "次數": int(r["n"]), "占比%": r["占比%"]})
        if rows:
            L.append(pd.DataFrame(rows).to_markdown(index=False))

    L.append("\n\n---\n_狀態統計，不含報酬、不含勝率。轉移機率是歷史頻率，"
             "不是預測。_\n")
    os.makedirs("reports", exist_ok=True)
    open("reports/STATES.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n寫入 reports/STATES.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("--years", default="")
    ap.add_argument("--gap", type=int, default=5,
                    help="episode 中斷幾個日曆天以內仍算同一段")
    ap.add_argument("--trans-gap", type=int, default=60,
                    help="轉移統計中，兩段相隔超過幾天就不算接續")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--report-start", default="2021-01-01")
    args = ap.parse_args()

    reg = cc.load_registry(args.registry)
    meta = {c["id"]: c for c in reg["combos"]}
    pri = {c: i for i, c in enumerate(reg["priority"])}
    order = sorted(meta, key=lambda c: (pri.get(meta[c]["cls"], 99), c))

    years = ([int(y) for y in args.years.split(",")] if args.years else
             sorted(int(os.path.basename(f)[9:13])
                    for f in glob.glob(f"{args.features}/features_*.parquet")))
    print(f"年份 {years}　優先序 {reg['priority']}")
    rates = []
    for y in years:
        _, r = build_year(y, args, reg, order, meta)
        if r is not None:
            rates.append(r)
    if rates:
        print(f"\n平均殘差率 {np.mean(rates):.1f}%　"
              f"（目標 <10%；高於此代表 60 組覆蓋不足，看 residual_*.parquet）")
    if args.report:
        report(args, meta)


if __name__ == "__main__":
    main()
