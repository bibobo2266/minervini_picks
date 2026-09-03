r"""
每日 Minervini 選股報表：CSV + 教學圖 ZIP

在 GitHub Actions 上 headless 執行，用的是 minervini_core / teach_core，
跟 Streamlit app 完全同一份判斷邏輯——不會出現「app 上是這樣、報表是那樣」。

排在 daily_adj 之後跑（17:23 更新資料 → 17:40 產報表），
所以讀到的是當天收盤。

用法：
    python scripts/report_minervini.py
    python scripts/report_minervini.py --top 10 --liq 5000
    python scripts/report_minervini.py --no-charts       # 只出 CSV，快
"""
import argparse
import datetime as dt
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minervini_core as M   # noqa: E402

# teach_core 在 build_charts 裡才 import。
# 它會拉進 matplotlib，而 watchlist_update.py 只用得到 scan_today、不畫圖，
# 在模組層 import 的話 watchlist 那支 workflow 就得多裝 matplotlib 才跑得動
# （2026-09-02 就是這樣掛掉的）。

OUT = "out"
ORDER = {"觸發": 0, "準備": 1, "觀察": 2}


def build_charts(ids: list, names: dict, outdir: str, weeks: int = 52) -> list:
    """對名單逐檔出教學圖。單檔失敗不讓整份報表掛掉——
    新股歷史不足、停牌沒資料都是正常會遇到的事。"""
    import teach_core as T      # 只有出圖才需要，見檔案上方說明
    import matplotlib.pyplot as plt

    made = []
    idx_mkt, rs_rank_all = T.market_context()
    for sid in ids:
        try:
            daily = T.load_one(sid)
            if daily.empty:
                print(f"  跳過 {sid}：無資料")
                continue
            wk = T.to_weekly(daily)
            if len(wk) < T.MA_WEEKS + T.SLOPE_LAG + 2:
                print(f"  跳過 {sid}：週線僅 {len(wk)} 根")
                continue
            rs, rs_nh = T.rs_line(wk, idx_mkt)
            rs_rank = float(rs_rank_all.get(sid, float("nan"))) \
                if len(rs_rank_all) else float("nan")
            segs = T.smooth_segments(T.stage_series(wk), 6)
            legs, ok = T.vcp_contractions(wk, 26, 6.0)
            dtbl, _ = T.deduct_table(wk, 5, 5)
            m = T.read_metrics(daily, wk, dtbl, 12, rs_rank, rs_nh)
            bx = T.box_stats(wk, 12, 20)
            vz = T.verdicts(m, bx, segs[-1][2] if segs else 0, ok, legs)
            fig = T.build_figure(sid, names.get(sid, ""), daily, wk, segs, dtbl,
                                 m, bx, vz, legs, ok, rs, rs_nh, True, True,
                                 weeks, 5)
            p = os.path.join(outdir, f"{sid}_{names.get(sid, '')}.png")
            fig.savefig(p, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            made.append(p)
            print(f"  圖 {sid} {names.get(sid, '')}")
        except Exception as e:
            print(f"  跳過 {sid}：{type(e).__name__} {e}")
    return made


def scan_today(liq: float = 5000, min_tt: int = 8, m=None, uni=None,
               liq_pct: float = 0) -> pd.DataFrame:
    """跑一次完整掃描，回傳非淘汰名單（含狀態）。
    抽出來讓 watchlist_update.py 共用——名單判定和報表必須看同一份數字，
    不然會出現「報表說觸發、追蹤表說沒有」。

    m / uni 是給 watchlist_replay.py 用的：回放時餵進已切到某一天的矩陣，
    這裡就會照那天的資料判定。不給就跟原本一樣抓最新。

    liq_pct > 0 時改用百分位流動性門檻（見 minervini_core.scan）。跨年度
    回測必須用它，否則母體大小會跟著市場成交量漂移。預設 0 = 沿用絕對金額，
    每日實跑的行為不變。"""
    if m is None:
        m, _ = M.build_matrices()
    if uni is None:
        uni = M.load_universe()
    base = M.scan(m, liq, liq_pct=liq_pct)
    if base.empty:
        return pd.DataFrame()

    rows = []
    for sid in base[base["TT分"] >= min_tt].index:
        px = pd.DataFrame({k: m[kk][sid] for k, kk in
                           [("close", "c"), ("max", "h"), ("min", "l"),
                            ("Trading_Volume", "v")]}).dropna()
        if len(px) < 60:
            continue
        stg, ma30w, _ = M.stage_of(px)
        f = M.vcp_foot(px)
        b = base.loc[sid]
        rows.append(dict(
            代號=sid,
            名稱=uni.loc[sid, "stock_name"] if sid in uni.index else "?",
            產業=uni.loc[sid, "industry_category"] if sid in uni.index else "",
            收盤=b["收盤"], RS=b["RS"], TT分=int(b["TT分"]), 階段=stg or 0,
            MA30W=round(float(ma30w), 2) if ma30w == ma30w else None,
            距高點=b["距高點"], 量比=b["量比"], 量增=bool(b["量增"]),
            VCP=f["ok"], 足跡=f["foot"], 樞紐價=f["pivot"], 近樞紐=f["near"],
            突破=bool(b["收盤"] >= (f["pivot"] or 1e9)),
            底部序=M.base_count(px), 均額億=b["均額億"], 新股=len(px) < 300,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["狀態"] = df.apply(M.classify, axis=1)
    df["晚期"] = df["底部序"] >= 4
    df = df[df["狀態"] != "淘汰"]
    return (df.assign(_o=df["狀態"].map(ORDER))
              .sort_values(["_o", "TT分", "RS"], ascending=[True, False, False])
              .drop(columns="_o").reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liq", type=float, default=5000, help="60日均額門檻（萬元）")
    ap.add_argument("--liq-pct", type=float, default=0,
                    help="改用百分位門檻：25 = 取成交值前 25%%。0 = 用 --liq")
    ap.add_argument("--min-tt", type=int, default=8, help="最低趨勢模板分數")
    ap.add_argument("--top", type=int, default=20, help="最多出幾張圖")
    ap.add_argument("--weeks", type=int, default=52, help="圖上顯示週數")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    today = dt.date.today().isoformat()

    m, _ = M.build_matrices()
    uni = M.load_universe()
    base = M.scan(m, args.liq, liq_pct=args.liq_pct)
    if base.empty:
        print("母體為空，不產報表"); sys.exit(1)

    cand = base[base["TT分"] >= args.min_tt].index
    print(f"母體 {len(base)} 檔 · TT≥{args.min_tt} 有 {len(cand)} 檔")

    rows = []
    for sid in cand:
        px = pd.DataFrame({k: m[kk][sid] for k, kk in
                           [("close", "c"), ("max", "h"), ("min", "l"),
                            ("Trading_Volume", "v")]}).dropna()
        if len(px) < 60:
            continue
        stg, _, _ = M.stage_of(px)
        f = M.vcp_foot(px)
        b = base.loc[sid]
        rows.append(dict(
            代號=sid,
            名稱=uni.loc[sid, "stock_name"] if sid in uni.index else "?",
            產業=uni.loc[sid, "industry_category"] if sid in uni.index else "",
            收盤=b["收盤"], RS=b["RS"], TT分=int(b["TT分"]), 階段=stg or 0,
            距高點=b["距高點"], 量比=b["量比"], 量增=bool(b["量增"]),
            VCP=f["ok"], 足跡=f["foot"], 樞紐價=f["pivot"], 近樞紐=f["near"],
            突破=bool(b["收盤"] >= (f["pivot"] or 1e9)),
            底部序=M.base_count(px), 均額億=b["均額億"], 新股=len(px) < 300,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("沒有標的通過門檻")
        pd.DataFrame().to_csv(f"{OUT}/minervini_{today}.csv", index=False)
        with open(f"{OUT}/summary.txt", "w") as fh:
            fh.write(f"Minervini 選股 {today}\n\n沒有標的通過門檻"
                     f"（TT≥{args.min_tt}、均額 {args.liq:,.0f} 萬）。\n")
        return

    df["狀態"] = df.apply(M.classify, axis=1)
    df["晚期"] = df["底部序"] >= 4
    df = df[df["狀態"] != "淘汰"]
    df = (df.assign(_o=df["狀態"].map(ORDER))
            .sort_values(["_o", "TT分", "RS"], ascending=[True, False, False])
            .drop(columns="_o").reset_index(drop=True))

    csv = f"{OUT}/minervini_{today}.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    cnt = df["狀態"].value_counts()
    print(f"名單 {len(df)} 檔：" +
          "、".join(f"{k} {int(v)}" for k, v in cnt.items()))

    # 摘要（信件內文用）
    lines = [f"Minervini 選股　{today}", "",
             f"母體 {len(base)} 檔　通過 {len(df)} 檔："
             + "、".join(f"{k} {int(v)}" for k, v in cnt.items()), ""]
    for stt in ["觸發", "準備", "觀察"]:
        g = df[df["狀態"] == stt]
        if g.empty:
            continue
        lines.append(f"【{stt}】{len(g)} 檔")
        for _, r in g.head(15).iterrows():
            flag = "".join(["·晚期" if r["晚期"] else "",
                            "·新股" if r["新股"] else "",
                            "·VCP" if r["VCP"] else ""])
            lines.append(f"  {r['代號']} {r['名稱']}　{r['收盤']:,.1f}　"
                         f"RS {r['RS']:.0f}　距高 {r['距高點']:+.1f}%　"
                         f"樞紐 {r['樞紐價'] or '—'}{flag}")
        if len(g) > 15:
            lines.append(f"  …另 {len(g) - 15} 檔見附件 CSV")
        lines.append("")
    lines += ["名單不是買點。附件教學圖裡的三欄結論是規則產生的，",
              "要買賣請自己再判一次。", ""]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines))

    if args.no_charts:
        print("--no-charts，略過出圖")
        return

    picks = df.head(args.top)["代號"].tolist()
    names = dict(zip(df["代號"], df["名稱"]))
    cdir = os.path.join(OUT, "charts")
    os.makedirs(cdir, exist_ok=True)
    print(f"出圖 {len(picks)} 檔（上限 {args.top}）")
    made = build_charts(picks, names, cdir, args.weeks)

    if made:
        zp = f"{OUT}/charts_{today}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in made:
                z.write(p, os.path.basename(p))
        print(f"已打包 {len(made)} 張圖 → {zp} "
              f"({os.path.getsize(zp) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
