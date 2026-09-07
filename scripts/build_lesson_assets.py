"""
scripts/build_lesson_assets.py

在 GitHub Actions 跑一次，產出：
  assets/lessons/pool.parquet          全市場歷史突破事件 + C01~C11 量測 + 後續結果
  assets/lessons/<CID>/pos_*.png       正例教材圖（全標註）
  assets/lessons/<CID>/neg_*.png       反例教材圖（全標註）
  assets/lessons/<CID>/cheatsheet.png  速查卡（可存手機相簿）
  assets/lessons/<CID>/index.json      該課教材清單

用法：
  python scripts/build_lesson_assets.py                 # 全部 12 課
  python scripts/build_lesson_assets.py --lessons C03   # 只做第 1 課
  python scripts/build_lesson_assets.py --n 10 --years 2018 2019 2020
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lesson_core as lc


LESSON_SELECTOR = lc.LESSON_SELECTOR


def build_pool(df: pd.DataFrame, taiex, min_events_per_stock: int = 0) -> pd.DataFrame:
    rows = []
    ids = [s for s in df["stock_id"].astype(str).unique()
           if s not in lc.TAIEX_CANDIDATES]
    for n, sid in enumerate(ids, 1):
        g = df[df["stock_id"].astype(str) == sid].sort_values("date").reset_index(drop=True)
        bo = lc.find_breakouts(g)
        if bo.empty:
            continue
        for _, r in bo.iterrows():
            i = int(r["i"])
            if i + 20 >= len(g):      # 沒有後續 20 日就不收（教材要能揭曉）
                continue
            f = lc.compute_features(g, i, taiex)
            f.update(lc.compute_outcomes(g, i))
            f.update({
                "stock_id": sid,
                "date": g["date"].iloc[i],
                "i": i,
                "prior_high": float(r["prior_high"]),
                "close": float(g["close"].iloc[i]),
            })
            rows.append(f)
        if n % 200 == 0:
            print(f"  ... {n}/{len(ids)} 檔，已收 {len(rows)} 筆事件", flush=True)
    pool = pd.DataFrame(rows)
    print(f"事件總數：{len(pool)}")
    return pool


def pick_examples(pool: pd.DataFrame, field: str, higher_is_better: bool, n: int):
    """挑正例與反例。刻意避開同一檔重複，讓教材看起來不像同一支股票。"""
    sub = pool.dropna(subset=[field]).copy()
    if sub.empty:
        return sub, sub
    sub = sub.sort_values(field, ascending=not higher_is_better)
    pos = sub.drop_duplicates("stock_id").head(n)
    neg = sub.iloc[::-1].drop_duplicates("stock_id").head(n)
    return pos, neg


def draw_cheatsheet(lesson: lc.Lesson, outpath: str):
    fig = plt.figure(figsize=(7.2, 10.2), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0.03, 0.03), 0.94, 0.94, facecolor="#f8fafc",
                           edgecolor="#334155", linewidth=2))
    ax.add_patch(Rectangle((0.03, 0.855), 0.94, 0.118, facecolor="#1e3a5f",
                           edgecolor="none"))
    ax.text(0.5, 0.935, f"第 {lesson.order} 課 · {lesson.cid} {lesson.name}",
            ha="center", fontsize=21, color="white", fontweight="bold")
    ax.text(0.5, 0.884, lesson.question, ha="center", fontsize=13, color="#cbd5e1")

    y = 0.805

    def block(title, body, color, height):
        nonlocal y
        ax.add_patch(Rectangle((0.07, y - height), 0.86, height,
                               facecolor="white", edgecolor=color, linewidth=1.5))
        ax.text(0.10, y - 0.032, title, fontsize=13, color=color, fontweight="bold")
        ax.text(0.10, y - 0.062, body, fontsize=11.2, color="#1f2937",
                va="top", wrap=True, linespacing=1.75)
        y -= height + 0.028

    idea = "\n".join(_wrap(lesson.idea, 26))
    block("心法 · 老手在問什麼", idea, "#1d4ed8", 0.055 + 0.033 * len(idea.split("\n")))
    block("量測 · Primary", f"{lesson.primary}\n" + "\n".join(_wrap(lesson.formula, 26)),
          "#7c3aed", 0.135)
    block("這樣算好", "\n".join(_wrap(lesson.good, 26)), "#047857", 0.115)
    block("這樣算不好", "\n".join(_wrap(lesson.bad, 26)), "#b91c1c", 0.115)

    ax.text(0.5, 0.055, "看不懂的時候切出來看一眼這張",
            ha="center", fontsize=10.5, color="#64748b", style="italic")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for ch in text.replace("\n", ""):
        line += ch
        if len(line) >= width and ch in "，。、；：）」":
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return out or [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", nargs="*", default=None, help="只做這幾課，例如 C03")
    ap.add_argument("--n", type=int, default=10, help="每課正例／反例各幾張")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--rebuild-pool", action="store_true")
    args = ap.parse_args()

    font = lc.setup_font()
    print(f"字型：{font}")

    pool_path = os.path.join(lc.ASSET_DIR, "pool.parquet")
    df = lc.load_prices(args.years)
    taiex = lc.load_taiex(df)
    print(f"價格資料：{df['stock_id'].nunique()} 檔 / {len(df):,} 列；"
          f"加權指數：{'有' if taiex is not None else '無（C06 會是空值）'}")

    if args.rebuild_pool or not os.path.exists(pool_path):
        pool = build_pool(df, taiex)
        os.makedirs(lc.ASSET_DIR, exist_ok=True)
        pool.to_parquet(pool_path, index=False)
    else:
        pool = pd.read_parquet(pool_path)
        print(f"沿用既有 pool：{len(pool)} 筆（要重算加 --rebuild-pool）")

    if pool.empty:
        print("pool 是空的，沒有可用事件，結束。")
        return

    todo = [l for l in lc.LESSONS
            if args.lessons is None or l.cid in args.lessons]

    for lesson in sorted(todo, key=lambda x: x.order):
        field, hib = LESSON_SELECTOR[lesson.cid]
        if field not in pool.columns:
            print(f"[{lesson.cid}] 缺欄位 {field}，跳過")
            continue
        outdir = os.path.join(lc.ASSET_DIR, lesson.cid)
        os.makedirs(outdir, exist_ok=True)

        pos, neg = pick_examples(pool, field, hib, args.n)
        index = {"cid": lesson.cid, "name": lesson.name, "order": lesson.order,
                 "selector_field": field, "positive": [], "negative": []}

        for kind, rows in (("pos", pos), ("neg", neg)):
            for rank, (_, r) in enumerate(rows.iterrows(), 1):
                sid = str(r["stock_id"])
                g = df[df["stock_id"].astype(str) == sid].sort_values("date").reset_index(drop=True)
                i = int(r["i"])
                feats = {c: r[c] for c in pool.columns if c not in ("stock_id", "date", "i")}
                fname = f"{kind}_{rank:02d}_{sid}_{pd.Timestamp(r['date']).date()}.png"
                path = os.path.join(outdir, fname)
                label = "正例" if kind == "pos" else "反例"
                lc.render_chart(
                    g, i, feats, mode="annotated",
                    title=f"{sid}　{pd.Timestamp(r['date']).date()}　【{label}】",
                    subtitle=f"{lesson.cid} {lesson.name}　{field} = {r[field]:.2f}"
                             f"　｜　後20日 {r.get('ret_20d_pct', float('nan')):+.1f}%"
                             f"　後60日 {r.get('ret_60d_pct', float('nan')):+.1f}%",
                    outpath=path,
                )
                index["positive" if kind == "pos" else "negative"].append({
                    "file": fname, "stock_id": sid,
                    "date": str(pd.Timestamp(r["date"]).date()),
                    "value": float(r[field]),
                    "ret_20d_pct": float(r.get("ret_20d_pct", np.nan)),
                    "ret_60d_pct": float(r.get("ret_60d_pct", np.nan)),
                })

        draw_cheatsheet(lesson, os.path.join(outdir, "cheatsheet.png"))
        lc.save_json(index, os.path.join(outdir, "index.json"))
        print(f"[{lesson.cid}] {lesson.name}：正例 {len(index['positive'])} 張 / "
              f"反例 {len(index['negative'])} 張 + 速查卡")

    print("完成。")


if __name__ == "__main__":
    main()
