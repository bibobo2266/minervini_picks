r"""
分年回測結果合併（Replay Merge）

watchlist_replay.py 一次只跑一年——七年放在同一個 job 會撞到 Actions 的
六小時上限，而且失敗一次就整段重來。分年跑成矩陣、各自產出 artifact，
再由這支合併。

合併後做的事只有一件：**逐年攤開，看符號一致性**。
把多年事件池在一起算單一中位數是錯的——觸發事件最多的必然是多頭年，
池在一起等於變相只測多頭。

輸出 REPLAY.md 是為了手機閱讀：GitHub 的 Markdown 在手機上看得下去，
CSV 不行。

用法：
    python scripts/replay_merge.py --in out/replay --out REPLAY.md
"""
import argparse
import contextlib
import glob
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watchlist_replay import gap_and_filters, yearly_stability  # noqa: E402


def load_all(indir: str) -> pd.DataFrame:
    """讀 indir 底下所有 */replay_events.csv。缺年份不當錯誤——
    某一年 job 掛掉時，其餘年份的結果仍然有意義，不該整份報告不出。"""
    files = sorted(glob.glob(os.path.join(indir, "*", "replay_events.csv")))
    if not files:
        raise SystemExit(f"{indir} 底下找不到任何 replay_events.csv")
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype={"代號": str})
        d["來源"] = os.path.basename(os.path.dirname(f))
        parts.append(d)
        print(f"  讀入 {f}：{len(d)} 筆")
    ev = pd.concat(parts, ignore_index=True)
    # 暖身期已經在各年的 replay 裡濾掉，這裡只防重複執行造成的重複列
    ev = ev.drop_duplicates(subset=["日期", "代號", "事件", "原因"])
    return ev.sort_values("日期").reset_index(drop=True)


def capture(fn, *a, **kw):
    """把函式印到 stdout 的內容收成字串，同時保留回傳值。
    這些分析函式本來就是設計成印在終端機看的，改成回傳結構化資料會讓
    watchlist_replay.py 的單機使用變難用。收 stdout 是比較小的代價。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def md_table(df: pd.DataFrame, index_name="年") -> str:
    if df is None or df.empty:
        return "_無資料_\n"
    d = df.reset_index()
    if d.columns[0] == "index":
        d = d.rename(columns={"index": index_name})
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(
        "" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))
        for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="out/replay")
    ap.add_argument("--out", default="REPLAY.md")
    ap.add_argument("--params", default="", help="寫進報告抬頭的執行參數字串")
    args = ap.parse_args()

    print("合併分年回測結果…")
    ev = load_all(args.indir)
    print(f"合計 {len(ev)} 筆事件")

    trg = ev[ev["事件"] == "觸發"]
    add = ev[ev["事件"] == "進榜"]
    rmv = ev[ev["事件"] == "移出"]

    yr, yr_txt = capture(yearly_stability, ev)
    _, gf_txt = capture(gap_and_filters, ev)

    has_open = "報酬60D開" in ev.columns
    span = f"{ev['日期'].min()} → {ev['日期'].max()}"

    lines = [
        "# 名單回測：逐年穩定性報告", "",
        f"區間 `{span}`　進榜 {len(add)}　觸發 {len(trg)}　移出 {len(rmv)}", "",
    ]
    if args.params:
        lines += [f"執行參數：`{args.params}`", ""]
    if not has_open:
        lines += ["> ⚠️ 事件表沒有 `報酬60D開`，以下全部是**收盤起算**的數字，"
                  "系統性高估。確認 `data/adj` 的 parquet 有 `open` 欄位、"
                  "且 `minervini_core.build_matrices` 有讀進來。", ""]

    lines += ["## 逐年矩陣", "", md_table(yr), "",
              "**微結構斷點**：逐筆交易 2020-03-23 上路（取代五秒一次集合競價）、"
              "當沖降稅 2017-04 起、盤中零股 2020-10。這些直接改變開盤價的形成"
              "方式，而跳空成本正是這份報告的核心數字。"
              "**2020 之前與之後不是同質樣本，跨越這條線做平均沒有意義。**", "",
              "看的是**符號一致性**不是幅度。某個分組 11 年裡有 8 年為負，"
              "那是規則；只有 1 年為負，那是雜訊。", "",
              "## 判定與跳空成本", "", "```", yr_txt.strip(), "", gf_txt.strip(),
              "```", "",
              "## 怎麼用這份報告", "",
              "1. 先看**跳空成本**。如果開盤起算的超額接近 0，那個週期的訊號"
              "做不出來——不是策略無效，是進場價拿不到。",
              "2. 再看逐年矩陣的**符號**。穩定為負的分組才值得寫進 "
              "`watchlist_update.py` 的 `MAX_RS` / `MIN_BASE`。",
              "3. 觸發數 < 100 的年份不要單獨解讀。",
              "4. 這裡沒有部位管理、沒有停損、沒有交易成本。加停損會同時砍掉"
              "左尾和右尾，而這套策略的期望值全在右尾——那要另外模擬。", ""]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    os.makedirs(args.indir, exist_ok=True)
    ev.to_csv(os.path.join(args.indir, "events_all.csv"),
              index=False, encoding="utf-8-sig")
    if yr is not None and len(yr):
        yr.to_csv(os.path.join(args.indir, "yearly_all.csv"),
                  encoding="utf-8-sig")

    print(f"\n已寫入 {args.out} 與 {args.indir}/events_all.csv")
    print(yr_txt)


if __name__ == "__main__":
    main()
