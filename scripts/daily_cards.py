#!/usr/bin/env python3
"""每日執行清單 — 三張卡的買進/在手/賣出清單。

盤後跑，產出隔日開盤要執行的動作。定義全部 import 自 probe_vcp，
實盤與回測共用同一份程式碼，不可能走樣。

## 為什麼不用狀態檔記錄在手部位

每天重放最近 REPLAY 個交易日的訊號，重新推算目前還開著的部位。
理由：
  1. 漏跑一天不會壞 —— 狀態檔一旦漏更新就永久偏掉，重放不會。
  2. 跟回測用同一套邏輯（同樣的 20 日去重、同樣的停損成交假設），
     不會出現「實盤清單跟回測對不起來」的情況。
  3. 任何一天都能重現，出事時可以直接比對。

代價是每天多算一次歷史，但只掃 REPLAY 天，成本可忽略。

## 停損成交假設（與回測一致，不可改）

停損價 = 進場日開盤價 x 0.88，掛停損單，當日盤中觸價即成交。
若開盤已低於停損價，以開盤價成交。
**不是**「今日觸及、明日開盤賣」—— 那會讓虧損分佈系統性劣於回測，
且劣化集中在跌停與跳空這些最糟的交易上。

## 輸出

out/daily_cards_YYYY-MM-DD.md   完整清單（推播正文）
out/daily_buys.csv              明日買進
out/daily_positions.csv         在手部位
out/summary.txt                 email 正文用的摘要
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_vcp as P          # noqa: E402  母體與訊號定義的單一來源

OUT = "out"
REPLAY = 90                    # 重放天數，需 > 持有期 60 再留緩衝
HOLD = 60                      # 交易日
STOP = P.STOP                  # 0.12
CARDS = ("卡二 裸250", "卡一 VCP", "卡三 營收")


def build():
    p, _ = P.prep('ELEC', revenue=True)
    di = p['_i'].values
    sids = p['stock_id'].values
    dates = np.array(sorted(p['date'].unique()))
    nd = len(dates)
    valid = p['px_ok'].values & p['liq_ok'].values & (p['size'].values == 'LARGE')
    sig = {
        "卡二 裸250": valid & p['brk250'].values,
        "卡一 VCP": valid & p['brk60'].values & (p['vc'].values < P.VC_TH),
        "卡三 營收": valid & p['brk60'].values & p['rev_ok'].values,
    }
    return p, di, sids, dates, nd, sig


def replay(p, di, sids, dates, nd, sig_mask, card, start_i):
    """重放 start_i 之後的訊號，回傳 (已平倉, 在手, 今日新訊號)。"""
    o = p['open'].values
    lo = p['min'].values
    idx_by_stock = {}
    for s, g in p.groupby('stock_id', sort=False):
        idx_by_stock[s] = g.index.values

    hit = np.flatnonzero(sig_mask & (di >= start_i))
    closed, open_pos, fresh = [], [], []
    last_by = {}
    for k in hit:
        s = sids[k]
        if s in last_by and di[k] - last_by[s] < P.NOREPEAT:
            continue
        last_by[s] = di[k]
        if di[k] == nd - 1:                       # 今日收盤剛觸發，明日買
            fresh.append(dict(card=card, stock_id=s, sig_date=dates[di[k]],
                              ref_close=None, i=int(di[k])))
            continue
        gi = idx_by_stock[s]
        pos = int(np.searchsorted(gi, k))
        if pos + 1 >= len(gi):
            continue
        e = o[gi[pos + 1]]
        if not np.isfinite(e) or e <= 0:
            continue
        sp = e * (1 - STOP)
        last = min(pos + 1 + HOLD, len(gi) - 1)
        exit_i = exit_px = why = None
        for j in range(pos + 1, last + 1):
            if lo[gi[j]] <= sp:
                exit_i, exit_px, why = j, min(o[gi[j]], sp), '停損'
                break
        if exit_i is None and (pos + 1 + HOLD) <= len(gi) - 1:
            exit_i, exit_px, why = last, o[gi[last]], '滿60日'
        rec = dict(card=card, stock_id=s, entry_date=dates[di[gi[pos + 1]]],
                   entry_px=round(e, 2), stop_px=round(sp, 2),
                   held=int(nd - 1 - di[gi[pos + 1]]))
        if exit_i is None:                         # 還開著
            cur = o[gi[-1]]
            rec.update(days_left=HOLD - rec['held'],
                       pnl_pct=round((cur / e - 1) * 100, 1))
            open_pos.append(rec)
        else:
            rec.update(exit_date=dates[di[gi[exit_i]]], why=why,
                       ret_pct=round((exit_px / e - 1) * 100, 1))
            closed.append(rec)
    return closed, open_pos, fresh


def main():
    os.makedirs(OUT, exist_ok=True)
    p, di, sids, dates, nd, sig = build()
    today = dates[-1]
    start_i = max(0, nd - 1 - REPLAY)
    close_px = None
    if 'close' in p.columns:
        close_px = p['close'].values

    buys, pos, exits = [], [], []
    for card in CARDS:
        c, o_, f = replay(p, di, sids, dates, nd, sig[card], card, start_i)
        buys += f
        pos += o_
        exits += [x for x in c if x['exit_date'] == today]

    # 跨卡合併：同一檔同一天被多張卡觸發，實盤只買一次。
    # 卡二與卡三重疊 45%，不合併會買成兩倍部位。
    def merge(rows, key):
        out = {}
        for r in rows:
            kk = (r['stock_id'], r[key])
            if kk in out:
                out[kk]['card'] += "＋" + r['card'].split()[0]
            else:
                out[kk] = dict(r)
        return list(out.values())

    buys = merge(buys, 'sig_date')
    pos = merge(pos, 'entry_date')
    exits = merge(exits, 'entry_date')

    # 明日到期（今日是第 60 天）
    due = [x for x in pos if x['days_left'] <= 0]
    pos = [x for x in pos if x['days_left'] > 0]

    name = pd.read_parquet(f"{P.ROOT}/data/universe.parquet")
    name['stock_id'] = name['stock_id'].astype(str)
    nm = dict(zip(name['stock_id'], name['stock_name']))
    lastopen = {}
    for s, g in p.groupby('stock_id', sort=False):
        lastopen[s] = g['open'].values[-1]

    L = [f"# 每日執行清單 {today}", ""]
    L += ["盤後產生，隔日開盤執行。停損一律掛單、當日盤中觸價成交（不是隔日賣）。", ""]

    L += [f"## 一、明日開盤買進（{len(buys)} 檔）", ""]
    if buys:
        L += ["| 代號 | 名稱 | 卡別 | 停損價 | 預計出場 |",
              "|---|---|---|---|---|"]
        for b in buys:
            L.append(f"| {b['stock_id']} | {nm.get(b['stock_id'],'?')} | {b['card']} "
                     f"| 開盤價 × {1-STOP:.2f} | 進場日 +{HOLD} 交易日 |")
        L += ["", "部位比例由使用者自行決定。換算公式："
                  "單筆部位上限 =（可承受單筆虧損 ÷ 總資產）÷ 停損幅度。",
              "停損價須待開盤成交後回填。"]
    else:
        L.append("（無）")
    pd.DataFrame(buys).to_csv(f"{OUT}/daily_buys.csv", index=False)

    L += ["", f"## 二、在手部位（{len(pos)} 檔）", ""]
    if pos:
        L += ["| 代號 | 名稱 | 卡別 | 進場日 | 成本 | 停損價 | 天數 | 剩餘 | 浮動損益 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in sorted(pos, key=lambda r: r['held'], reverse=True):
            L.append(f"| {x['stock_id']} | {nm.get(x['stock_id'],'?')} | {x['card']} "
                     f"| {x['entry_date']} | {x['entry_px']} | {x['stop_px']} "
                     f"| {x['held']}/{HOLD} | {x['days_left']} 日 | {x['pnl_pct']:+.1f}% |")
    else:
        L.append("（無）")
    pd.DataFrame(pos).to_csv(f"{OUT}/daily_positions.csv", index=False)

    L += ["", f"## 三、明日開盤賣出 — 滿 {HOLD} 交易日（{len(due)} 檔）", ""]
    if due:
        L += ["| 代號 | 名稱 | 卡別 | 進場日 | 成本 | 浮動損益 |", "|---|---|---|---|---|---|"]
        for x in due:
            L.append(f"| {x['stock_id']} | {nm.get(x['stock_id'],'?')} | {x['card']} "
                     f"| {x['entry_date']} | {x['entry_px']} | {x['pnl_pct']:+.1f}% |")
    else:
        L.append("（無）")

    L += ["", "## 四、今日已出場（對帳用）", ""]
    if exits:
        L += ["| 代號 | 卡別 | 進場日 | 出場原因 | 報酬 |", "|---|---|---|---|---|"]
        for x in exits:
            L.append(f"| {x['stock_id']} | {x['card']} | {x['entry_date']} "
                     f"| {x['why']} | {x['ret_pct']:+.1f}% |")
    else:
        L.append("（無）")

    L += ["", "---", "",
          "母體：電子硬體大型股（排除半導體），價格 ≥10 元、20日均額 ≥3,000 萬、族群內市值前 33%。",
          "同一檔 20 個交易日內不重複進場。詳見 ELEC_LARGE_FINDINGS.md 第十一節。"]

    md = "\n".join(L)
    with open(f"{OUT}/daily_cards_{today}.md", "w") as f:
        f.write(md)
    summary = (f"{today} 執行清單\n買進 {len(buys)} 檔 ｜ 在手 {len(pos)} 檔 ｜ "
               f"到期賣出 {len(due)} 檔 ｜ 今日出場 {len(exits)} 檔\n\n" + md)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(summary.rstrip() + "\n")
    print(md)


if __name__ == "__main__":
    main()
