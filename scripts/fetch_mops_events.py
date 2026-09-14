#!/usr/bin/env python3
"""
scripts/fetch_mops_events.py — 重大訊息 event log

來源：證交所官方 OpenAPI（keyless，不需金鑰，開放資料授權 1.0）
  上市 https://openapi.twse.com.tw/v1/opendata/t187ap04_L   「上市公司每日重大訊息」
  上櫃 https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O

⚠️ 這兩個端點是「當日滾動快照」，不是歷史庫。**抓不到過去的資料。**
   歷史重大訊息只在 MOPS 消費者入口（mops.twse.com.tw/mops/web/t05st01），
   那邊有 referer 牆與安全性檢查，不建議自動化。
   → 結論：event log 只能從「今天開始，每天累積」。這支要天天跑，漏一天就補不回來。

回傳欄位（官方 Swagger 定義，不是解析 HTML，改版風險低）：
  出表日期 / 發言日期 / 發言時間 / 公司代號 / 公司名稱 / 主旨 / 符合條款 / 事實發生日 / 說明

寫入 data/events/events.csv，**append-only，永不覆寫**。
event_id = sha1(代號+發言日期+發言時間+主旨前40字)，重跑同一天不會重複。

  python scripts/fetch_mops_events.py
  python scripts/fetch_mops_events.py --dry-run      # 只印不寫
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request

SRC = [
    ("twse", "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"),
    ("tpex", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"),
]
OUT = "data/events/events.csv"
COLS = ["event_id", "stock_id", "pub_date", "pub_time", "category", "polarity",
        "source", "source_url", "subject", "clause", "fact_date",
        "excerpt", "verification_state"]

# 分類只用主旨關鍵字粗分。分不出來就留 other，讓人工確認 —— 不要猜。
CATEGORY = [
    ("contract", ["合約", "契約", "訂單", "得標", "承攬", "簽約", "採購案", "標案"]),
    ("mou", ["備忘錄", "MOU", "意向書", "合作意向"]),
    ("guidance", ["財務預測", "營運展望", "法人說明會", "法說會", "財測"]),
    ("target", ["營收", "自結", "獲利", "每股盈餘", "EPS"]),
    ("cashflow", ["現金流", "應收", "資產減損", "呆帳", "背書保證", "資金貸與"]),
    ("governance", ["董事", "監察人", "辭任", "解任", "改選", "質押", "增資",
                    "減資", "私募", "併購", "合併", "處分", "取得或處分"]),
    ("capital", ["股利", "配息", "配股", "除權", "除息", "庫藏股"]),
    ("risk", ["訴訟", "裁罰", "處分書", "停工", "火災", "重大損害", "跳票", "違約"]),
]

# 方向只在關鍵字非常明確時才標，其餘一律留空。
# 手冊 F 類要求「事前定義負面事件」，不能先看漲跌才挑事件 —— 所以這裡不看價格。
POS_KW = ["得標", "取得訂單", "簽約", "獲頒", "通過認證", "上修", "創新高"]
NEG_KW = ["裁罰", "處分書", "訴訟", "跳票", "違約", "停工", "重大損害",
          "下修", "解任", "辭任", "減資彌補虧損"]


def roc_to_iso(s: str) -> str:
    """民國日期轉西元。1150521 或 115/05/21 都吃。轉不出來回空字串。"""
    s = (s or "").strip().replace("/", "").replace("-", "")
    if not s.isdigit() or len(s) not in (6, 7):
        return ""
    y = int(s[:-4]) + 1911
    return f"{y}-{s[-4:-2]}-{s[-2:]}"


def classify(subject: str, clause: str) -> str:
    t = f"{subject} {clause}"
    for cat, kws in CATEGORY:
        if any(k in t for k in kws):
            return cat
    return "other"


def polarity(subject: str, desc: str) -> str:
    t = f"{subject} {desc}"
    p = any(k in t for k in POS_KW)
    n = any(k in t for k in NEG_KW)
    if p and not n:
        return "positive"
    if n and not p:
        return "negative"
    return ""          # 含混或兩者皆有 → 留空，人工判


def fetch(url: str, retries: int = 3, timeout: int = 40):
    req = urllib.request.Request(url, headers={
        "User-Agent": "oldhand-events/1.0", "Accept": "application/json"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    print(f"[WARN] 抓取失敗 {url}：{last}")
    return []


def get(row: dict, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n]).strip()
    return ""


def normalize(row: dict, source: str, url: str) -> dict | None:
    sid = get(row, "公司代號", "SecuritiesCompanyCode", "Code")
    if len(sid) != 4 or not sid.isdigit():
        return None
    pub_date = roc_to_iso(get(row, "發言日期", "DateOfSpeech"))
    if not pub_date:
        return None
    pub_time = get(row, "發言時間", "TimeOfSpeech")
    subject = get(row, "主旨", "Subject", "主旨 ")
    clause = get(row, "符合條款", "Clause")
    desc = get(row, "說明", "Description")
    fact = roc_to_iso(get(row, "事實發生日", "DateOfEvent"))
    eid = hashlib.sha1(f"{sid}|{pub_date}|{pub_time}|{subject[:40]}"
                       .encode()).hexdigest()[:16]
    return {
        "event_id": eid, "stock_id": sid,
        "pub_date": pub_date, "pub_time": pub_time,
        "category": classify(subject, clause),
        "polarity": polarity(subject, desc),
        "source": source, "source_url": url,
        "subject": subject, "clause": clause, "fact_date": fact,
        # 摘錄壓到 300 字，全文留在 MOPS，這裡只要夠人工判讀
        "excerpt": desc[:300].replace("\n", " ").replace("\r", " "),
        # 分類與方向都是關鍵字猜的 → 一律 unverified，人工確認後才改 verified
        "verification_state": "unverified",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows, seen_src = [], []
    for source, url in SRC:
        data = fetch(url)
        if not isinstance(data, list) or not data:
            print(f"[WARN] {source} 回傳空的或格式不符，跳過")
            continue
        seen_src.append(source)
        n = 0
        for raw in data:
            r = normalize(raw, source, url)
            if r:
                rows.append(r)
                n += 1
        print(f"[OK] {source} {len(data)} 筆原始 → {n} 筆可用")

    if not rows:
        print("::warning::這次沒有抓到任何重大訊息。可能是假日，也可能是端點改版。")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing.add(r.get("event_id", ""))

    new = [r for r in rows if r["event_id"] not in existing]
    dup = len(rows) - len(new)

    # 盤後公告：發言時間晚於 13:30，combo 36 靠這個判定
    after = sum(1 for r in new if r["pub_time"] and r["pub_time"] >= "13:30")
    cats = {}
    for r in new:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    print(f"\n新增 {len(new)} 筆（重複略過 {dup}）　盤後公告 {after} 筆")
    print("分類：" + "、".join(f"{k} {v}" for k, v in sorted(cats.items(),
                                                          key=lambda x: -x[1])))
    if args.dry_run:
        for r in new[:5]:
            print(f"  {r['pub_date']} {r['pub_time']} {r['stock_id']} "
                  f"[{r['category']}] {r['subject'][:40]}")
        print("(dry-run，沒有寫檔)")
        return

    fresh = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if fresh:
            w.writeheader()
        for r in new:
            w.writerow({k: r.get(k, "") for k in COLS})
    print(f"寫入 {args.out}")

    if len(seen_src) < len(SRC):
        print("::warning::只有部分來源成功，上櫃或上市其中一邊沒抓到")


if __name__ == "__main__":
    main()
