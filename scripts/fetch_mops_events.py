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
import io
import json
import os
import sys
import time
import urllib.request

# 每個市場給多個候選端點，依序嘗試，第一個回得出東西的就用。
# 理由：上櫃那支 JSON 端點實測回空，但 mopsfin 的 CSV 版命名規則是確定的
# （t187ap03_L / _O / _R 都存在），所以拿 CSV 當退路。
SRC = [
    ("twse", ["https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
              "https://mopsfin.twse.com.tw/opendata/t187ap04_L.csv"]),
    ("tpex", ["https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O",
              "https://mopsfin.twse.com.tw/opendata/t187ap04_O.csv"]),
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


def norm_time(s: str) -> str:
    """發言時間正規化成 HH:MM:SS。

    ⚠️ 證交所回的是無冒號純數字，而且<b>長度不固定</b>：
       早上 7:00:04 回 "70004"（5 碼）、下午 13:30:00 回 "133000"（6 碼）。
       直接拿去做字串比較會錯（"70004" > "133000"），combo 36 的盤後判定會全錯。
       所以一律補零到 6 碼再切。
    """
    t = "".join(ch for ch in (s or "") if ch.isdigit())
    if not t:
        return ""
    if len(t) > 6:          # 有些來源帶毫秒
        t = t[:6]
    t = t.zfill(6)
    hh, mm, ss = t[:2], t[2:4], t[4:6]
    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59 and 0 <= int(ss) <= 59):
        return ""
    return f"{hh}:{mm}:{ss}"


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
    """回傳 list[dict]。JSON 與 CSV 都吃（看副檔名與內容自動判斷）。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; oldhand-events/1.1)",
        "Accept": "application/json, text/csv, */*"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8-sig", errors="replace")
            txt = raw.lstrip()
            if txt.startswith("[") or txt.startswith("{"):
                d = json.loads(txt)
                return d if isinstance(d, list) else [d]
            if "," in txt.split("\n", 1)[0]:
                return list(csv.DictReader(io.StringIO(raw)))
            print(f"[WARN] {url} 回傳的不是 JSON 也不是 CSV")
            return []
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    print(f"[WARN] 抓取失敗 {url}：{last}")
    return []


def fetch_first(urls: list[str]):
    """依序試候選端點，回傳 (資料, 成功的 url)。"""
    for u in urls:
        d = fetch(u, retries=2)
        if d:
            return d, u
    return [], urls[0]


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
    pub_time = norm_time(get(row, "發言時間", "TimeOfSpeech"))
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
    for source, urls in SRC:
        data, url = fetch_first(urls)
        if not data:
            print(f"[WARN] {source} 所有候選端點都回空，跳過")
            continue
        seen_src.append(source)
        n = 0
        for raw in data:
            r = normalize(raw, source, url)
            if r:
                rows.append(r)
                n += 1
        print(f"[OK] {source} {len(data)} 筆原始 → {n} 筆可用　（{url}）")

    if not rows:
        print("::warning::這次沒有抓到任何重大訊息。可能是假日，也可能是端點改版。")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            try:
                header = next(rd)
            except StopIteration:
                header = []
        # 舊版標頭是 9 欄、新版 13 欄。欄數或順序不符就直接停手，
        # 不能默默 append —— 那會讓每一欄從第 7 格開始整排錯位。
        if header != COLS:
            print("::error::既有 CSV 的標頭跟目前的欄位定義不符，拒絕寫入。")
            print(f"  檔案：{args.out}")
            print(f"  既有 {len(header)} 欄：{','.join(header)}")
            print(f"  目前 {len(COLS)} 欄：{','.join(COLS)}")
            print("  處理方式：把這個檔案刪掉（或改名備份），下次執行會用新標頭重建。")
            sys.exit(1)
        with open(args.out, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                existing.add(r.get("event_id", ""))

    # 去重要做兩層：對既有檔案，也對這一次抓到的批次本身。
    # 上市與上櫃是兩個端點，理論上不會重疊，但端點若曾短暫混供就會寫入兩筆同 id。
    new, batch = [], set()
    for r in rows:
        if r["event_id"] in existing or r["event_id"] in batch:
            continue
        batch.add(r["event_id"])
        new.append(r)
    dup = len(rows) - len(new)

    # 盤後公告：發言時間晚於 13:30，combo 36 靠這個判定
    after = sum(1 for r in new if r["pub_time"] and r["pub_time"] >= "13:30:00")
    notime = sum(1 for r in new if not r["pub_time"])
    cats = {}
    for r in new:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    # 端點是「滾動快照」，不保證含今天的公告。印出日期分佈，
    # 才看得出快照到底落後幾天 —— 這決定排程該排幾點。
    bydate = {}
    for r in rows:
        bydate[r["pub_date"]] = bydate.get(r["pub_date"], 0) + 1
    print("快照日期分佈：" + "、".join(f"{k} {v} 筆"
                                 for k, v in sorted(bydate.items())[-5:]))

    print(f"\n新增 {len(new)} 筆（重複略過 {dup}）　盤後公告 {after} 筆"
          + (f"　⚠️ 無發言時間 {notime} 筆" if notime else ""))
    print("分類：" + "、".join(f"{k} {v}" for k, v in sorted(cats.items(),
                                                          key=lambda x: -x[1])))
    if args.dry_run:
        for r in new[:5]:
            print(f"  {r['pub_date']} {r['pub_time']} {r['stock_id']} "
                  f"[{r['category']}] {r['subject'][:40]}")
        print("(dry-run，沒有寫檔)")
        return

    fresh = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
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
