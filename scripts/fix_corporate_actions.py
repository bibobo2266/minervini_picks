r"""
從價格自己偵測並修正未還原的公司行動

為什麼需要這支：
    FinMind 的 TaiwanStockPriceAdj 只還原「除權息類」事件（配股、配息）。
    股本變動類事件——減資、面額變更、股票分割——它不處理。

偵測原理：
    台股上市櫃股票單日漲跌幅上限 10%。單日變動超過 11% 在物理上只有一種可能
    ——未還原的公司行動。這條規則不需要任何 API、不需要訂閱、到期後永遠可用。

    雙向偵測：
      - 分割 / 面額變更 → 除權日價格「往下跳」，比值 > 1，歷史價要「除以」它。
      - 減資 / 反向分割 → 除權參考價「往上跳」（減資五成 = 價格翻倍），
        比值 < 1，除以小於 1 的數就等於把歷史價乘上去。

⚠️ 三個踩過的坑，寫在這裡免得重蹈：

1. universe 的 type 是「現在的市場別」，不是「當時的」。
   一檔 2016 年在興櫃、2019 年轉上市的股票，現在 type = twse，但它 2016 那段
   沒有漲跌幅限制。實測光靠 type 過濾，命中 3,460 筆，其中單一檔就佔 58 筆。
   一檔股票不可能有 58 次公司行動。

2. TaiwanStockInfo 的 date 欄不是上市日，是資料快照日。
   曾經想用它算掛牌天數來排除轉板前的興櫃期間，結果 3,147 檔裡 2,740 檔的值
   都是抓取當天，每個歷史事件的掛牌天數都變負的，3,460 筆全被判成「新上市」、
   建議修正 0 筆。這條路是死的，不要再試。

3. 改用價格自己的密集度推斷興櫃期間（現在的做法）。
   興櫃期間沒有漲跌幅限制，逾 11% 的日子密密麻麻；一旦掛牌上市櫃，這種日子
   幾乎絕跡，只剩零星的公司行動。所以：同檔前後 60 個交易日內出現 3 次以上
   命中，判定為興櫃期間；最後一次密集命中之後 30 個交易日才開始採信。
   這個做法不需要上市日、不需要任何 API，訂閱到期後照樣能用。
   實測 3,460 → 1,003 筆，而且逐年分佈變平（每年 40–110 筆），
   符合公司行動應有的樣子。

⚠️ 幅度門檻：11% 到 20% 那一段不能自動修
    實測 1,003 筆裡有 617 筆落在 11–20%，而且比值幾乎全部 snap 到邊界候選
    （0.900、1.111、1.176、0.850）——那是「剛好超過門檻」被硬湊出來的，不是
    真的公司行動。真的減資分割會是 0.75、0.80、1.333、1.538 這種數字。
    所以 CONFIRM_LIMIT 以下只列出來、不建議修，交給人工看。
    幅度逾 25% 的實測只有 65 筆、每年約 6 筆，那才是可以動的清單。

⚠️ 比值可以人工覆寫
    CLEAN 候選集是 5% 一格，遇到 1.87 這種官方值會 snap 到 1.818。CSV 裡留了
    「原始比值」欄（= 前收 / 當日收），你查到官方數字就直接改 ratio 欄再跑
    --apply，程式用的是 CSV 裡的值，不會覆蓋你的手改。

刻意分成兩段，不要一鍵修完：
    python scripts/fix_corporate_actions.py --detect   # 只偵測，產出清單
    python scripts/fix_corporate_actions.py --apply    # 依清單修正
    自動修正價格資料是不可逆的操作，中間需要一個人看過。

用法：
    python scripts/fix_corporate_actions.py --detect
    python scripts/fix_corporate_actions.py --apply --dry-run
    python scripts/fix_corporate_actions.py --apply
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data/adj"
EVENTS = os.path.join(DATA_DIR, "corporate_actions_detected.csv")
BACKUP = os.path.join(DATA_DIR, "_backup_before_fix")

MOVE_LIMIT = 0.11          # 台股漲跌幅上限 10%，留 1% 緩衝給資料誤差（雙向）
CONFIRM_LIMIT = 0.20       # 低於這個幅度不自動建議修正，交給人工
MIN_RATIO = 1.10           # 比值離 1 太近就不值得修
MAX_RATIO = 25.0           # 高於這個八成是資料錯誤，不是公司行動
MAX_HITS_PER_STOCK = 6     # 密集過濾後同檔仍命中超過這個數，還是可疑

DENSE_WINDOW = 60          # 興櫃判定的視窗（交易日）
DENSE_COUNT = 3            # 視窗內命中達這個數就算密集 → 興櫃期間
POST_LIST_BUFFER = 30      # 密集期結束後再跳過這麼多交易日才採信

# 乾淨比值候選：
#   > 1  →  分割 / 面額變更 / 大額配股，除權日價格往下跳
#   < 1  →  減資 / 反向分割，除權參考價往上跳
_BASE = sorted({round(1 / (1 - k), 4) for k in np.arange(0.05, 0.91, 0.05)}
               | {float(n) for n in range(2, 11)})
CLEAN = sorted(set(_BASE) | {round(1 / r, 6) for r in _BASE})


def ratio_ok(r) -> bool:
    """比值要離 1 夠遠、又不能離譜。兩個方向都檢查。"""
    if r is None or r != r:
        return False
    return (MIN_RATIO <= r <= MAX_RATIO) or (1 / MAX_RATIO <= r <= 1 / MIN_RATIO)


def snap_ratio(prev_close: float, close: float):
    """在「還原後單日漲跌 <= 10%」的合法區間內，挑最乾淨的比值。

    合法區間：close * ratio / prev_close - 1 落在 [-0.10, +0.10]
             → ratio ∈ [0.90 * prev/close, 1.10 * prev/close]
    對兩個方向都成立：往下跳時 prev/close > 1、候選在 1 以上；
    往上跳時 prev/close < 1、候選落在倒數那一側。
    """
    lo = 0.90 * prev_close / close
    hi = 1.10 * prev_close / close
    cands = [r for r in CLEAN if lo <= r <= hi]
    if not cands:
        return None
    mid = prev_close / close
    return min(cands, key=lambda r: abs(r - mid))


def load_all():
    fs = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))
    if not fs:
        raise SystemExit(f"{DATA_DIR} 底下沒有 prices_adj_*.parquet")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["stock_id", "date"]).reset_index(drop=True), fs


def emerging_cutoff(hits: pd.DataFrame) -> dict:
    """回傳每檔的興櫃期間結束位置（交易日序號）。

    興櫃沒有漲跌幅限制，逾 11% 的日子會成群出現；掛牌後只剩零星公司行動。
    所以找出「前後 DENSE_WINDOW 天內命中達 DENSE_COUNT 次」的那些命中，
    取最後一次的位置當分界。沒有任何密集命中的股票不設限。
    """
    cut = {}
    for sid, g in hits.groupby("stock_id"):
        idx = g["i"].values
        dense = np.array([(np.abs(idx - v) <= DENSE_WINDOW).sum() >= DENSE_COUNT
                          for v in idx])
        cut[sid] = int(idx[dense].max()) if dense.any() else None
    return cut


def detect(d: pd.DataFrame, uni=None) -> pd.DataFrame:
    d = d[d["stock_id"].astype(str).str.match(r"^[1-9]\d{3}$")].copy()
    d = d[d["close"] > 0]

    if uni is not None and "type" in uni.columns:
        listed = set(uni[uni["type"].isin(["twse", "tpex"])].index.astype(str))
        n0 = d["stock_id"].nunique()
        d = d[d["stock_id"].astype(str).isin(listed)]
        print(f"  排除現為興櫃與未知：{n0} → {d['stock_id'].nunique()} 檔")

    d = d.sort_values(["stock_id", "date"]).reset_index(drop=True)
    g = d.groupby("stock_id", sort=False)
    d["prev_close"] = g["close"].shift()
    d["chg"] = d["close"] / d["prev_close"] - 1
    d["i"] = g.cumcount()

    hits = d[(d["chg"].abs() > MOVE_LIMIT) & d["prev_close"].notna()].copy()
    if hits.empty:
        return pd.DataFrame()
    print(f"  原始命中 {len(hits)} 筆 / {hits['stock_id'].nunique()} 檔")

    cut = emerging_cutoff(hits)
    n_emerging = sum(1 for _, e in hits.iterrows()
                     if cut.get(str(e["stock_id"])) is not None
                     and e["i"] <= cut[str(e["stock_id"])] + POST_LIST_BUFFER)
    hits = hits[[cut.get(str(r.stock_id)) is None
                 or r.i > cut[str(r.stock_id)] + POST_LIST_BUFFER
                 for r in hits.itertuples()]]
    print(f"  推斷為興櫃期間而排除 {n_emerging} 筆 → 剩 {len(hits)} 筆 / "
          f"{hits['stock_id'].nunique()} 檔")
    if hits.empty:
        return pd.DataFrame()

    hit_count = hits["stock_id"].value_counts().to_dict()

    rows = []
    for e in hits.itertuples():
        sid = str(e.stock_id)
        prev_c, close_c = float(e.prev_close), float(e.close)
        ratio = snap_ratio(prev_c, close_c)

        if hit_count.get(sid, 0) > MAX_HITS_PER_STOCK:
            verdict = "同檔命中過多"
        elif abs(e.chg) <= CONFIRM_LIMIT:
            verdict = "幅度不足以確認"
        elif ratio is None:
            verdict = "找不到合理比值"
        elif not ratio_ok(ratio):
            verdict = "比值超出合理範圍"
        else:
            verdict = "建議修正"

        adj_chg = (close_c * ratio / prev_c - 1) * 100 if ratio else np.nan

        rows.append({
            "stock_id": sid,
            "event_date": pd.Timestamp(e.date).date().isoformat(),
            "prev_close": round(prev_c, 4),
            "open": round(float(getattr(e, "open", np.nan)), 4),
            "close": round(close_c, 4),
            "chg%": round(float(e.chg) * 100, 1),
            "ratio": round(ratio, 6) if ratio else np.nan,
            "原始比值": round(prev_c / close_c, 6),
            "還原後漲跌%": round(adj_chg, 1) if adj_chg == adj_chg else np.nan,
            "方向": "往下跳（分割／面額變更）" if e.chg < 0 else "往上跳（減資／反向分割）",
            "同檔命中數": hit_count.get(sid, 0),
            "判定": verdict,
        })

    out = pd.DataFrame(rows).sort_values(["event_date", "stock_id"])
    return out.reset_index(drop=True)


def apply_fixes(events: pd.DataFrame, dry_run: bool):
    """把事件日之前的 OHLC 除以比值。

    比值 > 1 → 歷史價被壓低（分割）。
    比值 < 1 → 歷史價被放大（減資），除以小於 1 的數就是乘上去。

    用的是 CSV 裡 ratio 欄的值，所以你手改過的官方數字會被照用。
    只處理判定為「建議修正」的列——想納入別的列，把那格改成「建議修正」。
    """
    use = events[events["判定"] == "建議修正"].copy()
    use = use[use["ratio"].notna()]
    if use.empty:
        print("沒有判定為「建議修正」的事件，不做任何修改。")
        return

    n_down = int((use["ratio"] > 1).sum())
    n_up = int((use["ratio"] < 1).sum())
    print(f"將套用 {len(use)} 筆修正（涉及 {use['stock_id'].nunique()} 檔）"
          f"；往下跳 {n_down} 筆、往上跳 {n_up} 筆")

    fs = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))

    if not dry_run:
        os.makedirs(BACKUP, exist_ok=True)
        for f in fs:
            shutil.copy2(f, os.path.join(BACKUP, os.path.basename(f)))
        print(f"已備份 {len(fs)} 個檔案到 {BACKUP}")

    # 同一檔可能有多次事件。較早的價格要吸收它之後所有事件的比值，
    # 由近而遠逐次相除即可（次序不影響結果，乘法可交換）。
    factor = {}
    for sid, g in use.groupby("stock_id"):
        evs = g.sort_values("event_date", ascending=False)
        factor[sid] = [(pd.Timestamp(r["event_date"]), float(r["ratio"]))
                       for _, r in evs.iterrows()]

    total = 0
    for f in fs:
        d = pd.read_parquet(f)
        d["date"] = pd.to_datetime(d["date"])
        changed = 0
        for sid, evs in factor.items():
            mask_sid = d["stock_id"].astype(str) == sid
            if not mask_sid.any():
                continue
            for ev_date, ratio in evs:
                m = mask_sid & (d["date"] < ev_date)
                if not m.any():
                    continue
                for c in ("open", "max", "min", "close"):
                    if c in d.columns:
                        d.loc[m, c] = (d.loc[m, c] / ratio).round(4)
                changed += int(m.sum())
        if changed:
            total += changed
            print(f"  {os.path.basename(f)} 修正 {changed:,} 列"
                  + ("（dry-run，未寫入）" if dry_run else ""))
            if not dry_run:
                d.to_parquet(f, index=False, compression="zstd")

    print(f"\n合計修正 {total:,} 列"
          + ("（dry-run）" if dry_run else f"。備份在 {BACKUP}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true", help="只偵測，產出清單")
    ap.add_argument("--apply", action="store_true", help="依清單套用修正")
    ap.add_argument("--dry-run", action="store_true", help="套用但不寫檔")
    args = ap.parse_args()

    if not (args.detect or args.apply):
        ap.error("要指定 --detect 或 --apply")

    if args.detect:
        d, fs = load_all()
        import minervini_core as M
        try:
            uni = M.load_universe()
        except Exception as ex:
            print(f"  讀不到 universe（{ex}），無法排除現為興櫃的股票")
            uni = None

        print(f"讀入 {len(fs)} 個年份檔，{len(d):,} 列，{d['stock_id'].nunique()} 檔")
        ev = detect(d, uni)
        if ev.empty:
            print("沒有留下任何疑似公司行動。")
            return

        ev.to_csv(EVENTS, index=False, encoding="utf-8-sig")
        print(f"\n寫入 {EVENTS}（{len(ev)} 筆）\n")
        print(ev["判定"].value_counts().to_string())
        print()
        print(ev["方向"].value_counts().to_string())

        cols = ["stock_id", "event_date", "prev_close", "close", "chg%",
                "ratio", "原始比值", "還原後漲跌%", "同檔命中數"]
        rec = ev[ev["判定"] == "建議修正"]
        print(f"\n建議修正 {len(rec)} 筆（全列）：")
        print(rec[cols].to_string(index=False) if len(rec) else "  無")

        print(f"\n幅度不足以確認 {int((ev['判定'] == '幅度不足以確認').sum())} 筆"
              f"（{MOVE_LIMIT:.0%}–{CONFIRM_LIMIT:.0%}），這段多半是雜訊，"
              "要修請自行把判定欄改成「建議修正」")
        print("⚠️ 停牌復牌首日、處置股解除處置當天也可能沒有漲跌幅限制，"
              "那不是公司行動，套用前請把那些列刪掉。")
        print("⚠️ 比值是 5% 一格 snap 出來的，查得到官方數字就直接改 ratio 欄，"
              "--apply 會照用你改的值。")

    if args.apply:
        if not os.path.exists(EVENTS):
            raise SystemExit(f"找不到 {EVENTS}，先跑 --detect")
        ev = pd.read_csv(EVENTS, dtype={"stock_id": str})
        apply_fixes(ev, args.dry_run)


if __name__ == "__main__":
    main()
