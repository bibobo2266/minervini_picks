r"""
從價格自己偵測並修正未還原的公司行動

為什麼需要這支：
    FinMind 的 TaiwanStockPriceAdj 只還原「除權息類」事件（配股、配息）。
    股本變動類事件——減資、面額變更、股票分割——它不處理。實測 5 個已知案例，
    官方還原修好 1 檔（6669，1:3），另外 4 檔（6696 1:10、4546 2.07、
    7855 1.87、6428 1.50）完全沒動。
    而自算因子版（build_adj.py 用 TaiwanStockDividendResult）5 檔修好 0 檔，
    因為那張表根本不含股本變動事件。

偵測原理：
    台股上市櫃股票單日漲跌幅上限 10%。單日變動超過 11% 在物理上只有一種可能
    ——未還原的公司行動。這條規則不需要任何 API、不需要訂閱、到期後永遠可用。

    ⚠️ 必須雙向偵測：
      - 分割 / 面額變更（10 元→1 元）→ 除權日價格「往下跳」，比值 > 1，
        修正方式是把歷史價「除以」比值。
      - 減資 / 反向分割 → 除權參考價「往上跳」（減資五成 = 價格翻倍），
        比值 < 1，修正方式等於把歷史價「乘上去」。
      舊版只抓下跌，減資那一整類事件完全看不到。

    例外：興櫃股票、上市前五個交易日沒有漲跌幅限制，±50% 對它們是合法的。
    universe.parquet 的 type 欄位直接標了 twse / tpex / emerging，用它排除
    興櫃比從價格猜可靠得多——539 檔興櫃如果用猜的會漏掉一大半。

    另一個誤判來源：停牌復牌首日、處置股解除處置，當天也可能沒有漲跌幅限制。
    這種不是公司行動，只能靠 --detect 產出的 CSV 人工剔除。兩段式流程就是
    為了擋這個。

修正方式：
    比值不能直接用「前收 / 當日價」算——當日開盤與收盤都已經含市場漲跌，
    跟除權參考價有落差。實測 6669：前收 7800、開 2790、收 2610，
    開盤推出 2.796、收盤推出 2.988，官方的正確值是 2.983。

    改用約束求解：真正的比值必須讓還原後的單日漲跌落在 ±10% 內（漲跌幅上限）。
    在這個區間裡挑最接近「乾淨比值」的候選——減資成數 1/(1-k)、整數分割倍數，
    以及它們的倒數（給往上跳的減資用）。

刻意分成兩段，不要一鍵修完：
    python scripts/fix_corporate_actions.py --detect   # 只偵測，產出清單
    python scripts/fix_corporate_actions.py --apply    # 依清單修正
    偵測結果是 CSV，你可以在套用前把誤判的列刪掉。自動修正價格資料是不可逆的
    操作，中間需要一個人看過。

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

MOVE_LIMIT = 0.11      # 台股漲跌幅上限 10%，留 1% 緩衝給資料誤差（雙向）
MIN_RATIO = 1.10       # 比值離 1 太近就不值得修（多半是資料雜訊）
MAX_RATIO = 25.0       # 高於這個八成是資料錯誤，不是公司行動
NEW_LIST_DAYS = 30     # 上市未滿這麼多天，前五日無漲跌幅限制的影響還在

# 乾淨比值候選：
#   > 1  →  分割 / 面額變更 / 大額配股，除權日價格往下跳
#   < 1  →  減資 / 反向分割，除權參考價往上跳
_BASE = sorted({round(1 / (1 - k), 4) for k in np.arange(0.05, 0.91, 0.05)}
               | {float(n) for n in range(2, 11)})
CLEAN = sorted(set(_BASE) | {round(1 / r, 6) for r in _BASE})


def ratio_ok(r: float) -> bool:
    """比值要離 1 夠遠、又不能離譜。兩個方向都檢查。"""
    if r is None or r != r:
        return False
    return (MIN_RATIO <= r <= MAX_RATIO) or (1 / MAX_RATIO <= r <= 1 / MIN_RATIO)


def snap_ratio(prev_close: float, close: float):
    """在「還原後單日漲跌 <= 10%」的合法區間內，挑最乾淨的比值。

    合法區間：close * ratio / prev_close - 1 必須落在 [-0.10, +0.10]
             → ratio ∈ [0.90 * prev/close, 1.10 * prev/close]

    這個式子對兩個方向都成立：價格往下跳時 prev/close > 1、候選在 1 以上；
    往上跳時 prev/close < 1、候選落在倒數那一側。
    """
    lo = 0.90 * prev_close / close
    hi = 1.10 * prev_close / close
    cands = [r for r in CLEAN if lo <= r <= hi]
    if not cands:
        return None, None
    mid = prev_close / close
    best = min(cands, key=lambda r: abs(r - mid))
    return best, abs(best - mid)


def load_all():
    fs = sorted(glob.glob(os.path.join(DATA_DIR, "prices_adj_*.parquet")))
    if not fs:
        raise SystemExit(f"{DATA_DIR} 底下沒有 prices_adj_*.parquet")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["stock_id", "date"]).reset_index(drop=True), fs


def detect(d: pd.DataFrame, uni=None) -> pd.DataFrame:
    """回傳疑似未還原公司行動的清單（雙向）。"""
    d = d[d["stock_id"].astype(str).str.match(r"^[1-9]\d{3}$")].copy()
    d = d[d["close"] > 0]

    if uni is not None and "type" in uni.columns:
        listed = set(uni[uni["type"].isin(["twse", "tpex"])].index.astype(str))
        n0 = d["stock_id"].nunique()
        d = d[d["stock_id"].astype(str).isin(listed)]
        print(f"  排除興櫃與未知：{n0} → {d['stock_id'].nunique()} 檔"
              "（興櫃無漲跌幅限制，±50% 對它們是合法的）")

    g = d.groupby("stock_id", sort=False)
    d["prev_close"] = g["close"].shift()
    d["chg"] = d["close"] / d["prev_close"] - 1
    # 一次算完每檔在資料裡的首日，取代舊版每筆命中重掃一次全表
    d["first_date"] = g["date"].transform("min")

    hits = d[(d["chg"].abs() > MOVE_LIMIT) & d["prev_close"].notna()].copy()
    if hits.empty:
        return pd.DataFrame()

    rows = []
    for _, e in hits.iterrows():
        prev_c, close_c = float(e["prev_close"]), float(e["close"])
        ratio, _gap = snap_ratio(prev_c, close_c)
        days_since_listing = (e["date"] - e["first_date"]).days
        direction = "往下跳（分割／面額變更）" if e["chg"] < 0 \
            else "往上跳（減資／反向分割）"

        if days_since_listing < NEW_LIST_DAYS:
            verdict = "疑似新上市"
            ratio = ratio if ratio is not None else np.nan
        elif ratio is None:
            verdict = "找不到合理比值"
        elif not ratio_ok(ratio):
            verdict = "比值超出合理範圍"
        else:
            verdict = "建議修正"

        adj_chg = (close_c * ratio / prev_c - 1) * 100 \
            if ratio is not None and ratio == ratio else np.nan

        rows.append({
            "stock_id": e["stock_id"],
            "event_date": e["date"].date().isoformat(),
            "prev_close": round(prev_c, 4),
            "open": round(float(e.get("open", np.nan)), 4),
            "close": round(close_c, 4),
            "chg%": round(float(e["chg"]) * 100, 1),
            "ratio": round(ratio, 6) if ratio is not None and ratio == ratio else np.nan,
            "還原後漲跌%": round(adj_chg, 1) if adj_chg == adj_chg else np.nan,
            "方向": direction,
            "上市天數": days_since_listing,
            "判定": verdict,
        })

    out = pd.DataFrame(rows).sort_values(["event_date", "stock_id"])
    return out.reset_index(drop=True)


def apply_fixes(events: pd.DataFrame, dry_run: bool):
    """把事件日之前的 OHLC 除以比值。

    比值 > 1 → 歷史價被壓低（分割）。
    比值 < 1 → 歷史價被放大（減資），除以小於 1 的數就是乘上去。

    只處理判定為「建議修正」的列。想納入其他列的，自己把 CSV 裡那一格改成
    「建議修正」再跑一次——這是刻意的摩擦，價格資料改壞很難救。
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

    # 同一檔可能有多次事件。由近而遠累乘，跟 build_adj 的因子邏輯一致：
    # 較早的價格要吸收它之後所有事件的比值。
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
            print(f"  讀不到 universe（{ex}），無法排除興櫃，結果會有大量誤判")
            uni = None

        print(f"讀入 {len(fs)} 個年份檔，{len(d):,} 列，"
              f"{d['stock_id'].nunique()} 檔")
        ev = detect(d, uni)
        if ev.empty:
            print("沒有偵測到單日變動逾 11% 的異常，資料乾淨。")
            return

        ev.to_csv(EVENTS, index=False, encoding="utf-8-sig")
        print(f"\n偵測到 {len(ev)} 筆，已寫入 {EVENTS}\n")
        print(ev["判定"].value_counts().to_string())
        print()
        print(ev["方向"].value_counts().to_string())

        cols = ["stock_id", "event_date", "prev_close", "close", "ratio",
                "chg%", "還原後漲跌%", "方向"]
        rec = ev[ev["判定"] == "建議修正"]
        print("\n判定為「建議修正」的（最近 20 筆）：")
        print(rec.tail(20)[cols].to_string(index=False) if len(rec) else "  無")

        other = ev[ev["判定"] != "建議修正"]
        if len(other):
            print("\n非「建議修正」的（最近 10 筆，人工看過覺得該修就改判定欄）：")
            print(other.tail(10)[cols].to_string(index=False))
            print(f"  要修的話，把 {EVENTS} 裡那一列的「判定」改成"
                  "「建議修正」再跑 --apply")

        print("\n⚠️ 停牌復牌首日、處置股解除處置當天也可能沒有漲跌幅限制，"
              "那不是公司行動，套用前請把那些列刪掉。")

    if args.apply:
        if not os.path.exists(EVENTS):
            raise SystemExit(f"找不到 {EVENTS}，先跑 --detect")
        ev = pd.read_csv(EVENTS, dtype={"stock_id": str})
        apply_fixes(ev, args.dry_run)


if __name__ == "__main__":
    main()
