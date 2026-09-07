"""
scripts/build_lesson_html.py

每課產生一個「獨立的 HTML 練習檔」：
  - 圖片用 base64 內嵌，單一檔案就能跑，不需要伺服器
  - 手機瀏覽器直接開，可離線
  - 每跑一次抽不同的題目（用 --seed 可重現）
  - 開頭有心法與看圖重點，答完即時揭曉，最後給總分與錯題回顧
  - 可把作答結果匯出成 CSV

用法：
  python scripts/build_lesson_html.py --lessons C03 --questions 20
  python scripts/build_lesson_html.py --questions 15          # 全部 12 課各一檔
  python scripts/build_lesson_html.py --lessons C03 --seed 42 # 固定題目
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lesson_core as lc

OUT_DIR = os.environ.get("LESSON_HTML_DIR", "lessons_html")

# 每課的作答選項與看圖重點
LESSON_UI = {
    "C03": ("這一根突破 K 的收盤品質是高還是低？",
            ["高（收盤貼近當天最高）", "低（收盤靠近當天最低）"],
            "只看最後一根 K。收盤價落在當天最高與最低之間的哪個位置？"
            "上影線越長，收盤品質越低。"),
    "C01": ("突破前 15 天的整理緊湊度是緊還是鬆？",
            ["緊（振幅小）", "鬆（振幅大）"],
            "看突破前那一段 K 棒的大小，不要看最後一根。"
            "又短又擠 = 緊；天天大紅大綠 = 鬆。"),
    "C05": ("整理期的高檔換手佔比是高還是低？",
            ["高（量集中在箱體上緣）", "低（量堆在箱體下緣）"],
            "看下面的成交量：大量的那幾天，價格是在整理區的上面還是下面？"),
    "C02": ("前高被測試的次數算多還是少？",
            ["多（≥ 3 次）", "少（≤ 2 次）"],
            "往左看，找出價格反覆撞到同一個水平然後被打回來的次數。"),
    "C08": ("這個突破的趨勢年齡算年輕還是老？",
            ["年輕（第一、二次創高）", "老（已經創高很多次）"],
            "看整張圖：這是從底部起來的第一段，還是已經一路墊高很久了？"),
    "C06": ("突破當天相對大盤是強還是弱？",
            ["強於大盤", "弱於或等於大盤"],
            "這題只看數字概念：個股漲跌減掉大盤漲跌。圖上看不出來的部分，"
            "先憑突破的力道猜。"),
    "C04": ("突破前有沒有出現「假跌破後快速收回」的洗盤？",
            ["有洗盤並收回", "沒有（或破了回不來）"],
            "找整理區下緣：有沒有一根探出去、隔兩三天又爬回箱體裡的？"),
    "C07": ("突破前 20 天的推進效率是高還是低？",
            ["高（走得順）", "低（上沖下洗）"],
            "看走勢是一路墊高，還是走半天原地打轉。"),
    "C11": ("整理期的長黑棒修復速度是快還是慢？",
            ["快（≤ 5 天收復）", "慢（很久或沒收復）"],
            "找整理期跌最兇的那根黑K，看之後幾天收盤能站回它的最高點。"),
    "C09": ("這個突破的資訊反應算正面還是負面？",
            ["正面（利空不跌／利多續強）", "負面（利多開高走低）"],
            "先看突破前後有沒有出現「跳空後守不住」的形狀。"),
    "C10": ("如果進場，前三天比較可能站穩還是被打？",
            ["站穩", "被打回突破線下"],
            "出場課。看突破那根的品質與量能，猜隔天的接受度。"),
    "C12": ("這一段動能看起來還在延續還是在老化？",
            ["還在延續", "在老化"],
            "出場課。看創高的間隔有沒有越拉越長、漲幅有沒有越來越小。"),
}


def png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=88)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(v, spec="{:.2f}"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return spec.format(v)
    except Exception:
        return "—"


def build_questions(lesson, pool, prices, n, rng):
    field, hib = lc.LESSON_SELECTOR.get(lesson.cid, ("close_quality", True))
    if field not in pool.columns:
        raise ValueError(f"pool 沒有欄位 {field}")
    sub = pool.dropna(subset=[field]).copy()
    before = len(sub)
    sub = sub[sub["stock_id"].astype(str).map(lc.is_teachable_symbol)]
    if sub.empty:
        raise ValueError(
            f"欄位 {field} 過濾後沒有樣本（原本 {before} 筆；"
            f"C06 需要加權指數資料）")

    med = float(sub[field].median())
    # 兩端各抽一半，避免全是模糊題；不足時放寬
    lo_q, hi_q = sub[field].quantile(0.30), sub[field].quantile(0.70)
    easy_lo = sub[sub[field] <= lo_q]
    easy_hi = sub[sub[field] >= hi_q]
    half = max(n // 2, 1)
    picks = []
    for part in (easy_hi, easy_lo):
        k = min(half, len(part))
        if k:
            picks.append(part.sample(k, random_state=int(rng.integers(1e9))))
    cand = pd.concat(picks) if picks else sub.sample(min(n, len(sub)))
    if len(cand) < n:
        extra = sub.drop(cand.index, errors="ignore")
        if len(extra):
            cand = pd.concat([cand, extra.sample(min(n - len(cand), len(extra)),
                                                 random_state=int(rng.integers(1e9)))])
    cand = cand.sample(frac=1, random_state=int(rng.integers(1e9))).head(n)

    prompt, options, howto = LESSON_UI[lesson.cid]
    qs = []
    for _, r in cand.iterrows():
        sid = str(r["stock_id"])
        g = (prices[prices["stock_id"].astype(str) == sid]
             .sort_values("date").reset_index(drop=True))
        i = int(r["i"])
        if i >= len(g):
            continue
        feats = {c: r[c] for c in pool.columns if c not in ("stock_id", "date", "i")}
        val = float(r[field])
        is_good = (val >= med) if hib else (val <= med)

        q_img = png_b64(lc.render_chart(g, i, {}, mode="plain", lookback=140,
                                        title=f"{sid}",
                                        subtitle="最後一根 = 突破日 T"))
        a_img = png_b64(lc.render_chart(g, i, feats, mode="reveal", forward=60,
                                        lookback=140,
                                        title=f"{sid}　{pd.Timestamp(r['date']).date()}",
                                        subtitle=f"{lesson.cid} {lesson.name}"))
        qs.append({
            "stock_id": sid,
            "date": str(pd.Timestamp(r["date"]).date()),
            "prompt": prompt,
            "options": options,
            "answer": 0 if is_good else 1,
            "field": field,
            "value": val,
            "median": med,
            "q_img": q_img,
            "a_img": a_img,
            "ret20": fmt(r.get("ret_20d_pct"), "{:+.1f}%"),
            "ret60": fmt(r.get("ret_60d_pct"), "{:+.1f}%"),
            "mae": fmt(r.get("mae_60d_pct"), "{:+.1f}%"),
        })
    return qs, howto


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#111827;--grey:#6b7280;--line:#e5e7eb;--ok:#15803d;--no:#b91c1c;--blue:#1d4ed8;--bg:#f8fafc}
*{box-sizing:border-box}
body{margin:0;padding:0 14px 60px;font-family:-apple-system,"PingFang TC","Noto Sans TC",sans-serif;
     color:var(--ink);background:#fff;line-height:1.65;max-width:900px;margin:0 auto}
h1{font-size:1.45rem;margin:22px 0 4px}
.sub{color:var(--grey);font-size:.86rem;margin-bottom:18px}
.card{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0;background:var(--bg)}
.card h3{margin:0 0 8px;font-size:1rem;color:var(--blue)}
.formula{font-family:ui-monospace,Menlo,monospace;background:#fff;border:1px solid var(--line);
         border-radius:8px;padding:9px 11px;font-size:.85rem;overflow-x:auto}
.good{color:var(--ok)}.bad{color:var(--no)}
img{width:100%;border:1px solid var(--line);border-radius:10px;margin:10px 0}
.qhead{display:flex;justify-content:space-between;align-items:baseline;margin-top:26px}
.qno{font-weight:700;font-size:1.05rem}
.meta{color:var(--grey);font-size:.8rem}
button.opt{display:block;width:100%;text-align:left;padding:13px 15px;margin:7px 0;border-radius:10px;
  border:1.5px solid var(--line);background:#fff;font-size:.97rem;cursor:pointer;font-family:inherit;color:var(--ink)}
button.opt:active{background:#f1f5f9}
button.opt.correct{border-color:var(--ok);background:#f0fdf4}
button.opt.wrong{border-color:var(--no);background:#fef2f2}
button.opt:disabled{cursor:default;opacity:.95}
.fb{border-radius:10px;padding:12px 14px;margin:10px 0;font-size:.92rem;display:none}
.fb.show{display:block}
.fb.ok{background:#f0fdf4;border:1px solid #bbf7d0}
.fb.no{background:#fef2f2;border:1px solid #fecaca}
.nums{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:.85rem;color:var(--grey)}
.nums b{color:var(--ink)}
.bar{position:fixed;left:0;right:0;bottom:0;background:#111827;color:#fff;padding:11px 16px;
     display:flex;justify-content:space-between;align-items:center;font-size:.9rem}
.bar button{background:#fff;color:#111827;border:0;border-radius:8px;padding:8px 14px;font-size:.86rem;
     font-family:inherit;cursor:pointer}
hr{border:0;border-top:1px solid var(--line);margin:26px 0}
.hint{font-size:.86rem;color:var(--grey);margin:6px 0 0}
</style></head><body>

<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>

<div class="card">
  <h3>心法 · 老手在問什麼</h3>
  <div>__IDEA__</div>
</div>

<div class="card">
  <h3>怎麼看</h3>
  <div>__HOWTO__</div>
  <div class="formula">__FORMULA__</div>
  <div style="margin-top:9px">
    <div class="good">✓ 這樣算好：__GOOD__</div>
    <div class="bad">✗ 這樣算不好：__BAD__</div>
  </div>
</div>

<hr>
<div id="quiz"></div>

<div class="bar">
  <span id="score">0 / 0　正確率 —</span>
  <span>
    <button onclick="exportCSV()">匯出 CSV</button>
    <button onclick="location.reload()">重看</button>
  </span>
</div>

<script>
const DATA = __DATA__;
const LESSON = __LESSON__;
let log = [];

function render(){
  const box = document.getElementById('quiz');
  DATA.forEach((q, idx) => {
    const d = document.createElement('div');
    d.innerHTML = `
      <div class="qhead"><span class="qno">第 ${idx+1} 題</span>
        <span class="meta">${q.stock_id}　${q.date}</span></div>
      <img src="data:image/png;base64,${q.q_img}" alt="題目圖">
      <div style="font-weight:600;margin:6px 0">${q.prompt}</div>
      <div id="opts${idx}"></div>
      <div class="fb" id="fb${idx}"></div>`;
    box.appendChild(d);
    const opts = d.querySelector('#opts'+idx);
    q.options.forEach((txt, oi) => {
      const b = document.createElement('button');
      b.className='opt'; b.textContent = txt;
      b.onclick = () => answer(idx, oi);
      opts.appendChild(b);
    });
  });
}

function answer(idx, oi){
  const q = DATA[idx];
  const opts = document.querySelectorAll('#opts'+idx+' button');
  if(opts[0].disabled) return;
  opts.forEach((b,i)=>{ b.disabled = true;
    if(i===q.answer) b.classList.add('correct');
    else if(i===oi) b.classList.add('wrong'); });

  const ok = (oi === q.answer);
  const fb = document.getElementById('fb'+idx);
  fb.className = 'fb show ' + (ok ? 'ok' : 'no');
  fb.innerHTML = `<b>${ok ? '答對' : '這題錯了'}</b>　實際 ${q.field} = ${q.value.toFixed(2)}
      （本課母體中位數 ${q.median.toFixed(2)}）
      <div class="nums"><span>後20日 <b>${q.ret20}</b></span>
      <span>後60日 <b>${q.ret60}</b></span>
      <span>最大回撤 <b>${q.mae}</b></span></div>
      <img src="data:image/png;base64,${q.a_img}" alt="揭曉圖">
      <div class="hint">上圖是全標註版：對照一下你剛才有沒有看到該看的地方。</div>`;

  log.push({q:idx+1, lesson:LESSON.cid, stock_id:q.stock_id, date:q.date,
            field:q.field, value:q.value, my_answer:q.options[oi], correct:ok});
  updateScore();
  fb.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function updateScore(){
  const n = log.length, c = log.filter(r=>r.correct).length;
  document.getElementById('score').textContent =
    `${c} / ${n}　正確率 ${n ? Math.round(c/n*100) : '—'}%`;
}

function exportCSV(){
  if(!log.length){ alert('還沒有作答紀錄'); return; }
  const cols = Object.keys(log[0]);
  const csv = '\\ufeff' + cols.join(',') + '\\n' +
    log.map(r=>cols.map(c=>JSON.stringify(r[c] ?? '')).join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
  a.download = `dojo_${LESSON.cid}_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

render();
</script></body></html>
"""


def build_html(lesson, qs, howto, outpath):
    page = (TEMPLATE
            .replace("__TITLE__", html.escape(f"第 {lesson.order} 課 · {lesson.cid} {lesson.name}"))
            .replace("__SUBTITLE__", html.escape(
                f"{lesson.question}　｜　{len(qs)} 題　｜　產生於 {datetime.now():%Y-%m-%d %H:%M}"))
            .replace("__IDEA__", html.escape(lesson.idea))
            .replace("__HOWTO__", html.escape(howto))
            .replace("__FORMULA__", html.escape(f"{lesson.primary} = {lesson.formula}"))
            .replace("__GOOD__", html.escape(lesson.good))
            .replace("__BAD__", html.escape(lesson.bad))
            .replace("__DATA__", json.dumps(qs, ensure_ascii=False))
            .replace("__LESSON__", json.dumps({"cid": lesson.cid, "name": lesson.name},
                                              ensure_ascii=False)))
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(page)
    return outpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", nargs="*", default=None)
    ap.add_argument("--questions", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    lc.setup_font()
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())
    rng = np.random.default_rng(seed)
    print(f"seed = {seed}（要重現同一份題目就加 --seed {seed}）")

    pool_path = os.path.join(lc.ASSET_DIR, "pool.parquet")
    if not os.path.exists(pool_path):
        raise SystemExit(f"找不到 {pool_path}，請先跑 build_lesson_assets.py --rebuild-pool")
    pool = pd.read_parquet(pool_path)
    prices = lc.load_prices()
    print(f"事件池 {len(pool)} 筆")

    todo = [l for l in sorted(lc.LESSONS, key=lambda x: x.order)
            if args.lessons is None or l.cid in args.lessons]

    ok, fail = [], []
    for lesson in todo:
        try:
            qs, howto = build_questions(lesson, pool, prices, args.questions, rng)
            if not qs:
                raise ValueError("抽不到任何題目")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = os.path.join(args.outdir,
                                f"{lesson.order:02d}_{lesson.cid}_{lesson.name}_{stamp}.html")
            build_html(lesson, qs, howto, path)
            size = os.path.getsize(path) / 1024
            print(f"  ✓ {lesson.cid} {lesson.name}：{len(qs)} 題　{size:.0f} KB　{path}")
            ok.append(lesson.cid)
        except Exception as e:
            fail.append((lesson.cid, str(e)))
            print(f"  ✗ {lesson.cid} {lesson.name} 失敗：{e}")
            traceback.print_exc(limit=2)

    print(f"\n完成 {len(ok)} 課；失敗 {len(fail)} 課")
    for cid, msg in fail:
        print(f"  {cid}: {msg}")


if __name__ == "__main__":
    main()
