#!/usr/bin/env python3
"""
scripts/build_handbook.py — 產生《老手看盤 · 詞彙速查手冊（引擎版）》PDF

沿用原手冊版面：一課一頁、深藍標題列、等級表、為什麼／不理會的後果、指示表。
兩處新增：等級表多一欄「台股實際百分位」；每課末列出「哪幾組 combo 用到它」。
校準數字取自 combo_registry.json 的 calibration 區塊。

  python scripts/build_handbook.py -o 老手速查_引擎版.pdf
"""
from __future__ import annotations

import argparse
import ast
import json
import os

CSS = """
@page { size: A4; margin: 18mm 15mm 15mm 15mm;
  @top-left { content: "老手看盤 · 詞彙速查手冊（引擎版）"; font-size: 8.5pt; color: #555; }
  @top-right { content: "第 " counter(page) " 頁"; font-size: 8.5pt; color: #555; } }
body { font-family: "Noto Sans CJK TC","Noto Sans CJK SC",sans-serif;
  font-size: 10pt; color: #111; line-height: 1.45; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
h1.cover { font-size: 22pt; margin: 60mm 0 4mm; }
p.sub { color:#555; font-size:10pt; margin:0 0 2mm; }
.bar { background:#12314f; color:#fff; padding:7px 12px; display:flex;
  justify-content:space-between; align-items:baseline; margin-bottom:0; }
.bar .t { font-size:14pt; font-weight:700; }
.bar .q { font-size:10pt; opacity:.92; }
table { width:100%; border-collapse:collapse; margin:0 0 3mm; }
td, th { border:1px solid #c9d2da; padding:5px 8px; vertical-align:top; font-size:9.5pt; }
th { background:#e7ecf1; font-weight:700; text-align:left; }
td.lab { background:#f5f7f9; font-weight:700; width:16%; }
td.field { font-family:"DejaVu Sans Mono",monospace; font-size:9pt; }
.g { color:#1a7f37; font-weight:700; }
.o { color:#b35c00; font-weight:700; }
.r { color:#b3261e; font-weight:700; }
.n { color:#555; font-weight:700; }
.pct { text-align:center; white-space:nowrap; }
.bad { background:#fdecea; }
.warn { background:#fff6e5; }
.note { font-size:8.5pt; color:#444; background:#f5f7f9;
  border-left:3px solid #12314f; padding:6px 9px; margin:0 0 3mm; }
.combo { font-size:8.5pt; color:#222; }
.combo b { color:#12314f; }
ul { margin:2mm 0 3mm 5mm; padding:0; }
li { font-size:9.5pt; margin-bottom:1.5mm; }
"""

# 欄位 → 課別對照，用來從 registry 反查「哪幾組 combo 用到它」
FIELD2C = {
    "c01_tight": "C01", "c01_base_depth": "C01+", "c02_tests": "C02", "c03_cq": "C03",
    "shake_depth": "C04", "shake_recovered": "C04", "shake_broken": "C04",
    "c05_volratio": "C05+", "c05_contract": "C05+", "c05_high_zone": "C05",
    "c06_rs": "C06", "c06_rs20": "C06+", "c06_rs_ind": "C06ind", "ind_ret": "C06ind",
    "c07_er": "C07", "c07_net": "C07", "c08_bo_count": "C08",
    "c08_days_from_low": "C08+", "c08_gain_from_low": "C08+",
    "c09_excess": "C09", "ev_ret": "C09", "c10_mae3": "C10", "c10_mfe3": "C10",
    "c11_bars_recover": "C11", "c11_days_elapsed": "C11",
    "c12_days_since_high": "C12+", "c12_decay": "C12",
}


def combo_usage(reg):
    use = {}
    for c in reg["combos"]:
        seen = set()
        for e in c["when"]:
            for n in ast.walk(ast.parse(e, mode="eval")):
                if isinstance(n, ast.Name) and n.id in FIELD2C:
                    seen.add(FIELD2C[n.id])
        for s in seen:
            use.setdefault(s, []).append(c["id"])
    return {k: sorted(v) for k, v in use.items()}


def pctof(cal, field, thr):
    f = cal.get(field)
    if not f:
        return None
    t = f["thresholds"].get(str(thr)) or f["thresholds"].get(str(float(thr)))
    return None if t is None else t["pct"]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lesson_html(L, cal, use):
    h = ['<div class="page">']
    h.append(f'<div class="bar"><span class="t">第 {L["no"]} 課　{L["cid"]} {L["title"]}</span>'
             f'<span class="q">{L["q"]}</span></div>')
    h.append("<table>")
    h.append(f'<tr><td class="lab">欄位</td><td class="field">{L["field"]}'
             f'　<span style="font-family:inherit">（{L["fieldzh"]}）</span></td></tr>')
    h.append(f'<tr><td class="lab">怎麼算</td><td>{L["how"]}</td></tr>')
    if L.get("engine"):
        h.append(f'<tr><td class="lab">引擎差異</td><td class="warn">{L["engine"]}</td></tr>')
    h.append("</table>")

    h.append("<table>")
    h.append("<tr><th style='width:14%'>等級</th><th style='width:18%'>數值</th>"
             "<th style='width:14%'>台股實際<br>百分位</th><th>長什麼樣</th></tr>")
    for lv in L["levels"]:
        cls = lv.get("cls", "n")
        p = pctof(cal, L.get("calfield"), lv.get("thr")) if lv.get("thr") is not None else None
        pcell = "—" if p is None else f"P{p:g}"
        rowcls = ' class="bad"' if lv.get("bad") else ""
        h.append(f'<tr{rowcls}><td class="{cls}">{lv["name"]}</td><td>{lv["val"]}</td>'
                 f'<td class="pct">{pcell}</td><td>{lv["look"]}</td></tr>')
    for extra in L.get("companion", []):
        h.append(f'<tr><td class="lab">{extra[0]}</td><td>{extra[1]}</td>'
                 f'<td class="pct">{extra[2]}</td><td>{extra[3]}</td></tr>')
    h.append("</table>")

    if L.get("verdict"):
        h.append(f'<div class="note">{L["verdict"]}</div>')

    h.append("<table>")
    h.append(f'<tr><td class="lab">為什麼</td><td>{L["why"]}</td></tr>')
    h.append(f'<tr><td class="lab">不理會的<br>後果</td><td>{L["cost"]}</td></tr>')
    h.append("</table>")

    h.append("<table>")
    h.append("<tr><th style='width:16%'>指示</th><th>條件</th></tr>")
    for s in L["signals"]:
        h.append(f'<tr><td class="{s[0]}">{s[1]}</td><td>{s[2]}</td></tr>')
    h.append("</table>")

    ids = use.get(L["cid"], [])
    line = (f'<b>{len(ids)} 組 combo 用到本課主欄</b>：{" ".join(ids)}' if ids
            else "<b>目前沒有 combo 使用本課主欄</b>")
    for tag, label in [(L["cid"] + "+", "配套欄"), (L["cid"] + "ind", "產業基準")]:
        e = use.get(tag)
        if e:
            line += f'<br><b>{label}</b>（{len(e)} 組）：{" ".join(e)}'
    h.append(f'<div class="combo">{line}</div>')
    h.append("</div>")
    return "".join(h)


def build(reg):
    cal = reg.get("calibration", {}).get("fields", {})
    use = combo_usage(reg)
    T = reg["thresholds"]

    LESSONS = [
        dict(no=1, cid="C03", title="突破推力", q="這根 K 乾不乾淨？",
             field="close_quality", fieldzh="收盤品質", calfield="c03_cq",
             how="(收盤 − 最低) ÷ (最高 − 最低)，值域 0 ~ 1",
             engine="最高＝最低時分母為零 → 記為<b>缺值</b>，不強填 1。缺值不算不符、也不算符合。",
             levels=[
                 dict(name="很好", val="≥ 0.85", thr=0.85, cls="g", look="收盤幾乎貼著當日最高，上影線幾乎沒有"),
                 dict(name="及格", val="0.70 ~ 0.85", thr=0.7, cls="n", look="有一小段上影線，尾盤有點賣壓但守住"),
                 dict(name="警戒", val="0.40 ~ 0.70", thr=None, cls="o", look="收在中段，多空拉鋸，方向沒定"),
                 dict(name="不及格", val="≤ 0.40", thr=0.4, cls="r", look="長上影線，衝高被打回，收在下緣"),
             ],
             verdict="校準結果：0.85 落在 P84、0.70 落在 P73，<b>刻度準確</b>。"
                     "但 0.40 落在 P49.6 —— 幾乎一半的交易日都在「不及格」區，"
                     "這條線<b>幾乎沒有篩選力</b>，實務上要更嚴才有意義。",
             why="收盤是一天當中唯一「所有人都同意」的價格。盤中衝高誰都做得到，但要撐到 13:30 還在高點，"
                 "代表整個交易時段買方都沒鬆手、也沒人急著在尾盤倒貨。上影線的物理意義就是：有人在那個價位大量賣出，把價格壓了回來。",
             cost="收盤品質低的突破，當天最後那批買在高點的人已經套住，他們就是明天的賣壓。追這種突破常常買在當日最高。",
             signals=[("g", "↑ 偏多", "≥ 0.85 且帶量（量比 ≥ 1.5）"),
                      ("o", "→ 中性", "0.40 ~ 0.70，等隔天確認"),
                      ("r", "↓ 偏空", "≤ 0.40，尤其量還特別大——衝高爆量收黑是典型出貨形狀")]),

        dict(no=2, cid="C01", title="壓縮整理", q="噴之前有沒有被壓成彈簧？",
             field="tightness_15d_pct", fieldzh="整理緊湊度", calfield="c01_tight",
             how="過去 15 天 (最高−最低) ÷ 收盤 的平均，換算成 %",
             engine=f"<b>原手冊整組刻度是美股振幅尺度，台股不適用。</b>台股日振幅中位數 15.6%。"
                    f"引擎已改為台股同義分位：壓縮線 <b>{T['tight_compress']}%</b>（P11）、"
                    f"寬幅線 <b>{T['tight_wide']}%</b>（P75）、淺 base <b>{T['base_shallow']}%</b>（P15）。",
             levels=[
                 dict(name="很好", val="≤ 2.5%（原）", thr=2.5, cls="g", bad=True, look="整理期像睡著，K 棒又短又擠"),
                 dict(name="及格", val="2.5% ~ 3.5%（原）", thr=3.5, cls="n", bad=True, look="正常整理"),
                 dict(name="偏鬆", val="3.5% ~ 5%（原）", thr=5.0, cls="o", bad=True, look="還在震，籌碼沒完全沉澱"),
                 dict(name="不及格", val="≥ 5%（原）", thr=None, cls="r", bad=True, look="天天大漲大跌，分歧很大"),
             ],
             companion=[("引擎壓縮線", f"≤ {T['tight_compress']}%", "P11", "台股「緊」的實際樣子，約每 9 天出現 1 天"),
                        ("引擎寬幅線", f"≥ {T['tight_wide']}%", "P75", "真正的高震盪"),
                        ("配套：Base 深度", "≤ 15% 原／≤ 22% 引擎", "P4.7／P15", "> 30%（P32.5，合理）太深，那是下跌不是整理")],
             verdict="⚠️ <b>這是全手冊唯一整組壞掉的刻度。</b>原線 2.5／3.0／3.5／5.0 分別落在 "
                     "P0.4／P0.8／P1.4／P4.5，等於 95% 的台股交易日都被判「不及格」。"
                     "combo 07 與 13 在舊刻度下十年 0 次命中，原因就在這裡。",
             why="振幅是「意見分歧程度」的量化。振幅收斂代表想賣的人賣得差不多、想買的人也不急，浮動籌碼被時間磨光。"
                 "這時上方沒有源源不絕的賣單，一有買盤價格就容易快速移動。整理越久，越多人的成本集中在同一個窄區間。",
             cost="整理期還在三溫暖就突破，代表分歧仍大，上方隨時有人想跑。這種突破常兩三天就打回原形，"
                  "而且因為波動本來就大，停損很容易被掃到。",
             signals=[("g", "↑ 偏多", f"≤ {T['tight_compress']}%（引擎線）且均線糾結"),
                      ("o", "→ 橫盤", f"≤ {T['tight_compress']}% 但相對強弱為負——彈簧壓緊了卻沒人扣扳機，可以緊很久"),
                      ("r", "↓ 偏空", f"≥ {T['tight_wide']}%，突破可信度低")]),

        dict(no=3, cid="C05", title="量價換手", q="量縮得夠不夠？高檔換手夠不夠？",
             field="high_zone_turnover_share_20d", fieldzh="高檔換手佔比", calfield=None,
             how="過去 20 天成交量落在箱體上 1/3 價格帶的比例（%）",
             engine="⛔ <b>主欄永久停用。</b>只靠每日 OHLCV 無法知道每一筆量成交在哪一個價位，"
                    "手冊自己在第 4 頁載明這點。任何用到主欄的 combo 一律降級為 PARTIAL（01、07）。"
                    "<b>下面兩個配套欄可用，且刻度準確。</b>",
             levels=[
                 dict(name="很好", val="≥ 45%", thr=None, cls="g", bad=True, look="⛔ 無法計算"),
                 dict(name="及格", val="30% ~ 45%", thr=None, cls="n", bad=True, look="⛔ 無法計算"),
                 dict(name="不及格", val="≤ 20%", thr=None, cls="r", bad=True, look="⛔ 無法計算"),
             ],
             companion=[("配套：量縮比", "≤ 0.60", "P14", "近 5 日均量 ÷ 近 20 日均量（原手冊為 15/60 日中位數）"),
                        ("配套：突破量比", "≥ 1.5 倍", "P74", "今日量 ÷ 前 20 日均量"),
                        ("配套：爆量", "≥ 3.0 倍", "P90", "搭配低收盤品質時是出貨疑慮"),
                        ("配套：無量", "< 1.2 倍", "P64", "未明顯放量")],
             verdict="配套三條線 1.2／1.5／3.0 分別落在 P64／P74／P90，量縮比 0.60 落在 P14。"
                     "<b>四條都合理，可以照原手冊用。</b>只有主欄不能用。",
             why="整理期量縮代表「賣方沒興趣了」——不是沒人買，是沒人願意在這價位賣。"
                 "量縮到極致後的第一根爆量，就是新買方集中進場的那一刻。",
             cost="整理期量沒縮，代表還有人持續在賣，突破時要吃掉的賣單更多。無量突破多半是零星買盤把價格墊上去，"
                  "沒有真實承接，一有賣壓就掉回去。",
             signals=[("g", "↑ 偏多", "量縮 ≤ 0.6 ＋ 突破量 ≥ 1.5 倍（高檔換手無法驗證）"),
                      ("o", "→ 觀察", "突破量 1.2 ~ 1.5 倍，力道不明確"),
                      ("r", "↓ 偏空", "突破日爆量 ≥ 3 倍但收盤品質 ≤ 0.4——量能用在出貨不是進貨")]),

        dict(no=4, cid="C02", title="阻力消耗", q="前高被撞幾次？賣壓吃乾淨沒？",
             field="prior_high_tests_count", fieldzh="前高測試次數", calfield="c02_tests",
             how="過去 60 天內，最高價觸及 250 日前高 2% 以內的波峰次數",
             engine=f"引擎凍結兩個原手冊沒定義的規則：<b>觸及帶 {T.get('touch_band', 2)}%</b>"
                    f"（registry 的 touch_band_pct）、<b>波峰去重 5 個交易日</b>。改這兩個數字會讓新舊卡片不可比。",
             levels=[
                 dict(name="很好", val="3 ~ 5 次", thr=3, cls="g", look="撞了好幾次，上面的解套賣單被磨得差不多"),
                 dict(name="及格", val="2 次", thr=None, cls="n", look="有測試過，但還不夠"),
                 dict(name="太早", val="1 次", thr=None, cls="o", look="第一次碰前高，上方賣壓完全沒被消耗"),
                 dict(name="太強", val="≥ 6 次", thr=6, cls="r", look="撞這麼多次還過不去，那裡的賣壓是真的重"),
             ],
             companion=[("測頂上限", "5 次", "P79", "3~5 次區間的上緣"),
                        ("配套", "回檔一次比一次淺", "—", "引擎用 higher_lows 旗標（連續 3 個波段低點遞增）")],
             verdict="3 次落在 P66、5 次 P79、6 次 P85。<b>刻度合理，照原手冊用。</b>",
             why="前高之所以是壓力，是因為有一群人套在那裡等解套。撞第一次時這些人幾乎都還在；"
                 "撞到第三、四次，急著解套的人大多走了。上方掛單變薄，同樣的買盤力道就能推得更遠。",
             cost="第一次碰前高就追，你是在跟一整批等解套的人對作。常見形狀是衝上去一兩天就被打回箱體內。",
             signals=[("g", "↑ 偏多", "3 ~ 5 次，且每次回檔越來越淺、貼著前高不掉"),
                      ("o", "→ 觀察", "1 ~ 2 次，等它再測一次"),
                      ("r", "↓ 偏空", "≥ 6 次且回檔越來越深——不是在消耗賣壓，是在走弱")]),

        dict(no=5, cid="C08", title="趨勢年齡", q="這是第一段還是末升段？",
             field="major_breakout_count_250d", fieldzh="250日內突破次數", calfield="c08_bo_count",
             how="過去 250 天內，收盤突破 250 日前高的事件次數",
             engine="引擎的突破事件<b>去重 10 個交易日</b>（連續創高不重複計數）。"
                    "⚠️ 原手冊寫「C08 有否決權、高齡一律降級」，<b>60 組版已取消這條，引擎也沒有實作否決</b>。"
                    "高齡是風險背景，不是自動看空。",
             levels=[
                 dict(name="很好", val="≤ 1 次", thr=1, cls="g", look="剛從底部起來"),
                 dict(name="及格", val="2 ~ 3 次", thr=3, cls="n", look="趨勢確立中"),
                 dict(name="老", val="≥ 4 次", thr=4, cls="r", look="已經漲一大段，晚期"),
             ],
             companion=[("配套：距底部", "20 ~ 80 天", "—", "距 250 日低點天數"),
                        ("配套：漲幅", "≤ 50%", "P48", "⚠️ 篩選力弱，近半數股票都在這一區"),
                        ("配套：高風險", "> 100%", "P74", "自 250 日低點的漲幅，合理")],
             verdict="1 次落在 P32、4 次 P66，合理。但 <b>3 次落在 P55、漲幅 50% 落在 P48</b>，"
                     "這兩條線各自把市場切成一半，篩選力弱。",
             why="股價從底部起漲會經歷「沒人相信 → 開始有人注意 → 大家都知道」三階段。"
                 "同一根漂亮的突破 K，在第一段和第五段的市場結構完全不同。",
             cost="末升段的突破「看起來最漂亮」——經過多次上漲，所有技術指標都完美。這是最容易買在最高點的地方。",
             signals=[("g", "↑ 偏多", "≤ 1 次 ＋ 距底部 20~80 天 ＋ 漲幅 ≤ 50%"),
                      ("o", "→ 中性", "2 ~ 3 次"),
                      ("r", "↓ 偏空", "≥ 4 次或自底部漲幅 > 100%——這是風險背景，不是自動否決")]),

        dict(no=6, cid="C06", title="相對強弱", q="是大盤帶它漲，還是它自己走？",
             field="rs_excess_taiex_pct", fieldzh="相對大盤強弱", calfield="c06_rs",
             how="個股當日漲跌 % − 加權指數當日漲跌 %，單位是<b>百分點</b>",
             engine="⚠️ <b>指數落後就整欄缺值。</b>index_taiex.parquet 若沒跟上行情日期，"
                    "本欄變 NaN，D 類六組（19–24）會整類靜默失效。看盤卡會印警告。"
                    "另有配套 20 日累積超額 c06_rs20。",
             levels=[
                 dict(name="很好", val="≥ +2 個百分點", thr=2.0, cls="g", look="明顯自己走"),
                 dict(name="及格", val="0 ~ +2", thr=None, cls="n", look="略強於大盤"),
                 dict(name="不及格", val="< 0", thr=0.0, cls="r", look="比大盤弱，這根突破多半是被大盤抬上去的"),
             ],
             companion=[("加分項", "大盤跌它不跌", "—", "大盤單日 −1% 以上而個股收紅，最有價值的訊號（combo 19）")],
             verdict="+2 落在 P79，<b>刻度準確</b>。0 落在 P51.6 —— 本來就該接近一半，這不是缺陷。"
                     "<b>本欄有 31 組 combo 用到，是使用率第二高的欄位。</b>",
             why="任何一天的個股漲跌，約一半來自大盤，另一半才是它自己。真正有人在買的股票，"
                 "特徵是大盤回檔時它橫著不跌——這種承接在大盤好的時候看不出來，只有逆風時才顯形。",
             cost="只有 beta 沒有 alpha 的股票，大盤一轉弱就原形畢露，而且跌得比大盤還快。",
             signals=[("g", "↑ 偏多", "≥ +2 個百分點，或大盤重挫日仍收紅"),
                      ("o", "→ 中性", "0 ~ +2"),
                      ("r", "↓ 偏空", "< 0，尤其大盤漲它漲更少")]),

        dict(no=7, cid="C04", title="結構防守", q="被打下去有沒有立刻爬回來？",
             field="shakeout_depth_pct", fieldzh="洗盤深度", calfield=None,
             how="跌破箱底 L 後、3 日內重新收回的破底幅度（%）",
             engine="引擎的箱底 L = <b>過去 60 日最低價（不含當日）</b>，掃描窗 20 日。"
                    "另存 shake_recovered、shake_broken 兩個旗標。"
                    "⚠️ 原手冊把 &gt;8% 直接視為真破底，<b>引擎把「深度」與「收回與否」分開記</b>，不直接判死。",
             levels=[
                 dict(name="很好", val="1% ~ 5%", thr=None, cls="g", look="淺淺探一下就收回，典型的最後洗盤"),
                 dict(name="可接受", val="5% ~ 8%", thr=None, cls="n", look="洗得比較兇，但仍收回"),
                 dict(name="危險", val="> 8%", thr=None, cls="o", look="幅度大，但已收回仍記為已收回（combo 26）"),
                 dict(name="不及格", val="破了沒收回", thr=None, cls="r", look="shake_broken = True，結構已壞"),
             ],
             companion=[("配套：收回天數", "≤ 3 天", "—", "引擎的 shakeout_recover_days = 3")],
             verdict="本欄沒有跨股可比的分位（多數日子沒有洗盤事件，值為缺值）。"
                     "<b>無事件時是缺值，不是 0% 淺破</b>——這一點原手冊也特別強調過。",
             why="一次向下假跌破會觸發沒信心那批人的停損，把持股轉移給願意在低點承接的人。"
                 "關鍵在「快速收回」——跌破後回不來，那不是洗盤，是趨勢真的轉向了。",
             cost="把真破底誤認成洗盤，是虧大錢最常見的方式之一。差別只有一個：有沒有在 3 天內收回。",
             signals=[("g", "↑ 偏多", "1~5% 淺洗 ＋ 3 天內收回"),
                      ("o", "→ 觀察", "破了還在掙扎，等它站回箱底之上"),
                      ("r", "↓ 偏空", "破 8% 以上且未收回")]),

        dict(no=8, cid="C07", title="推進效率", q="走得順不順？",
             field="price_progress_efficiency_20d", fieldzh="推進效率", calfield="c07_er",
             how="20 日淨位移 ÷ 20 日路徑總長，值域 0 ~ 1",
             engine="⚠️ <b>引擎改了公式。</b>原手冊寫「淨位移 ÷ 真實區間總和」卻稱考夫曼效率比，口徑有疑義。"
                    "引擎採<b>標準 Kaufman ER</b> = |收盤ₜ − 收盤ₜ₋₂₀| ÷ Σ|每日收盤變動|。"
                    "另存帶正負的 c07_net（淨位移）判方向——效率高不等於往上。",
             levels=[
                 dict(name="很好", val="≥ 0.40", thr=0.4, cls="g", look="一路墊高，走勢順暢"),
                 dict(name="及格", val="0.25 ~ 0.40", thr=None, cls="n", look="正常"),
                 dict(name="不及格", val="≤ 0.15", thr=0.15, cls="r", look="上沖下洗，走半天原地打轉"),
             ],
             verdict="0.40 落在 P81、0.15 落在 P36，<b>換公式後居然剛好對上原手冊的分級</b>。"
                     "但這是巧合不是驗證：舊公式的 0.40 跟新公式的 0.40 不是同一把尺。",
             why="同樣漲 20%，一路順順走上去跟震盪半年才到，背後的市場結構完全不同。"
                 "效率低代表買賣力道相當，每一步都要打一場仗。",
             cost="低效率的股票就算方向對了，實際交易起來也會虧損，因為被洗出場的次數比獲利次數多。",
             signals=[("g", "↑ 偏多", "≥ 0.40 且淨位移為正"),
                      ("o", "→ 中性", "0.25 ~ 0.40"),
                      ("r", "↓ 偏空", "≥ 0.40 但淨位移為負——那是順暢下跌（combo 17）")]),

        dict(no=9, cid="C11", title="壓力修復", q="這檔是不是慣性騙人？",
             field="bars_to_recover_largest_down_day", fieldzh="長黑收復天數", calfield=None,
             how="近 60 日最大單日跌幅那天的最高價，之後幾根收盤才收復",
             engine="<b>−1 是狀態碼（尚未收復），不是 −1 天。</b>引擎另存 c11_days_elapsed（已經過幾天），"
                    "未滿 20 天不能算「慢逾 20 日」。掃描窗 60 個交易日。",
             levels=[
                 dict(name="很好", val="≤ 5 天", thr=None, cls="g", look="抗壓性強，被打下去很快站回來"),
                 dict(name="及格", val="6 ~ 15 天", thr=None, cls="n", look="正常"),
                 dict(name="未分級", val="16 ~ 20 天", thr=None, cls="n", look="原稿未分級，引擎不自行補等級"),
                 dict(name="不及格", val="> 20 天 或 −1", thr=None, cls="r", look="−1 代表始終沒收復"),
             ],
             verdict="本欄是天數不是比率，沒有跨股橫斷面分位。"
                     "<b>combo 28（快速修復但尚未突破）就是這一課</b>，它是全市場最黏的狀態，"
                     "平均連續亮 13.3 天。它講的是個股自己的傷口癒合，跟大盤無關。",
             why="每檔股票都有自己的慣性。願意在下跌時加碼的人多，修復就快。這個特性相當穩定，"
                 "可以拿來預判它未來遇到壓力時的行為。",
             cost="修復慢的股票，進場後只要遇到一次大盤回檔，就得等很久才能回到成本，期間很容易先被停損掃出去。",
             signals=[("g", "↑ 偏多", "≤ 5 天"),
                      ("o", "→ 中性", "6 ~ 15 天"),
                      ("r", "↓ 偏空", "> 20 天或從未收復")]),

        dict(no=10, cid="C09", title="資訊反應", q="利空打不打得下去？利多是不是出貨？",
             field="resilience_excess_ret", fieldzh="利空抗跌度", calfield=None,
             how="負面事件日或大盤重挫日的個股超額報酬（%）",
             engine="⛔ <b>需要 event log。</b>data/events/events.csv 沒有對應日期的紀錄就沒有值，"
                    "F 類六組（31–36）目前全部啞掉。事件要有<b>發言時間</b>才能分辨盤中或盤後（combo 36）。"
                    "來源建議：公開資訊觀測站重大訊息。",
             levels=[
                 dict(name="很好", val="≥ 0%", thr=None, cls="g", look="利空日還能收平或收紅"),
                 dict(name="及格", val="−1% ~ 0%", thr=None, cls="n", look="小跌，仍相對抗跌"),
                 dict(name="不及格", val="跌幅大於大盤", thr=None, cls="r", look="沒有支撐"),
                 dict(name="警訊", val="利多開高走低", thr=None, cls="r", look="好消息當天收盤品質 ≤ 0.4"),
             ],
             verdict="⚠️ <b>≥0 只表示不弱於基準，不等於個股收平或上漲。</b>必須另看絕對報酬"
                     "（combo 31 與 32 的差別就在這裡）。事件分類、公開時間、基準與反應窗都要事先定義，"
                     "不能先看漲跌才挑事件。",
             why="消息本身不重要，市場對消息的反應才重要。壞消息出來股價不跌，代表想賣的人早賣完了。"
                 "好消息出來開高走低，代表有人趁大家興奮時把貨倒出去。",
             cost="只看新聞內容做決定，會系統性地買在利多的最高點。",
             signals=[("g", "↑ 偏多", "利空日不跌，或利多後續三天仍守住"),
                      ("o", "→ 中性", "反應跟大盤差不多"),
                      ("r", "↓ 偏空", "利多開高走低（尤其爆量），或利空跌幅大於大盤")]),

        dict(no=11, cid="C10", title="事後認可", q="進場後前三天該不該砍？（出場課，進場時不可用）",
             field="mfe_mae_ratio_3d", fieldzh="前三日盈虧比", calfield=None,
             how="進場後 3 日最大浮盈 ÷ 最大回撤",
             engine="⛔ <b>T 日一律清空。</b>引擎目前只算 c10_mae3 / c10_mfe3 兩個分量，"
                    "<b>沒有算比值本身</b>（原手冊也沒給比值的分級線）。看盤卡在 T 日把兩欄設為缺值，防前視偏誤。"
                    "只有回頭做歷史統計時才會有值。",
             levels=[
                 dict(name="很好", val="次日站穩突破價之上", thr=None, cls="g", look="市場認可這個新價格"),
                 dict(name="及格", val="回檔但守住突破價", thr=None, cls="n", look="正常換手"),
                 dict(name="不及格", val="隔天跌回突破線下", thr=None, cls="r", look="市場否決了這次突破"),
             ],
             companion=[("配套", "前 3 日回撤 ≤ 4%", "—", "⚠️ 4% 是 MAE 幅度的教學參考，不是 ratio 門檻，也不是停損指令")],
             verdict="⚠️ <b>這一課含有進場當下不存在的資訊。</b>用它做進場決策等於偷看未來"
                     "（combo 60 就是抓這件事的守門員）。",
             why="突破是一個提議：「這檔股票應該值更多錢」。接下來幾天是市場投票。"
                 "如果隔天就跌回突破價下方，代表提議被否決——而你買在了那個壓力區裡面。",
             cost="死抱著已經跌回箱體內的突破、等它慢慢跌到停損，是把一次小損失拖成大損失。",
             signals=[("g", "↑ 續抱", "次日站穩，且缺口沒被補"),
                      ("o", "→ 觀察", "在突破價附近震盪，給它三天"),
                      ("r", "↓ 出場", "跌回突破價下方並收盤")]),

        dict(no=12, cid="C12", title="動能老化", q="什麼時候該走？（出場課，進場時不可用）",
             field="efficiency_decay", fieldzh="效率衰退", calfield="c12_days_since_high",
             how="突破後前 5 日效率比 − 之後 10 日效率比",
             engine="主欄 c12_decay <b>可用</b>（窗未完成時為缺值）。"
                    "⚠️ 但只有 1 組 combo（46）用到主欄，其餘 7 組用的是<b>配套的「距最近新高日數」</b>。"
                    "手冊明說創高間隔是<b>診斷欄位，不等於 efficiency_decay</b>，兩者不可互換。",
             levels=[
                 dict(name="健康", val="創高間隔 < 15 天", thr=15, cls="g", look="還在持續推進"),
                 dict(name="警戒", val="15 ~ 30 天", thr=30, cls="o", look="動能開始鈍化"),
                 dict(name="老化", val="> 30 天", thr=None, cls="r", look="上漲越來越吃力"),
             ],
             companion=[("配套", "高點是否持續擴張", "—", "引擎用 higher_highs 旗標"),
                        ("主欄 c12_decay", "> 0", "—", "後段效率低於前段（combo 46）")],
             verdict="15 天落在 P28、30 天落在 P38。<b>兩條線都偏鬆</b>——超過六成的交易日"
                     "距上次新高已超過 30 天，因為大部分股票大部分時間根本不在創高。"
                     "這一課要跟「目前是否處於上升趨勢」一起看，單獨用會誤判。",
             why="上升趨勢的健康程度可以用「創新高的頻率」來量。當創高間隔越拉越長、每次推進幅度越來越小，"
                 "代表接手的買盤在變少——不是崩盤，是燃料在耗盡。",
             cost="動能股的下跌通常很快。等到 K 線走壞才出場，往往已經回吐一大段。",
             signals=[("g", "↑ 續抱", "創高間隔穩定 < 15 天，高點持續擴張"),
                      ("o", "→ 減碼", "間隔拉長到 15 ~ 30 天"),
                      ("r", "↓ 出場", "間隔 > 30 天，或跌破 MA20 並收盤")]),
    ]

    # ---- 附錄一：11 大類個性
    CLS = [
        ("A", "突破前", "T 前", "醞釀期。回答「這個盤整在蓄力還是在爛掉」", "C01 C02 C05配套"),
        ("B", "突破當日", "T 收盤", "判定這根突破的品質。只描述今天，不預測明天", "C03 C05配套 C01"),
        ("C", "趨勢背景", "T 收盤", "位置與路徑。兩者都不能單獨預告頂部", "C08 C07 C06"),
        ("D", "相對強弱", "T 收盤", "扣掉大盤看它自己。指數缺值時整類靜默失效", "C06"),
        ("E", "結構防守", "T 前或 T", "破了沒、收回沒。決定「拿不拿得住」", "C04 C11"),
        ("F", "事件反應", "事件後", "⛔ 需 event log，目前幾乎全啞", "C09"),
        ("G", "突破後接受", "T+1 起", "市場認不認這個新價格", "C10 C03"),
        ("H", "突破後動能", "T 後", "燃料還剩多少", "C12配套 C06"),
        ("I", "基本面交叉", "公告後", "⛔ 需公告原文，目前全啞", "C03 C05配套"),
        ("J", "矛盾與品質", "任何時點", "驗證器不是型態。平常 0 次命中是正常的，不進報酬統計", "全部"),
        ("K", "資料長出來的", "任何時點", "手冊沒有、殘差分群補的。優先序排最後", "C12配套 C03 C11"),
    ]
    byc = {}
    for c in reg["combos"]:
        byc.setdefault(c["cls"], []).append(c)
    ap_ = ['<div class="page"><div class="bar"><span class="t">附錄一　11 個大類的個性</span>'
           '<span class="q">先認大類，再 map 到 combo</span></div>',
           '<table><tr><th style="width:5%">類</th><th style="width:13%">名稱</th>'
           '<th style="width:9%">幾組</th><th style="width:11%">時點</th>'
           '<th>個性</th><th style="width:17%">主要看哪幾課</th></tr>']
    for cid, nm, when, per, fields in CLS:
        g = byc.get(cid, [])
        rng = f'{g[0]["id"]}–{g[-1]["id"]}' if g else "—"
        ap_.append(f'<tr><td class="lab" style="text-align:center">{cid}</td><td><b>{nm}</b></td>'
                   f'<td>{len(g)} 組<br><span style="font-size:8pt;color:#666">{rng}</span></td>'
                   f'<td>{when}</td><td>{per}</td><td style="font-size:8.5pt">{fields}</td></tr>')
    ap_.append("</table>")

    dirs = {"多": [], "空": [], "中性": [], "驗證": []}
    for c in reg["combos"]:
        dirs.setdefault(c.get("dir", "中性"), []).append(c["id"])
    ap_.append('<div class="bar" style="margin-top:5mm"><span class="t">方向速查</span>'
               '<span class="q">方向不是機率</span></div>')
    ap_.append('<table><tr><th style="width:14%">方向</th><th style="width:9%">組數</th><th>combo</th></tr>')
    for k, cls in [("多", "g"), ("空", "r"), ("中性", "o"), ("驗證", "n")]:
        v = dirs.get(k, [])
        ap_.append(f'<tr><td class="{cls}">{k}</td><td>{len(v)}</td>'
                   f'<td style="font-size:9pt">{" ".join(sorted(v))}</td></tr>')
    ap_.append("</table>")
    ap_.append('<div class="note"><b>「偏多」只表示現有結構較支持向上情境，不表示上漲機率大於五成。</b>'
               '偏空同理。橫盤與資料不足是兩種不同狀態。<br>'
               '<b>同時命中多空兩邊時要並列保留，不做多數決。</b>'
               '正確寫法是「向上結構仍在，但位置高且推進受阻，延續待確認」，'
               '不是「4 項偏多 2 項偏空所以 78 分」。</div>')
    ap_.append('<div class="note">主狀態的挑選順序（引擎已實作）：'
               '<b>J 可用性 → E 結構位置 → G 突破後 → B 當日 → A 背景 → C 趨勢 → '
               'D 相對 → H 動能 → F 事件 → I 基本面 → K 補位</b>。'
               '其餘命中的存為 secondary，是補充不是加權。</div></div>')

    # ---- 附錄二：65 組索引（combo → 類 → 方向 → 主要看哪幾課）
    C_ORDER = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
               "C09", "C10", "C11", "C12"]
    rev = {}
    for c in reg["combos"]:
        seen = []
        for e in c["when"]:
            for n in ast.walk(ast.parse(e, mode="eval")):
                if isinstance(n, ast.Name) and n.id in FIELD2C:
                    tag = FIELD2C[n.id]
                    base = tag.replace("+", "").replace("ind", "")
                    lab = base + ("+" if tag.endswith("+") else "")
                    if lab not in seen:
                        seen.append(lab)
        rev[c["id"]] = sorted(seen, key=lambda x: (C_ORDER.index(x[:3]), x))
    DIRCLS = {"多": "g", "空": "r", "中性": "o", "驗證": "n"}
    idx = ['<div class="page"><div class="bar"><span class="t">附錄二　65 組索引</span>'
           '<span class="q">看到某組，就知道回哪一課</span></div>',
           '<div class="note">「主要看哪幾課」加號代表配套欄（例如 C05+ = 量比／量縮比，'
           '因為 C05 主欄停用）。空白代表該組不依賴 12 課的數值欄位。</div>',
           '<table><tr><th style="width:7%">#</th><th style="width:5%">類</th>'
           '<th style="width:30%">名稱</th><th style="width:9%">方向</th>'
           '<th>主要看哪幾課</th></tr>']
    for c in reg["combos"]:
        d = c.get("dir", "中性")
        idx.append(f'<tr><td><b>{c["id"]}</b></td><td style="text-align:center">{c["cls"]}</td>'
                   f'<td>{c["name"]}</td><td class="{DIRCLS.get(d, "n")}">{d}</td>'
                   f'<td style="font-size:8.5pt">{" ".join(rev[c["id"]]) or "—"}</td></tr>')
        if c["id"] in ("06", "12", "18", "24", "30", "36", "42", "48", "54", "60"):
            idx.append('</table><table><tr><th style="width:7%">#</th><th style="width:5%">類</th>'
                       '<th style="width:30%">名稱</th><th style="width:9%">方向</th>'
                       '<th>主要看哪幾課</th></tr>')
    idx.append("</table></div>")

    cov = reg.get("calibration", {})
    parts = [f'<div class="page"><h1 class="cover">老手看盤 · 詞彙速查手冊</h1>'
             f'<p class="sub" style="font-size:14pt;color:#12314f;font-weight:700">引擎版</p>'
             f'<p class="sub">沿用原手冊版面，一課一頁。每個門檻旁邊加上<b>台股全市場實際百分位</b>，'
             f'每課末列出<b>哪幾組 combo 用到它</b>。</p>'
             f'<p class="sub">校準樣本：{cov.get("sample_start", "?")} 起、20 日均額 ≥ 5,000 萬、'
             f'759,704 檔-日　｜　校準日期 {cov.get("built_at", "?")}</p>'
             f'<p class="sub">原手冊第 17 頁說這些數字未經統計驗證。現在驗過了——'
             f'<b>22 條合理、5 條壞掉、3 課停用</b>。</p>'
             f'<div class="note"><b>缺值不是偏空，是「不知道」。</b>引擎遇到缺值標 PARTIAL，'
             f'不算符合也不算不符，更不會當成 0。</div>'
             f'<div class="note">門檻與定義以 combo_registry.json 為準。本文件若與 registry 不符，'
             f'以 registry 為準。</div></div>']
    for L in LESSONS:
        parts.append(lesson_html(L, cal, use))
    parts.append("".join(ap_))
    parts.append("".join(idx))
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="combo_registry.json")
    ap.add_argument("-o", "--out", default="老手速查_引擎版.pdf")
    args = ap.parse_args()
    reg = json.load(open(args.registry, encoding="utf-8"))
    html = build(reg)
    from weasyprint import HTML
    HTML(string=html).write_pdf(args.out)
    print(f"寫入 {args.out}　({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
