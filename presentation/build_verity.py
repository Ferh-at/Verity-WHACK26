"""Builds presentation/Verity.pptx from the eight-slide talk. Run: python presentation/build_verity.py"""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("Verity.pptx")
FONT = "Segoe UI"
TEAM = "verity    Ferhat Celik - Reyansh Mathur - Efe Ulker"

INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x6B, 0x72, 0x80)
RULE = RGBColor(0xD1, 0xD5, 0xDB)
AI = RGBColor(0x3B, 0x5B, 0xDB)
AI_BG = RGBColor(0xE8, 0xED, 0xFC)
ENG = RGBColor(0x0F, 0x76, 0x6E)
ENG_BG = RGBColor(0xE1, 0xF3, 0xF0)
ENG_MID = RGBColor(0x5E, 0xB3, 0xA8)
SEALED = RGBColor(0x1F, 0x29, 0x37)
CODE_BG = RGBColor(0xF1, 0xF2, 0xF4)
LUCK = RGBColor(0xC2, 0x41, 0x0C)
GREY_BAR = RGBColor(0x9C, 0xA3, 0xAF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
REJECTED = RGBColor(0xB9, 0x1C, 0x1C)
FRAGILE = RGBColor(0xB4, 0x53, 0x09)
PROMISING = RGBColor(0x1D, 0x4E, 0xD8)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, body, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, gap=5):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = body if isinstance(body, list) else [body]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(gap)
        for chunk, style in (line if isinstance(line, list) else [(line, {})]):
            r = p.add_run()
            r.text = chunk
            r.font.name = FONT
            r.font.size = Pt(style.get("size", size))
            r.font.bold = style.get("bold", bold)
            r.font.color.rgb = style.get("color", color)
    return tb


def band(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.1
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def connector(slide, x1, y1, x2, y2, color=MUTED, width=1.75, head=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def title(slide, kicker, heading):
    text(slide, 0.7, 0.36, 11.9, 0.28, kicker.upper(), 12, True, MUTED)
    text(slide, 0.7, 0.64, 11.9, 0.52, heading, 28, True)


def footer(slide, n):
    text(slide, 0.7, 7.15, 9.5, 0.24, TEAM, 11, color=MUTED)
    if n:
        text(slide, 12.15, 7.15, 0.5, 0.24, str(n), 11, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def node(slide, x, y, w, h, head, sub, ink, edge):
    band(slide, x, y, w, h, WHITE, edge, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, x + 0.14, y + 0.1, w - 0.28, h - 0.16,
         [[(head, {"bold": True, "size": 15, "color": ink})], [(sub, {"size": 12, "color": INK})]], gap=2)


# 1
s = prs.slides.add_slide(BLANK)
text(s, 0.8, 1.55, 11.5, 1.55, ["Finding patterns in data is easy.", "Knowing which ones to trust is hard."], 36, True)
connector(s, 0.8, 3.7, 2.3, 3.7, ENG, 3)
text(s, 0.8, 3.95, 11.5, 0.75, "Verity", 48, True, ENG)
text(s, 0.8, 4.8, 11, 0.5, "An AI researcher that tests its own ideas and reports which ones survive.", 20, color=MUTED)
text(s, 0.8, 6.55, 8, 0.28, "WHACK 26  ·  Brevan Howard track", 13, color=MUTED)
footer(s, None)
notes(s, "Finding patterns in market data is easy. Knowing which ones to trust is hard. "
         "Test enough ideas and some will look brilliant by luck. Verity tests its own ideas and reports which ones survive.")

# 2
s = prs.slides.add_slide(BLANK)
title(s, "How it works", "Who decides, and what crosses the line")
text(s, 0.7, 1.22, 12, 0.36, [[("The statistics engine is deterministic. ", {"bold": True, "color": ENG}),
                              ("Same data and same rules in, same verdict out. Fixed seeds. No model inside a calculation.",
                               {})]], 15)

band(s, 2.05, 1.72, 10.55, 1.28, AI_BG)
band(s, 2.05, 3.14, 10.55, 1.28, ENG_BG)
text(s, 0.7, 1.72, 1.25, 1.28, [[("AI", {"bold": True, "size": 14, "color": AI})],
                                [("proposes and reviews", {"size": 11, "color": MUTED})]],
     anchor=MSO_ANCHOR.MIDDLE, gap=2)
text(s, 0.7, 3.14, 1.25, 1.28, [[("Engine", {"bold": True, "size": 14, "color": ENG})],
                                [("tests and decides", {"size": 11, "color": MUTED})]],
     anchor=MSO_ANCHOR.MIDDLE, gap=2)
node(s, 2.25, 1.9, 2.7, 0.92, "Generator", "Writes about 20 rules", AI, AI)
node(s, 6.55, 1.9, 2.85, 0.92, "Reviewer", "Reads the discovery sheet", AI, AI)
node(s, 2.25, 3.32, 2.7, 0.92, "Discovery tests", "2003 to 2014, every round", ENG, ENG)
node(s, 6.55, 3.32, 2.85, 0.92, "Later data", "Runs only after review stops", ENG, ENG)
node(s, 10.0, 3.32, 2.35, 0.92, "Verdict", "One of five labels", ENG, ENG)

connector(s, 3.6, 2.82, 3.6, 3.32, MUTED, 1.75, head=True)
text(s, 3.75, 2.9, 1.5, 0.28, "rules", 11, True, MUTED)
connector(s, 4.95, 3.55, 6.55, 2.36, AI, 1.75, head=True)
text(s, 5.15, 2.55, 1.35, 0.4, "2003–2014 sheet", 11, True, AI)
connector(s, 7.9, 2.82, 7.9, 3.32, MUTED, 1.75, head=True)
text(s, 8.05, 2.88, 1.55, 0.28, "names", 11, True, MUTED)
connector(s, 9.4, 3.78, 10.0, 3.78, MUTED, 1.75, head=True)
connector(s, 6.55, 4.28, 4.95, 4.28, ENG, 1.5, head=True)
text(s, 4.9, 4.32, 1.7, 0.24, "another round", 11, True, ENG)

text(s, 0.7, 4.68, 5.85, 2.25, [
    [("AI to engine", {"bold": True, "size": 16})],
    [("Only structured rules cross. Six fields: feature, series, lookback, transform, asset, direction. "
      "The sentence of reasoning stays on screen and is never tested.", {"size": 14, "color": MUTED})],
    [("A request for another test names the rule and the test. The model does not hand over a p-value, "
      "a Sharpe ratio, or a verdict.", {"size": 14, "color": MUTED})],
], gap=6)
text(s, 6.85, 4.68, 5.8, 2.25, [
    [("Engine to AI", {"bold": True, "size": 16})],
    [("After each discovery round, one row per rule: status, the tests it failed, its t-statistic, "
      "its bootstrap p-value, and the luck bar for the whole search.", {"size": 14, "color": MUTED})],
    [("Nothing from 2015 onward is in that sheet. Prices stay on the engine side. The reply is another "
      "test, new rules, or names for the sealed period.", {"size": 14, "color": MUTED})],
], gap=6)
footer(s, 2)
notes(s, "Two sides. The AI proposes and reviews. The engine tests and decides, and it is deterministic: "
         "same data, same rules, same verdict, because the random seeds are fixed and the model never enters a calculation.\n\n"
         "What the AI sends: rules in six fields. The written reason is not tested. It cannot send a number and have that number become the result.\n\n"
         "What the engine sends back: the 2003 to 2014 sheet only. Status, failed tests, t-statistic, p-value, luck bar. "
         "The AI then asks for another test, sends new rules, or names finalists. Validation and the holdout run after it stops.")

# 3
s = prs.slides.add_slide(BLANK)
title(s, "The rule", "One belief, one fixed form")
text(s, 0.7, 1.28, 12, 0.4, "Every idea the model writes has to be exactly these six fields. Anything else is dropped before a test runs.",
     16, color=MUTED)
parts = [("Feature", "momentum, level,\nchange, z-score, correlation"),
         ("Series", "16 market and\nmacro series"),
         ("Lookback", "1, 3, 6,\n12, 24, 36 months"),
         ("Transform", "sign, or a\nclipped z-score"),
         ("Asset", "S&P, Nasdaq, bonds,\ngold, dollar, bitcoin"),
         ("Direction", "long or short\nwhen the signal is high")]
pw, gap, py = 1.85, 0.18, 1.82
for i, (name, allowed) in enumerate(parts):
    x = 0.7 + i * (pw + gap)
    band(s, x, py, pw, 0.48, AI, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, py, pw, 0.48, name, 15, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, x + 0.06, py + 0.54, pw - 0.12, 0.7, allowed, 12, color=MUTED, align=PP_ALIGN.CENTER)
    if i < 5:
        text(s, x + pw - 0.02, py, gap + 0.04, 0.48, "×", 16, True, MUTED, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

text(s, 0.7, 3.55, 12, 0.3, "A RULE FROM A LIVE RUN", 12, True, ENG)
values = ["momentum", "Bitcoin", "1 month", "sign", "US dollar", "+1"]
for i, value in enumerate(values):
    x = 0.7 + i * (pw + gap)
    band(s, x, 3.9, pw, 0.5, ENG_BG, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.06, 3.9, pw - 0.12, 0.5, value, 14, True, INK, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
text(s, 0.7, 4.52, 12, 0.36, "Positive Bitcoin month, long the dollar next month. Negative month, short it.", 16, True)

band(s, 0.7, 5.1, 11.95, 1.85, CODE_BG, None, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 0.95, 5.28, 11.45, 1.55, [
    [("Position at month-end t", {"bold": True, "size": 13, "color": MUTED})],
    [("p(t)  =  direction  ×  transform( feature(t) )", {"bold": True, "size": 20}),
     ("      between −1 and +1", {"size": 14, "color": MUTED})],
    [("Return earned over the next month", {"bold": True, "size": 13, "color": MUTED})],
    [("net(t+1)  =  p(t) × r(t+1)  −  0.0002 × |p(t) − p(t−1)|", {"bold": True, "size": 20})],
], gap=4)
footer(s, 3)
notes(s, "The model can only submit rules in these six fields. This one came from a live run on the thesis that "
         "a rise in Bitcoin lifts the dollar. Feature is one-month Bitcoin momentum, transform is the sign, "
         "the asset is the dollar index, direction is long when momentum is positive.\n\n"
         "The position is set at month-end from data known that day, and it earns next month's dollar return "
         "minus 2 basis points per unit traded. The model's sentence of reasoning is shown and never tested.")

# 4
s = prs.slides.add_slide(BLANK)
title(s, "The data", "Three periods, three different jobs")
x0, per, yb = 0.7, 11.95 / 23.83, 1.4
for years, name, fill in ((12.0, "Discovery", ENG), (5.0, "Validation", ENG_MID), (6.83, "Holdout", SEALED)):
    w = years * per
    band(s, x0, yb, w - 0.05, 0.52, fill)
    text(s, x0, yb, w - 0.05, 0.52, name, 15, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    x0 += w
x0 = 0.7
for year, pos, align in (("2003", 0, PP_ALIGN.LEFT), ("2015", 12, PP_ALIGN.LEFT), ("2020", 17, PP_ALIGN.LEFT)):
    text(s, 0.7 + pos * per, 1.12, 1.1, 0.26, year, 12, True, MUTED, align)
text(s, 0.7 + 11.95 - 0.8, 1.12, 0.8, 0.26, "2026", 12, True, MUTED, PP_ALIGN.RIGHT)

cards = [
    (ENG, "Discovery  ·  2003 to 2014",
     "The only years a rule is allowed to be built on.",
     ["Every round of testing happens here: leak, significance, flukes, the luck bar.",
      "The reviewer receives this period and nothing later.",
      "A result that exists only in this window is not a finding."]),
    (ENG, "Validation  ·  2015 to 2019",
     "A repeat check on years the model never saw.",
     ["It runs once, after the reviewer has stopped, on every rule.",
      "Same direction of profit, and a t-statistic above 1.",
      "Fail here and the label is Fragile. The rule is not rewritten."]),
    (SEALED, "Holdout  ·  2020 to 2026",
     "Closed for the whole search. One look, then shut.",
     ["Only rules the reviewer names can be opened, at most three.",
      "The look is logged before the return is computed.",
      "The same rule cannot be opened a second time."]),
]
for i, (color, head, lead, body) in enumerate(cards):
    x = 0.7 + i * 4.05
    band(s, x, 2.15, 3.9, 4.75, WHITE, RULE)
    band(s, x, 2.15, 3.9, 0.08, color)
    text(s, x + 0.22, 2.4, 3.46, 1.15, [[(head, {"bold": True, "size": 16, "color": color if color != ENG_MID else ENG})],
                                       [(lead, {"bold": True, "size": 14})]], gap=6)
    text(s, x + 0.22, 3.7, 3.46, 2.9, body, 14, color=MUTED, gap=10)
footer(s, 4)
notes(s, "Three jobs, not three copies of the same test. Discovery, 2003 to 2014, is the only window the rules "
         "are built on and the only window the reviewer sees. Validation, 2015 to 2019, runs once after the "
         "reviewer stops: same direction, t-statistic above 1, or the rule is Fragile. The holdout from 2020 "
         "stays sealed. At most three named rules get one look, logged before the number is computed, and that "
         "rule cannot be opened again.")

# 5
s = prs.slides.add_slide(BLANK)
title(s, "Discovery tests", "Four checks, in this order")
checks = [
    ("1", "Edit the future", "Rejected", REJECTED,
     "At 24 dates, scale everything after that date by about 5% and recompute.",
     "Fail if the signal on any earlier month moves by more than one part in a billion.",
     "A centred average uses next month, so it moves. All 500 planted leaks failed here."),
    ("2", "Significance", "Rejected, or Can’t tell yet", REJECTED,
     "Mean net return must be positive.",
     "p  =  (1 + number of resamples with t* ≥ t) / 1,001",
     "t is the Newey–West t-statistic. 1,000 resamples keep months in blocks. Need p < 0.05."),
    ("3", "Fluke check", "Rejected", REJECTED,
     "Drop the biggest 1% of months. Then drop each calendar year in turn.",
     "Every version still needs a positive mean and p < 0.05.",
     "A rule that made its money in 2008 fails as soon as 2008 is removed."),
    ("4", "Luck hurdle", "Fragile", FRAGILE,
     "Shift the rules out of time, 200 times. Each time, keep the best t in the whole search.",
     "Bar  =  95th percentile of those 200 best scores.  Need  t  >  bar.",
     "Then a Benjamini–Yekutieli q-value under 0.10, because the rules are correlated."),
]
for i, (n, name, fails, color, a, b, c) in enumerate(checks):
    x = 0.7 + (i % 2) * 6.3
    y = 1.32 + (i // 2) * 2.8
    band(s, x, y, 6.1, 2.65, WHITE, RULE)
    band(s, x + 0.18, y + 0.18, 0.38, 0.38, ENG if n != "4" else LUCK, None, MSO_SHAPE.OVAL)
    text(s, x + 0.18, y + 0.18, 0.38, 0.38, n, 14, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, x + 0.68, y + 0.18, 2.5, 0.38, name, 16, True, INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 3.15, y + 0.2, 2.75, 0.34, fails, 12, True, color, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
    text(s, x + 0.2, y + 0.7, 5.7, 1.8, [[(a, {"size": 13})], [(b, {"bold": True, "size": 13})],
                                         [(c, {"size": 13, "color": MUTED})]], gap=4)
footer(s, 5)
notes(s, "Four checks, in order, all on 2003 to 2014. First, edit the future: if scaling later data moves an "
         "earlier signal, the rule is rejected. That caught every one of 500 planted leaks. Second, significance: "
         "positive average net return, and a block-bootstrap p-value under 0.05, where p is one plus the number "
         "of resamples that beat the real t-statistic, over 1,001. Third, the fluke check: still true with the "
         "biggest months removed and with each year removed. Fourth, the luck hurdle: 200 scrambled replays of "
         "the whole search, and the rule must beat the 95th percentile of the best score luck found, then pass "
         "a Benjamini–Yekutieli q-value under 0.10.")

# 6
s = prs.slides.add_slide(BLANK)
title(s, "The reviewer", "It runs by itself, at the end of every round")
text(s, 0.7, 1.26, 12, 0.55, "Nobody presses a button. When discovery finishes, the sheet goes to the model and it has to answer. "
                             "The answer is one of these three, chosen from what the sheet shows.", 16)
moves = [
    ("Challenge", "A rule is still alive, and the result looks too neat.",
     "The model points at that rule and sends it back for one more test. The engine runs the test. "
     "This move can remove a rule. It cannot promote one."),
    ("Refine", "Nothing has survived, or a failure suggests a different rule.",
     "The model writes new rules in the same six fields and hands them back. They join the same search, "
     "so the luck bar is rebuilt for every rule, including the old ones."),
    ("Nominate", "A rule has survived, and the model is ready to stop.",
     "It names those rules. Only then does the engine leave 2003–2014, run 2015–2019, and open 2020 "
     "for the names."),
]
for i, (name, when, how) in enumerate(moves):
    x = 0.7 + i * 4.15
    band(s, x, 2.05, 4.0, 4.0, AI_BG, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.25, 2.25, 3.5, 0.45, name, 22, True, AI)
    text(s, x + 0.25, 2.8, 3.5, 0.85, when, 15, True)
    text(s, x + 0.25, 3.75, 3.5, 2.0, how, 14, color=INK)
text(s, 0.7, 6.25, 12, 0.7, "Up to three rounds. On the last one it can nominate or stop, and it cannot add rules.",
     15, True, MUTED)
footer(s, 6)
notes(s, "The reviewer is called automatically when a discovery round ends. Challenge: a rule is still standing "
         "but the result looks too neat, so it goes back for one more test, which can only hurt it. "
         "Refine: nothing survived, or a failure points somewhere else, so it writes new rules and they re-enter "
         "the same search. Nominate: something survived and it wants to stop, so it names those rules and only "
         "then does 2015 onward get touched. Three rounds maximum.")

# 7
s = prs.slides.add_slide(BLANK)
title(s, "The verdict", "Five labels, then what earns each one")
labels = [("Rejected", "Failed a test, or no effect in a long enough sample.", REJECTED),
          ("Can’t tell yet", "No effect, and too little data to see one.", MUTED),
          ("Fragile", "Looked real, then failed luck or a later period.", FRAGILE),
          ("Promising", "Cleared 2019. The sealed years were not confirmed.", PROMISING),
          ("Replicated", "Named, and the one look after 2020 passed.", ENG)]
for i, (name, meaning, color) in enumerate(labels):
    x = 0.7 + i * 2.5
    text(s, x, 1.28, 2.35, 0.3, name, 13, True, color)
    text(s, x, 1.58, 2.35, 0.85, meaning, 11, color=INK)
rows = [
    ("Future edit moved the signal", "Discovery", "Rejected", REJECTED),
    ("Flat for 12 years", "Discovery", "Rejected", REJECTED),
    ("Flat, on a short history", "Discovery", "Can’t tell yet", MUTED),
    ("Lost to scrambled luck", "Discovery", "Fragile", FRAGILE),
    ("Died after 2015", "Validation", "Fragile", FRAGILE),
    ("Passed. Holdout never opened", "Validation", "Promising", PROMISING),
    ("Held after 2020", "Holdout", "Replicated", ENG),
]
text(s, 0.7, 2.55, 6.2, 0.28, "FOR EXAMPLE", 11, True, MUTED)
text(s, 7.3, 2.55, 2.6, 0.28, "PERIOD", 11, True, MUTED)
text(s, 10.2, 2.55, 2.3, 0.28, "LABEL", 11, True, MUTED)
for i, (what, where, label, color) in enumerate(rows):
    y = 2.9 + i * 0.55
    if i:
        connector(s, 0.7, y, 12.55, y, RULE, 0.75)
    text(s, 0.7, y, 6.3, 0.52, what, 15, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.3, y, 2.6, 0.52, where, 15, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.2, y, 2.3, 0.52, label, 15, True, color, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 7)
notes(s, "Five labels, applied by code in a fixed order, so the same evidence always gets the same label. "
         "Rejected: a test failed, or there was no effect in a sample long enough to show one. "
         "Can't tell yet: no effect, and too little data. Fragile: it looked real, then lost to luck or to "
         "2015–2019. Promising: it cleared 2019 and the sealed years were not confirmed. "
         "Replicated: it was named and the one look after 2020 passed.\n\n"
         "The rows are examples of what earns each label. A failed holdout stays Promising. It does not get rewritten as Rejected.")

# 8
s = prs.slides.add_slide(BLANK)
title(s, "The proof", "Scored where the answer was planted")
text(s, 0.7, 1.24, 6.3, 0.7, "On real markets the truth is unknown, so a mistake cannot be counted. "
                            "Here the real rules and the fakes were planted first.", 14, color=MUTED)
data = CategoryChartData()
data.categories = ["Naive t-test", "Verity"]
data.add_series("Share of trusted rules that were fake", (0.7813, 0.0033))
chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.55), Inches(1.95), Inches(6.5), Inches(2.55), data).chart
chart.has_legend = False
chart.has_title = False
axis = chart.value_axis
axis.maximum_scale, axis.minimum_scale = 1.0, 0.0
axis.has_major_gridlines = False
axis.visible = False
chart.category_axis.format.line.color.rgb = RULE
chart.category_axis.tick_labels.font.size = Pt(12)
chart.category_axis.tick_labels.font.bold = True
chart.category_axis.tick_labels.font.name = FONT
plot = chart.plots[0]
plot.gap_width = 80
plot.has_data_labels = True
labels = plot.data_labels
labels.number_format, labels.number_format_is_linked = "0.00%", False
labels.position = XL_LABEL_POSITION.OUTSIDE_END
labels.font.size, labels.font.bold, labels.font.name = Pt(16), True, FONT
for point, color in zip(plot.series[0].points, (GREY_BAR, ENG)):
    point.format.fill.solid()
    point.format.fill.fore_color.rgb = color
text(s, 0.7, 4.5, 6.2, 0.4, "Trusted rules that were fake. Mean of 50 datasets, 10,000 rules.", 12, color=MUTED)

text(s, 7.3, 1.24, 5.3, 0.32, "How the number was made", 15, True)
steps = ["50 synthetic markets, 300 months, split the same way as the real timeline.",
         "200 rules in each: 10 real, at Sharpe 0.5 to 2.0, and 190 fakes.",
         "The fakes include noise, copies, leaks, one-month spikes, and hidden market exposure.",
         "Same engine code. No model in the loop. Compared with a t-test at p < 0.05."]
for i, line_ in enumerate(steps):
    y = 1.64 + i * 0.58
    band(s, 7.3, y, 0.32, 0.32, ENG, None, MSO_SHAPE.OVAL)
    text(s, 7.3, y, 0.32, 0.32, str(i + 1), 11, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, 7.75, y - 0.04, 4.85, 0.52, line_, 13, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.3, 4.05, 5.3, 0.85, [[("97%", {"bold": True, "size": 18, "color": ENG}),
                               (" of Sharpe 2.0 signals found.   ", {"size": 13, "color": MUTED}),
                               ("0%", {"bold": True, "size": 18, "color": LUCK}),
                               (" of Sharpe 0.5.", {"size": 13, "color": MUTED})]], gap=2)

band(s, 0.7, 5.05, 11.95, 1.95, CODE_BG, None, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 0.95, 5.18, 11.5, 0.3, "WHERE THIS GOES WRONG", 12, True, LUCK)
limits = [
    ("The model is not blind.", " Its training includes 2003–2014, so a rule can be remembered. Only months after June 2024 are past its cutoff."),
    ("This chart is not the live loop.", " These 10,000 rules never met the reviewer. The factor check was on here and is off in the app."),
    ("Saying no has a cost.", " No Sharpe 0.5 signal was found. About 8% of real signals were killed by a robustness check. One fake in 201 still passed."),
]
text(s, 0.95, 5.5, 11.5, 1.4,
     [[(head, {"bold": True, "size": 13}), (body, {"size": 13})] for head, body in limits], gap=3)
footer(s, 8)
notes(s, "Synthetic data because on real markets you cannot count mistakes. Fifty datasets, ten real and 190 fake "
         "rules each, ten thousand in total, through the engine with no model. A naive t-test trusted fakes "
         "78 percent of the time. Verity, 0.33 percent on average. It found 97 percent of the Sharpe 2 signals "
         "and none of the Sharpe 0.5 ones.\n\n"
         "Three limits to say out loud if asked. The model has read 2003 to 2014, so generation is not blind; "
         "only post-June 2024 is past the cutoff we set. The benchmark is the engine alone, with the factor "
         "check on, and the app runs the reviewer with that check off. And the cost of saying no: weak real "
         "signals are missed, about 8 percent of real signals were rejected by a robustness check, and one of "
         "201 trusted rules was still a fake. Pooled across seeds that one fake is 0.50 percent, against the "
         "0.33 percent average of the fifty datasets.\n\n"
         "If asked about the regime check: it was built to catch rules that only worked early, and it almost "
         "never fired. Other checks rejected those rules. None of them were trusted.")

for slide in prs.slides:
    pass

prs.save(OUT)
print(OUT)
