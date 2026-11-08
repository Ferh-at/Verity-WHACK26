"""Build the three-minute deck. Run: python presentation/build_deck.py."""
import argparse
import json
import math
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("Discovery_Lab.pptx")
MOCKS = OUT.parent.parent / "mocks"
FONT = "Segoe UI"

INK = RGBColor.from_string("111827")
MUTED = RGBColor.from_string("526071")
RULE = RGBColor.from_string("D1D5DB")
AI = RGBColor.from_string("3B5BDB")
AI_BG = RGBColor.from_string("E8EDFC")
ENG = RGBColor.from_string("0F766E")
ENG_BG = RGBColor.from_string("E1F3F0")
CODE_BG = RGBColor.from_string("F1F2F4")
LUCK = RGBColor.from_string("C2410C")
GREY_BAR = RGBColor.from_string("9CA3AF")
REJECTED = RGBColor.from_string("B91C1C")
WHITE = RGBColor.from_string("FFFFFF")


def text(slide, x, y, w, h, body, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, gap=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, value in enumerate(body if isinstance(body, list) else [body]):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if i:
            paragraph.space_before = Pt(gap)
        run = paragraph.add_run()
        run.text = value
        run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
        run.font.color.rgb = color
    return box


def band(slide, x, y, w, h, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False


def line(slide, x1, y1, x2, y2, color=MUTED, width=1.5, head=False):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(width)
    if head:
        xml = connector.line._get_or_add_ln()
        xml.append(xml.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def title(slide, kicker, heading):
    text(slide, 0.8, 0.49, 11.7, 0.31, kicker.upper(), 12, True, MUTED)
    text(slide, 0.8, 0.86, 11.75, 0.73, heading, 32, True)


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def column_chart(slide, x, y, w, h, categories, label, values, colors, gap_width):
    data = CategoryChartData()
    data.categories = categories
    data.add_series(label, values)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data
    ).chart
    chart.has_legend = False
    chart.has_title = False
    chart.font.name, chart.font.size, chart.font.color.rgb = FONT, Pt(14), MUTED
    chart.value_axis.minimum_scale = 0
    chart.value_axis.has_major_gridlines = False
    for axis in (chart.category_axis, chart.value_axis):
        axis.major_tick_mark = XL_TICK_MARK.NONE
        axis.minor_tick_mark = XL_TICK_MARK.NONE
        axis.tick_labels.font.name = FONT
        axis.format.line.color.rgb = RULE
    chart.plots[0].gap_width = gap_width
    for point, color in zip(chart.series[0].points, colors):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = color
        point.format.line.fill.background()
    # Positive axis IDs keep these charts readable by consumers that parse axId as unsigned.
    axis_ids = {}
    for element in chart._element.xpath(".//c:axId | .//c:crossAx"):
        original = element.get("val")
        element.set("val", axis_ids.setdefault(original, str(100 + len(axis_ids))))
    return chart


def luck_chart(slide, fixture):
    scores, cutoff = fixture["null_max"], fixture["hurdle"]
    step = 0.25
    left_bins = math.ceil((cutoff - min(scores)) / step)
    right_bins = math.ceil((max(scores) - cutoff) / step)
    n_bins = left_bins + right_bins
    first_edge = cutoff - left_bins * step
    centers = [f"{first_edge + (i + 0.5) * step:.2f}" for i in range(n_bins)]
    counts = [0] * n_bins
    for score in scores:
        counts[min(n_bins - 1, int((score - first_edge) / step))] += 1
    chart = column_chart(slide, 0.8, 2.18, 8.35, 3.85, centers, "Null replays", counts,
                         [GREY_BAR] * left_bins + [LUCK] * right_bins, gap_width=10)
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.value_axis.tick_labels.font.size = Pt(12)
    chart.value_axis.major_unit = 10
    chart.value_axis.maximum_scale = math.ceil(max(counts) / 10) * 10 + 10
    chart.value_axis.tick_labels.number_format = "0"
    # The cutoff falls on a bin edge. Anchor the overlay to the chart's manual plot bounds.
    plot_left, plot_top, plot_width, plot_height = 0.08, 0.07, 0.9, 0.81
    layout = chart._element.xpath(".//c:plotArea/c:layout")[0]
    manual = layout.makeelement(qn("c:manualLayout"), {})
    layout.append(manual)
    for name, value in (
        ("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"),
        ("wMode", "factor"), ("hMode", "factor"), ("x", plot_left), ("y", plot_top),
        ("w", plot_width), ("h", plot_height),
    ):
        manual.append(manual.makeelement(qn(f"c:{name}"), {"val": str(value)}))
    cutoff_x = 0.8 + 8.35 * (plot_left + plot_width * left_bins / n_bins)
    line(slide, cutoff_x, 2.18 + 3.85 * plot_top, cutoff_x, 2.18 + 3.85 * (plot_top + plot_height), LUCK, 2)


def build(output):
    benchmark_dir = MOCKS / "recorded_benchmark"
    metrics = json.loads((benchmark_dir / "benchmark_metrics.json").read_text(encoding="utf-8"))
    noise = json.loads((benchmark_dir / "noise_search.json").read_text(encoding="utf-8"))
    fixture = json.loads((MOCKS / "sample_hurdle.json").read_text(encoding="utf-8"))
    naive, lab, recall = metrics["naive"], metrics["ours"], metrics["recall_by_sr"]

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    blank = deck.slide_layouts[6]

    s = deck.slides.add_slide(blank)
    text(s, 0.8, 1.5, 11.65, 0.7, "How much of the best backtest is luck?", 38, True)
    line(s, 0.8, 2.75, 2.3, 2.75, ENG, 3)
    text(s, 0.8, 3.08, 11.65, 0.88, "Discovery Lab", 47, True, ENG)
    text(s, 0.8, 4.15, 9.9, 0.94, "An AI researcher that tests trading ideas against what chance can produce.",
         23, color=MUTED)
    text(s, 0.8, 6.76, 8, 0.3, "WHACK 26 · Brevan Howard track", 12, color=MUTED)
    notes(s, "0:00–0:20\n\n"
          "If an AI tries twenty trading rules, one can look good by chance. Discovery Lab tests whether "
          "the best-looking rule beats what chance could produce from the same search.")

    s = deck.slides.add_slide(blank)
    title(s, "Research loop", "One thesis, one trial ledger")
    text(s, 0.8, 1.64, 11.75, 0.44, "Does Bitcoin momentum predict the US dollar?", 23, True, AI)
    text(s, 0.8, 2.17, 11.75, 0.42, "One candidate goes long the dollar after a positive Bitcoin month and short after a negative one.", 17)
    steps = [
        ("AI proposes", "Writes rules in a fixed format.", AI, AI_BG),
        ("Engine tests", "Scores candidates on discovery data.", ENG, ENG_BG),
        ("AI reviews", "Challenges, refines, or nominates.", AI, AI_BG),
        ("Engine confirms", "Checks later data and assigns verdicts.", ENG, ENG_BG),
    ]
    step_x = [0.8, 3.78, 6.76, 9.74]
    for i, (heading, description, ink, fill) in enumerate(steps):
        x = step_x[i]
        band(s, x, 3.06, 2.75, 1.42, fill)
        text(s, x + 0.15, 3.22, 2.45, 0.4, heading, 19, True, ink)
        text(s, x + 0.15, 3.72, 2.45, 0.62, description, 15)
        if i < len(steps) - 1:
            line(s, x + 2.75, 3.77, step_x[i + 1], 3.77, head=True)
    line(s, 8.13, 4.48, 8.13, 4.95, AI)
    line(s, 8.13, 4.95, 5.16, 4.95, AI)
    line(s, 5.16, 4.95, 5.16, 4.48, AI, head=True)
    text(s, 3.78, 5.11, 5.73, 0.4, "Refinements enter the same ledger", 17, True, AI, PP_ALIGN.CENTER)
    text(s, 0.8, 6.12, 11.7, 0.44, "The reviewer sees discovery performance. Later tests run after the review stops.",
         17, color=MUTED)
    notes(s, "0:20–0:55\n\n"
          "Here the question is whether Bitcoin momentum predicts the dollar. One rule goes long the dollar "
          "next month after a positive Bitcoin month and short after a negative one. The reviewer sees discovery "
          "performance and can challenge, refine, or nominate. New rules enter the same ledger. Once review "
          "stops, the engine tests later data and assigns the verdict.\n\n"
          "Sources: agents/researcher.py, engine/research.py, runs/latest.json. The saved thesis was phrased "
          "in causal terms. The backtest tests a predictive rule.")

    s = deck.slides.add_slide(blank)
    title(s, "Search-level test", "The luck hurdle")
    text(s, 0.8, 1.63, 11.7, 0.48,
         f"Best score from each of {len(fixture['null_max'])} shifted replays of the same set of rules", 20, True)
    luck_chart(s, fixture)
    text(s, 0.96, 6.12, 8.03, 0.31, "Best score per replay (Newey–West t, bin midpoints)", 14,
         color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 9.45, 2.55, 2.98, 0.87, f"{fixture['hurdle']:.2f}", 46, True, LUCK)
    text(s, 9.45, 3.43, 2.98, 0.7, "95th-percentile cutoff", 20, True, LUCK)
    text(s, 9.45, 4.29, 2.98, 1.26, "A candidate’s observed score must clear this bar.", 19)
    text(s, 0.8, 6.67, 11.7, 0.34, "Illustrative distribution from a bundled synthetic run.", 13, color=MUTED)
    notes(s, "0:55–1:25\n\n"
          "We shift the tested rules’ positions against market returns two hundred times and keep the best "
          "score from each replay. A candidate has to beat the ninety-fifth percentile of those best scores. "
          "Adding distinct rules can raise that bar for earlier candidates too. This distribution is from a "
          "bundled synthetic example.\n\n"
          "Sources: engine/hurdle.py, engine/research.py, mocks/sample_hurdle.json. The shifts reuse the fixed "
          "set of tested rules. The model does not make new decisions in a null replay.")

    s = deck.slides.add_slide(blank)
    title(s, "Product demo", "Editing the future exposes a leak")
    text(s, 0.8, 1.66, 11.75, 0.52, "Decision month · December 2014", 23, True)
    text(s, 0.8, 2.27, 11.75, 0.46, "Multiply only data after that month by 1.30.", 19, color=MUTED)
    headers = [(0.8, 5.3, "RULE / FEATURE"), (6.15, 2.0, "BEFORE"), (8.37, 2.0, "AFTER"), (10.58, 1.9, "RESULT")]
    for x, w, label in headers:
        text(s, x, 3.1, w, 0.34, label, 13, True, MUTED)
    for i, (name, before, after, outcome, color) in enumerate((
        ("Planted centred S&P 500 average", "−0.36%", "+9.33%", "LEAK", REJECTED),
        ("One-month Bitcoin momentum", "−15.30%", "−15.30%", "PASS", ENG),
    )):
        y = 3.61 + i * 1.04
        line(s, 0.8, y - 0.13, 12.48, y - 0.13, RULE, 1)
        text(s, 0.8, y + 0.1, 5.3, 0.52, name, 20, True)
        text(s, 6.15, y, 2.0, 0.63, before, 27, True)
        text(s, 8.37, y, 2.0, 0.63, after, 27, True, color)
        text(s, 10.58, y + 0.1, 1.9, 0.48, outcome, 20, True, color)
    text(s, 0.8, 6.13, 11.75, 0.55, "Recorded feature values at the decision month, rounded to two decimal places.",
         15, color=MUTED)
    notes(s, "1:25–2:15\n\n"
          "Watch what happens when we change only the future. The planted centred average changes at the "
          "decision month because its window includes next month. The Bitcoin momentum feature stays fixed. "
          "That is the difference the audit is looking for.\n\n"
          "Demo cue: Research, then Replay last run, then Edit the future. Use the recorded Bitcoin/dollar run "
          "and select December 2014 with the multiplier at 1.30. Rerun first with the planted centred SPX "
          "control and then with the one-month Bitcoin momentum rule (hy_836c116fcf). The default audit month "
          "has no data for that Bitcoin feature. Use this slide as the recorded result if the replay differs.\n\n"
          "Sources: app/pages/1_Research.py, data/audit.py, runs/latest.json, and the market cache. "
          "The saved run stopped after one round with 17 Rejected and 3 Can’t tell yet verdicts.")

    s = deck.slides.add_slide(blank)
    title(s, "Proof", "Synthetic benchmark of the statistical pipeline")
    text(s, 0.8, 1.67, 11.75, 0.48, "Share of accepted rules that were planted fakes", 22, True)
    chart = column_chart(s, 0.8, 2.18, 7.74, 3.87, ["Naive t-test", "Discovery Lab"], "Share of accepted rules that were fake",
                         [naive["fdr_mean"], lab["fdr_mean"]], [GREY_BAR, ENG], gap_width=70)
    chart.value_axis.maximum_scale = 1.0
    chart.value_axis.visible = False
    chart.category_axis.tick_labels.font.size = Pt(16)
    plot = chart.plots[0]
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = "0.00%", False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.name, labels.font.size, labels.font.bold = FONT, Pt(25), True
    text(s, 8.9, 2.29, 3.56, 0.61, "Planted signals found", 19, True)
    for i, (sharpe, rate) in enumerate((("2.0", recall["2.0"]), ("1.0", recall["1.0"]))):
        y = 3.08 + 1.22 * i
        text(s, 8.9, y, 1.61, 0.64, f"{rate:.0%}", 34, True, ENG)
        text(s, 10.58, y + 0.18, 1.88, 0.45, f"Sharpe {sharpe}", 17)
    text(s, 0.8, 6.46, 11.75, 0.54, "Mean across 50 synthetic datasets · 10,000 candidates · statistical checks only",
         15, color=MUTED)
    notes(s, "2:15–3:00\n\n"
          f"Across fifty synthetic datasets, an average of {naive['fdr_mean']:.2%} of the rules accepted by the "
          f"naive t-test were planted fakes. For the lab’s statistical pipeline it was {lab['fdr_mean']:.2%}. "
          f"It found {recall['2.0']:.0%} of the strongest planted signals and {recall['1.0']:.0%} at Sharpe one. "
          "You can inspect every rule tried, the evidence behind its verdict, and what the sample had enough "
          "power to detect.\n\n"
          "Sources: engine/benchmark.py and mocks/recorded_benchmark/benchmark_metrics.json. “Accepted” means "
          "Promising or Replicated for the lab, and one-sided p < 0.05 for the naive t-test. The benchmark "
          "uses the batch pipeline with the factor check enabled and has no model in the loop. End the "
          "three-minute talk here.")

    s = deck.slides.add_slide(blank)
    title(s, "Q&A · Rule specification", "The Bitcoin momentum rule")
    text(s, 0.8, 1.72, 11.7, 0.46, "At month-end, measure Bitcoin’s one-month return.", 22, True)
    text(s, 0.8, 2.55, 11.7, 1.35, [
        "Feature · one-month Bitcoin momentum",
        "Position · sign of the feature × direction (+1)",
        "Asset · US Dollar Index (DXY)",
    ], 19, gap=13)
    band(s, 0.8, 4.34, 11.7, 1.25, CODE_BG)
    text(s, 1.05, 4.53, 11.2, 0.42, "Next month’s net return = held position × DXY return − trading costs",
         19, True)
    text(s, 1.05, 5.05, 11.2, 0.35, "The backtest charges 2 basis points per unit of position change.", 15, color=MUTED)
    notes(s, "Backup. Sources: engine/backtest.py, engine/features.py, config.py, runs/latest.json.")

    s = deck.slides.add_slide(blank)
    title(s, "Q&A · Data and confirmation", "Chronological splits")
    for i, (heading, description, fill) in enumerate((
        ("Discovery · 2003–2014", "Engine tests candidates. The reviewer receives discovery performance.", ENG_BG),
        ("Validation · 2015–2019", "After review ends, a representative needs positive net return and Newey–West t > 1.", CODE_BG),
        ("Holdout · 2020–2026", "Up to three nominations. Only qualifying rules get one logged look per spec.", AI_BG),
    )):
        y = 1.81 + i * 1.29
        band(s, 0.8, y, 11.7, 1.1, fill)
        text(s, 1.04, y + 0.15, 11.15, 0.38, heading, 19, True)
        text(s, 1.04, y + 0.58, 11.15, 0.37, description, 16)
    text(s, 0.8, 6.0, 11.7, 0.62,
         "Default windows shown. Assets with short histories use a 60/20/20 split. Monthly macro series use a one-month release lag.",
         16, color=MUTED)
    notes(s, "Backup. Sources: config.py, data/fetch.py, engine/backtest.py, engine/research.py, engine/vault.py. "
          "The vault logs a reservation for the spec hash before evaluating the holdout. The release lag "
          "approximates availability and the FRED download uses the current series.")

    s = deck.slides.add_slide(blank)
    title(s, "Q&A · Verdict rules", "How the engine assigns a verdict")
    outcomes = [
        ("Rejected", "Fails a hard check, or lacks significance in a sample with sufficient power.", REJECTED),
        ("Can’t tell yet", "Lacks significance and the sample has insufficient power.", MUTED),
        ("Fragile", "Passes discovery significance but misses a search-level gate or validation.", LUCK),
        ("Promising", "Clears the discovery and validation gates. Replication is still unconfirmed.", AI),
        ("Replicated", "Clears the discovery and validation gates, then passes the one-shot holdout.", ENG),
    ]
    for i, (label, description, color) in enumerate(outcomes):
        y = 1.8 + 0.93 * i
        if i:
            line(s, 0.8, y - 0.13, 12.5, y - 0.13, RULE, 1)
        text(s, 0.8, y, 2.7, 0.53, label, 20, True, color)
        text(s, 3.55, y, 8.95, 0.64, description, 17)
    notes(s, "Backup. The power boundary is an annual Sharpe of 0.75 at 80% power. Code applies hard failures "
          "first, including non-positive discovery mean. Sources: config.py, engine/verdict.py, engine/power.py.")

    s = deck.slides.add_slide(blank)
    title(s, "Q&A · Benchmark details", "Denominators and separate tests")
    noise_successes = round(noise["share_with_trusted"] * noise["n_searches"])
    for i, (heading, description) in enumerate((
        ("50 datasets, 200 rules each", "Each synthetic dataset has 10 planted signals and 190 traps or noise rules."),
        (f"{lab['false_trusted']} of {lab['trusted']} accepted rules was fake",
         f"Pooled across all seeds ({lab['fdr_pooled']:.2%}). The headline averages the 50 per-dataset rates."),
        (f"{noise_successes} of {noise['n_searches']} pure-noise searches accepted a rule",
         f"A separate experiment used {noise['candidates']} noise candidates per search."),
    )):
        y = 1.84 + 1.44 * i
        text(s, 0.8, y, 11.7, 0.5, heading, 23, True, ENG)
        text(s, 0.8, y + 0.53, 11.7, 0.63, description, 17)
    notes(s, "Backup. Sources: mocks/recorded_benchmark/benchmark_metrics.json, "
          "mocks/recorded_benchmark/noise_search.json, engine/benchmark.py. The pure-noise search disables the leak audit.")

    for i, slide in enumerate(deck.slides, 1):
        if i > 1:
            text(slide, 12.2, 6.97, 0.59, 0.28, str(i), 11, color=MUTED, align=PP_ALIGN.RIGHT)
    deck.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    build(parser.parse_args().output)
