"""Build the LinkedIn document post (swipeable PDF, 1080x1350 pages): `python -m src.carousel`.

Every number comes from outputs written by the pipeline (summary.json); the code pages are
pulled from the real source files with `inspect`, so what you see is what ran.
"""
from __future__ import annotations

import ast
import inspect
import io
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from . import cohort, config, explain, features, modeling, plots, target

W, H, DPI = 1080, 1350, 100
MARGIN = 80
INK, MUTED, BLUE, ORANGE, BG = plots.INK, plots.MUTED, plots.C_BLUE, plots.C_YES, "#ffffff"
CODE_BG, CODE_FG, CODE_CMT, CODE_KW, CODE_STR = "#0f172a", "#e2e8f0", "#7c8aa0", "#7fb8ff", "#f4b183"
KEYWORDS = {"def", "return", "for", "in", "if", "else", "import", "from", "and", "not", "lambda", "None",
            "True", "False", "with", "as", "or"}


class Page:
    """Top-down text layout in pixel coordinates (y measured from the top)."""

    def __init__(self, footer: str = ""):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.y = 90
        self.ax.text(MARGIN, H - 45, "Data: OpenAlex (CC0)", fontsize=14, color=MUTED, va="center")
        if footer:
            self.ax.text(W - MARGIN, H - 45, footer, fontsize=14, color=MUTED, va="center", ha="right")

    def _wrap(self, text: str, fs: float, width_px: int = W - 2 * MARGIN) -> list[str]:
        chars = int(width_px / (fs * 0.80))
        return textwrap.wrap(text, chars) or [""]

    def title(self, text: str, fs: float = 40, color: str = INK):
        for line in self._wrap(text, fs):
            self.ax.text(MARGIN, self.y, line, fontsize=fs, fontweight="bold", color=color, va="top")
            self.y += fs * 1.75
        self.y += 20

    def para(self, text: str, fs: float = 25, color: str = INK, gap: float = 22, bold: bool = False):
        for line in self._wrap(text, fs):
            self.ax.text(MARGIN, self.y, line, fontsize=fs, color=color, va="top",
                         fontweight="bold" if bold else "normal")
            self.y += fs * 1.85
        self.y += gap

    def bullets(self, items: list[str], fs: float = 25, gap: float = 14):
        for it in items:
            lines = self._wrap(it, fs, W - 2 * MARGIN - 40)
            self.ax.text(MARGIN, self.y, "•", fontsize=fs, color=ORANGE, va="top")
            for line in lines:
                self.ax.text(MARGIN + 40, self.y, line, fontsize=fs, color=INK, va="top")
                self.y += fs * 1.85
            self.y += gap

    def big(self, number: str, caption: str, color: str = BLUE):
        self.ax.text(MARGIN, self.y, number, fontsize=64, fontweight="bold", color=color, va="top")
        self.y += 105
        self.para(caption, fs=24, color=MUTED, gap=26)

    def code(self, snippets: list[tuple[str, str]], fs: float = 12.5):
        """Dark panel with syntax-lightened code. snippets = [(file label, source)]."""
        probe = self.ax.text(0, 0, "M" * 200, fontsize=fs, family="DejaVu Sans Mono")
        self.fig.canvas.draw()
        char_px = probe.get_window_extent(self.fig.canvas.get_renderer()).width / 200  # measured, not guessed
        probe.remove()
        max_chars = int((W - 2 * MARGIN - 50) / char_px)
        rows: list[tuple[str, str]] = []  # (kind, text)
        for label, src in snippets:
            rows.append(("label", f"# {label}"))
            for ln in src.splitlines():
                if len(ln) <= max_chars:
                    rows.append(("code", ln))
                    continue
                indent = len(ln) - len(ln.lstrip())
                wrapped = textwrap.wrap(ln, max_chars, subsequent_indent=" " * (indent + 4),
                                        break_long_words=False, break_on_hyphens=False)
                rows.extend(("code", w) for w in wrapped)
            rows.append(("code", ""))
        line_h = fs * 1.62
        top = self.y
        height = len(rows) * line_h + 40
        self.ax.add_patch(FancyBboxPatch((MARGIN - 10, top), W - 2 * MARGIN + 20, height,
                                         boxstyle="round,pad=0,rounding_size=14", fc=CODE_BG, ec="none"))
        y = top + 24
        for kind, text in rows:
            if kind == "label":
                self.ax.text(MARGIN + 10, y, text, fontsize=fs, family="DejaVu Sans Mono", color="#9fb3c8",
                             va="top", fontweight="bold")
            else:
                self._draw_code_line(text, y, fs, char_px)
            y += line_h
        self.y = top + height + 20
        return self.y

    def _draw_code_line(self, text: str, y: float, fs: float, char_px: float):
        pat = re.compile(r"(#.*$)|(\"[^\"]*\"|'[^']*')|\b(" + "|".join(sorted(KEYWORDS)) + r")\b")
        pos, x0 = 0, MARGIN + 10
        for m in pat.finditer(text):
            if m.start() > pos:
                self._seg(text[pos:m.start()], pos, y, fs, x0, char_px, CODE_FG)
            color = CODE_CMT if m.group(1) else CODE_STR if m.group(2) else CODE_KW
            self._seg(m.group(0), m.start(), y, fs, x0, char_px, color)
            pos = m.end()
        if pos < len(text):
            self._seg(text[pos:], pos, y, fs, x0, char_px, CODE_FG)

    def _seg(self, s, col, y, fs, x0, char_px, color):
        self.ax.text(x0 + col * char_px, y, s, fontsize=fs, family="DejaVu Sans Mono", color=color, va="top")

    def image(self) -> Image.Image:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
        plt.close(self.fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- author footer
AUTHOR = "Maryam Mahmoudi"
GITHUB = "github.com/maryammahmoudi1993"
LINKEDIN = "linkedin.com/in/maryam-mahmoudi-data-scientist"
REPO = "github.com/maryammahmoudi1993/xai-sketchbook/tree/main/sketches/01-early-career-signals"
BAND = 170


def add_band(im: Image.Image) -> Image.Image:
    """Append an author/links band under every page (including the pre-rendered figures)."""
    bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 27)
    reg = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)
    out = Image.new("RGB", (im.width, im.height + BAND), "white")
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    y0 = im.height
    d.line([(MARGIN, y0 + 6), (im.width - MARGIN, y0 + 6)], fill="#d5dae1", width=2)
    d.text((MARGIN, y0 + 22), AUTHOR, font=bold, fill=INK)
    for i, (k, v) in enumerate((("GitHub", GITHUB), ("LinkedIn", LINKEDIN), ("Code", REPO))):
        d.text((MARGIN, y0 + 66 + i * 32), f"{k}:", font=reg, fill=MUTED)
        d.text((MARGIN + 95, y0 + 66 + i * 32), v, font=reg, fill=INK)
    return out


# ---------------------------------------------------------------- code excerpts
def source(func, drop_doc: bool = True) -> str:
    """Verbatim source of a function, optionally without its docstring."""
    src = textwrap.dedent(inspect.getsource(func))
    if not drop_doc:
        return src
    tree = ast.parse(src)
    fn = tree.body[0]
    if ast.get_docstring(fn) is not None:
        d = fn.body[0]
        lines = src.splitlines()
        del lines[d.lineno - 1:d.end_lineno]
        src = "\n".join(lines)
    return src


# ---------------------------------------------------------------- pages
def pct(x, d=1):
    return f"{x * 100:.{d}f}"


def build_pages(s: dict, fig_dir: Path) -> list[Image.Image]:
    m = s["metrics"]
    lg = m["lightgbm|without_early_citations"]
    lr = m["logistic|without_early_citations"]
    g = {(r["tier_group"], r["has_top_coauthor"]): r for r in s["groups"]}
    top_no, top_yes = g[("Top-ranked institution", 0)], g[("Top-ranked institution", 1)]
    low_no, low_yes = g[("Lower-ranked institution", 0)], g[("Lower-ranked institution", 1)]
    did, adj, ov = s["did"], s["adj"], s["overall_gap"]
    fn = s["funnel"]
    pages: list[Image.Image] = []
    n_pages = 14

    def mk(i):
        return Page(footer=f"{i}/{n_pages}")

    # 1 cover
    p = mk(1)
    p.y = 330
    p.title("Can you spot a future top-cited AI researcher in their first 3 years?", fs=46)
    p.para(f"I followed {s['n_cohort']:,} AI researchers for 15 years using open data and asked what their "
           "first three years reveal.", fs=27, color=MUTED)
    p.y += 60
    p.para("Swipe for the method, the code and the results  →", fs=26, color=ORANGE, bold=True)
    p.y = 1150
    p.para("Association, not cause. Preliminary results from one field.", fs=20, color=MUTED)
    pages.append(p.image())

    # 2 question
    p = mk(2)
    p.title("Two questions")
    p.bullets(["Which signals in the first 3 years of an AI career predict becoming a top-5% cited researcher by year 15?",
               "Does working with a top researcher early matter more for people at lower-ranked institutions?"], fs=28, gap=28)
    p.para(f"\"Top 5%\" here means at least {s['threshold']:,.0f} citations, counting every paper published in the "
           f"researcher's first 15 years. {s['n_pos']} of {s['n_cohort']:,} researchers ({pct(s['n_pos'] / s['n_cohort'])}%) "
           "reach it.", fs=25)
    p.para("I report whatever the data shows, including the parts that did not come out.", fs=25, color=MUTED)
    pages.append(p.image())

    # 3 cohort
    p = mk(3)
    p.title("Who is in the study")
    p.bullets([f"{fn['candidates_screened']:,} candidate authors, found on a random sample of AI papers from 2008-2011",
               f"{fn['t0_in_range']:,} of them published their first indexed paper in 2008-2011",
               f"{fn['cohort_after_min_early_works']:,} published at least 2 papers in their first 3 years: this is the cohort"],
              fs=26, gap=22)
    p.para(f"People who later stopped publishing stay in the data with the label \"not top 5%\". "
           f"{pct(s['share_stopped'], 0)}% of the cohort published nothing after year 3. Dropping them would make careers "
           "look better than they were (survivorship bias).", fs=25)
    p.para("Source: OpenAlex, a free open catalogue of research papers (CC0).", fs=22, color=MUTED)
    pages.append(p.image())

    # 4 features
    p = mk(4)
    p.title("What the model sees")
    p.para("Only information from the first 3 years (no peeking at the future):", fs=25, gap=14)
    labels = [v[0] for k, v in features.DATA_DICTIONARY.items() if k != "early_citations"]
    p.bullets(labels + ["Citations to the early papers (also tested without)"], fs=23, gap=4)
    p.y += 14
    p.para(f"Train on researchers who started in 2008-09 (n={s['n_train']:,}). Test on those who started in "
           f"2010-11 (n={s['n_test']:,}): the model never sees the test group.", fs=24, color=MUTED)
    pages.append(p.image())

    # 5 code: cohort + target
    p = mk(5)
    p.title("The code: cohort and outcome", fs=36)
    p.code([("src/cohort.py", source(cohort.select_cohort)), ("src/target.py", source(target.build_targets))])
    pages.append(p.image())

    # 6 code: model + SHAP
    p = mk(6)
    p.title("The code: model and explanation", fs=36)
    p.code([("src/modeling.py", source(modeling.temporal_split) + "\n\n" + source(modeling.make_lgbm)),
            ("src/explain.py", source(explain.shap_explain)),
            ("src/explain.py", source(explain.group_assignment))])
    pages.append(p.image())

    # 7 metrics
    p = mk(7)
    p.title("Result 1: the first 3 years do carry a signal")
    p.big(f"{lg['roc_auc'][0]:.2f}", f"AUC on researchers the model never saw (95% range {lg['roc_auc'][1]:.2f}-{lg['roc_auc'][2]:.2f}). "
          "0.50 would be a coin flip, 1.00 perfect.")
    p.para("AUC asks: pick one future top-5% researcher and one other at random; how often does the model score the "
           "top-5% one higher?", fs=24)
    p.para(f"But the model is far from a crystal ball: precision-recall AUC is {lg['pr_auc'][0]:.2f} against "
           f"{lg['base_rate']:.2f} for guessing. Most top-5% researchers are not identifiable at year 3. "
           f"A plain logistic regression does about as well (AUC {lr['roc_auc'][0]:.2f}).", fs=24)
    pages.append(p.image())
    pages.append(Image.open(fig_dir / "4_metrics_card.png").convert("RGB"))

    # 9 shap explain + fig
    p = mk(9)
    p.title("Reading the SHAP chart")
    p.para("SHAP splits each prediction into contributions, one per signal.", fs=25)
    p.bullets(["Each dot is one researcher.", "Right of zero: that signal pushed the prediction toward top 5%. Left: away from it.",
               "Red = a high value of the signal, blue = low. Grey = unknown.",
               "Signals are sorted by overall influence, most influential on top."], fs=25, gap=14)
    p.para("Explains the model, not the world: the model uses whatever statistically goes with the outcome.",
           fs=23, color=MUTED)
    pages.append(p.image())
    pages.append(Image.open(fig_dir / "2_shap_beeswarm.png").convert("RGB"))

    # 11 2x2 explain + fig
    p = mk(11)
    p.title("Result 2: a top coauthor goes with better odds")
    p.para("No model needed: just count who reached the top 5%.", fs=25, gap=14)
    p.bullets([f"Top-ranked institution: {pct(top_no['p_top5'])}% without a top coauthor vs {pct(top_yes['p_top5'])}% with one",
               f"Lower-ranked institution: {pct(low_no['p_top5'])}% vs {pct(low_yes['p_top5'])}%"], fs=26, gap=18)
    p.para(f"Overall gap: {ov[0] * 100:.1f} percentage points (95% range {ov[1] * 100:.1f} to {ov[2] * 100:.1f}). It survives "
           f"adjusting for the other signals (log-odds {adj[0]:.2f}, range {adj[1]:.2f} to {adj[2]:.2f}), and its direction "
           "matches Li et al. 2019 (Nature Communications).", fs=24)
    p.para("\"Top coauthor\" = an early coauthor in the top 5% by current h-index. Papers written together also add to the "
           "researcher's own citations, which builds part of the link in.", fs=22, color=MUTED)
    pages.append(p.image())
    pages.append(Image.open(fig_dir / "1_top_coauthor_by_institution.png").convert("RGB"))

    # 13 DiD
    p = mk(13)
    p.title("Result 3: does it matter MORE at lower-ranked institutions?")
    p.big("Can't tell", f"The gap is {s['gap_top'][0] * 100:.1f} points at top-ranked and {s['gap_lower'][0] * 100:.1f} at lower-ranked "
          f"institutions. Difference: {did[0] * 100:.1f} points (95% range {did[1] * 100:.1f} to {did[2] * 100:+.1f}).", color=ORANGE)
    p.para("The range includes zero, so the data do not support the idea that a top coauthor matters more for people "
           "from lower-ranked institutions, and they do not rule it out either.", fs=24)
    p.para(f"Only {low_yes['n']} lower-ranked researchers had a top coauthor ({low_yes['n_top5']} of them reached the top 5%). "
           "That is too few to separate a real difference from noise.", fs=24, color=MUTED)
    pages.append(p.image())

    # 14 limits + repro
    p = mk(14)
    p.title("What this does not show")
    p.bullets(["Correlation, not causation: strong researchers choose strong groups and vice versa.",
               "One field (AI) and one database. No claim beyond that.",
               f"Many conferences ({', '.join(s['hand_unmatched'][:4])}, ...) are not matched in OpenAlex, so venue quality is understated.",
               f"{pct(s['feat']['share_unknown_institution'], 0)}% of researchers have no institution; the 2x2 leaves out {s['n_excluded']:,}.",
               "Author-name mix-ups in OpenAlex can merge or split careers."], fs=23, gap=10)
    p.para("Code, data steps and every number are open: see the links below.", fs=22, color=ORANGE, bold=True)
    p.para("python -m src.pipeline  |  python -m src.carousel", fs=21, color=MUTED)
    pages.append(p.image())
    return pages


def main(argv=None) -> Path:
    s = json.loads((config.ROOT / "summary.json").read_text(encoding="utf-8"))
    if s["dry_run"]:
        raise SystemExit("summary.json is from a dry run; run the full pipeline first.")
    pages = [add_band(im) for im in build_pages(s, config.FIG_DIR)]
    out = config.ROOT / "LinkedIn_carousel.pdf"
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=72.0)
    prev = config.RESULTS_DIR / "carousel_preview"
    prev.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(pages, 1):
        im.save(prev / f"page_{i:02d}.png")
    print(f"Wrote {out} ({len(pages)} pages)")
    return out


if __name__ == "__main__":
    main()
