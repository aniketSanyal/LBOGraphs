"""Builds the manifold-benchmark report PDF."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
EQ = os.path.join(HERE, "eq")
os.makedirs(EQ, exist_ok=True)
RES = r"d:\Work\Graphs\code\cluster_results\results"
OUT = r"d:\Work\Graphs\code\manifold_benchmark_report.pdf"

# ---- fonts -------------------------------------------------------------------
TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Sans", os.path.join(TTF, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", os.path.join(TTF, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", os.path.join(TTF, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Sans-BoldItalic", os.path.join(TTF, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(TTF, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans-Italic", boldItalic="Sans-BoldItalic")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#8a8984")
RULE = colors.HexColor("#dcdad4")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_BG = colors.HexColor("#eef4fc")
WARN_BG = colors.HexColor("#fdf3ec")
WARN = colors.HexColor("#c4521f")
HEAD_BG = colors.HexColor("#f3f2ef")
GOOD_BG = colors.HexColor("#e6f2ff")

ss = {
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=14.2, textColor=INK,
                           spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.2, leading=11.5, textColor=INK2),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.2, leading=11.5, textColor=INK2,
                          spaceBefore=3, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=20, textColor=INK,
                         spaceBefore=4, spaceAfter=10, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=5, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=9.8, leading=13, textColor=INK2,
                         spaceBefore=6, spaceAfter=3, keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9.6, leading=14, textColor=INK,
                             leftIndent=14, bulletIndent=3, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10.4, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Sans-Bold", fontSize=8, leading=10.4, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10.2, textColor=INK,
                           backColor=HEAD_BG, borderPadding=(5, 5, 5, 5), leftIndent=5,
                           rightIndent=5, spaceBefore=4, spaceAfter=8),
    "q": ParagraphStyle("q", fontName="Sans-Bold", fontSize=9.6, leading=13.5, textColor=INK,
                        spaceBefore=7, spaceAfter=2),
    "a": ParagraphStyle("a", fontName="Sans", fontSize=9.4, leading=13.8, textColor=INK,
                        leftIndent=12, spaceAfter=4),
}

story = []


def P(text, style="body"):
    story.append(Paragraph(text, ss[style]))


def B(items):
    for it in items:
        story.append(Paragraph(it, ss["bullet"], bulletText="\u2022"))


def H1(text):
    story.append(Paragraph(text, ss["h1"]))


def H2(text):
    story.append(Paragraph(text, ss["h2"]))


def H3(text):
    story.append(Paragraph(text, ss["h3"]))


def box(title, lines, kind="info"):
    bg, edge = (ACCENT_BG, ACCENT) if kind == "info" else (WARN_BG, WARN)
    content = [Paragraph(f"<b>{title}</b>", ss["cellb"])]
    for ln in lines:
        content.append(Paragraph(ln, ParagraphStyle("bx", parent=ss["body"], fontSize=9,
                                                     leading=13, spaceAfter=2)))
    t = Table([[content]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 3))
    story.append(t)
    story.append(Spacer(1, 8))


def fig(name, caption, width_mm=170):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = width_mm * mm
    img = Image(path, width=width, height=width * h / w)
    story.append(KeepTogether([img, Paragraph(caption, ss["cap"])]))


_eq_n = [0]


def eq(latex, size=12, label=True):
    _eq_n[0] += 1
    path = os.path.join(EQ, f"eq{_eq_n[0]}.png")
    f = plt.figure(figsize=(0.01, 0.01))
    f.text(0, 0, f"${latex}$", fontsize=size, color="#0b0b0b")
    f.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(f)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = w / 300 * 72 * 0.95
    img = Image(path, width=width, height=width * h / w)
    num = Paragraph(f"({_eq_n[0]})", ParagraphStyle("eqn", parent=ss["small"], alignment=2))
    t = Table([[img, num if label else ""]], colWidths=[150 * mm, 20 * mm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)


def table(rows, widths, header_rows=1, highlight=None, align_right_from=1, font_size=8, zebra=True):
    """rows: list of lists of str (may contain markup). highlight: set of (r, c) to bold+tint."""
    highlight = highlight or set()
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, cell in enumerate(row):
            st = ss["cellh"] if r < header_rows else (ss["cellb"] if (r, c) in highlight else ss["cell"])
            st = ParagraphStyle(f"c{r}{c}", parent=st, fontSize=font_size, leading=font_size * 1.3,
                                alignment=(2 if (c >= align_right_from and r >= header_rows) else 0))
            out.append(Paragraph(str(cell), st))
        data.append(out)
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=header_rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK2),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fafaf8")))
    for (r, c) in highlight:
        style.append(("BACKGROUND", (c, r), (c, r), GOOD_BG))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 4))


# ---- data ----------------------------------------------------------------------
def load(fn):
    return json.load(open(os.path.join(RES, fn)))


smooth = load("manifold_v2_lowmodes_results_1969722.json")
rough = load("manifold_v2_noise_results_1969723.json")
rough_run1 = load("manifold_v2_noise_results_1938876.json")
v1 = load("manifold_results_1936768.json")

TASKS = [f"{d}/{s}" for d in ("Sphere", "Torus") for s in ("Diffusion", "Reaction-Diffusion", "Beam")]
MODELS = ["Ours", "SFNO / FNO2d", "FNO", "CNO", "LNO"]
HEADERS = ["Ours", "SFNO (sphere) / FNO2d (torus)", "FNO", "CNO", "LNO"]


def geo(task):
    return "SFNO" if task.startswith("Sphere") else "FNO2d"


def e(run, task, model):
    res = run["results"] if "results" in run else run
    key = geo(task) if model == "SFNO / FNO2d" else model
    return res[task][key]["final_error"]


def results_table(run, note_floor=False):
    rows = [["Domain", "PDE"] + HEADERS + (["Our floor\u2020"] if note_floor else [])]
    hl = set()
    for i, t in enumerate(TASKS):
        d, s = t.split("/")
        vals = [e(run, t, m) for m in MODELS]
        best = int(np.argmin(vals))
        hl.add((i + 1, 2 + best))
        row = [d, s] + [f"{v:.4f}" for v in vals]
        if note_floor:
            fl = run["data_stats"][t]["uT_outside_basis"]
            row.append(f"{fl:.4f}")
        rows.append(row)
    widths = ([17, 31, 17, 30, 17, 17, 17] if not note_floor else [17, 31, 16, 28, 15, 15, 15, 18])
    table(rows, widths, highlight=hl, align_right_from=2)


# ================================================================================
# Cover
# ================================================================================
story.append(Spacer(1, 40 * mm))
story.append(Paragraph("Learning PDE solution operators on curved surfaces",
                       ParagraphStyle("t", fontName="Sans-Bold", fontSize=24, leading=30, textColor=INK)))
story.append(Spacer(1, 6))
story.append(Paragraph("A heat-kernel neural operator on the sphere and the torus: method, experiments, "
                       "results and how to defend them",
                       ParagraphStyle("st", fontName="Sans", fontSize=13, leading=18, textColor=INK2)))
story.append(Spacer(1, 14 * mm))
cover_rows = [
    ["Project", "Graphs / neural-operator benchmark (d:\\Work\\Graphs\\code)"],
    ["Author", "Aniket"],
    ["Experiments", "October 21\u201330, 2026, TUM cluster (atcremers47), 1 GPU per job"],
    ["Runs reported", "1936768 (manifold v1), 1938876 + 1969723 (rough ICs), 1969722 (smooth ICs)"],
    ["Code", "model.py, equations.py, ref_models/, benchmark_manifolds.ipynb, benchmark_manifolds_v2.ipynb"],
]
t = Table([[Paragraph(a, ss["cellb"]), Paragraph(b, ss["cell"])] for a, b in cover_rows],
          colWidths=[32 * mm, 138 * mm])
t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story.append(t)
story.append(Spacer(1, 16 * mm))
box("How to use this document", [
    "Sections 1\u20136 explain the problem, the method, the setup and the history of changes, in the order you "
    "would present them. Section 7 has every result table, section 8 the learning curves, section 9 example "
    "predictions, section 10 what the results do and do not show, and section 11 the questions you should "
    "expect, with answers. "
    "Every number in the result tables is read directly from the JSON files the cluster jobs wrote.",
])
story.append(NextPageTemplate("normal"))
story.append(PageBreak())

# ---- contents ------------------------------------------------------------------
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle("toc1", fontName="Sans-Bold", fontSize=9.6, leading=14, leftIndent=0, spaceBefore=3),
    ParagraphStyle("toc2", fontName="Sans", fontSize=8.6, leading=11.5, leftIndent=14, textColor=INK2),
]
story.append(Paragraph("Contents", ParagraphStyle("tocTitle", parent=ss["h1"])))
story.append(toc)
story.append(PageBreak())

# ================================================================================
# 0. Summary
# ================================================================================
H1("Summary")
P("We built a neural operator that learns how a PDE moves a field forward in time on a curved surface. "
  "It works in the surface's own vibration modes (the eigenfunctions of the Laplace\u2013Beltrami operator) "
  "and builds every layer from heat kernels, i.e. from smoothing the field for different amounts of time. "
  "We tested it on three PDEs (diffusion, reaction\u2013diffusion, a beam/plate equation) on two closed "
  "surfaces (a sphere and a torus), against five baselines, with two kinds of input data.")
rows = [
    ["Comparison", "Result", "What it means"],
    ["Ours vs grid models that ignore the geometry (FNO, CNO, LNO on a vertex sequence)",
     "Ours wins all 12 task/data combinations, by 1.7\u201330\u00d7 (smallest margin: beam with rough data)",
     "Respecting the surface's geometry matters a great deal."],
    ["Ours vs geometry-aware models (SFNO on the sphere, FNO2d on the torus)",
     "They win 11 of 12. On smooth data by 3\u201319\u00d7; on rough data by 1.4\u20138\u00d7. "
     "Ours wins only torus reaction\u2013diffusion with rough data (0.026 vs 0.035).",
     "Our model is not yet competitive with the best specialised models."],
    ["Our model on its own",
     "2\u20134% error on every diffusion-type task, with 4,860 parameters (60\u2013180\u00d7 fewer than SFNO/FNO2d); "
     "no overfitting; still improving at the last epoch",
     "Small and stable, but under-trained and slow (O(V\u00b2) cost)."],
    ["Beam with rough data",
     "Ours 0.56\u20130.59; 22\u201331% of the target is outside our 96 modes",
     "A hard limit of the truncated basis, plus the same training limit."],
]
table(rows, [55, 62, 53], align_right_from=99)
box("The one-sentence version for the talk", [
    "\u201cA 5k-parameter operator built from the surface's own heat kernels learns PDE dynamics on curved "
    "surfaces to 2\u20133% error and beats every model that ignores the geometry, but specialised spectral "
    "models with 60\u2013180\u00d7 more parameters are still 3\u201319\u00d7 more accurate, and our learning "
    "curves show we have not reached our model's limit yet.\u201d",
])
P("Honest framing matters here. The first manifold run (section 7.3) looked like a clean win, but only "
  "because the baselines could not see the geometry. We then added geometry-aware baselines and rough "
  "inputs precisely to test that, and the result changed. Presenting both, in that order, is the "
  "strongest position: it shows the comparison was stress-tested.")
story.append(PageBreak())

# ================================================================================
# 1. Problem
# ================================================================================
H1("1. The problem: learning a PDE's solution operator on a surface")
P("A <b>PDE solution operator</b> is the map from a starting field to the field some time later. "
  "For example, for the heat equation it takes a temperature pattern u(0) on a surface and returns the "
  "pattern u(T) after it has diffused for time T. A <b>neural operator</b> learns this map from examples: "
  "we show it many pairs (u(0), u(T)) computed by a numerical solver, and it learns to predict u(T) for new "
  "u(0) directly, without running the solver.")
P("Most well-known neural operators (FNO, CNO) assume the data lives on a <b>regular grid on a flat domain</b>: "
  "they use the Fast Fourier Transform or image convolutions, which need equally spaced points in "
  "rows and columns. A surface such as a sphere, a torus, an organ or a car body is <b>not flat</b> and is "
  "usually given as a <b>triangle mesh</b> with irregularly placed vertices. There is no natural grid.")
P("Our idea: instead of borrowing the Fourier basis of a flat grid, use the surface's own natural basis, "
  "the eigenfunctions of its Laplace\u2013Beltrami operator, and build the network from heat kernels on "
  "the surface. Then the model is defined on any mesh, and every operation respects the geometry.")
H2("1.1 Why \u201cnon-Euclidean\u201d")
P("Euclidean means flat: straight lines, the usual distances, a global grid. On a sphere or torus, "
  "distances are measured along the surface, there are no straight lines, and you cannot cover the "
  "surface with a single regular grid without distortion (the sphere's poles are the classic example). "
  "All of our earlier 1D experiments (benchmark.ipynb, appendix A) were on a flat interval, where our model "
  "has no advantage because its basis there <i>is</i> just the Fourier sine basis. The surfaces are where "
  "the idea should matter.")

# ================================================================================
# 2. Laplace-Beltrami
# ================================================================================
H1("2. The Laplace\u2013Beltrami operator and its eigenpairs")
P("The Laplacian \u0394 measures how much a value at a point differs from the average around it; it is the "
  "operator behind diffusion, waves and vibrations. On a curved surface the same idea is called the "
  "<b>Laplace\u2013Beltrami operator</b>. On a triangle mesh with V vertices it is discretised with the standard "
  "<b>cotangent formula</b> (we use the robust_laplacian library), which gives two V\u00d7V sparse matrices:")
B(["<b>S, the stiffness matrix</b>: how strongly neighbouring vertices are coupled (built from the angles of "
   "the triangles). Symmetric and positive semi-definite.",
   "<b>M, the mass matrix</b>: diagonal; M<sub>vv</sub> is the surface area that belongs to vertex v. It "
   "corrects for uneven vertex spacing, so sums over vertices become integrals over the surface."])
P("The discrete operator is L = \u2212M<super>\u22121</super>S (code: <font name='Mono'>MeshDomain.laplacian</font> "
  "in equations.py). Its eigenpairs solve the <b>generalised eigenproblem</b>")
eq(r"S\,\phi_k = \lambda_k\, M\,\phi_k ,\qquad 0=\lambda_0\leq\lambda_1\leq\lambda_2\leq\ldots")
P("We solve it through the equivalent symmetric matrix M<super>\u2212\u00bd</super>SM<super>\u2212\u00bd</super> "
  "with ARPACK (scipy eigsh, shift-invert), keep the <b>96 smallest</b> eigenpairs and map back with "
  "\u03a6 = M<super>\u2212\u00bd</super>U. The resulting eigenvectors are orthonormal with respect to the mass matrix:")
eq(r"\Phi^{\top} M\, \Phi = I \quad(\text{checked numerically: error } \approx 2\times10^{-14})")
P("<b>What the eigenpairs mean.</b> Each eigenvector \u03c6<sub>k</sub> is a standing-wave pattern on the "
  "surface (figure 2); its eigenvalue \u03bb<sub>k</sub> is its squared frequency: small \u03bb = broad, smooth "
  "pattern; large \u03bb = fine, wiggly pattern. \u03bb<sub>0</sub> = 0 is the constant pattern. Together they "
  "are the surface's version of the Fourier basis: any reasonable field can be written as a sum of them.")
fig("eigenvalues.png",
    "<b>Figure 1.</b> The 96 eigenvalues we keep. <b>Left, sphere:</b> blue dots are our mesh's eigenvalues, the "
    "dashed line the exact values l(l+1) of a perfect sphere. They come in groups of 1, 3, 5, 7, ... equal values "
    "(the spherical harmonics of degree l), exactly as theory says, which validates the discretisation. "
    "The mesh values fall slightly below the exact ones at higher l (about 79 vs 90 at l = 9) because 642 "
    "vertices resolve fine patterns less accurately. <b>Right, torus:</b> no closed formula, but the structure is "
    "visible: pairs of equal values (patterns rotated around the ring), and the first pattern that varies "
    "around the thin tube appears near 1/r\u00b2 \u2248 11.")
fig("eigenvectors.png",
    "<b>Figure 2.</b> Some eigenvectors drawn on the surfaces (red positive, blue negative). Low k = smooth, "
    "large-scale patterns; higher k = finer patterns. Sphere k = 1 is the l = 1 harmonic (one half positive, "
    "one negative); torus k = 1 varies once around the big ring.")
H2("2.1 Numbers to remember")
table([
    ["", "Sphere", "Torus"],
    ["Vertices / triangles", "642 / 1,280", "640 / 1,280"],
    ["Surface area (sum of M)", "12.51 (exact 4\u03c0 = 12.57)", "11.74 (exact 4\u03c0\u00b2Rr = 11.84)"],
    ["\u03bb<sub>1</sub> (first non-zero)", "2.00 (exact 2)", "1.03"],
    ["\u03bb<sub>95</sub> (largest kept)", "79.5", "91.7"],
    ["M-orthogonality error of \u03a6", "1.8e-14", "1.8e-14 to 2.4e-14"],
], [60, 55, 55])

# ================================================================================
# 3. Our operator
# ================================================================================
H1("3. Our model: the heat-kernel integral operator")
H2("3.1 The heat kernel")
P("If you put a unit of heat at vertex u and let it diffuse on the surface for time t, the amount that "
  "arrives at vertex v is the <b>heat kernel</b> K<sub>t</sub>(v, u). In the eigenbasis it has a simple formula:")
eq(r"K_t(v,u) \;=\; \sum_{k=0}^{95} e^{-t\lambda_k}\,\phi_k(v)\,\phi_k(u)")
P("Applying it to a field f (\u201csmoothing f for time t\u201d) is three cheap steps, which is how HeatKernel.forward "
  "in model.py computes it:")
eq(r"\hat f_k=\sum_u \phi_k(u)\,m_u\,f(u)\;\;\longrightarrow\;\; \hat f_k\,e^{-t\lambda_k}\;\;\longrightarrow\;\; "
   r"(K_t f)(v)=\sum_k \phi_k(v)\,\hat f_k\,e^{-t\lambda_k}")
P("In words: break the field into surface patterns, shrink each pattern by e<super>\u2212t\u03bb</super> "
  "(fine patterns shrink fastest), put it back together. Small t barely changes the field; large t keeps only "
  "the smoothest patterns. Because the eigenvectors come from the mesh, this smoothing follows the surface "
  "(heat flows around the sphere, not through it).")
fig("heat_filters.png",
    "<b>Figure 3.</b> The 12 heat-kernel \u201cfilters\u201d each layer can use: how much each pattern (eigenvalue \u03bb) "
    "is kept after smoothing for time t<sub>r</sub>. The times are log-spaced from 1/\u03bb<sub>max</sub> (lightest: barely "
    "smooths) to 3/\u03bb<sub>1</sub> (darkest: removes everything except the mean). They are computed from each "
    "surface's own spectrum, so the same rule works on any mesh.")
H2("3.2 One layer")
P("A layer (IntegralOperator in model.py) takes a field with C<sub>in</sub> channels at every vertex and "
  "returns one with C<sub>out</sub> channels:")
eq(r"y_o(v)\;=\;\sum_{r=1}^{12}\;\sum_{c}\;W_r[c,o]\;\sum_{u}\;K_{t_r}(v,u)\;m_u\;"
   r"\varphi_c\left(f(v),f(u)\right)\;f_c(u)", size=12.5)
P("Read it from the inside out:")
B(["<b>\u03c6(f(v), f(u))</b> is a small neural network (2 linear layers, 32 hidden units, ReLU) that looks at the "
   "values at the receiving vertex v and the sending vertex u and outputs a weight per channel. It decides "
   "<i>how much of u's value v should listen to</i>. This is what makes the layer nonlinear and lets it "
   "depend on the data (similar to attention).",
   "<b>K<sub>t<sub>r</sub></sub>(v, u) m<sub>u</sub></b> spreads those weighted values over the surface with heat kernel "
   "number r, i.e. with smoothing time t<sub>r</sub>. m<sub>u</sub> is the vertex area (from M).",
   "<b>W<sub>r</sub>[c, o]</b> are learned numbers: for every smoothing time r, how much input channel c "
   "contributes to output channel o. So each layer learns its own mixture of 12 smoothing scales.",
   "<b>Fixed vs learned:</b> the eigenpairs and the 12 times are fixed, computed from the mesh. Only W and "
   "\u03c6 are learned."])
H2("3.3 The full network")
P("Four layers are stacked: channels go 4 \u2192 8 \u2192 8 \u2192 8 \u2192 1, with a LeakyReLU between layers and "
  "none after the last. The 4 input channels are the field value plus the vertex's x, y, z coordinates. "
  "Total: <b>4,860 parameters</b>.")
table([
    ["Layer", "Channels", "W (12 \u00d7 C<sub>in</sub> \u00d7 C<sub>out</sub>)", "\u03c6 network", "Total"],
    ["1", "4 \u2192 8", "384", "8\u00b732+32 + 32\u00b74+4 = 420", "804"],
    ["2", "8 \u2192 8", "768", "16\u00b732+32 + 32\u00b78+8 = 808", "1,576"],
    ["3", "8 \u2192 8", "768", "808", "1,576"],
    ["4", "8 \u2192 1", "96", "808", "904"],
    ["", "", "", "<b>Total</b>", "<b>4,860</b>"],
], [16, 22, 42, 62, 28])
H2("3.4 Properties worth stating in the talk")
B(["<b>Works on any mesh.</b> Nothing assumes a grid, a sphere or a torus; give it a mesh and it computes "
   "the eigenpairs.",
   "<b>Vertex order does not matter</b> (by construction: renumbering the vertices renumbers everything consistently). "
   "Grid models depend entirely on the order.",
   "<b>Output lies in the span of the 96 kept eigenvectors.</b> Patterns finer than \u03bb<sub>95</sub> cannot be "
   "produced. This gives a computable error floor (section 4.4).",
   "<b>Each layer's linear part is a symmetric smoothing operator.</b> Heat kernels only damp patterns; "
   "they cannot shift or rotate them. \u03c6 is the only part that breaks this.",
   "<b>Cost is O(V\u00b2).</b> \u03c6 is evaluated for every pair of vertices: 642\u00b2 \u2248 412,000 pairs per sample per "
   "layer. This is why training takes ~22\u201355 min per task versus under 2 min for the baselines, and why we "
   "accumulate gradients over micro-batches (section 5)."])
# ================================================================================
# 4. Domains and data
# ================================================================================
H1("4. Domains, PDEs and data")
H2("4.1 The two surfaces")
table([
    ["Domain", "Mesh", "Vertices", "Why this domain"],
    ["Sphere", "icosphere (trimesh), 3 subdivisions, radius 1", "642",
     "Curved, closed, no boundary; the classic case where flat grids fail (poles). Exact eigenvalues known."],
    ["Torus", "trimesh torus, R = 1, r = 0.3, 40 \u00d7 16 sections", "640",
     "Different topology (a hole); two very different length scales (ring vs tube)."],
], [18, 52, 18, 82], align_right_from=99)
P("Both have about 640 vertices so their results are directly comparable. Both are closed surfaces: there is "
  "no boundary, so no boundary conditions are needed (unlike the 1D interval, which had u = 0 at both ends).")
H2("4.2 The three PDEs")
P("The PDE code (equations.py) is unchanged from the 1D benchmark. Every equation only uses "
  "<font name='Mono'>domain.laplacian</font>, so moving it to a surface just means passing a MeshDomain. "
  "Each is solved with scipy's RK45 (method of lines).")
table([
    ["PDE", "Equation", "Type", "Parameters (both surfaces)", "Time T"],
    ["Diffusion", "\u2202<sub>t</sub>u = D \u0394u", "linear, smoothing", "D = 0.1", "1.0"],
    ["Reaction\u2013diffusion", "\u2202<sub>t</sub>u = D \u0394u + k u\u00b2", "nonlinear", "D = 0.1, k = 0.5", "1.0"],
    ["Beam / plate", "\u03c1A \u2202<sub>t</sub>\u00b2w = \u2212EI \u0394\u00b2w,  \u2202<sub>t</sub>w(0) = 0",
     "wave-like, oscillating, no damping", "EI = 0.04, \u03c1A = 1", "1.0"],
], [28, 50, 34, 40, 18], align_right_from=99)
box("Why the physical parameters differ from the 1D benchmark", [
    "On the unit interval the relevant eigenvalues are (k\u03c0)\u00b2 \u2248 10\u2013250; on the surfaces the starting "
    "patterns have \u03bb \u2248 1\u201312. With the 1D values (D = 0.02, EI = 10<super>\u22124</super>, T<sub>beam</sub> = 0.05) the fields "
    "would hardly change and the task would be trivial. We rescaled so that every task changes the field "
    "substantially and checked it: the relative change \u2016u(T) \u2212 u(0)\u2016/\u2016u(0)\u2016 is 0.54\u20131.45 for every task "
    "(table 4). A first attempt used smaller torus values (D = 0.02, EI = 0.002); the torus fields moved only "
    "13\u201319%, and since the two surfaces have similar low spectra we set both to the same values.",
])
H2("4.3 Two kinds of initial condition")
P("<b>Smooth (IC_TYPE = lowmodes).</b> A random combination of eigenvectors 1\u201315 (constant excluded), with "
  "standard-normal coefficients, scaled to max |u| = 1. This is the surface version of the 1D benchmark's "
  "\u201csum of the first few sine waves\u201d. Caveat: these fields lie entirely inside our model's basis.")
P("<b>Rough (IC_TYPE = noise).</b> Independent random values at every vertex, smoothed by a short heat flow "
  "e<super>0.02\u00b7\u0394</super>, mean removed, scaled to max |u| = 1 (the same recipe as main.ipynb). About 21\u201323% "
  "of each field lies <i>outside</i> our 96 modes. We picked the smoothing time 0.02 deliberately: at 0.01 "
  "the fields are so rough that even the 642-vertex mesh cannot represent them well, and at 0.03 only ~9% "
  "lies outside our basis, which would not test anything.")
fig("initial_conditions.png",
    "<b>Figure 4.</b> One smooth and one rough initial condition on the sphere, and their coefficients in our "
    "96 eigenvectors. The smooth field is exactly zero beyond mode 15; the rough field has energy in every mode "
    "and beyond.")
P("Each dataset has 128 samples: 96 for training and 32 for testing, with seed 0. Both kinds of data are "
  "generated by the same notebook (benchmark_manifolds_v2.ipynb) so the comparison is like for like.")
H2("4.4 How hard is each task? (measured)")
rows = [["Domain / PDE", "change", "u(0) outside 96 modes", "u(T) outside 96 modes", "SFNO grid floor",
         "change", "u(0) outside", "u(T) outside", "SFNO floor"]]
for t in TASKS:
    a, b = smooth["data_stats"][t], rough["data_stats"][t]
    rows.append([t.replace("/", " / "),
                 f"{a['change']:.2f}", f"{a['u0_outside_basis']:.4f}", f"{a['uT_outside_basis']:.4f}",
                 f"{a['sfno_floor']:.4f}" if "sfno_floor" in a else "\u2013",
                 f"{b['change']:.2f}", f"{b['u0_outside_basis']:.3f}", f"{b['uT_outside_basis']:.4f}",
                 f"{b['sfno_floor']:.4f}" if "sfno_floor" in b else "\u2013"])
story.append(Paragraph("<b>Table 4.</b> Left four numbers: smooth data. Right four: rough data. "
                       "\u201cchange\u201d = \u2016u(T)\u2212u(0)\u2016/\u2016u(0)\u2016. \u201coutside 96 modes\u201d = relative size of the part our basis "
                       "cannot represent. \u201cSFNO floor\u201d = error from mapping mesh \u2192 SFNO grid \u2192 mesh with no network.",
                       ss["cap"]))
table(rows, [38, 14, 20, 20, 18, 14, 16, 16, 14], font_size=7.4)
P("The key line is the beam with rough data: <b>31% (sphere) and 22% (torus) of the target lies outside our "
  "basis</b>, so our error there cannot go below those numbers, whatever the training does. Diffusion and "
  "reaction\u2013diffusion destroy fine patterns, so their targets are 99.98% inside the basis even when the input "
  "is rough; there the difficulty is only on the input side. The beam does not damp anything, so fine "
  "patterns survive to time T.")
# ================================================================================
# 5. Baselines and protocol
# ================================================================================
H1("5. Baselines and training protocol")
H2("5.1 The five baselines")
table([
    ["Model", "Source", "How it sees the surface", "Params"],
    ["<b>FNO</b>", "Li et al., ICLR 2021 (official 1D code)",
     "As a 1D sequence of vertices. FFT along the sequence, 16 modes, width 64.", "287,617"],
    ["<b>CNO</b>", "Raonic et al., NeurIPS 2023 (1D)",
     "As a 1D sequence; U-Net-style convolutions.", "107,657"],
    ["<b>LNO</b>", "Cao et al., Nat. Mach. Intell. 2024 (1D)",
     "As a 1D sequence; pole\u2013residue (Laplace-domain) kernel.", "1,325"],
    ["<b>SFNO</b> (sphere only)", "Bonev et al., ICML 2023 (idea); ref_models/sfno.py",
     "FNO with the FFT replaced by the spherical harmonic transform on a 32 \u00d7 64 grid; one weight matrix per "
     "degree l \u2264 16 (rotation-equivariant). Width 64.", "304,129"],
    ["<b>FNO2d</b> (torus only)", "Li et al., official 2D code; ref_models/fno2d.py",
     "2D FFT on the torus's own 40 \u00d7 16 angle grid. The mesh vertices are exactly that grid, so "
     "no interpolation. Modes 12 \u00d7 9, width 32, no padding (the torus is periodic).", "893,537"],
    ["<b>Ours</b>", "model.py", "Laplace\u2013Beltrami eigenbasis of the mesh (section 3).", "4,860"],
], [28, 44, 78, 20], align_right_from=3)
H2("5.2 How the 1D baselines are given the vertices")
P("FNO, CNO and LNO need a 1D ordering. Trimesh's own icosphere order follows the subdivision hierarchy and "
  "jumps across the surface, so we "
  "sort the vertices into a <b>snake-like raster</b>: latitude bands on the sphere, rings around the big circle "
  "on the torus, alternating direction so consecutive vertices are neighbours (figure 5). On average one step "
  "in the sequence is 1.06 (sphere) and 0.74 (torus) mesh-edge lengths. This is the most favourable fair "
  "ordering we could give them, but neighbours on the surface can still be far apart in the sequence "
  "(between bands), and the FFT treats the sequence as periodic.")
fig("raster_order.png",
    "<b>Figure 5.</b> The vertex order the 1D baselines see (light = early in the sequence, dark = late). The "
    "orange line follows the first 120 steps: latitude rings on the sphere, tube rings on the torus.",
    width_mm=150)
H2("5.3 How SFNO and FNO2d are connected to the mesh")
B(["<b>FNO2d:</b> trimesh builds the torus on a 40 \u00d7 16 grid of angles, so the vertices are reshaped into "
   "that grid by a permutation (verified to be exact) and back again. Nothing is lost.",
   "<b>SFNO:</b> needs a latitude\u2013longitude (Gauss\u2013Legendre) grid. Mesh \u2192 grid: least-squares fit of "
   "spherical harmonics up to degree 19 at the vertices, evaluated on the grid. Grid \u2192 mesh: spherical "
   "harmonic expansion evaluated at the vertices. We first tried plain barycentric interpolation on the "
   "triangles, but it lost 1.6% even on very smooth fields, which would have handicapped SFNO unfairly; the "
   "least-squares fit loses 0.03\u20130.04% on smooth data (table 4). The harmonics are computed by our own "
   "recurrence (the cluster's scipy 1.18 no longer has sph_harm) and were checked: orthonormal to 1e-14, and "
   "they are eigenfunctions of our mesh Laplacian with eigenvalues 0, 2, 5.97, 11.83."])
H2("5.4 Training and evaluation (identical for every model)")
table([
    ["Setting", "Value"],
    ["Samples", "128 per task: 96 train, 32 test (seed 0)"],
    ["Optimiser", "Adam, learning rate 10<super>\u22123</super>, halved every 250 epochs"],
    ["Epochs / batch size", "1000 / 32 (3 optimiser steps per epoch)"],
    ["Loss", "mean squared error on normalised fields"],
    ["Normalisation", "inputs and targets standardised with the training set's mean and std (one scalar each); "
                      "xyz coordinates (scaled to [\u22121, 1]) appended as 3 input channels"],
    ["Metric", "relative L2 error on the 32 test samples: \u2016prediction \u2212 truth\u2016 / \u2016truth\u2016 (normalised units)"],
    ["Reported number", "error after the <b>final</b> epoch (not the best epoch, so no selection on the test set)"],
    ["Our model's memory", "gradients accumulated over micro-batches of 8 (v1) / 16 (v2) samples; the update is "
                           "mathematically identical to a full batch of 32. Peak GPU memory 6.8\u20136.9 GiB at 16."],
    ["Hardware", "1 GPU (24 GB), 6 CPU cores, 32 GB RAM per job; one job per data type"],
], [42, 128], align_right_from=99)
# ================================================================================
# 6. Timeline
# ================================================================================
H1("6. What we ran, what we changed, and why")
H2("6.1 Timeline of runs")
table([
    ["Run (job id)", "Date", "Notebook / settings", "Models", "Duration", "Status"],
    ["1D benchmark (3 runs)", "Aug 29", "benchmark.ipynb, interval [0, 1]", "Ours, CNO, FNO, LNO", "15–25 min each",
     "done; context only (appendix A)"],
    ["Manifold v1 (1936768)", "Oct 21", "benchmark_manifolds.ipynb, smooth ICs", "Ours, CNO, FNO, LNO", "5 h 33 min",
     "done; no plots saved"],
    ["v2 rough, run 1 (1938876)", "Oct 21\u201322", "v2, IC_TYPE = noise", "all 6", "5 h 30 min",
     "done; no plots saved"],
    ["v2 smooth, first try (1938875)", "Oct 21", "v2, lowmodes, FNO + SFNO + FNO2d", "\u2013", "\u2013",
     "cancelled before starting"],
    ["v2 smooth (1969722)", "Oct 29\u201330", "v2, IC_TYPE = lowmodes", "all 6", "5 h 50 min", "done; curves saved"],
    ["v2 rough, run 2 (1969723)", "Oct 29\u201330", "v2, IC_TYPE = noise", "all 6", "2 h 27 min", "done; curves saved"],
], [36, 18, 44, 30, 18, 24], align_right_from=99, font_size=7.6)
H2("6.2 Every change we made, and the reason")
table([
    ["Change", "Why"],
    ["PDEs moved from the interval to MeshDomain sphere and torus", "Test the model where its geometric basis should matter."],
    ["Eigenpairs from the generalised problem S\u03c6 = \u03bbM\u03c6, M-orthonormal",
     "Correct inner product on an uneven mesh; HeatKernel's projection assumes \u03a6<super>\u22a4</super>M\u03a6 = I."],
    ["Diffusion times start from the first non-zero eigenvalue",
     "Closed surfaces have \u03bb<sub>0</sub> = 0; the 1D rule 3/\u03bb<sub>0</sub> would divide by zero."],
    ["Physical parameters rescaled (D = 0.1, EI = 0.04, T = 1)", "Surface eigenvalues are ~10\u00d7 smaller; otherwise the fields barely move (section 4.2)."],
    ["Smooth ICs from 15 lowest eigenvectors", "Surface analogue of the 1D sine-mode ICs."],
    ["xyz coordinates as input channels", "Surface analogue of the 1D coordinate channel, given to every model."],
    ["Raster vertex ordering for the 1D baselines", "Give sequence models the most neighbour-preserving order possible."],
    ["Gradient accumulation for our model (micro-batch 8, then 16)", "O(V\u00b2) memory; identical update to batch 32."],
    ["Out-of-memory guard (records NaN, continues)", "One failure should not lose a 5-hour run."],
    ["v2: added SFNO and FNO2d", "v1 only beat models that cannot see the geometry; a fair test needs ones that can."],
    ["v2: added rough ICs", "Smooth ICs lie exactly in our basis, which flatters our model."],
    ["SFNO mesh\u2192grid map: barycentric \u2192 least-squares spherical harmonics", "Barycentric lost 1.6% on smooth fields, unfair to SFNO."],
    ["Per-model peak-memory reset", "v1 reported a run-wide maximum (3.4 GiB for every model), which was misleading."],
    ["Removed MPLBACKEND=Agg from run_benchmark.sbatch; save histories + PNGs",
     "Every run before Oct 29 lost its plots (Agg stops Jupyter from capturing them) and no per-epoch losses were stored."],
    ["run_benchmark.sbatch takes NOTEBOOK as a parameter", "Same script runs any notebook; output names carry the notebook name."],
], [80, 90], align_right_from=99, font_size=7.6)
story.append(PageBreak())

# ================================================================================
# 8. Results
# ================================================================================
H1("7. Results")
P("All numbers are test relative L2 errors after epoch 1000 (lower is better). The best model in each row is "
  "highlighted. \u201cSFNO / FNO2d\u201d is the geometry-aware baseline: SFNO on the sphere, FNO2d on the torus.")
fig("results_dotplot.png",
    "<b>Figure 6.</b> All final test errors on a log scale (each step of the grid is a factor of 10). Left: smooth "
    "inputs (run 1969722). Right: rough inputs (run 1969723). Dotted lines: our model's error floor from basis "
    "truncation (only non-negligible for the beam with rough data).")
H2("7.1 Smooth initial conditions (run 1969722)")
results_table(smooth)
P("SFNO and FNO2d are the best on every task, 3\u201319\u00d7 more accurate than ours (SFNO 0.0011 vs our 0.0203 on "
  "sphere diffusion). Ours is second everywhere, 4\u20137\u00d7 better than FNO and far ahead of CNO and LNO.")
H2("7.2 Rough initial conditions (run 1969723)")
results_table(rough, note_floor=True)
P("\u2020 Our floor = part of the target outside our 96 modes. The 1D baselines collapse (FNO goes from 0.08 to "
  "0.76 on sphere diffusion). Ours barely changes on diffusion and reaction\u2013diffusion (0.020 \u2192 0.024 on "
  "sphere diffusion), which shows that a rough input is not what limits it. The geometry-aware models lose "
  "more (SFNO 0.0011 \u2192 0.0067) but still win 5 of 6. Ours wins torus reaction\u2013diffusion (0.026 vs 0.035); with "
  "a single seed this 1.3\u00d7 gap should not be claimed as significant. On the beam ours (0.56/0.59) is well "
  "above its floor (0.31/0.22), so truncation explains only part of the error.")
H2("7.3 The first manifold run (v1, 1936768): the comparison without geometry-aware baselines")
rows = [["Domain", "PDE", "Ours", "FNO", "CNO", "LNO", "Ours vs best baseline"]]
hl = set()
for i, t in enumerate(TASKS):
    d, s = t.split("/")
    vals = [v1[t][m]["final_error"] for m in ("Ours", "FNO", "CNO", "LNO")]
    hl.add((i + 1, 2 + int(np.argmin(vals))))
    rows.append([d, s] + [f"{v:.4f}" for v in vals] + [f"{min(vals[1:]) / vals[0]:.1f}\u00d7 better"])
table(rows, [16, 32, 20, 20, 20, 20, 42], highlight=hl, align_right_from=2)
P("Taken alone this looks like a clear win (3\u20136\u00d7 better than the best of these baselines on every task). It is a fair result against these baselines, but those "
  "baselines cannot see the surface. Present it as \u201cgeometry matters\u201d, not as \u201cbetter than the state of the art\u201d.")
H2("7.4 Are the runs reproducible?")
rows = [["Domain / PDE", "Ours: v1 vs v2 (smooth)", "FNO: v1 vs v2 (smooth)", "Ours: rough run 1 vs 2",
         "SFNO/FNO2d: rough run 1 vs 2"]]
for t in TASKS:
    rows.append([t.replace("/", " / "),
                 f"{v1[t]['Ours']['final_error']:.4f} / {e(smooth, t, 'Ours'):.4f}",
                 f"{v1[t]['FNO']['final_error']:.4f} / {e(smooth, t, 'FNO'):.4f}",
                 f"{e(rough_run1, t, 'Ours'):.4f} / {e(rough, t, 'Ours'):.4f}",
                 f"{e(rough_run1, t, 'SFNO / FNO2d'):.4f} / {e(rough, t, 'SFNO / FNO2d'):.4f}"])
table(rows, [40, 33, 33, 32, 32], align_right_from=1, font_size=7.6)
P("The two rough-data runs agree to 3\u20134 decimals (only GPU non-determinism differs). Between v1 and v2 on "
  "smooth data, FNO agrees to 1–4% on the sphere (0.0757 vs 0.0764 on diffusion), so the datasets are "
  "effectively the same. On the torus FNO's test error rises during training (overfitting, figure 7), so its "
  "final value is more sensitive (0.131 vs 0.164 on reaction–diffusion). Ours agrees to within ~10% "
  "(e.g. 0.0228 vs 0.0203); v2 likely differs because it used micro-batches of 16 instead of 8, which "
  "changes the floating-point summation order of a model that is still improving. <b>Not tested:</b> different random seeds "
  "(initialisation and data split). The large gaps (3\u201319\u00d7) are far beyond any plausible seed effect; "
  "the small ones (e.g. 1.3\u00d7 on torus reaction\u2013diffusion) are not.")
_cost_start = len(story)
H2("7.5 Cost")
rows = [["Model", "Params", "Time / task, smooth run", "Time / task, rough run 2"]]
for m, key in (("Ours", "Ours"), ("SFNO", "SFNO"), ("FNO2d", "FNO2d"), ("FNO", "FNO"), ("CNO", "CNO"), ("LNO", "LNO")):
    ts = [smooth["results"][t][key]["seconds"] for t in TASKS if key in smooth["results"][t]]
    tr = [rough["results"][t][key]["seconds"] for t in TASKS if key in rough["results"][t]]
    p = [smooth["results"][t][key]["n_parameters"] for t in TASKS if key in smooth["results"][t]][0]
    rows.append([m, f"{p:,}", f"{np.mean(ts) / 60:.1f} min", f"{np.mean(tr) / 60:.1f} min"])
table(rows, [40, 35, 48, 47], align_right_from=1)
_k = story[_cost_start:]
del story[_cost_start:]
story.append(KeepTogether(_k))
P("Our model is the smallest except LNO but by far the slowest, because of the all-pairs \u03c6 network (section 3.4). "
  "The rough run happened on a faster node, which is why both columns differ for every model.")
# ================================================================================
# 9. Loss curves
# ================================================================================
H1("8. Learning curves")
P("These curves were recorded by the October 29\u201330 reruns (earlier runs did not save them; section 6.2).")
fig("curves_test_smooth.png",
    "<b>Figure 7.</b> Test error during training, smooth inputs. Ours (blue) is still going down at epoch 1000 on "
    "every task. SFNO/FNO2d (orange) drop fastest and lowest. LNO and CNO stall early.")
fig("curves_test_rough.png",
    "<b>Figure 8.</b> Test error during training, rough inputs. FNO (green) gets <i>worse</i> as training goes on "
    "(overfitting). On the beam, FNO2d reaches its best around epoch 80 and then also overfits. Dotted line: our "
    "basis floor for the beam.")
fig("train_vs_test.png",
    "<b>Figure 9.</b> Training error (faint) vs test error (bold) for two representative tasks. For ours the two "
    "lines lie on top of each other: no overfitting. For FNO, and for FNO2d on the rough beam, training error "
    "keeps falling while test error stays high or rises.")
H2("8.1 What the curves tell us")
rows = [["Task", "Ours test @250 / 500 / 1000", "Ours train \u2248 test?", "Ours last 100 epochs",
         "Geometry-aware @1000 (train / test)"]]
for run, lab in ((smooth, "smooth"), (rough, "rough")):
    for t in ("Sphere/Diffusion", "Torus/Reaction-Diffusion", "Sphere/Beam"):
        h = run["histories"][t]["Ours"]
        te = np.array(h["test_error"]); tr = np.sqrt(np.array(h["train_loss"]))
        g = run["histories"][t][geo(t)]
        rows.append([f"{t.replace('/', ' / ')} ({lab})",
                     f"{te[249]:.4f} / {te[499]:.4f} / {te[-1]:.4f}",
                     f"{tr[-1]:.4f} vs {te[-1]:.4f}",
                     f"{(te[-1] - te[-100]) / te[-100] * 100:+.1f}%",
                     f"{np.sqrt(g['train_loss'][-1]):.4f} / {g['test_error'][-1]:.4f}"])
table(rows, [46, 38, 30, 22, 34], align_right_from=1, font_size=7.4)
B(["<b>Ours is under-trained, not over-fitted.</b> Test error still falls 1\u20134% in the last 100 epochs on the "
   "diffusion-type tasks, and the training error (\u221aMSE, which approximates relative error because targets are "
   "normalised to unit variance) is almost the same as the test error. More epochs or a better learning-rate "
   "schedule would very likely lower our numbers; how much is untested.",
   "<b>The ~2% level is not a property of the data</b>: it is the same for smooth and rough inputs, while our "
   "basis floor on those tasks is 0.0001. The bottleneck is how fast the 4-layer \u03c6 stack learns.",
   "<b>SFNO generalises well</b> (train \u2248 test); FNO2d overfits on rough data (train 0.002, test 0.019\u20130.20).",
   "<b>1D baselines</b>: FNO and CNO overfit on rough data; LNO underfits everywhere (train \u2248 test \u2248 0.36\u20131.0)."])

# ================================================================================
# 10. Predictions
# ================================================================================
H1("9. What the predictions look like")
P("Each figure shows one test sample (the first of the 32) as the cluster run drew it: one row per PDE, ground "
  "truth on the left, then each model's prediction on the same colour scale. The small tables give that "
  "sample's error per model, since the labels inside the images are small in print.")
fig("predictions_v2_lowmodes_Sphere_1969722.jpg",
    "<b>Figure 10.</b> Sphere, smooth inputs. Ours and SFNO are visually indistinguishable from the truth; CNO "
    "and LNO get the large-scale pattern wrong, and LNO collapses to almost nothing on the beam.")
table([
    ["This sample", "Ours", "SFNO", "FNO", "CNO", "LNO"],
    ["Diffusion", "0.025", "0.001", "0.088", "0.504", "0.477"],
    ["Reaction–diffusion", "0.033", "0.002", "0.125", "0.490", "0.519"],
    ["Beam", "0.025", "0.002", "0.142", "1.011", "0.989"],
], [45, 25, 25, 25, 25, 25])
story.append(Spacer(1, 6))
fig("predictions_v2_noise_Torus_1969723.jpg",
    "<b>Figure 11.</b> Torus, rough inputs. On diffusion and reaction–diffusion ours and FNO2d match the truth. "
    "On the beam (bottom row) fine detail survives to time T: FNO2d captures most of it, ours produces only the "
    "smooth part (its 96-mode limit), and the 1D models produce noise.")
table([
    ["This sample", "Ours", "FNO2d", "FNO", "CNO", "LNO"],
    ["Diffusion", "0.025", "0.013", "0.639", "0.260", "0.676"],
    ["Reaction–diffusion", "0.024", "0.015", "0.571", "0.301", "0.667"],
    ["Beam", "0.591", "0.197", "1.384", "1.150", "0.975"],
], [45, 25, 25, 25, 25, 25])
# ================================================================================
# 11. Interpretation
# ================================================================================
H1("10. What the experiments show, and what they do not")
H2("10.1 Claims you can defend")
B(["<b>Geometry matters.</b> Every model that sees the surface (ours, SFNO, FNO2d) is far better than every model "
   "that sees a vertex sequence, on all 12 task/data combinations.",
   "<b>Our model learns PDE dynamics on curved surfaces with 4,860 parameters</b>, to 2\u20134% error on all "
   "diffusion-type tasks and on the smooth beam, on two different topologies.",
   "<b>It is robust to rough inputs on smoothing PDEs</b> (error essentially unchanged), and it does not overfit "
   "(train \u2248 test on every task), unlike FNO, CNO and FNO2d on rough data.",
   "<b>It is general.</b> It needs only a mesh. SFNO needs the domain to be a sphere; FNO2d needs the torus's "
   "special grid structure. Neither exists for an arbitrary surface such as a scanned object."])
H2("10.2 Claims you should not make")
B(["\u201cOur model beats state-of-the-art neural operators.\u201d It does not: SFNO and FNO2d win 11 of 12.",
   "\u201cOur model wins on torus reaction\u2013diffusion.\u201d The gap (0.026 vs 0.035) is small and single-seed.",
   "\u201cThe first manifold run (v1) shows we beat FNO/CNO/LNO, so we are best.\u201d Those baselines cannot see the geometry.",
   "\u201cOur model handles fine-scale dynamics.\u201d It cannot: its output is limited to 96 modes (beam with rough data)."])
H2("10.3 Why the geometry-aware baselines win (best current explanation)")
B(["<b>Capacity:</b> 60\u2013180\u00d7 more parameters, wide (32\u201364 channel) latent features, a lifting layer and a "
   "nonlinear decoder; ours has 8 channels and no encoder/decoder.",
   "<b>Architecture:</b> FNO-style layers have a pointwise \u201cskip\u201d path (W\u00b7f) next to the spectral path and "
   "free weights per mode. Each of our layers is only an integral (no skip path), and its spectral weights are "
   "tied to 12 decaying exponentials.",
   "<b>Training:</b> everyone's smooth-data test error is still falling slowly at epoch 1000 (ours by 1–4% "
   "per 100 epochs, SFNO by up to 5%), but theirs is falling from a level 3–19× lower. Longer training "
   "would help both; whether it closes the gap is untested.",
   "<b>Data favour them a little:</b> on smooth data the targets are low-degree spherical harmonics, which is "
   "exactly SFNO's basis; the torus fields are naturally periodic on FNO2d's grid."])
H2("10.4 Limitations of the study")
B(["Single seed (0) for initialisation and data split; no error bars.",
   "One mesh resolution per surface (~640 vertices); no test of transfer to finer/coarser meshes.",
   "Hyperparameters taken from the 1D benchmark, not tuned for any model on the surfaces.",
   "Only two surfaces, both with special structure that a specialised baseline can exploit.",
   "128 samples per task is small; 1000 epochs is not enough for our model to converge."])
# ================================================================================
# 12. Q&A
# ================================================================================
H1("11. Questions to expect, with answers")
QA = [
    ("Why not just use FNO on the vertices?",
     "We did: it fails (0.08\u20130.18 on smooth data, 0.6\u20132.1 on rough data), because the FFT assumes equally spaced, "
     "periodic 1D data and surface neighbours are not sequence neighbours. Even with the best ordering we could build, "
     "it cannot see the surface."),
    ("Your v1 baselines were unfair, weren't they?",
     "Yes, as a test of \u201cbest model\u201d they were, which is why v2 adds SFNO (sphere) and FNO2d (torus), both "
     "geometry-aware, with the same data and protocol. We report both; v1 supports \u201cgeometry matters\u201d, v2 "
     "shows where we stand against specialised models."),
    ("Then why is your model interesting if SFNO wins?",
     "SFNO only exists for the sphere and FNO2d only for grids like the torus's. Ours runs unchanged on any "
     "triangle mesh, is 60\u2013180\u00d7 smaller, does not overfit, and has not finished training. The next experiment "
     "that decides its value is an irregular surface where no specialised baseline exists."),
    ("Did you give SFNO a fair interface to the mesh?",
     "Yes. We measured the loss from moving data between mesh and grid with no network: 0.03\u20130.04% on smooth "
     "data, 2.6% only on the rough beam (table 4). Our first, barycentric, version lost 1.6% and we replaced it "
     "because that would have been unfair to SFNO."),
    ("Don't the smooth initial conditions favour your model?",
     "Yes, they lie exactly in our basis. That is why we added rough ones with 21\u201323% of their size outside "
     "it. Our error on diffusion-type tasks did not change; on the beam it got much worse, for a reason we can "
     "quantify (the target's 22\u201331% outside our basis)."),
    ("Why did you change the PDE parameters from the 1D benchmark?",
     "The surfaces' eigenvalues are about ten times smaller than on the unit interval, so the 1D values would "
     "barely change the fields. We chose values that change every field by 54\u2013145% and printed that check in "
     "every run."),
    ("Why is your model so slow?",
     "\u03c6 is a small network evaluated on every pair of vertices, which costs O(V\u00b2): about 412,000 pairs per "
     "sample per layer at 642 vertices. A cheaper variant would apply \u03c6 per vertex or on mesh neighbours only."),
    ("Is the 2% error a hard limit of the method?",
     "No evidence of that. The basis can represent the targets to 0.01%, train and test error are equal, and the "
     "test error was still falling at epoch 1000. The limit is training speed of the current architecture."),
    ("How do I know the results are not a fluke?",
     "Two independent rough-data runs agree to 3\u20134 decimals, and FNO's smooth-data errors on the sphere match the "
     "earlier v1 run to 1\u20134%. What we have not done is vary the random seed; the large gaps (3\u201319\u00d7) are far beyond "
     "seed noise, the small ones are not."),
    ("What does \u201crelative L2 error\u201d mean here?",
     "\u2016prediction \u2212 truth\u2016 / \u2016truth\u2016 over all 32 test samples, after normalising fields with training "
     "statistics. 0.02 means the error is 2% of the size of the true field. We report the last epoch, not the best "
     "one, so the test set is never used to pick a model."),
    ("Why are the eigenvalues on the sphere slightly below l(l+1)?",
     "Discretisation: 642 vertices resolve fine patterns less accurately, so higher eigenvalues come out low "
     "(about 79 vs 90 at l = 9). The low ones match closely (2.00, 5.97, 11.83 vs 2, 6, 12)."),
]
for q, a in QA:
    story.append(KeepTogether([Paragraph(q, ss["q"]), Paragraph(a, ss["a"])]))

# ================================================================================
# 13. Next steps
# ================================================================================
H1("12. Next steps (in order of value)")
B(["<b>Train ours longer</b> (e.g. 3000 epochs or a cosine schedule) on 2\u20133 tasks to find where it actually converges.",
   "<b>Ablation on Diffusion:</b> 1 layer with \u03c6 frozen at 1 vs 4 layers; with and without xyz inputs; with a "
   "pointwise skip path W\u00b7f added. In main.ipynb a single frozen block reached 0.2–0.6% on sphere heat flow.",
   "<b>Parameter-matched baselines:</b> SFNO/FNO2d shrunk to ~5k parameters (e.g. width 8), to separate capacity from architecture.",
   "<b>An irregular mesh</b> (e.g. a scanned object): no SFNO/FNO2d exists there; this is where our model's generality would show.",
   "<b>Seeds:</b> 3 seeds for the key rows, to get error bars.",
   "<b>Beam:</b> more modes (192+) to lower the basis floor, tested on one task first because cost grows with V\u00b2.",
   "<b>Cheaper \u03c6:</b> evaluate it per vertex or on mesh neighbours instead of all pairs."])

# ================================================================================
# Appendix
# ================================================================================
story.append(PageBreak())
H1("Appendix A. Context: the 1D (Euclidean) benchmark")
P("Before the surfaces, the same four models were compared on the unit interval (benchmark.ipynb, run 1687359, "
  "4 layers, 96 modes, coordinate channel). There our basis is just the sine basis, so our model is a "
  "restricted FNO, and FNO wins every row. On the three ODEs the 128 points are time steps, not space, and "
  "our symmetric heat-kernel filters are a poor fit for forward-in-time dynamics.")
table([
    ["System", "Ours", "CNO", "FNO", "LNO"],
    ["Diffusion", "0.0245", "0.0481", "<b>0.0047</b>", "0.4739"],
    ["Reaction\u2013diffusion", "0.0347", "0.0587", "<b>0.0056</b>", "0.4249"],
    ["Beam", "0.0154", "0.0139", "<b>0.0014</b>", "0.0124"],
    ["Pendulum (ODE)", "0.1562", "0.0455", "<b>0.0074</b>", "0.4193"],
    ["Duffing (ODE)", "0.1361", "0.0618", "<b>0.0059</b>", "0.3636"],
    ["Lorenz (ODE)", "0.3998", "0.3489", "<b>0.1561</b>", "0.4235"],
], [50, 30, 30, 30, 30])
H1("Appendix B. Reproducing the experiments")
P("Files (d:\\Work\\Graphs\\code): <font name='Mono'>model.py</font> (our operator), "
  "<font name='Mono'>equations.py</font> (PDEs, MeshDomain), <font name='Mono'>ref_models/</font> "
  "(cno.py, fno.py, lno.py, sfno.py, fno2d.py), <font name='Mono'>benchmark_manifolds.ipynb</font> (v1), "
  "<font name='Mono'>benchmark_manifolds_v2.ipynb</font> (v2), <font name='Mono'>run_benchmark.sbatch</font>. "
  "Cluster folder: /storage/user/sany/graphs_benchmark. Results: cluster_results/results/.")
story.append(Paragraph(
    "ssh tum<br/>cd /storage/user/sany/graphs_benchmark<br/><br/>"
    "# smooth inputs, all six models<br/>"
    "sbatch --time=12:00:00 --export=ALL,NOTEBOOK=benchmark_manifolds_v2.ipynb,IC_TYPE=lowmodes run_benchmark.sbatch<br/><br/>"
    "# rough inputs, all six models<br/>"
    "sbatch --time=12:00:00 --export=ALL,NOTEBOOK=benchmark_manifolds_v2.ipynb,IC_TYPE=noise run_benchmark.sbatch<br/><br/>"
    "# options: MODELS=Ours,SFNO,...  OURS_MICRO_BATCH=8  BENCHMARK_SMOKE=1 (2-epoch test)<br/><br/>"
    "# fetch results (from PowerShell on Windows)<br/>"
    "scp \"tum:/storage/user/sany/graphs_benchmark/results/*\" cluster_results/results/", ss["code"]))
table([
    ["Hyperparameter", "Value"],
    ["Our model", "4 layers, latent 8 channels, \u03c6 hidden 32, 96 modes, 12 diffusion times, LeakyReLU between layers"],
    ["FNO / CNO / LNO", "modes 16 width 64 / 3 levels, 2 res-blocks, multiplier 16 / modes 16 width 4"],
    ["SFNO", "32 \u00d7 64 Gauss\u2013Legendre grid, degrees \u2264 16 weighted, width 64, harmonic fit degree 19"],
    ["FNO2d", "40 \u00d7 16 torus grid, modes 12 \u00d7 9, width 32, no padding"],
    ["Data", "128 samples (96/32), seed 0, smooth: modes 1\u201315; rough: noise smoothed with t = 0.02"],
    ["PDEs", "D = 0.1, k = 0.5, EI = 0.04, \u03c1A = 1, T = 1, RK45 (scipy defaults)"],
], [40, 130], align_right_from=99)


# ---- page templates ------------------------------------------------------------
class Doc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            text = flowable.getPlainText()
            if style == "h1":
                self.notify("TOCEntry", (0, text, self.page))
            elif style == "h2":
                self.notify("TOCEntry", (1, text, self.page))


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 287 * mm, "Learning PDE solution operators on curved surfaces")
    canvas.drawRightString(190 * mm, 287 * mm, "Graphs benchmark \u00b7 October 2026")
    canvas.setStrokeColor(RULE)
    canvas.line(20 * mm, 285 * mm, 190 * mm, 285 * mm)
    canvas.drawCentredString(105 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 287 * mm, 210 * mm, 10 * mm, stroke=0, fill=1)
    canvas.restoreState()


doc = Doc(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
          bottomMargin=16 * mm, title="Learning PDE solution operators on curved surfaces",
          author="Aniket", subject="Heat-kernel neural operator: manifold PDE benchmark")
frame = Frame(20 * mm, 16 * mm, 170 * mm, 265 * mm, id="f")
doc.addPageTemplates([PageTemplate("cover", [frame], onPage=on_cover),
                      PageTemplate("normal", [frame], onPage=on_page)])
doc.multiBuild(story)
print("wrote", OUT)
