"""Figures for the manifold-benchmark report."""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, expm_multiply

CODE = r"d:\Work\Graphs\code"
RES = os.path.join(CODE, "cluster_results", "results")
OUT = os.path.join(os.path.dirname(__file__), "fig")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CODE)
from equations import MeshDomain  # noqa: E402

# ---- style -----------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#ffffff"
SERIES = {  # validated order: blue, orange, aqua, yellow, magenta
    "Ours": "#2a78d6",
    "Geometry-aware": "#eb6834",
    "FNO": "#1baf7a",
    "CNO": "#eda100",
    "LNO": "#e87ba4",
}
MARKERS = {"Ours": "o", "Geometry-aware": "D", "FNO": "s", "CNO": "^", "LNO": "v"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


def geo_name(domain):
    return "SFNO" if domain == "Sphere" else "FNO2d"


# ---- domains and eigenpairs (same construction as the notebooks) -----------
def build(name):
    if name == "Sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    else:
        mesh = trimesh.creation.torus(1.0, 0.3, major_sections=40, minor_sections=16)
    return MeshDomain(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))


def eigen(dom, k=96):
    m = dom.mass_diagonal
    Mi = sp.diags(1 / np.sqrt(m))
    vals, vecs = eigsh((Mi @ dom.stiffness @ Mi).tocsc(), k=k, sigma=-1e-6, which="LM")
    o = np.argsort(vals)
    return np.maximum(vals[o], 0), np.asarray(Mi @ vecs[:, o]), m


doms = {n: build(n) for n in ("Sphere", "Torus")}
eig = {n: eigen(d) for n, d in doms.items()}

# F1: eigenvalue spectra ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
vals = eig["Sphere"][0]
exact = np.array([l * (l + 1) for l in range(12) for _ in range(2 * l + 1)])[:96]
ax = axes[0]
ax.plot(np.arange(96), exact, color=MUTED, lw=1.2, ls="--", label="exact l(l+1)")
ax.plot(np.arange(96), vals, "o", ms=3, color=SERIES["Ours"], label="mesh (642 vertices)")
ax.set_title("Sphere: Laplace-Beltrami eigenvalues")
ax.set_xlabel("mode index k")
ax.set_ylabel("eigenvalue  \u03bb\u2096")
ax.legend(loc="upper left")
ax.annotate("multiplets of size 2l+1\n(1, 3, 5, 7, ...)", xy=(30, vals[30]), xytext=(45, 18),
            fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax = axes[1]
tv = eig["Torus"][0]
ax.plot(np.arange(96), tv, "o", ms=3, color=SERIES["Ours"])
ax.set_title("Torus (R = 1, r = 0.3): eigenvalues")
ax.set_xlabel("mode index k")
ax.set_ylabel("eigenvalue  \u03bb\u2096")
ax.axhline(1 / 0.3 ** 2, color=MUTED, lw=0.8, ls=":")
ax.text(2, 1 / 0.3 ** 2 + 2, "first tube mode ~ 1/r\u00b2 \u2248 11", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "eigenvalues.png")

# F2: eigenvectors on the surfaces -------------------------------------------
def surface_scatter(ax, pts, val, title, s=7):
    lim = np.abs(val).max()
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=val, cmap="RdBu_r", vmin=-lim, vmax=lim,
               s=s, linewidths=0)
    ax.set_box_aspect(np.ptp(pts, axis=0))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, fontweight="normal", color=INK2, pad=0)


fig = plt.figure(figsize=(7.4, 3.3))
picks = {"Sphere": [1, 4, 9, 30], "Torus": [1, 3, 9, 30]}
for row, name in enumerate(("Sphere", "Torus")):
    vals, Phi, _ = eig[name]
    pts = doms[name].points
    for col, k in enumerate(picks[name]):
        ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
        if name == "Torus":
            ax.view_init(elev=35, azim=30)
        surface_scatter(ax, pts, Phi[:, k], f"{name}  k={k}  \u03bb={vals[k]:.1f}")
fig.subplots_adjust(wspace=0.0, hspace=0.15)
save(fig, "eigenvectors.png")

# F3: heat-kernel filters -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, name in zip(axes, ("Sphere", "Torus")):
    vals = eig[name][0]
    first = vals[vals > 1e-8][0]
    times = np.logspace(np.log10(1 / vals[-1]), np.log10(3 / first), 12)
    lam = np.linspace(0, vals[-1], 400)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for i, t in enumerate(times):
        ax.plot(lam, np.exp(-t * lam), color=cmap(0.15 + 0.85 * i / 11), lw=1.4)
    ax.set_title(f"{name}: the 12 heat filters e^(-t\u00b7\u03bb)")
    ax.set_xlabel("eigenvalue \u03bb")
    ax.set_ylabel("filter gain")
    ax.text(vals[-1] * 0.55, 0.85, f"t from {times[0]:.4f}\n(light) to {times[-1]:.2f}\n(dark)",
            fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "heat_filters.png")

# F4: vertex ordering seen by the 1D baselines -----------------------------------
def snake(band, angle):
    key = np.where(band % 2 == 0, angle, -angle)
    return np.lexsort((key, band))


def sphere_order(v):
    polar = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
    az = np.arctan2(v[:, 1], v[:, 0])
    nb = int(round(np.sqrt(len(v))))
    return snake(np.minimum((polar / np.pi * nb).astype(int), nb - 1), az)


def torus_order(v):
    major = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
    minor = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]) - 1.0)
    return snake(np.round(major / (2 * np.pi) * 40).astype(int) % 40, minor)


fig = plt.figure(figsize=(7.4, 3.0))
for i, (name, fn) in enumerate((("Sphere", sphere_order), ("Torus", torus_order))):
    pts = doms[name].points
    order = fn(pts)
    seq = pts[order]
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    if name == "Torus":
        ax.view_init(elev=40, azim=30)
    rank = np.empty(len(order)); rank[order] = np.arange(len(order))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rank, cmap=cmap, s=6, linewidths=0)
    ax.plot(seq[:120, 0], seq[:120, 1], seq[:120, 2], color=SERIES["Geometry-aware"], lw=0.9)
    ax.set_box_aspect(np.ptp(pts, axis=0))
    ax.set_axis_off()
    ax.set_title(name, fontsize=9, color=INK2)
save(fig, "raster_order.png")

# F5: the two kinds of initial condition -------------------------------------------
rng = np.random.default_rng(3)
fig = plt.figure(figsize=(7.4, 2.7))
dom = doms["Sphere"]; vals, Phi, m = eig["Sphere"]
low = Phi[:, 1:16] @ rng.normal(size=15); low /= np.abs(low).max()
noise = expm_multiply(0.02 * dom.laplacian, rng.normal(size=dom.n_points))
noise -= (m @ noise) / m.sum(); noise /= np.abs(noise).max()
for i, (u, t) in enumerate(((low, "smooth: 15 lowest modes"), (noise, "rough: smoothed white noise"))):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    surface_scatter(ax, dom.points, u, t, s=9)
ax = fig.add_subplot(1, 3, 3)
for u, lab, col in ((low, "smooth", SERIES["Ours"]), (noise, "rough", SERIES["Geometry-aware"])):
    c = Phi.T @ (m * u)
    ax.semilogy(np.arange(96), np.abs(c) + 1e-6, "o", ms=2.5, color=col, label=lab)
ax.set_title("coefficients in our 96 modes", fontsize=9)
ax.text(20, 4e-6, "smooth: exactly 0 above mode 15", fontsize=7, color=INK2)
ax.set_xlabel("mode index k")
ax.set_ylim(1e-6, 10)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45))
fig.tight_layout()
save(fig, "initial_conditions.png")

# ---- results ----------------------------------------------------------------------
def load(fn):
    return json.load(open(os.path.join(RES, fn)))


runs = {
    "smooth": load("manifold_v2_lowmodes_results_1969722.json"),
    "rough": load("manifold_v2_noise_results_1969723.json"),
}
TASKS = [f"{d}/{s}" for d in ("Sphere", "Torus") for s in ("Diffusion", "Reaction-Diffusion", "Beam")]
ORDER = ["Ours", "Geometry-aware", "FNO", "CNO", "LNO"]


def err(run, task, model):
    d = task.split("/")[0]
    key = geo_name(d) if model == "Geometry-aware" else model
    return runs[run]["results"][task][key]["final_error"]


# F6: dot plot of final test errors ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
for ax, run in zip(axes, ("smooth", "rough")):
    for j, model in enumerate(ORDER):
        y = np.arange(len(TASKS)) + (j - 2) * 0.13
        x = [err(run, t, model) for t in TASKS]
        ax.scatter(x, y, s=34, marker=MARKERS[model], color=SERIES[model],
                   edgecolor=SURFACE, linewidth=1.0, zorder=3,
                   label=model if model != "Geometry-aware" else "SFNO (sphere) / FNO2d (torus)")
    if run == "rough":
        for i, t in enumerate(TASKS):
            floor = runs[run]["data_stats"][t]["uT_outside_basis"]
            if floor > 0.01:
                ax.plot([floor, floor], [i - 0.4, i + 0.4], color=INK2, lw=1.2, ls=":")
                ax.text(floor * 0.92, i + 0.47, "our floor", fontsize=6.5, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 3)
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels([t.replace("/", " / ").replace("Reaction-Diffusion", "Reaction-Diff.") for t in TASKS])
    if run == "smooth":
        ax.invert_yaxis()  # shared y-axis: invert once, not per panel
    ax.set_title("Smooth initial conditions" if run == "smooth" else "Rough initial conditions")
    ax.grid(axis="y", visible=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.supxlabel("test relative L2 error (log scale, lower is better)", fontsize=9, color=INK2, y=0.07)
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "results_dotplot.png")

# F7: learning curves (test error) per run and domain --------------------------------
for run in ("smooth", "rough"):
    H = runs[run]["histories"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    for r, d in enumerate(("Sphere", "Torus")):
        for c, s in enumerate(("Diffusion", "Reaction-Diffusion", "Beam")):
            ax = axes[r, c]
            hs = H[f"{d}/{s}"]
            for model in ORDER:
                key = geo_name(d) if model == "Geometry-aware" else model
                h = hs[key]
                ax.semilogy(np.arange(1, len(h["test_error"]) + 1), h["test_error"],
                            color=SERIES[model], lw=1.4,
                            label=model if model != "Geometry-aware" else "SFNO / FNO2d")
            if run == "rough":
                floor = runs[run]["data_stats"][f"{d}/{s}"]["uT_outside_basis"]
                if floor > 0.01:
                    ax.axhline(floor, color=INK2, lw=1.0, ls=":")
            ax.set_title(f"{d} / {s}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("test rel. L2")
            if r == 1:
                ax.set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"curves_test_{run}.png")

# F8: train vs test for Ours and the geometry-aware baseline (generalisation gap) ----
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
for ax, run in zip(axes, ("smooth", "rough")):
    H = runs[run]["histories"]
    for task, ls in (("Sphere/Diffusion", "-"), ("Torus/Beam", "--")):
        for model in ("Ours", "Geometry-aware", "FNO"):
            key = geo_name(task.split("/")[0]) if model == "Geometry-aware" else model
            h = H[task][key]
            ep = np.arange(1, len(h["train_loss"]) + 1)
            ax.semilogy(ep, np.sqrt(h["train_loss"]), color=SERIES[model], lw=1.0, ls=ls, alpha=0.55)
            ax.semilogy(ep, h["test_error"], color=SERIES[model], lw=1.6, ls=ls)
    ax.set_title(("Smooth" if run == "smooth" else "Rough") + " ICs: train (faint) vs test (bold)", fontsize=9)
    ax.set_xlabel("epoch")
axes[0].set_ylabel("relative error")
from matplotlib.lines import Line2D  # noqa: E402
leg = [Line2D([], [], color=SERIES[m], lw=1.6, label=l) for m, l in
       (("Ours", "Ours"), ("Geometry-aware", "SFNO / FNO2d"), ("FNO", "FNO (1D)"))]
leg += [Line2D([], [], color=INK2, lw=1.2, ls="-", label="Sphere / Diffusion"),
        Line2D([], [], color=INK2, lw=1.2, ls="--", label="Torus / Beam")]
fig.legend(handles=leg, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "train_vs_test.png")

# F9: downscale the cluster prediction figures -------------------------------------
from PIL import Image  # noqa: E402
for f in ("predictions_v2_lowmodes_Sphere_1969722.png", "predictions_v2_noise_Torus_1969723.png",
          "predictions_v2_lowmodes_Torus_1969722.png", "predictions_v2_noise_Sphere_1969723.png"):
    im = Image.open(os.path.join(RES, f)).convert("RGB")
    # squeeze out the large white gaps between rows/columns so labels stay legible
    a = np.asarray(im)
    ink = (a < 245).any(axis=2)

    def bands(mask, gap):
        idx = np.flatnonzero(mask)
        keep = np.zeros_like(mask)
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                keep[max(start - 6, 0):prev + 7] = True
                start = i
            prev = i
        keep[max(start - 6, 0):prev + 7] = True
        return keep

    a = a[bands(ink.any(axis=1), 25)]
    cols = np.flatnonzero((a < 245).any(axis=2).any(axis=0))
    a = a[:, max(cols[0] - 8, 0):cols[-1] + 9]
    im = Image.fromarray(a)
    im.thumbnail((2200, 2200))
    im.save(os.path.join(OUT, f.replace(".png", ".jpg")), quality=88)
    print("saved", f, im.size)
