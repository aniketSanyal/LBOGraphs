"""Cut the cluster prediction grids into one image per (system, ground truth / model).

The notebook only saved the per-vertex predictions as rendered composites
(rows = systems, columns = ground truth + models), so we slice those.
"""
import glob
import json
import os
import re

import numpy as np
from PIL import Image

RES = r"d:\Work\Graphs\code\cluster_results\results"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig", "panels")
SYSTEMS = ["Diffusion", "Reaction-Diffusion", "Beam"]


def bands(mask, gap):
    """[start, stop) runs of True in mask, merging runs separated by < gap."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def columns_for(run_id, domain):
    js = glob.glob(os.path.join(RES, f"manifold_v2_*_results_{run_id}.json"))[0]
    d = json.load(open(js))
    present = d["results"][f"{domain}/{SYSTEMS[0]}"]
    return ["Ground truth"] + [m for m in d["models"] if m in present]


for f in sorted(glob.glob(os.path.join(RES, "predictions_v2_*.png"))):
    ic, domain, run_id = re.match(r"predictions_v2_(.+)_(Sphere|Torus)_(\d+)\.png",
                                  os.path.basename(f)).groups()
    names = columns_for(run_id, domain)
    a = np.asarray(Image.open(f).convert("RGB"))
    ink = (a < 245).any(axis=2)

    # vertical bands: suptitle, then (title, surface) pairs per system row
    rows = bands(ink.any(axis=1), 15)[1:]
    assert len(rows) == 2 * len(SYSTEMS), (f, rows)
    dest = os.path.join(OUT, f"{ic}_{run_id}", domain)
    os.makedirs(dest, exist_ok=True)
    for r, system in enumerate(SYSTEMS):
        y0, y1 = rows[2 * r][0], rows[2 * r + 1][1]
        strip = ink[y0:y1]
        cols = bands(strip.any(axis=0), 60)
        assert len(cols) == len(names), (f, system, cols)
        for (x0, x1), name in zip(cols, names):
            cell = a[max(y0 - 10, 0):y1 + 10, max(x0 - 10, 0):x1 + 10]
            fn = f"{system}_{name.replace(' ', '_')}.png"
            Image.fromarray(cell).save(os.path.join(dest, fn))
    print("split", os.path.basename(f), "->", dest)
