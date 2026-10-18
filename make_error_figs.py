"""Reviewer-requested error analysis.
  1. Confusion matrices (row-normalised, %) of every dense model on the test split -> results/confusion_dense.csv
  2. Error-map figure: for 5 test images, columns = image | GT | M4 prediction | M4 error map
     error map colours: dark grey = correct, orange = crop predicted as weed, blue = weed predicted as crop,
     red = plant (crop or weed) missed as soil, yellow = soil predicted as plant (false alarm)
     -> paper/figs/fig_error_maps_dense.png   and the same for pruned M4 (structured 30 %) -> fig_error_maps_pruned.png
"""
import csv
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

import config as C
import data as D
from infer import load, predict, colorize, read_test

OUT = C.ROOT / "paper" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
IDS = [1, 13, 26, 39, 54]
CROP = 768
_, TEST = D.read_split()

# ---------------------------------------------------------------- 1. confusion matrices
rows = []
for name in ([] if (C.RESULTS_DIR / "confusion_dense.csv").exists() else C.MODELS):
    m = load(name, 0)
    cm = np.zeros((3, 3), np.int64)
    for i in TEST:
        img, gt = read_test(i)
        pred = predict(m, img)
        cm += np.bincount(3 * gt.ravel().astype(int) + pred.ravel().astype(int), minlength=9).reshape(3, 3)
    rn = 100 * cm / cm.sum(1, keepdims=True)
    for t, tn in enumerate(C.CLASS_NAMES):
        rows.append(dict(model=name, true=tn, **{f"pred_{pn}": round(rn[t, p], 2) for p, pn in enumerate(C.CLASS_NAMES)}, pixels=int(cm[t].sum())))
    print(name); print("  rows=true, cols=pred (soil, crop, weed) in %"); print(np.round(rn, 2))
    del m; torch.cuda.empty_cache()
if rows:
    with open(C.RESULTS_DIR / "confusion_dense.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("saved results/confusion_dense.csv")


# ---------------------------------------------------------------- 2. error maps
ERR = np.array([[255, 140, 0],     # crop -> weed   orange
                [230, 0, 0],       # crop -> soil   red
                [0, 90, 255],      # weed -> crop   blue
                [170, 0, 200],     # weed -> soil   purple
                [255, 225, 0],     # soil -> crop   yellow
                [0, 200, 220]], np.uint8)   # soil -> weed   cyan
LEG = ["crop predicted as weed", "crop predicted as soil",
       "weed predicted as crop", "weed predicted as soil",
       "soil predicted as crop", "soil predicted as weed"]
PAIRS = [(1, 2), (1, 0), (2, 1), (2, 0), (0, 1), (0, 2)]   # (true, predicted)
KER = np.ones((3, 3), np.uint8)


def error_map(img, gt, pred):
    """Six confusion types drawn on a lightened greyscale copy of the photo; each dilated by 1 px for visibility."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    base = np.clip(0.55 * g + 115, 0, 255).astype(np.uint8)
    out = np.stack([base] * 3, -1)
    for k, (t, p) in enumerate(PAIRS):
        m = cv2.dilate(((gt == t) & (pred == p)).astype(np.uint8), KER, iterations=1).astype(bool)
        out[m] = ERR[k]
    return out


def cc(a):
    h, w = a.shape[:2]; y = (h - CROP) // 2; x = (w - CROP) // 2
    return a[y:y + CROP, x:x + CROP]


def grid(model, fname, title):
    cols = ["Image", "Ground truth", "Prediction", "Error map"]
    fig, axes = plt.subplots(len(IDS), 4, figsize=(2.05 * 4, 2.05 * len(IDS) + 0.45), squeeze=False)
    for r, i in enumerate(IDS):
        img, gt = read_test(i)
        pred = predict(model, img)
        panels = [cc(img), cc(colorize(gt)), cc(colorize(pred)), cc(error_map(img, gt, pred))]
        for c, p in enumerate(panels):
            ax = axes[r][c]; ax.imshow(p); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0: ax.set_title(cols[c], fontsize=9, pad=4)
        axes[r][0].set_ylabel(f"img {i:03d}", fontsize=8)
    handles = [mpatches.Patch(color=ERR[k] / 255, ec="#555", lw=0.5, label=LEG[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.965, bottom=0.075, wspace=0.04, hspace=0.06)
    fig.savefig(OUT / fname, dpi=220); plt.close(fig); print("saved", OUT / fname)


grid(load("M4_unet_mbv3", 0), "fig_error_maps_dense.png", "Dense M4 (U-Net/MobileNetV3-L)")
grid(load("M4_unet_mbv3", 0, "structured/iterative", 30), "fig_error_maps_pruned.png", "M4 after 30 % iterative structured pruning")
