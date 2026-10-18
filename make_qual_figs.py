"""Qualitative figure grids for the paper.

  fig_qual_models.png  : 5 test images x [image | GT | M1..M6 dense predictions]
  fig_qual_prune_<mode>.png : 5 test images x [image | GT | dense | pruned @ sparsities]   (mode = unstructured / structured, iterative)

Usage:  python make_qual_figs.py            (writes to paper/figs/)
"""
import sys
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
IDS = [1, 13, 26, 39, 54]                     # 5 test images spread over the split
CROP = 768                                    # central square crop so cells are square and legible
LABEL = {"M1_unet_r34": "M1 U-Net/R34", "M2_deeplabv3p_r50": "M2 DeepLabV3+", "M3_pspnet_r50": "M3 PSPNet",
         "M4_unet_mbv3": "M4 U-Net/MBv3", "M5_segformer_b0": "M5 SegFormer-B0", "M6_segformer_b2": "M6 SegFormer-B2"}


def center_crop(a):
    h, w = a.shape[:2]; y = (h - CROP) // 2; x = (w - CROP) // 2
    return a[y:y + CROP, x:x + CROP]


def grid(cols, rows, fname, title_fs=8):
    """cols: list of (title, fn(img_rgb) -> rgb array); rows: image ids."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.55 * len(cols), 1.55 * len(rows) + 0.25), squeeze=False)
    for r, i in enumerate(rows):
        img, gt = read_test(i)
        img, gt = center_crop(img), center_crop(gt)
        for c, (title, fn) in enumerate(cols):
            ax = axes[r][c]; ax.imshow(fn(img, gt)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0: ax.set_title(title, fontsize=title_fs, pad=3)
        axes[r][0].set_ylabel(f"img {i:03d}", fontsize=7)
    handles = [mpatches.Patch(color=np.array(c) / 255, label=n) for c, n in zip([[0, 0, 0], [0, 200, 0], [220, 0, 0]], ["soil", "crop", "weed"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.05, wspace=0.04, hspace=0.06)
    fig.savefig(OUT / fname, dpi=220); plt.close(fig); print("saved", OUT / fname)


def pred_fn(model):
    return lambda img, gt: colorize(predict(model, img))


# ---------------- Fig: six dense models
cols = [("Image", lambda img, gt: img), ("Ground truth", lambda img, gt: colorize(gt))]
for name in ([] if (OUT / "fig_qual_models.png").exists() else C.MODELS):
    m = load(name, 0)
    cols.append((LABEL[name], pred_fn(m)))
if not (OUT / "fig_qual_models.png").exists(): grid(cols, IDS, "fig_qual_models.png", title_fs=7.5)
torch.cuda.empty_cache()

# ---------------- Fig: pruned M4, iterative schedule
for mode, levels in [("unstructured", [20, 40, 60, 80]), ("structured", [10, 20, 30, 50])]:
    cols = [("Image", lambda img, gt: img), ("Ground truth", lambda img, gt: colorize(gt)), ("Dense (0 %)", pred_fn(load("M4_unet_mbv3", 0)))]
    for s in levels:
        cols.append((f"{s} % pruned", pred_fn(load("M4_unet_mbv3", 0, f"{mode}/iterative", s))))
    grid(cols, IDS, f"fig_qual_prune_{mode}.png")
    torch.cuda.empty_cache()
