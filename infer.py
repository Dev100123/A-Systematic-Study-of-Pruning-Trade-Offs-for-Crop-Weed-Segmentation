"""Qualitative inference: predicted masks for dense and pruned checkpoints (paper Fig. 5),
plus a per-class IoU heatmap across sparsity levels.

Usage:
  # single image / whole test split with the dense model -> overlays in results/inference/<model>/dense/
  python infer.py --model M6_segformer_b2 --seed 0
  python infer.py --model M6_segformer_b2 --seed 0 --ids 5 12
  # comparison panel: image | GT | dense | 30% | 50% | 70%  (structured/iterative checkpoints)
  python infer.py --model M6_segformer_b2 --seed 0 --pruned structured/iterative --sparsities 30 50 70 --ids 5 12
  # your own image(s)
  python infer.py --model M6_segformer_b2 --seed 0 --images /path/a.png /path/b.jpg
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config as C
import data as D
import models as M
from metrics import SegMetrics
from train import DEV

PALETTE = np.array([[0, 0, 0], [0, 200, 0], [220, 0, 0]], np.uint8)   # soil black, crop green, weed red


def load(name, seed, pruned=None, sparsity=None):
    run_dir = C.RUNS_DIR / name / f"seed{seed}"
    if pruned is None:
        m = M.build(name, pretrained=False)
        m.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    else:
        mode, sched = pruned.split("/")
        ck = torch.load(run_dir / f"prune_{mode}_{sched}" / f"sparsity_{sparsity}.pt", map_location="cpu", weights_only=False)
        if isinstance(ck, torch.nn.Module):
            m = ck
        else:
            ck = {k: v for k, v in ck.items() if not k.endswith(("total_ops", "total_params"))}  # thop buffers
            m = M.build(name, pretrained=False); m.load_state_dict(ck)
    return m.to(DEV).eval()


@torch.no_grad()
def predict(model, img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb uint8 HxWx3 -> label map HxW (uint8). Pads to at least one tile."""
    H, W = img_rgb.shape[:2]
    ph, pw = max(0, C.TILE - H), max(0, C.TILE - W)
    padded = cv2.copyMakeBorder(img_rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT) if (ph or pw) else img_rgb
    x = D.eval_tf()(image=padded)["image"][None].to(DEV)
    logits = D.sliding_window_predict(model, x)
    return logits.argmax(1)[0, :H, :W].cpu().numpy().astype(np.uint8)


def colorize(lbl): return PALETTE[lbl]


def overlay(img, lbl, alpha=0.45):
    col = colorize(lbl)
    out = img.copy()
    mask = lbl > 0
    out[mask] = (alpha * col[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
    return out


def read_test(i):
    img = cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "images" / f"{i:03d}_image.png")), cv2.COLOR_BGR2RGB)
    gt = D.rgb_to_label(cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "annotations" / f"{i:03d}_annotation.png")), cv2.COLOR_BGR2RGB))
    return img, gt


def run_single(a):
    model = load(a.model, a.seed, a.pruned, a.sparsities[0] if a.pruned else None)
    tag = "dense" if a.pruned is None else f"{a.pruned.replace('/', '_')}_{a.sparsities[0]}"
    out = C.RESULTS_DIR / "inference" / a.model / tag; out.mkdir(parents=True, exist_ok=True)
    if a.images:                                    # arbitrary user images, no GT
        for p in a.images:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            pred = predict(model, img)
            stem = out / (p.split("/")[-1].rsplit(".", 1)[0])
            cv2.imwrite(f"{stem}_mask.png", cv2.cvtColor(colorize(pred), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{stem}_overlay.png", cv2.cvtColor(overlay(img, pred), cv2.COLOR_RGB2BGR))
            print("saved", stem)
        return
    ids = a.ids or D.read_split()[1]
    met = SegMetrics()
    for i in tqdm(ids, desc="infer"):
        img, gt = read_test(i)
        pred = predict(model, img)
        met.update(torch.from_numpy(pred), torch.from_numpy(gt))
        panel = np.concatenate([img, colorize(gt), colorize(pred), overlay(img, pred)], 1)
        cv2.imwrite(str(out / f"{i:03d}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{i:03d}_mask.png"), cv2.cvtColor(colorize(pred), cv2.COLOR_RGB2BGR))
    print({k: round(v, 4) for k, v in met.compute().items()}); print("saved ->", out)


def run_comparison(a):
    """Fig 5: rows = images, cols = image | GT | dense | pruned @ each sparsity. Also per-class IoU heatmap."""
    ids = a.ids or D.read_split()[1][:3]
    levels = [None] + list(a.sparsities)
    models = [load(a.model, a.seed)] + [load(a.model, a.seed, a.pruned, s) for s in a.sparsities]
    cols = ["Image", "Ground truth", "Dense"] + [f"{s}% pruned" for s in a.sparsities]
    fig, axes = plt.subplots(len(ids), len(cols), figsize=(2.6 * len(cols), 2.0 * len(ids)), squeeze=False)
    per_class = np.zeros((len(levels), C.NUM_CLASSES))
    mets = [SegMetrics() for _ in levels]
    for r, i in enumerate(ids):
        img, gt = read_test(i)
        axes[r][0].imshow(img); axes[r][1].imshow(colorize(gt))
        for k, m in enumerate(models):
            pred = predict(m, img); mets[k].update(torch.from_numpy(pred), torch.from_numpy(gt))
            axes[r][2 + k].imshow(overlay(img, pred))
        for c in range(len(cols)):
            axes[r][c].axis("off")
            if r == 0: axes[r][c].set_title(cols[c], fontsize=9)
    for k, mt in enumerate(mets):
        res = mt.compute(); per_class[k] = [res[f"IoU_{c}"] for c in C.CLASS_NAMES]
    out = C.RESULTS_DIR / "inference"; out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / f"fig5_{a.model}_{a.pruned.replace('/', '_')}.pdf", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 0.5 + 0.35 * len(levels)))
    im = ax.imshow(per_class, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(C.NUM_CLASSES)); ax.set_xticklabels(C.CLASS_NAMES)
    ax.set_yticks(range(len(levels))); ax.set_yticklabels(["dense"] + [f"{s}%" for s in a.sparsities])
    for y in range(len(levels)):
        for x in range(C.NUM_CLASSES): ax.text(x, y, f"{per_class[y, x]:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, label="IoU"); fig.tight_layout()
    fig.savefig(out / f"fig5b_perclass_{a.model}_{a.pruned.replace('/', '_')}.pdf"); plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pruned", default=None, help="mode/schedule, e.g. structured/iterative")
    ap.add_argument("--sparsities", nargs="*", type=int, default=[30, 50, 70])
    ap.add_argument("--ids", nargs="*", type=int, default=None, help="CWFID image ids (default: test split)")
    ap.add_argument("--images", nargs="*", default=None, help="arbitrary image paths")
    a = ap.parse_args()
    if a.pruned and len(a.sparsities) > 1 and not a.images:
        run_comparison(a)
    else:
        run_single(a)
