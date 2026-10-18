"""Leakage check for CWFID image 028 (listed in both official splits).
Evaluates the 6 dense models (3 seeds) and the M4 structured-30% / unstructured-80% iterative checkpoints (3 seeds)
on: all 21 official test images, the 20 images not in the training list, and image 028 alone.
Writes results/test_no28.csv incrementally."""
import csv, sys
import torch
import config as C, data as D, models as M
from train import evaluate, DEV
from infer import load

def main():
    TR, TEST = D.read_split()
    LEAK = sorted(set(TR) & set(TEST)); CLEAN = [i for i in TEST if i not in LEAK]
    print("leaked ids:", LEAK, "| clean test ids:", len(CLEAN), flush=True)
    subsets = {"all21": D.FullImageDataset(ids=TEST), "clean20": D.FullImageDataset(ids=CLEAN), "img028": D.FullImageDataset(ids=LEAK)}

    out = C.RESULTS_DIR / "test_no28.csv"
    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["model", "seed", "mode", "sparsity", "subset", "mIoU", "IoU_soil", "IoU_crop", "IoU_weed"]); w.writeheader()

    def rec(tag, model, seed, sparsity=0, mode="dense"):
        for name, ds in subsets.items():
            r = evaluate(model, ds)
            row = dict(model=tag, seed=seed, mode=mode, sparsity=sparsity, subset=name, mIoU=round(100 * r["mIoU"], 2),
                       IoU_soil=round(100 * r["IoU_soil"], 2), IoU_crop=round(100 * r["IoU_crop"], 2), IoU_weed=round(100 * r["IoU_weed"], 2))
            w.writerow(row); f.flush(); print(row, flush=True)

    for name in C.MODELS:
        for seed in C.SEEDS:
            m = M.build(name, pretrained=False)
            m.load_state_dict(torch.load(C.RUNS_DIR / name / f"seed{seed}" / "best.pt", map_location="cpu")); m.to(DEV).eval()
            rec(name, m, seed); del m; torch.cuda.empty_cache()

    for mode, s in (("structured", 30), ("unstructured", 80)):
        for seed in C.SEEDS:
            m = load("M4_unet_mbv3", seed, f"{mode}/iterative", s)
            rec("M4_unet_mbv3", m, seed, s, f"{mode}/iterative"); del m; torch.cuda.empty_cache()
    f.close(); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
