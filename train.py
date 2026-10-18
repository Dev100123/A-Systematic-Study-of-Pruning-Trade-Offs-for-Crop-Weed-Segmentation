"""Train baseline models (all architectures x seeds) and evaluate on the CWFID test split.

Usage:
  python train.py                      # all models, all seeds
  python train.py --models M5_segformer_b0 --seeds 0
  python train.py --epochs 2 --limit 16   # smoke test
"""
import argparse
import csv
import json
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config as C
import data as D
import models as M
from metrics import SegMetrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loss():
    ce = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")
    return lambda out, y: ce(out, y) + dice(out, y)


@torch.no_grad()
def evaluate(model, ds=None) -> dict:
    model.eval()
    ds = ds or D.FullImageDataset()
    met = SegMetrics()
    for img, lbl in tqdm(DataLoader(ds, batch_size=1, num_workers=2), desc="eval", leave=False):
        img = img.to(DEV)
        logits = D.sliding_window_predict(model, img)
        met.update(logits.argmax(1).cpu(), lbl)
    return met.compute()


def fit(model, train_ds, epochs, lr, log_prefix="", limit=None):
    """Generic training loop used for both baseline training and post-pruning fine-tune."""
    if limit:
        train_ds = Subset(train_ds, list(range(min(limit, len(train_ds)))))
    dl = DataLoader(train_ds, batch_size=C.BATCH, shuffle=True, num_workers=C.NUM_WORKERS, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * len(dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    loss_fn = make_loss()
    model.to(DEV)
    for ep in range(epochs):
        model.train(); tot = 0.0; t0 = time.time()
        bar = tqdm(dl, desc=f"{log_prefix} ep {ep + 1}/{epochs}", leave=False)
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            bar.set_postfix(loss=f"{tot / i:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        print(f"{log_prefix} ep {ep + 1}/{epochs} loss {tot / max(1, len(dl)):.4f} ({time.time() - t0:.0f}s)")
    return model


def run(name, seed, epochs, limit=None):
    seed_all(seed)
    out_dir = C.RUNS_DIR / name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = M.build(name)
    t0 = time.time()
    fit(model, D.TileDataset(D.train_tf()), epochs, C.LR, log_prefix=f"[{name} s{seed}]", limit=limit)
    train_time = time.time() - t0
    torch.save(model.state_dict(), out_dir / "best.pt")
    res = evaluate(model)
    res.update(model=name, seed=seed, params_M=M.count_params(model), gflops=M.count_flops(model),
               size_MB=M.size_mb(model), train_time_s=train_time)
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


def write_table(rows):
    """Aggregate mean±std across seeds -> results/table2_baselines.csv"""
    import pandas as pd
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "baselines_raw.csv", index=False)
    agg = df.groupby("model").agg(["mean", "std"])
    agg.to_csv(C.RESULTS_DIR / "table2_baselines.csv")
    print(agg[["mIoU", "params_M", "gflops"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=None, help="use only N tiles (smoke test)")
    a = ap.parse_args()
    D.download(); D.tile_dataset()
    rows = [run(n, s, a.epochs, a.limit) for n in a.models for s in a.seeds]
    write_table(rows)
