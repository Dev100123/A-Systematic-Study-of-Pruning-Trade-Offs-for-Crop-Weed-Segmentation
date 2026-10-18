"""Pruning sweep: unstructured (global magnitude) and structured (channel) pruning,
one-shot vs. iterative prune-and-fine-tune.

Usage:
  python prune.py --model M5_segformer_b0 --seed 0 --mode unstructured --schedule iterative
  python prune.py --model M1_unet_r34 --seed 0 --mode structured --schedule oneshot
  python prune.py --model ... --epochs 1 --limit 16     # smoke test
Outputs: results/prune_<model>_<mode>_<schedule>_seed<seed>.csv and pruned checkpoints in runs/.
"""
import argparse
import copy
import csv
import json

import torch
import torch.nn.utils.prune as tnp
import torch_pruning as tp

import config as C
import data as D
import models as M
from train import DEV, evaluate, fit, seed_all

# ----------------------------------------------------------------------------- unstructured

def prunable_modules(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]


def unstructured_prune(model, amount):
    """Global L1 magnitude pruning to *total* sparsity `amount` over conv/linear weights.
    Re-applying on an already-pruned model composes masks (torch handles it), so for the
    iterative schedule we pass the *incremental* amount relative to remaining weights."""
    tnp.global_unstructured(prunable_modules(model), pruning_method=tnp.L1Unstructured, amount=amount)
    return model


def bake_masks(model):
    for m, n in prunable_modules(model):
        if tnp.is_pruned(m):
            tnp.remove(m, n)
    return model


def sparsity_of(model):
    tot = nz = 0
    for m, _ in prunable_modules(model):
        w = m.weight
        tot += w.numel(); nz += int((w != 0).sum())
    return 1 - nz / tot

# ----------------------------------------------------------------------------- structured

def structured_prune(model, ratio, example, steps=1, ignored=None):
    """Channel pruning with Torch-Pruning (dependency-graph aware). `ratio` = fraction of
    channels removed per prunable layer. Returns the physically smaller model."""
    model.eval()
    ignored = ignored or []
    # never prune the stem (first conv): it is only 16 channels wide in mobile encoders
    ignored.append([m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][0])
    for m in model.modules():
        # never prune the final classification layer(s)
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == C.NUM_CLASSES:
            ignored.append(m)
        # transformer attention blocks (SegFormer MiT) are left intact: channel pruning them breaks
        # the head-dimension bookkeeping; MLP / conv parts of the encoder and the decoder are pruned.
        if "attention" in type(m).__name__.lower():
            ignored.append(m)
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=imp, iterative_steps=steps,
        pruning_ratio=ratio, ignored_layers=ignored, global_pruning=False, round_to=None,  # round_to=8 turned 10 % into 50 % on 16-ch layers
    )
    for _ in range(steps):
        pruner.step()
    return model

# ----------------------------------------------------------------------------- sweep

def record(model, tag, extra):
    res = evaluate(model)
    res.update(extra)
    # size must be measured with masks baked in: torch.nn.utils.prune keeps weight_orig + weight_mask
    # attached, which doubles the serialised size and does not reflect the saved checkpoint.
    size_model = model
    if any(tnp.is_pruned(m) for m, _ in prunable_modules(model)):
        size_model = bake_masks(copy.deepcopy(model))
    res.update(params_M=M.count_params(model), nonzero_params_M=M.count_nonzero_params(model),
               gflops=M.count_flops(model), size_MB=M.size_mb(size_model), sparsity_measured=sparsity_of(model))
    print(f"[{tag}] mIoU={res['mIoU']:.4f} params={res['params_M']:.2f}M nz={res['nonzero_params_M']:.2f}M GFLOPs={res['gflops']:.2f}")
    return res


def sweep(name, seed, mode, schedule, ft_epochs, limit):
    seed_all(seed)
    base = M.build(name, pretrained=False)
    base.load_state_dict(torch.load(C.RUNS_DIR / name / f"seed{seed}" / "best.pt", map_location="cpu"))
    base.to(DEV)
    train_ds = D.TileDataset(D.train_tf())
    example = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=DEV)
    rows = [record(base, "dense", dict(sparsity=0.0, mode=mode, schedule=schedule, model=name, seed=seed))]
    out_dir = C.RUNS_DIR / name / f"seed{seed}" / f"prune_{mode}_{schedule}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = copy.deepcopy(base)
    prev = 0.0
    for s in C.SPARSITIES:
        if schedule == "oneshot":
            model = copy.deepcopy(base)              # always start from dense
        if mode == "unstructured":
            if schedule == "oneshot":
                unstructured_prune(model, s)
            else:                                    # incremental amount on remaining weights
                unstructured_prune(model, (s - prev) / (1 - prev))
        else:
            if schedule == "oneshot":
                structured_prune(model, s, example)
            else:
                structured_prune(model, (s - prev) / (1 - prev), example)
        if schedule == "iterative" and ft_epochs > 0:
            fit(model, train_ds, ft_epochs, C.PRUNE_FINETUNE_LR, log_prefix=f"[{name} {mode} s={s:.2f}]", limit=limit)
        rows.append(record(model, f"{mode}/{schedule} {s:.2f}", dict(sparsity=s, mode=mode, schedule=schedule, model=name, seed=seed)))
        ck = copy.deepcopy(model)
        if mode == "unstructured":
            bake_masks(ck)
        if mode == "structured":
            torch.save(ck, out_dir / f"sparsity_{int(s * 100)}.pt")
        else:  # drop thop's total_ops / total_params buffers so the state dict loads strictly
            torch.save({k: v for k, v in ck.state_dict().items() if not k.endswith(("total_ops", "total_params"))}, out_dir / f"sparsity_{int(s * 100)}.pt")
        prev = s

    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    path = C.RESULTS_DIR / f"prune_{name}_{mode}_{schedule}_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["unstructured", "structured"], default="unstructured")
    ap.add_argument("--schedule", choices=["oneshot", "iterative"], default="iterative")
    ap.add_argument("--epochs", type=int, default=C.PRUNE_FINETUNE_EPOCHS, help="fine-tune epochs per step")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sweep(a.model, a.seed, a.mode, a.schedule, a.epochs, a.limit)
