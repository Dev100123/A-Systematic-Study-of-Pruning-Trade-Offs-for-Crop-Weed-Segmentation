#!/usr/bin/env bash
# Full experiment pipeline. Steps 1-3 on the training GPU, steps 4-5 on the Jetson Orin Nano.
set -e
cd "$(dirname "$0")"

# 1. data + all baselines (6 models x 3 seeds) -> results/table2_baselines.csv
python data.py
python train.py

# 2. pick the best-mIoU model (A) and best small model (B) from table2, then prune both.
#    Edit these two names after inspecting results/table2_baselines.csv
A=M4_unet_mbv3
B=M5_segformer_b0
for m in $A $B; do
  for seed in 0 1 2; do
    python prune.py --model $m --seed $seed --mode unstructured --schedule iterative
    python prune.py --model $m --seed $seed --mode unstructured --schedule oneshot --epochs 0
    python prune.py --model $m --seed $seed --mode structured   --schedule iterative
    python prune.py --model $m --seed $seed --mode structured   --schedule oneshot --epochs 0
  done
done

# 3. tables + figures + qualitative Fig 5
python plots.py
python infer.py --model $A --seed 0 --pruned structured/iterative --sparsities 30 50 70
python infer.py --model $B --seed 0 --pruned structured/iterative --sparsities 30 50 70

# 4. export ONNX (host) -> copy runs/ to the Jetson -> build engines there
python export.py --model $A --seed 0
python export.py --model $A --seed 0 --pruned structured/iterative --sparsity 30
python export.py --model $A --seed 0 --pruned structured/iterative --sparsity 50
python export.py --model $B --seed 0
python export.py --model $B --seed 0 --pruned structured/iterative --sparsity 30
python export.py --model $B --seed 0 --pruned structured/iterative --sparsity 50

# 5. on the Jetson:
#   python export.py ... (same commands; trtexec present -> builds fp32/fp16/int8 engines)
#   python benchmark_jetson.py --engines "runs/*/seed0/export/*.engine"
#   python plots.py   # adds fig4_pareto.pdf
