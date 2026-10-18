#!/usr/bin/env bash
# Pruning sweep only (step 2 of run_all.sh), using the conda env that has torch + torch_pruning.
# Run from Git Bash:  bash run_pruning.sh
# Produces results/prune_<model>_<mode>_<schedule>_seed<k>.csv and pruned checkpoints under runs/.
set -e
cd "$(dirname "$0")"
PY=/c/Users/rajib/miniconda3/envs/cre/python.exe

MODELS="M4_unet_mbv3 M5_segformer_b0"   # A = best accuracy, B = smallest (from Table 2)
SEEDS="0 1 2"

for m in $MODELS; do
  for s in $SEEDS; do
    # one-shot (no fine-tune): fast, isolates raw pruning damage
    $PY prune.py --model $m --seed $s --mode unstructured --schedule oneshot --epochs 0
    $PY prune.py --model $m --seed $s --mode structured   --schedule oneshot --epochs 0
    # iterative prune -> fine-tune 10 epochs per level (8 levels = 80 epochs each)
    $PY prune.py --model $m --seed $s --mode unstructured --schedule iterative
    $PY prune.py --model $m --seed $s --mode structured   --schedule iterative
  done
done

# tables + figures: results/table3_pruning.csv, fig2_miou_vs_sparsity.pdf, fig3_oneshot_vs_iterative.pdf
$PY plots.py
# qualitative panels for Fig 5
$PY infer.py --model M4_unet_mbv3   --seed 0 --pruned structured/iterative --sparsities 30 50 70
$PY infer.py --model M5_segformer_b0 --seed 0 --pruned structured/iterative --sparsities 30 50 70
