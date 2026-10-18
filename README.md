# Pruning Semantic Segmentation Models for Crop/Weed Segmentation (CWFID)

Training, pruning and evaluation pipeline behind the paper:

> Flora Amato, Rajib Chandra Ghosh, Semanto Mondal.
> **Compressing Semantic Segmentation Models for Agricultural IoT: A Systematic Study of Pruning Trade-Offs for Crop–Weed Segmentation.**
> *Journal of Imaging* 12(10), 495, 2026. DOI: [10.3390/jimaging12100495](https://doi.org/10.3390/jimaging12100495)

The study trains six semantic segmentation architectures on the Crop/Weed Field Image Dataset (CWFID) under one identical protocol, selects the most favourable compact model (U-Net with a MobileNetV3-Large encoder, "M4"), and prunes it with **unstructured global magnitude pruning** and **dependency-aware structured channel pruning**, each under a **one-shot** and an **iterative prune-and-fine-tune** schedule, at eight sparsity levels from 10 % to 80 %. Every configuration is run with three random seeds. The per-seed result files used in the paper are included under `results/`.

If you use this code or the result files, please cite the paper (see [Citation](#citation)).

---

## Contents

- [Main findings](#main-findings)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Dataset](#dataset)
- [Reproducing the paper](#reproducing-the-paper)
- [Pruning protocol](#pruning-protocol)
- [Checkpoints and outputs](#checkpoints-and-outputs)
- [Inference on your own images](#inference-on-your-own-images)
- [Edge deployment scripts (not used in the paper)](#edge-deployment-scripts-not-used-in-the-paper)
- [Notes and caveats](#notes-and-caveats)
- [Citation](#citation)

---

## Main findings

Dense baselines on the official CWFID test split (mean ± std over three seeds; complexity for one 512 × 512 input):

| ID | Architecture | Encoder | mIoU (%) | Params (M) | GFLOPs |
|----|--------------|---------|---------:|-----------:|-------:|
| M1 | U-Net | ResNet-34 | 87.80 ± 1.15 | 24.44 | 62.9 |
| M2 | DeepLabV3+ | ResNet-50 | 85.51 ± 0.22 | 26.68 | 73.8 |
| M3 | PSPNet | ResNet-50 | 81.70 ± 0.10 | 24.31 | 23.8 |
| **M4** | **U-Net** | **MobileNetV3-Large** | **88.85 ± 0.46** | **6.69** | **24.7** |
| M5 | SegFormer | MiT-B0 | 87.70 ± 0.20 | 3.71 | 13.5 |
| M6 | SegFormer | MiT-B2 | 88.67 ± 0.53 | 24.72 | 42.4 |

Pruning M4 (iterative schedule = 10 fine-tune epochs per sparsity level; three seeds):

| Regime | Sparsity | mIoU (%) | Params (M) | Non-zero (M) | GFLOPs | Size (MB) |
|--------|---------:|---------:|-----------:|-------------:|-------:|----------:|
| Dense | 0 % | 88.85 ± 0.46 | 6.69 | 6.69 | 24.7 | 27.1 |
| Unstructured, iterative | 80 % | 89.04 ± 0.14 | 6.69 | 1.36 | 24.7 | 27.1 |
| Structured, iterative | 30 % | 84.64 ± 0.97 | 3.27 | 3.27 | 11.6 | 13.5 |
| Structured, iterative | 40 % | 75.78 ± 14.09 | 2.41 | 2.41 | 8.4 | 10.0 |
| Structured, one-shot | 10 % | 34.13 ± 4.89 | 5.41 | 5.41 | 19.7 | 22.0 |

- The compact M4 matches the much larger SegFormer-B2 (the difference is not statistically significant across seeds) with 3.7× fewer parameters.
- Iterative unstructured pruning keeps accuracy flat up to 80 % weight sparsity, but it leaves tensor shapes, FLOPs, checkpoint size and measured latency unchanged on dense hardware.
- Structured channel pruning physically shrinks the network (2.1× fewer FLOPs at 30 %) at a cost of 4.21 mIoU points, becomes unstable at 40 % and collapses beyond that under the fixed fine-tuning budget.
- One-shot structured pruning without fine-tuning fails already at 10 % channel sparsity.
- Crop IoU degrades well before mIoU signals failure, so per-class metrics are reported throughout.

Full tables: `results/table2_baselines.csv`, `results/table3_pruning.csv`, `results/latency_local.csv`, `results/confusion_dense.csv`, `results/test_no28.csv`.

---

## Repository layout

| Path | Purpose | Paper output |
|------|---------|--------------|
| `config.py` | All hyper-parameters: paths, model list, seeds, sparsity levels, fine-tune budget | configuration table |
| `data.py` | Clones CWFID, tiles training images (512², stride 384), datasets, augmentations, sliding-window inference | Sec. 3 |
| `models.py` | Model factory via `segmentation_models_pytorch`; parameter, FLOP and size counters | Table 1 |
| `metrics.py` | Confusion-matrix based mIoU, per-class IoU, F1 and pixel accuracy | all tables |
| `train.py` | Trains all models × seeds, evaluates on the official test split | Table 2 |
| `prune.py` | Unstructured (global L1) and structured (Torch-Pruning channel) pruning, one-shot vs iterative | Table 3, pruning figures |
| `plots.py` | Aggregates the per-seed CSVs into `table3_pruning.csv` and the sparsity curves | Table 3, figures |
| `infer.py` | Predicted masks / overlays / comparison panels for dense and pruned checkpoints; also runs on arbitrary images | qualitative figures |
| `make_qual_figs.py` | Qualitative grids: six dense models, and M4 at increasing unstructured / structured sparsity | qualitative figures |
| `make_error_figs.py` | Row-normalised confusion matrices of all dense models and colour-coded error maps for dense and 30 % pruned M4 | confusion / error-map figures |
| `bench_latency.py` | Wall-clock latency of dense vs pruned M4 on the workstation GPU and CPU (PyTorch eager, FP32, batch 1) | latency table |
| `eval_no28.py` | Split-overlap check: re-evaluates all models with and without CWFID image 028 (listed in both official splits) | reviewer response |
| `export.py` | ONNX export and TensorRT FP32 / FP16 / INT8 engine build | not used in the paper |
| `benchmark_jetson.py` | TensorRT engine benchmark on Jetson Orin Nano (latency, power, energy, mIoU) | not used in the paper |
| `run_all.sh` | Full sequence from data download to figures | – |
| `run_pruning.sh` | Pruning sweep only (step 2 of `run_all.sh`) | – |
| `results/` | Per-seed CSVs, aggregated tables, figures, inference panels | – |
| `runs/` | Dense and pruned checkpoints per model and seed (created by training) | – |
| `paper/` | Manuscript sources, figures, tables, reviewer response, and the scripts that draw the workflow diagrams | – |

---

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended. The experiments in the paper were run on an NVIDIA GeForce RTX 5080 Laptop GPU (16 GB) with an Intel Core Ultra 9 CPU and 32 GB of RAM.

```bash
pip install torch torchvision                      # pick the CUDA build for your system from pytorch.org
pip install segmentation-models-pytorch timm torch-pruning albumentations opencv-python thop pandas matplotlib pyyaml tqdm numpy
```

Optional, only for the edge deployment scripts: `onnx`, `tensorrt`, `pycuda` (on a Jetson these ship with JetPack).

The scripts also run on CPU, which is useful for smoke tests but far too slow for the full study.

---

## Dataset

The Crop/Weed Field Image Dataset (CWFID) by Haug and Ostermann contains 60 field images (1296 × 966 px) of a carrot field with pixel-level soil / crop / weed annotations, and an official split of 40 training and 21 test images.

`python data.py` clones <https://github.com/cwfid/dataset> into `data/cwfid/` and tiles the 40 training images into 480 overlapping 512 × 512 patches under `data/tiles/train/`. Test images are kept at full resolution and evaluated by overlapping sliding-window inference with averaged logits. RGB annotation colours are mapped to labels 0 = soil, 1 = crop, 2 = weed.

CWFID is distributed for **non-commercial research only** and its authors ask to be cited when the data are used (BibTeX in [Citation](#citation)).

---

## Reproducing the paper

All commands are run from the repository root. Run a smoke test first:

```bash
python train.py --models M4_unet_mbv3 --seeds 0 --epochs 1 --limit 16
python prune.py --model M4_unet_mbv3 --seed 0 --mode structured --schedule iterative --epochs 1 --limit 16
```

### 1. Data and model complexity (Table 1)

```bash
python data.py      # clone + tile CWFID
python models.py    # prints params / GFLOPs / checkpoint size for the six architectures
```

### 2. Dense baselines (Table 2)

```bash
python train.py     # 6 models x 3 seeds x 60 epochs
```

Training protocol: cross-entropy + Dice loss, AdamW (weight decay 1e-4), OneCycle schedule with peak LR 3e-4 and 10 % warm-up, batch 8, 60 epochs, mixed precision, ImageNet-pretrained encoders, augmentations from `data.train_tf()`. One run takes roughly 17 to 20 minutes on the GPU above. Each run writes `runs/<model>/seed<k>/best.pt` and `metrics.json`; the script aggregates everything into `results/baselines_raw.csv` and `results/table2_baselines.csv`.

### 3. Pruning sweep on M4 (Table 3, pruning figures)

```bash
for s in 0 1 2; do
  python prune.py --model M4_unet_mbv3 --seed $s --mode unstructured --schedule oneshot   --epochs 0
  python prune.py --model M4_unet_mbv3 --seed $s --mode structured   --schedule oneshot   --epochs 0
  python prune.py --model M4_unet_mbv3 --seed $s --mode unstructured --schedule iterative
  python prune.py --model M4_unet_mbv3 --seed $s --mode structured   --schedule iterative
done
python plots.py     # -> results/table3_pruning.csv, fig2_miou_vs_sparsity.pdf, fig3_oneshot_vs_iterative.pdf
```

`run_pruning.sh` wraps the same loop. Edit its `PY=` line to point at your Python interpreter and its `MODELS=` line to the models you want (the published study prunes M4 only).

### 4. Qualitative figures, error analysis, latency, split check

```bash
python make_qual_figs.py    # paper/figs/fig_qual_models.png, fig_qual_prune_{unstructured,structured}.png
python make_error_figs.py   # results/confusion_dense.csv, paper/figs/fig_error_maps_{dense,pruned}.png
python bench_latency.py     # results/latency_local.csv  (dense vs pruned M4 on GPU and CPU)
python eval_no28.py         # results/test_no28.csv      (all 21 test images / 20 without 028 / 028 alone)
python infer.py --model M4_unet_mbv3 --seed 0 --pruned structured/iterative --sparsities 30 50 70   # comparison panel + per-class heatmap
```

`run_all.sh` chains steps 1 to 4 (its last two steps call the deployment scripts described below).

---

## Pruning protocol

Sparsity levels are `config.SPARSITIES` = 10, 20, ..., 80 %. Pruning always starts from the dense checkpoint `runs/<model>/seed<k>/best.pt`.

**Unstructured.** Global L1-magnitude pruning over the weights of all `Conv2d` and `Linear` layers (`torch.nn.utils.prune.global_unstructured`). Zeroed weights keep their tensor shape, so parameter count, FLOPs and checkpoint size do not change; the non-zero count is reported separately.

**Structured.** Per-layer L1 channel pruning with the dependency-graph pruner of [Torch-Pruning](https://github.com/VainF/Torch-Pruning), which propagates each channel removal to all coupled layers. The stem convolution, the final classification layer, and any attention module (SegFormer encoders) are excluded. The result is a physically smaller `nn.Module`.

**One-shot.** Prune the dense model directly to the target sparsity, no fine-tuning (`--epochs 0`). Isolates the raw damage of pruning.

**Iterative.** Prune the model from the previous level to the next level and fine-tune for `PRUNE_FINETUNE_EPOCHS` = 10 epochs (AdamW, OneCycle, peak LR 1e-4, same loss). The incremental ratio at each step is `(s_t - s_{t-1}) / (1 - s_{t-1})`, so the cumulative sparsity lands exactly on `s_t`. The model at 80 % has therefore received 80 fine-tune epochs in total.

Each sweep writes `results/prune_<model>_<mode>_<schedule>_seed<k>.csv` with mIoU, per-class IoU and F1, pixel accuracy, parameters, non-zero parameters, GFLOPs, checkpoint size and measured sparsity for the dense model and every level.

---

## Checkpoints and outputs

```
runs/<model>/seed<k>/
  best.pt                                  dense state_dict
  metrics.json                             test-split metrics + complexity of the dense model
  prune_<mode>_<schedule>/sparsity_<pct>.pt
      unstructured : state_dict with zeros baked in (load into models.build(name))
      structured   : whole pickled nn.Module (torch.load(..., weights_only=False))
  export/                                  ONNX / TensorRT engines, only if export.py was run

results/
  baselines_raw.csv, table2_baselines.csv  dense baselines (per seed / mean ± std)
  prune_*.csv, table3_pruning.csv          pruning sweep (per seed / aggregated)
  fig2_miou_vs_sparsity.pdf, fig3_oneshot_vs_iterative.pdf
  latency_local.csv                        GPU + CPU latency of dense and pruned M4
  confusion_dense.csv                      row-normalised confusion matrices
  test_no28.csv                            split-overlap check
  inference/<model>/<tag>/                 image | GT | prediction | overlay panels
```

To load any checkpoint in your own code:

```python
from infer import load
dense  = load("M4_unet_mbv3", seed=0)
pruned = load("M4_unet_mbv3", seed=0, pruned="structured/iterative", sparsity=30)
```

---

## Inference on your own images

```bash
python infer.py --model M4_unet_mbv3 --seed 0 --images field1.jpg field2.png
python infer.py --model M4_unet_mbv3 --seed 0 --pruned structured/iterative --sparsities 30 --images field1.jpg
```

Writes `<name>_mask.png` and `<name>_overlay.png` to `results/inference/<model>/<tag>/`. Images smaller than 512 px are reflection-padded; larger images are processed by sliding window. Colours: black = soil, green = crop, red = weed.

---

## Edge deployment scripts (not used in the paper)

`export.py` and `benchmark_jetson.py` export dense or pruned checkpoints to ONNX, build TensorRT FP32 / FP16 / INT8 engines (INT8 with entropy calibration on training tiles) and measure latency, GPU memory, power via `tegrastats` and energy per image on a Jetson Orin Nano. They were smoke-tested but **no on-device results are reported in the published paper**; the latency numbers in the paper come from `bench_latency.py` on the training workstation.

```bash
python export.py --model M4_unet_mbv3 --seed 0                                            # dense -> ONNX (+ engines if trtexec is present)
python export.py --model M4_unet_mbv3 --seed 0 --pruned structured/iterative --sparsity 30
python benchmark_jetson.py --engines "runs/M4_unet_mbv3/seed0/export/*.engine"              # -> results/table5_device.csv
```

Fix the board power mode first (`sudo nvpmodel -m 0 && sudo jetson_clocks`) and record it with the results.

---

## Notes and caveats

- **Image 028 appears in both the train and test lists of the official CWFID split.** The paper uses the split exactly as published; `eval_no28.py` shows that removing image 028 from the test set changes mIoU by only a few tenths of a point for every model.
- **Unstructured sparsity is not a speed-up** on dense hardware. FLOPs, checkpoint size and measured latency are unchanged at 80 % sparsity. Only structured pruning reduces them.
- **Fine-tuning budget.** The collapse of structured pruning beyond 30 % is observed under a fixed budget of 10 epochs per level. A larger budget or smaller steps would probably move the usable limit upward; this was not tested.
- **SegFormer structured pruning.** Attention blocks are skipped automatically; only MLP, convolutional and decoder layers are pruned. The published pruning results concern the purely convolutional M4 only.
- Seeds, epochs, learning rates and sparsity levels live in `config.py`; the `--epochs` and `--limit` flags of `train.py` and `prune.py` are meant for smoke tests only.
- The result CSVs store mIoU and IoU as fractions in [0, 1]; the paper reports them in percent.

---

## Citation

If you use this code, the result files or the trained / pruned checkpoints, please cite:

```bibtex
@Article{jimaging12100495,
AUTHOR = {Amato, Flora and Chandra Ghosh, Rajib and Mondal, Semanto},
TITLE = {Compressing Semantic Segmentation Models for Agricultural IoT: A Systematic Study of Pruning Trade-Offs for Crop–Weed Segmentation},
JOURNAL = {Journal of Imaging},
VOLUME = {12},
YEAR = {2026},
NUMBER = {10},
ARTICLE-NUMBER = {495},
URL = {https://www.mdpi.com/2313-433X/12/10/495},
ISSN = {2313-433X},
DOI = {10.3390/jimaging12100495}
}
```

Please also cite the dataset:

```bibtex
@inproceedings{haug15,
  author={Haug, Sebastian and Ostermann, J{\"o}rn},
  title={A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks},
  year={2015},
  booktitle={Computer Vision - ECCV 2014 Workshops},
  doi={10.1007/978-3-319-16220-1_8},
  url={http://dx.doi.org/10.1007/978-3-319-16220-1_8},
  pages={105--116},
}
```

Structured pruning uses Torch-Pruning (Fang et al., *DepGraph: Towards Any Structural Pruning*, CVPR 2023) and all architectures are built with [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## Contact

Rajib Chandra Ghosh, Department of Electrical Engineering and Information Technology (DIETI), University of Naples Federico II: rajib.chandraghosh@unina.it
