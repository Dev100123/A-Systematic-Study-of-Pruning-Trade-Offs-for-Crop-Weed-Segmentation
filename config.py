"""Central configuration for the CWFID pruning study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cwfid"          # raw dataset (git clone target)
TILE_DIR = ROOT / "data" / "tiles"          # tiled patches
RUNS_DIR = ROOT / "runs"                    # checkpoints + logs
RESULTS_DIR = ROOT / "results"              # csv / figures

CWFID_REPO = "https://github.com/cwfid/dataset.git"

NUM_CLASSES = 3                              # 0 soil, 1 crop, 2 weed
CLASS_NAMES = ["soil", "crop", "weed"]
TILE = 512
TILE_STRIDE = 384                            # overlap tiling for train
IMG_SIZE = 512

SEEDS = [0, 1, 2]
EPOCHS = 60
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Six candidate architectures (encoder, decoder) — all via segmentation_models_pytorch
MODELS = {
    "M1_unet_r34":        dict(arch="Unet",       encoder="resnet34",       weights="imagenet"),
    "M2_deeplabv3p_r50":  dict(arch="DeepLabV3Plus", encoder="resnet50",   weights="imagenet"),
    "M3_pspnet_r50":      dict(arch="PSPNet",     encoder="resnet50",       weights="imagenet"),
    "M4_unet_mbv3":       dict(arch="Unet",       encoder="timm-mobilenetv3_large_100", weights="imagenet"),
    "M5_segformer_b0":    dict(arch="Segformer",  encoder="mit_b0",         weights="imagenet"),
    "M6_segformer_b2":    dict(arch="Segformer",  encoder="mit_b2",         weights="imagenet"),
}

# Pruning study
SPARSITIES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
PRUNE_FINETUNE_EPOCHS = 10                   # per iterative step
PRUNE_FINETUNE_LR = 1e-4
