"""CWFID download, tiling, dataset and augmentations.

CWFID layout (github.com/cwfid/dataset):
  images/001_image.png            RGB 1296x966
  annotations/001_annotation.png  RGB mask: green=crop, red=weed, black=soil
  train_test_split.yaml           {train: [ids], test: [ids]}
"""
import subprocess
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import config as C

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def download():
    if not (C.DATA_DIR / "images").exists():
        C.DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", C.CWFID_REPO, str(C.DATA_DIR)], check=True)
    print("CWFID present at", C.DATA_DIR)


def rgb_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """green (0,255,0)->1 crop ; red (255,0,0)->2 weed ; else 0 soil."""
    r, g, b = mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]
    lbl = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    lbl[(g > 127) & (r < 127)] = 1
    lbl[(r > 127) & (g < 127)] = 2
    return lbl


def read_split():
    with open(C.DATA_DIR / "train_test_split.yaml") as f:
        s = yaml.safe_load(f)
    return [int(i) for i in s["train"]], [int(i) for i in s["test"]]


def tile_dataset():
    """Tile train images with overlap; keep test images whole (evaluated by sliding window)."""
    train_ids, _ = read_split()
    out = C.TILE_DIR / "train"
    if out.exists() and any(out.glob("*_img.png")):
        print("tiles exist"); return
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in train_ids:
        img = cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "images" / f"{i:03d}_image.png")), cv2.COLOR_BGR2RGB)
        lbl = rgb_to_label(cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "annotations" / f"{i:03d}_annotation.png")), cv2.COLOR_BGR2RGB))
        H, W = lbl.shape
        ys = list(range(0, H - C.TILE + 1, C.TILE_STRIDE)) + [H - C.TILE]
        xs = list(range(0, W - C.TILE + 1, C.TILE_STRIDE)) + [W - C.TILE]
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                cv2.imwrite(str(out / f"{i:03d}_{y}_{x}_img.png"), cv2.cvtColor(img[y:y + C.TILE, x:x + C.TILE], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out / f"{i:03d}_{y}_{x}_lbl.png"), lbl[y:y + C.TILE, x:x + C.TILE])
                n += 1
    print(f"wrote {n} train tiles")


def train_tf():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.5),
        A.GaussianBlur(p=0.1),
        A.Normalize(MEAN, STD), ToTensorV2(),
    ])


def eval_tf():
    return A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])


class TileDataset(Dataset):
    def __init__(self, tf):
        self.imgs = sorted(C.TILE_DIR.glob("train/*_img.png"))
        self.tf = tf

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(p).replace("_img.png", "_lbl.png"), cv2.IMREAD_GRAYSCALE)
        o = self.tf(image=img, mask=lbl)
        return o["image"], o["mask"].long()


class FullImageDataset(Dataset):
    """Whole CWFID images (test split by default) for sliding-window evaluation."""
    def __init__(self, ids=None):
        if ids is None:
            _, ids = read_split()
        self.ids = ids
        self.tf = eval_tf()

    def __len__(self): return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        img = cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "images" / f"{i:03d}_image.png")), cv2.COLOR_BGR2RGB)
        lbl = rgb_to_label(cv2.cvtColor(cv2.imread(str(C.DATA_DIR / "annotations" / f"{i:03d}_annotation.png")), cv2.COLOR_BGR2RGB))
        o = self.tf(image=img, mask=lbl)
        return o["image"], o["mask"].long()


@torch.no_grad()
def sliding_window_predict(model, img: torch.Tensor, tile=C.TILE, stride=C.TILE_STRIDE, num_classes=C.NUM_CLASSES):
    """img: (1,3,H,W) normalised. Returns logits (1,C,H,W) averaged over overlapping tiles."""
    _, _, H, W = img.shape
    logits = torch.zeros(1, num_classes, H, W, device=img.device)
    count = torch.zeros(1, 1, H, W, device=img.device)
    ys = sorted(set(list(range(0, H - tile + 1, stride)) + [H - tile]))
    xs = sorted(set(list(range(0, W - tile + 1, stride)) + [W - tile]))
    for y in ys:
        for x in xs:
            out = model(img[:, :, y:y + tile, x:x + tile])
            logits[:, :, y:y + tile, x:x + tile] += out
            count[:, :, y:y + tile, x:x + tile] += 1
    return logits / count


if __name__ == "__main__":
    download()
    tile_dataset()
