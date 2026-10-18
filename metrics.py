"""Confusion-matrix based segmentation metrics."""
import numpy as np
import torch

import config as C


class SegMetrics:
    def __init__(self, n=C.NUM_CLASSES):
        self.n = n
        self.cm = np.zeros((n, n), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        p = pred.detach().cpu().numpy().ravel()
        t = target.detach().cpu().numpy().ravel()
        self.cm += np.bincount(self.n * t + p, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self) -> dict:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        out = {"mIoU": float(iou.mean()), "pixel_acc": float(tp.sum() / cm.sum()), "mF1": float(f1.mean())}
        for i, c in enumerate(C.CLASS_NAMES):
            out[f"IoU_{c}"] = float(iou[i]); out[f"F1_{c}"] = float(f1[i])
        return out
