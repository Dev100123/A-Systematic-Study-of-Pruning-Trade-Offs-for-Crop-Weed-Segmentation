"""On-device benchmark for TensorRT engines on Jetson Orin Nano:
latency (ms), FPS, GPU memory, average power (W) via tegrastats, energy per image (mJ),
and mIoU of the engine on the CWFID test split.

Usage (on the Jetson):
  python benchmark_jetson.py --engines runs/M5_segformer_b0/seed0/export/*.engine
Writes results/table5_device.csv
"""
import argparse
import csv
import glob
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import torch

import config as C
import data as D
from metrics import SegMetrics

WARMUP, ITERS = 100, 1000


class TRTModel:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inp = self.engine.get_tensor_name(0); self.out = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.inp)); self.out_shape = tuple(self.engine.get_tensor_shape(self.out))
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * 4)
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * 4)
        self.h_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)), dtype=np.float32)
        self.ctx.set_tensor_address(self.inp, int(self.d_in)); self.ctx.set_tensor_address(self.out, int(self.d_out))
        self.stream = cuda.Stream()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        cuda.memcpy_htod_async(self.d_in, np.ascontiguousarray(x, dtype=np.float32), self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream); self.stream.synchronize()
        return self.h_out.reshape(self.out_shape)


class Tegrastats:
    """Samples VDD_IN (total board power, mW) from tegrastats while running."""
    def __enter__(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True); return self
    def __exit__(self, *a):
        self.p.terminate(); out = self.p.stdout.read()
        vals = [int(m) for m in re.findall(r"VDD_IN[^0-9]*(\d+)mW", out)]
        self.avg_w = np.mean(vals) / 1000 if vals else float("nan")


def latency(m):
    x = np.random.randn(*m.in_shape).astype(np.float32)
    for _ in range(WARMUP): m(x)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); m(x); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3, np.percentile(ts, 95) * 1e3


def miou(m):
    met = SegMetrics()
    for img, lbl in D.FullImageDataset():
        img = img[None]
        _, _, H, W = img.shape
        logits = np.zeros((C.NUM_CLASSES, H, W), np.float32); cnt = np.zeros((1, H, W), np.float32)
        ys = sorted(set(list(range(0, H - C.TILE + 1, C.TILE_STRIDE)) + [H - C.TILE]))
        xs = sorted(set(list(range(0, W - C.TILE + 1, C.TILE_STRIDE)) + [W - C.TILE]))
        for y in ys:
            for x in xs:
                logits[:, y:y + C.TILE, x:x + C.TILE] += m(img[:, :, y:y + C.TILE, x:x + C.TILE].numpy())[0]
                cnt[:, y:y + C.TILE, x:x + C.TILE] += 1
        met.update(torch.from_numpy((logits / cnt).argmax(0)), lbl)
    return met.compute()["mIoU"]


def gpu_mem_mb():
    try:
        free, total = cuda.mem_get_info(); return (total - free) / 1e6
    except Exception:
        return float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--engines", nargs="+", required=True); a = ap.parse_args()
    rows = []
    for e in sorted(sum([glob.glob(p) for p in a.engines], [])):
        m = TRTModel(e)
        mem0 = gpu_mem_mb()
        with Tegrastats() as ts:
            med, p95 = latency(m)
        acc = miou(m)
        row = dict(engine=Path(e).name, latency_ms=round(med, 2), p95_ms=round(p95, 2), fps=round(1000 / med, 1),
                   gpu_mem_MB=round(gpu_mem_mb() - mem0, 1), power_W=round(ts.avg_w, 2),
                   energy_mJ=round(ts.avg_w * med, 2), mIoU=round(acc, 4), size_MB=round(Path(e).stat().st_size / 1e6, 1))
        print(row); rows.append(row)
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(C.RESULTS_DIR / "table5_device.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
