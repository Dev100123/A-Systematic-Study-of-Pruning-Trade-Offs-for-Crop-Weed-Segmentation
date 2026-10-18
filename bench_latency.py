"""Latency of dense vs structurally pruned M4 on the available hardware (PyTorch eager, batch 1, 512x512, FP32).
Writes results/latency_local.csv. Reviewer 2, comment 3."""
import csv, time, platform
import numpy as np, torch
import config as C
from infer import load

def bench(model, dev, iters=100, warm=20):
    model = model.to(dev).eval(); x = torch.randn(1, 3, 512, 512, device=dev)
    with torch.no_grad():
        for _ in range(warm): model(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); model(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return 1e3 * np.median(ts), 1e3 * np.percentile(ts, 95)

rows = []
cfgs = [("dense", None, None)] + [(f"structured/iterative {s}%", "structured/iterative", s) for s in (10, 20, 30)] + [("unstructured/iterative 80%", "unstructured/iterative", 80)]
for dev in [torch.device("cuda"), torch.device("cpu")]:
    torch.set_num_threads(8)
    for tag, pr, sp in cfgs:
        m = load("M4_unet_mbv3", 0, pr, sp)
        med, p95 = bench(m, dev, iters=100 if dev.type == "cuda" else 20, warm=20 if dev.type == "cuda" else 3)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        rows.append(dict(device=torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor(), config=tag, params_M=round(n, 2), median_ms=round(med, 2), p95_ms=round(p95, 2), fps=round(1000 / med, 1)))
        print(rows[-1], flush=True)
        del m; torch.cuda.empty_cache()
with open(C.RESULTS_DIR / "latency_local.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("saved results/latency_local.csv")
