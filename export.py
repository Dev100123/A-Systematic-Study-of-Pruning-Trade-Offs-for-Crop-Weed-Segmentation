"""Export dense / pruned checkpoints to ONNX and build TensorRT engines (FP32 / FP16 / INT8).

Usage (on the Jetson, or any machine with TensorRT):
  python export.py --model M5_segformer_b0 --seed 0                          # dense
  python export.py --model M5_segformer_b0 --seed 0 --pruned structured/iterative --sparsity 50
Builds:  runs/<model>/seed<seed>/export/<tag>.onnx and <tag>_{fp32,fp16,int8}.engine
INT8 uses trtexec entropy calibration with a calibration cache from train tiles.
"""
import argparse
import shutil
import subprocess

import numpy as np
import torch

import config as C
import data as D
import models as M


def load_model(name, seed, pruned=None, sparsity=None):
    run_dir = C.RUNS_DIR / name / f"seed{seed}"
    if pruned is None:
        m = M.build(name, pretrained=False)
        m.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        tag = "dense"
    else:
        mode, schedule = pruned.split("/")
        ck = torch.load(run_dir / f"prune_{mode}_{schedule}" / f"sparsity_{sparsity}.pt", map_location="cpu", weights_only=False)
        if isinstance(ck, torch.nn.Module):          # structured: whole (smaller) model was saved
            m = ck
        else:                                        # unstructured: state_dict with zeros baked in
            m = M.build(name, pretrained=False); m.load_state_dict(ck)
        tag = f"{mode}_{schedule}_{sparsity}"
    return m.eval(), tag


def to_onnx(model, path):
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    torch.onnx.export(model, x, str(path), opset_version=18, input_names=["input"], output_names=["logits"],
                      dynamic_axes=None, do_constant_folding=True)
    print("onnx ->", path)


def write_calib_images(n=64):
    """Dump n normalised train tiles as raw float32 .bin for trtexec INT8 calibration is not
    supported directly; instead we build INT8 with a python calibrator (see build_int8)."""
    ds = D.TileDataset(D.eval_tf())
    xs = np.stack([ds[i][0].numpy() for i in range(min(n, len(ds)))]).astype(np.float32)
    p = C.RUNS_DIR / "calib.npy"; np.save(p, xs); return p


def build_engine(onnx, engine, precision, calib=None):
    """FP32/FP16 via trtexec; INT8 via TensorRT python API with entropy calibrator."""
    if precision in ("fp32", "fp16"):
        cmd = ["trtexec", f"--onnx={onnx}", f"--saveEngine={engine}", "--workspace=4096"]
        if precision == "fp16": cmd.append("--fp16")
        subprocess.run(cmd, check=True)
        return
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    assert parser.parse(open(onnx, "rb").read()), [parser.get_error(i) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.FP16)

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self, arr):
            super().__init__(); self.arr = arr; self.i = 0
            import pycuda.autoinit, pycuda.driver as cuda  # noqa
            self.cuda = cuda; self.dev = cuda.mem_alloc(arr[0].nbytes)
        def get_batch_size(self): return 1
        def get_batch(self, names):
            if self.i >= len(self.arr): return None
            self.cuda.memcpy_htod(self.dev, np.ascontiguousarray(self.arr[self.i])); self.i += 1
            return [int(self.dev)]
        def read_calibration_cache(self): return None
        def write_calibration_cache(self, cache): open(str(engine) + ".calib", "wb").write(cache)

    cfg.int8_calibrator = Calib(np.load(calib))
    ser = builder.build_serialized_network(net, cfg)
    open(engine, "wb").write(ser)
    print("engine ->", engine)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pruned", default=None, help="e.g. structured/iterative")
    ap.add_argument("--sparsity", type=int, default=None, help="percent, e.g. 50")
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "int8"])
    a = ap.parse_args()
    model, tag = load_model(a.model, a.seed, a.pruned, a.sparsity)
    out = C.RUNS_DIR / a.model / f"seed{a.seed}" / "export"; out.mkdir(parents=True, exist_ok=True)
    onnx = out / f"{tag}.onnx"; to_onnx(model, onnx)
    if shutil.which("trtexec") is None:
        print("trtexec not found — ONNX exported only (run engine build on the Jetson)."); raise SystemExit
    calib = write_calib_images()
    for p in a.precisions:
        build_engine(onnx, out / f"{tag}_{p}.engine", p, calib)
