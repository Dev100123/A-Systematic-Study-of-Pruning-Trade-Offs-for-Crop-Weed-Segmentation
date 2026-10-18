"""Model factory + complexity counters."""
import segmentation_models_pytorch as smp
import torch

import config as C


def build(name: str, pretrained: bool = True) -> torch.nn.Module:
    cfg = C.MODELS[name]
    cls = getattr(smp, cfg["arch"])
    return cls(encoder_name=cfg["encoder"],
               encoder_weights=cfg["weights"] if pretrained else None,
               in_channels=3, classes=C.NUM_CLASSES)


def count_params(model) -> float:
    """Millions of parameters (counts only non-zero if pruning masks are baked in)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_nonzero_params(model) -> float:
    """Non-zero parameters in millions; honours torch.nn.utils.prune masks (uses m.weight, not weight_orig)."""
    nz = 0
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if n.endswith("_orig"):
                nz += int((getattr(m, n[:-5]) != 0).sum())
            elif not n.endswith("_mask"):
                nz += int((p != 0).sum())
    return nz / 1e6


def count_flops(model, size=C.IMG_SIZE) -> float:
    """GFLOPs (MACs*2) for a single size x size image, via thop."""
    from thop import profile
    model = model.eval()
    dev = next(model.parameters()).device
    x = torch.randn(1, 3, size, size, device=dev)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return 2 * macs / 1e9


def size_mb(model) -> float:
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


if __name__ == "__main__":
    for n in C.MODELS:
        m = build(n, pretrained=False)
        print(f"{n:22s} params={count_params(m):6.2f}M  GFLOPs={count_flops(m):7.2f}  size={size_mb(m):6.1f}MB")
