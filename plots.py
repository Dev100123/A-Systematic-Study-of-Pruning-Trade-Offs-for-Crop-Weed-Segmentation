"""Aggregate result CSVs into paper tables and figures.

  python plots.py
Produces in results/: table3_pruning.csv, fig2_miou_vs_sparsity.pdf, fig3_oneshot_vs_iterative.pdf,
fig4_pareto.pdf (if table5_device.csv exists).
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

R = C.RESULTS_DIR
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def load_prune():
    fs = glob.glob(str(R / "prune_*.csv"))
    if not fs: return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["sparsity_pct"] = (df.sparsity * 100).round().astype(int)
    return df


def table3(df):
    g = df.groupby(["model", "mode", "schedule", "sparsity_pct"]).agg(
        mIoU=("mIoU", "mean"), mIoU_std=("mIoU", "std"), params_M=("params_M", "mean"),
        nonzero_M=("nonzero_params_M", "mean"), gflops=("gflops", "mean"), size_MB=("size_MB", "mean")).reset_index()
    dense = g[g.sparsity_pct == 0].set_index(["model", "mode", "schedule"]).mIoU
    g["delta_mIoU"] = g.apply(lambda r: r.mIoU - dense.loc[(r.model, r["mode"], r.schedule)], axis=1)
    g.round(4).to_csv(R / "table3_pruning.csv", index=False)
    return g


def fig_miou_vs_sparsity(g):
    models = g.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for (mode, sched), d in g[(g.model == m)].groupby(["mode", "schedule"]):
            ax.errorbar(d.sparsity_pct, d.mIoU, yerr=d.mIoU_std, marker="o", ms=3, capsize=2, label=f"{mode}/{sched}")
        ax.set_title(m); ax.set_xlabel("Sparsity (%)")
    axes[0][0].set_ylabel("mIoU"); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(R / "fig2_miou_vs_sparsity.pdf"); plt.close(fig)


def fig_oneshot_vs_iterative(g):
    d = g[g.schedule.isin(["oneshot", "iterative"])]
    if d.schedule.nunique() < 2: return
    fig, ax = plt.subplots(figsize=(4, 3))
    for (m, mode, s), dd in d.groupby(["model", "mode", "schedule"]):
        ax.plot(dd.sparsity_pct, dd.delta_mIoU, marker="o", ms=3, ls="-" if s == "iterative" else "--", label=f"{m} {mode} {s}")
    ax.axhline(0, c="k", lw=0.5); ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Δ mIoU vs dense"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(R / "fig3_oneshot_vs_iterative.pdf"); plt.close(fig)


def fig_pareto():
    p = R / "table5_device.csv"
    if not p.exists(): return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(d.latency_ms, d.mIoU, s=20)
    for _, r in d.iterrows(): ax.annotate(r.engine.replace(".engine", ""), (r.latency_ms, r.mIoU), fontsize=5)
    ax.set_xlabel("Latency on Jetson Orin Nano (ms)"); ax.set_ylabel("mIoU")
    fig.tight_layout(); fig.savefig(R / "fig4_pareto.pdf"); plt.close(fig)


if __name__ == "__main__":
    df = load_prune()
    if df is not None:
        g = table3(df); fig_miou_vs_sparsity(g); fig_oneshot_vs_iterative(g)
        print(g[["model", "mode", "schedule", "sparsity_pct", "mIoU", "delta_mIoU", "gflops"]].to_string(index=False))
    fig_pareto()
    print("done ->", R)
