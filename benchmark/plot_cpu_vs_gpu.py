"""
Generate comparison plots: CPU-confirm vs GPU-confirm results.

Usage:
    python benchmark/plot_cpu_vs_gpu.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CPU_DIR = RESULTS_DIR / "cpu-confirm"
GPU_DIR = RESULTS_DIR / "gpu-confirm"
OUT_DIR = RESULTS_DIR / "comparison"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

COLORS = {
    "ray": "#E74C3C",
    "monarch": "#3498DB",
}
STYLES = {
    "cpu": {"linestyle": "-", "marker": "o"},
    "gpu": {"linestyle": "--", "marker": "s"},
}


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def extract(manifest, framework, metric="elapsed_s"):
    """Return {(workers, horizon): value} for given framework."""
    return {
        (e["workers"], e["horizon"]): e[metric]
        for e in manifest
        if e["framework"] == framework and e["status"] == "OK"
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cpu = load_manifest(CPU_DIR / "manifest.json")
    gpu = load_manifest(GPU_DIR / "manifest.json")

    workers_list = [1, 2, 4, 8]
    horizons = [128, 256, 512]

    # ── Figure 1: Elapsed time vs workers, one subplot per horizon ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("PPO Training Time: CPU vs GPU (30 iterations, hidden_dim=64)", fontsize=12)

    for ax, h in zip(axes, horizons):
        for fw, color in COLORS.items():
            cpu_data = extract(cpu, fw)
            gpu_data = extract(gpu, fw)
            cpu_y = [cpu_data.get((w, h), np.nan) for w in workers_list]
            gpu_y = [gpu_data.get((w, h), np.nan) for w in workers_list]
            ax.plot(workers_list, cpu_y, color=color, label=f"{fw} CPU",
                    **STYLES["cpu"])
            ax.plot(workers_list, gpu_y, color=color, label=f"{fw} GPU",
                    **STYLES["gpu"])
        ax.set_title(f"horizon={h}")
        ax.set_xlabel("Workers")
        ax.set_xticks(workers_list)
        ax.set_ylabel("Elapsed (s)" if h == 128 else "")
        ax.grid(True, alpha=0.3)
        if h == 128:
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cpu_vs_gpu_elapsed.png")
    print(f"  Saved fig_cpu_vs_gpu_elapsed.png")

    # ── Figure 2: Speedup ratio (Monarch/Ray) for CPU vs GPU ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Ray/Monarch Time Ratio (>1 = Monarch faster)", fontsize=12)

    for ax, h in zip(axes, horizons):
        ray_cpu = extract(cpu, "ray")
        mon_cpu = extract(cpu, "monarch")
        ray_gpu = extract(gpu, "ray")
        mon_gpu = extract(gpu, "monarch")

        cpu_ratio = [ray_cpu.get((w, h), np.nan) / mon_cpu.get((w, h), np.nan)
                     for w in workers_list]
        gpu_ratio = [ray_gpu.get((w, h), np.nan) / mon_gpu.get((w, h), np.nan)
                     for w in workers_list]

        ax.plot(workers_list, cpu_ratio, color="#2ECC71", marker="o",
                label="CPU mode")
        ax.plot(workers_list, gpu_ratio, color="#9B59B6", marker="s",
                label="GPU mode")
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"horizon={h}")
        ax.set_xlabel("Workers")
        ax.set_xticks(workers_list)
        ax.set_ylabel("Ray time / Monarch time" if h == 128 else "")
        ax.grid(True, alpha=0.3)
        if h == 128:
            ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_speedup_ratio.png")
    print(f"  Saved fig_speedup_ratio.png")

    # ── Figure 3: Scaling efficiency (time at Nw / time at 1w) ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Scaling Cost: Time(Nw) / Time(1w) at horizon=512", fontsize=12)

    h = 512
    configs = [
        ("Ray CPU", extract(cpu, "ray"), COLORS["ray"], STYLES["cpu"]),
        ("Ray GPU", extract(gpu, "ray"), COLORS["ray"], STYLES["gpu"]),
        ("Monarch CPU", extract(cpu, "monarch"), COLORS["monarch"], STYLES["cpu"]),
        ("Monarch GPU", extract(gpu, "monarch"), COLORS["monarch"], STYLES["gpu"]),
    ]

    ax = axes[0, 0]
    for label, data, color, style in configs:
        base = data.get((1, h), np.nan)
        ratios = [data.get((w, h), np.nan) / base for w in workers_list]
        ax.plot(workers_list, ratios, color=color, label=label, **style)
    ax.set_xlabel("Workers")
    ax.set_ylabel("Time / Time(1 worker)")
    ax.set_xticks(workers_list)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Scaling cost (horizon=512)")

    # Bar chart: absolute times at 8 workers, horizon=512
    ax = axes[0, 1]
    labels = ["Ray\nCPU", "Ray\nGPU", "Monarch\nCPU", "Monarch\nGPU"]
    vals = [
        extract(cpu, "ray").get((8, 512), 0),
        extract(gpu, "ray").get((8, 512), 0),
        extract(cpu, "monarch").get((8, 512), 0),
        extract(gpu, "monarch").get((8, 512), 0),
    ]
    colors = [COLORS["ray"], COLORS["ray"], COLORS["monarch"], COLORS["monarch"]]
    alphas = [1.0, 0.5, 1.0, 0.5]
    bars = ax.bar(labels, vals, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.set_ylabel("Elapsed (s)")
    ax.set_title("8 workers, horizon=512")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Bar chart: absolute times at 1 worker, horizon=512
    ax = axes[1, 0]
    vals_1w = [
        extract(cpu, "ray").get((1, 512), 0),
        extract(gpu, "ray").get((1, 512), 0),
        extract(cpu, "monarch").get((1, 512), 0),
        extract(gpu, "monarch").get((1, 512), 0),
    ]
    bars = ax.bar(labels, vals_1w, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.set_ylabel("Elapsed (s)")
    ax.set_title("1 worker, horizon=512")
    for bar, v in zip(bars, vals_1w):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Summary table as text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "Key findings (horizon=512):\n\n"
        f"CPU is faster than GPU for all configs\n"
        f"  (hidden_dim=64 too small for GPU)\n\n"
        f"Monarch wins in both modes:\n"
        f"  CPU 8w: {extract(cpu,'monarch').get((8,512),0):.1f}s vs {extract(cpu,'ray').get((8,512),0):.1f}s "
        f"({extract(cpu,'ray').get((8,512),0)/extract(cpu,'monarch').get((8,512),0):.1f}x)\n"
        f"  GPU 8w: {extract(gpu,'monarch').get((8,512),0):.1f}s vs {extract(gpu,'ray').get((8,512),0):.1f}s "
        f"({extract(gpu,'ray').get((8,512),0)/extract(gpu,'monarch').get((8,512),0):.1f}x)\n\n"
        f"Ray scaling cost (1w→8w):\n"
        f"  CPU: {extract(cpu,'ray').get((8,512),0)/extract(cpu,'ray').get((1,512),0):.2f}x\n"
        f"  GPU: {extract(gpu,'ray').get((8,512),0)/extract(gpu,'ray').get((1,512),0):.2f}x\n\n"
        f"Monarch scaling cost (1w→8w):\n"
        f"  CPU: {extract(cpu,'monarch').get((8,512),0)/extract(cpu,'monarch').get((1,512),0):.2f}x\n"
        f"  GPU: {extract(gpu,'monarch').get((8,512),0)/extract(gpu,'monarch').get((1,512),0):.2f}x"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scaling_comparison.png")
    print(f"  Saved fig_scaling_comparison.png")

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
