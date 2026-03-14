"""
Generate comparison plots across all CPU/GPU × hidden_dim runs.

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


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def extract(manifest, framework):
    """Return {(workers, horizon): elapsed_s} for given framework."""
    return {
        (e["workers"], e["horizon"]): e["elapsed_s"]
        for e in manifest
        if e["framework"] == framework and e["status"] == "OK"
    }


def safe_div(a, b):
    if b and b != 0:
        return a / b
    return np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all four runs
    runs = {}
    for name in ["cpu-confirm", "gpu-confirm", "cpu-large", "gpu-large"]:
        p = RESULTS_DIR / name / "manifest.json"
        if p.exists():
            runs[name] = load_manifest(p)
            print(f"  Loaded {name} ({len(runs[name])} experiments)")
        else:
            print(f"  SKIP {name} (not found)")

    workers_list = [1, 2, 4, 8]
    horizons = [128, 256, 512]

    # ── Figure 1: hidden_dim=64 CPU vs GPU elapsed ──
    if "cpu-confirm" in runs and "gpu-confirm" in runs:
        cpu64 = runs["cpu-confirm"]
        gpu64 = runs["gpu-confirm"]
        _plot_elapsed(cpu64, gpu64, 64, workers_list, horizons)

    # ── Figure 2: hidden_dim=1024 CPU vs GPU elapsed ──
    if "cpu-large" in runs and "gpu-large" in runs:
        cpu1k = runs["cpu-large"]
        gpu1k = runs["gpu-large"]
        _plot_elapsed(cpu1k, gpu1k, 1024, workers_list, horizons)

    # ── Figure 3: Speedup ratio (Ray/Monarch) across all 4 configs ──
    _plot_speedup_all(runs, workers_list, horizons)

    # ── Figure 4: GPU vs CPU speedup — does GPU actually help? ──
    _plot_gpu_benefit(runs, workers_list)

    # ── Figure 5: Bar chart comparison at 8 workers, horizon=512 ──
    _plot_bar_comparison(runs)

    print(f"\nAll figures saved to {OUT_DIR}")


def _plot_elapsed(cpu_data, gpu_data, hidden_dim, workers_list, horizons):
    styles_cpu = {"linestyle": "-", "marker": "o"}
    styles_gpu = {"linestyle": "--", "marker": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(f"PPO Training Time: CPU vs GPU (hidden_dim={hidden_dim})", fontsize=12)

    for ax, h in zip(axes, horizons):
        for fw, color in COLORS.items():
            cd = extract(cpu_data, fw)
            gd = extract(gpu_data, fw)
            ax.plot(workers_list, [cd.get((w, h), np.nan) for w in workers_list],
                    color=color, label=f"{fw} CPU", **styles_cpu)
            ax.plot(workers_list, [gd.get((w, h), np.nan) for w in workers_list],
                    color=color, label=f"{fw} GPU", **styles_gpu)
        ax.set_title(f"horizon={h}")
        ax.set_xlabel("Workers")
        ax.set_xticks(workers_list)
        ax.set_ylabel("Elapsed (s)" if h == horizons[0] else "")
        ax.grid(True, alpha=0.3)
        if h == horizons[0]:
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fname = f"fig_elapsed_dim{hidden_dim}.png"
    fig.savefig(OUT_DIR / fname)
    print(f"  Saved {fname}")


def _plot_speedup_all(runs, workers_list, horizons):
    """Ray/Monarch ratio for each (device, hidden_dim) combo."""
    configs = []
    labels = []
    colors_line = []
    markers = []

    if "cpu-confirm" in runs and "gpu-confirm" in runs:
        configs.append(("cpu-confirm", "CPU dim=64"))
        colors_line.append("#2ECC71")
        markers.append("o")
        configs.append(("gpu-confirm", "GPU dim=64"))
        colors_line.append("#27AE60")
        markers.append("s")
    if "cpu-large" in runs and "gpu-large" in runs:
        configs.append(("cpu-large", "CPU dim=1024"))
        colors_line.append("#9B59B6")
        markers.append("o")
        configs.append(("gpu-large", "GPU dim=1024"))
        colors_line.append("#8E44AD")
        markers.append("s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Ray/Monarch Time Ratio (>1 = Monarch faster)", fontsize=12)

    for ax, h in zip(axes, horizons):
        for (run_name, label), color, marker in zip(configs, colors_line, markers):
            data = runs[run_name]
            ray_d = extract(data, "ray")
            mon_d = extract(data, "monarch")
            ratios = [safe_div(ray_d.get((w, h), np.nan), mon_d.get((w, h), np.nan))
                      for w in workers_list]
            ax.plot(workers_list, ratios, color=color, marker=marker, label=label)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"horizon={h}")
        ax.set_xlabel("Workers")
        ax.set_xticks(workers_list)
        ax.set_ylabel("Ray time / Monarch time" if h == horizons[0] else "")
        ax.grid(True, alpha=0.3)
        if h == horizons[0]:
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_speedup_all.png")
    print(f"  Saved fig_speedup_all.png")


def _plot_gpu_benefit(runs, workers_list):
    """Does GPU help? Plot CPU_time / GPU_time for each framework × hidden_dim."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle("GPU Benefit: CPU time / GPU time (>1 = GPU faster)", fontsize=12)

    h = 512  # most compute-intensive horizon

    pairs = [
        (("cpu-confirm", "gpu-confirm"), "dim=64", ["#E74C3C", "#3498DB"]),
        (("cpu-large", "gpu-large"), "dim=1024", ["#C0392B", "#2980B9"]),
    ]

    for ax, ((cpu_name, gpu_name), label, fw_colors) in zip(axes, pairs):
        if cpu_name not in runs or gpu_name not in runs:
            continue
        for fw, color in zip(["ray", "monarch"], fw_colors):
            cd = extract(runs[cpu_name], fw)
            gd = extract(runs[gpu_name], fw)
            ratios = [safe_div(cd.get((w, h), np.nan), gd.get((w, h), np.nan))
                      for w in workers_list]
            ax.plot(workers_list, ratios, color=color, marker="o", label=fw)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="break-even")
        ax.set_title(f"horizon=512, {label}")
        ax.set_xlabel("Workers")
        ax.set_xticks(workers_list)
        ax.set_ylabel("CPU time / GPU time" if label == "dim=64" else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gpu_benefit.png")
    print(f"  Saved fig_gpu_benefit.png")


def _plot_bar_comparison(runs):
    """Bar chart: all configs at 8 workers, horizon=512."""
    h, w = 512, 8

    groups = []
    for run_name, device_label, dim in [
        ("cpu-confirm", "CPU", 64), ("gpu-confirm", "GPU", 64),
        ("cpu-large", "CPU", 1024), ("gpu-large", "GPU", 1024),
    ]:
        if run_name not in runs:
            continue
        for fw in ["ray", "monarch"]:
            val = extract(runs[run_name], fw).get((w, h), 0)
            groups.append({
                "label": f"{fw}\n{device_label}\nd={dim}",
                "val": val,
                "color": COLORS[fw],
                "alpha": 1.0 if device_label == "CPU" else 0.6,
            })

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(groups))
    bars = ax.bar(x, [g["val"] for g in groups],
                  color=[g["color"] for g in groups])
    for bar, g in zip(bars, groups):
        bar.set_alpha(g["alpha"])
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=9)
    ax.set_ylabel("Elapsed (s)")
    ax.set_title(f"All Configurations at {w} workers, horizon={h}", fontsize=12)
    for bar, g in zip(bars, groups):
        ax.text(bar.get_x() + bar.get_width()/2, g["val"] + 0.5,
                f"{g['val']:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_bar_all_configs.png")
    print(f"  Saved fig_bar_all_configs.png")


if __name__ == "__main__":
    main()
