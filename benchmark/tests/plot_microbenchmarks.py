"""
Generate figures from microbenchmark results.

Produces 4 figures:
  M1: Scheduling latency bar chart (Ray vs Monarch, per worker count)
  M2: Serialization cost vs payload size (log-scale X, lines per framework)
  M3: Per-worker rollout time box plot (per framework per worker count)
  M4: Straggler ratio line chart (max/mean vs worker count)

Usage:
    python benchmark/tests/plot_microbenchmarks.py
    python benchmark/tests/plot_microbenchmarks.py --results-dir path/to/microbenchmarks
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match existing plot.py style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

FW_STYLE = {
    "ray":     {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "Ray"},
    "monarch": {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "Monarch"},
}
FRAMEWORKS = ["ray", "monarch"]


def _load_json(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Figure M1: Scheduling latency bar chart
# ---------------------------------------------------------------------------

def fig_scheduling_latency(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Scheduling Latency (no-op ping)", y=1.02)

    bar_data = {}  # fw -> {workers: mean_ms}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"scheduling_{fw}.json")
        if not data:
            continue
        bar_data[fw] = {r["workers"]: r for r in data}

    if not bar_data:
        plt.close(fig)
        print("  Skipped figM1 (no scheduling data)")
        return

    all_workers = sorted({w for d in bar_data.values() for w in d})
    x = np.arange(len(all_workers))
    width = 0.35

    for i, fw in enumerate(FRAMEWORKS):
        if fw not in bar_data:
            continue
        means = [bar_data[fw].get(w, {}).get("mean_ms", 0) for w in all_workers]
        p99s = [bar_data[fw].get(w, {}).get("p99_ms", 0) for w in all_workers]
        style = FW_STYLE[fw]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, color=style["color"],
                       label=f"{style['label']} (mean)", alpha=0.8)
        # P99 as error bar cap
        ax.errorbar(x + offset, means,
                     yerr=[np.zeros(len(means)), [p - m for p, m in zip(p99s, means)]],
                     fmt="none", ecolor=style["color"], capsize=3, alpha=0.6)

    ax.set_xlabel("Workers")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_workers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "figM1_scheduling_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM1_scheduling_latency.png")


# ---------------------------------------------------------------------------
# Figure M2: Serialization cost vs payload size
# ---------------------------------------------------------------------------

def fig_serialization(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Serialization Round-Trip Time", y=1.02)

    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"serialization_{fw}.json")
        if not data:
            continue
        style = FW_STYLE[fw]

        # Model states
        model_data = [r for r in data if r["payload_type"] == "model_state"]
        if model_data:
            xs = [r["payload_bytes"] for r in model_data]
            ys = [r["roundtrip_mean_ms"] for r in model_data]
            axes[0].plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

        # Rollouts
        rollout_data = [r for r in data if r["payload_type"] == "rollout"]
        if rollout_data:
            xs = [r["payload_bytes"] for r in rollout_data]
            ys = [r["roundtrip_mean_ms"] for r in rollout_data]
            axes[1].plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

    for ax, title in zip(axes, ["Model State", "Rollout Tensor"]):
        ax.set_title(title)
        ax.set_xlabel("Payload Size (bytes)")
        ax.set_ylabel("Round-trip Time (ms)")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM2_serialization.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM2_serialization.png")


# ---------------------------------------------------------------------------
# Figure M3: Per-worker rollout time box plot
# ---------------------------------------------------------------------------

def fig_worker_times(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Per-Worker Rollout Time Distribution", y=1.02)

    all_data = {}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"worker_scaling_{fw}.json")
        if data:
            all_data[fw] = data

    if not all_data:
        plt.close(fig)
        print("  Skipped figM3 (no worker scaling data)")
        return

    all_workers = sorted({r["workers"] for d in all_data.values() for r in d})
    positions = []
    box_data = []
    colors = []
    labels = []
    group_positions = []

    for wi, w in enumerate(all_workers):
        group_center = wi * 3
        group_positions.append(group_center)
        for fi, fw in enumerate(FRAMEWORKS):
            if fw not in all_data:
                continue
            entry = next((r for r in all_data[fw] if r["workers"] == w), None)
            if not entry or "per_batch_worker_times" not in entry:
                continue
            flat = [t for batch in entry["per_batch_worker_times"] for t in batch]
            pos = group_center + (fi - 0.5) * 0.8
            positions.append(pos)
            box_data.append(flat)
            colors.append(FW_STYLE[fw]["color"])
            labels.append(f"{FW_STYLE[fw]['label']} {w}w")

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                         patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(w) for w in all_workers])
    ax.set_xlabel("Workers")
    ax.set_ylabel("Rollout Time (s)")
    ax.grid(True, alpha=0.3, axis="y")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FW_STYLE[fw]["color"], alpha=0.6, label=FW_STYLE[fw]["label"])
        for fw in FRAMEWORKS if fw in all_data
    ]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(out_dir / "figM3_worker_times.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM3_worker_times.png")


# ---------------------------------------------------------------------------
# Figure M4: Straggler ratio line chart
# ---------------------------------------------------------------------------

def fig_straggler_ratio(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Straggler Ratio (max / mean worker time)", y=1.02)

    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"worker_scaling_{fw}.json")
        if not data:
            continue
        style = FW_STYLE[fw]
        xs = [r["workers"] for r in data]
        ys = [r["straggler_ratio_mean"] for r in data]
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], label=style["label"])

    ax.set_xlabel("Workers")
    ax.set_ylabel("Straggler Ratio (max / mean)")
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle=":")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM4_straggler_ratio.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM4_straggler_ratio.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent.parent / "results" / "microbenchmarks")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir or results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run benchmark/tests/run_microbenchmarks.py first.")
        return

    print(f"Loading microbenchmark results from {results_dir}")
    fig_scheduling_latency(results_dir, out_dir)
    fig_serialization(results_dir, out_dir)
    fig_worker_times(results_dir, out_dir)
    fig_straggler_ratio(results_dir, out_dir)
    print(f"\nAll figures saved under {out_dir}")


if __name__ == "__main__":
    main()
