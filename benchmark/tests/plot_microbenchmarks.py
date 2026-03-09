"""
Generate figures from microbenchmark results.

Produces 4 figures mapping to the 4 architectural differences:
  M1: Dispatch overhead — per-task (Ray) vs bulk (Monarch) scaling with worker count
  M2: Broadcast efficiency — object store vs controller-mediated transfer
  M3: Straggler sensitivity — framework overhead across variance levels
  M4: Iteration loop — amortized per-cycle coordination tax

Usage:
    python benchmark/tests/plot_microbenchmarks.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# M1: Dispatch Overhead
# ---------------------------------------------------------------------------

def fig_dispatch(results_dir: Path, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("M1: Dispatch Overhead (no-op tasks)", y=1.02)

    fw_data = {}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"dispatch_{fw}.json")
        if data:
            fw_data[fw] = {r["workers"]: r for r in data}

    if not fw_data:
        plt.close(fig)
        print("  Skipped figM1 (no dispatch data)")
        return

    # Left: absolute dispatch time vs workers
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        workers = sorted(fw_data[fw].keys())
        means = [fw_data[fw][w]["mean_ms"] for w in workers]
        stds = [fw_data[fw][w]["std_ms"] for w in workers]
        ax1.errorbar(workers, means, yerr=stds, capsize=3,
                     color=style["color"], linestyle=style["linestyle"],
                     marker=style["marker"], label=style["label"])

    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Dispatch + Collect Time (ms)")
    ax1.set_title("Absolute Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: normalized to 1-worker baseline (shows scaling slope)
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        workers = sorted(fw_data[fw].keys())
        means = [fw_data[fw][w]["mean_ms"] for w in workers]
        baseline = means[0] if means[0] > 0 else 1
        normalized = [m / baseline for m in means]
        ax2.plot(workers, normalized,
                 color=style["color"], linestyle=style["linestyle"],
                 marker=style["marker"], label=style["label"])

    ax2.plot(sorted(list(fw_data.values())[0].keys()),
             sorted(list(fw_data.values())[0].keys()),
             "k:", alpha=0.3, label="Linear scaling")
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Time / Time(1 worker)")
    ax2.set_title("Normalized Scaling")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM1_dispatch_overhead.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM1_dispatch_overhead.png")


# ---------------------------------------------------------------------------
# M2: Broadcast Efficiency
# ---------------------------------------------------------------------------

def fig_broadcast(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("M2: Broadcast Efficiency (same data to all workers)", y=1.02)

    fw_data = {}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"broadcast_{fw}.json")
        if data:
            fw_data[fw] = data

    if not fw_data:
        plt.close(fig)
        print("  Skipped figM2 (no broadcast data)")
        return

    # Left: time vs payload size, one line per framework (at max workers)
    ax = axes[0]
    max_workers = max(r["workers"] for d in fw_data.values() for r in d)
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        subset = [r for r in fw_data[fw] if r["workers"] == max_workers]
        subset.sort(key=lambda r: r["payload_bytes"])
        xs = [r["payload_bytes"] / 1000 for r in subset]  # KB
        ys = [r["total_mean_ms"] for r in subset]
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], label=f"{style['label']} ({max_workers}w)")

    ax.set_xlabel("Payload Size (KB)")
    ax.set_ylabel("Broadcast Time (ms)")
    ax.set_title(f"Time vs Payload ({max_workers} workers)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: time vs worker count, one line per framework (at largest payload)
    ax = axes[1]
    max_payload = max(r["payload_bytes"] for d in fw_data.values() for r in d)
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        subset = [r for r in fw_data[fw] if r["payload_bytes"] == max_payload]
        subset.sort(key=lambda r: r["workers"])
        xs = [r["workers"] for r in subset]
        ys = [r["total_mean_ms"] for r in subset]
        payload_label = f"{max_payload/1e6:.1f}MB"
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], label=f"{style['label']} ({payload_label})")

    ax.set_xlabel("Workers")
    ax.set_ylabel("Broadcast Time (ms)")
    ax.set_title(f"Time vs Workers (largest payload)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM2_broadcast_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM2_broadcast_efficiency.png")


# ---------------------------------------------------------------------------
# M3: Straggler Sensitivity
# ---------------------------------------------------------------------------

def fig_straggler(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("M3: Straggler Sensitivity (variable worker durations)", y=1.02)

    fw_data = {}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"straggler_{fw}.json")
        if data:
            fw_data[fw] = data

    if not fw_data:
        plt.close(fig)
        print("  Skipped figM3 (no straggler data)")
        return

    variance_names = ["uniform", "low_var", "med_var", "high_var", "extreme"]

    # Left: overhead (ms) vs variance level, one line per (framework, worker_count)
    ax = axes[0]
    all_workers = sorted({r["workers"] for d in fw_data.values() for r in d})
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        for w in all_workers:
            subset = [r for r in fw_data[fw]
                      if r["workers"] == w and r["variance"] in variance_names]
            subset.sort(key=lambda r: variance_names.index(r["variance"]))
            if not subset:
                continue
            xs = list(range(len(subset)))
            ys = [r["overhead_mean_ms"] for r in subset]
            alpha = 0.4 + 0.6 * (all_workers.index(w) / max(len(all_workers) - 1, 1))
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], alpha=alpha,
                    label=f"{style['label']} {w}w")

    ax.set_xticks(range(len(variance_names)))
    ax.set_xticklabels(variance_names, rotation=30, ha="right")
    ax.set_xlabel("Variance Level")
    ax.set_ylabel("Framework Overhead (ms)")
    ax.set_title("Overhead = wall_clock - max(worker_times)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: overhead as % of wall time
    ax = axes[1]
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        for w in all_workers:
            subset = [r for r in fw_data[fw]
                      if r["workers"] == w and r["variance"] in variance_names]
            subset.sort(key=lambda r: variance_names.index(r["variance"]))
            if not subset:
                continue
            xs = list(range(len(subset)))
            ys = [r["overhead_pct"] for r in subset]
            alpha = 0.4 + 0.6 * (all_workers.index(w) / max(len(all_workers) - 1, 1))
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], alpha=alpha,
                    label=f"{style['label']} {w}w")

    ax.set_xticks(range(len(variance_names)))
    ax.set_xticklabels(variance_names, rotation=30, ha="right")
    ax.set_xlabel("Variance Level")
    ax.set_ylabel("Overhead (% of wall time)")
    ax.set_title("Overhead Fraction")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM3_straggler_overhead.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM3_straggler_overhead.png")


# ---------------------------------------------------------------------------
# M4: Iteration Loop Overhead
# ---------------------------------------------------------------------------

def fig_iteration_loop(results_dir: Path, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("M4: Iteration Loop Overhead (broadcast + compute + gather)", y=1.02)

    fw_data = {}
    for fw in FRAMEWORKS:
        data = _load_json(results_dir / f"iteration_loop_{fw}.json")
        if data:
            fw_data[fw] = {r["workers"]: r for r in data}

    if not fw_data:
        plt.close(fig)
        print("  Skipped figM4 (no iteration loop data)")
        return

    # Left: total cycle time and overhead vs workers
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        workers = sorted(fw_data[fw].keys())
        cycles = [fw_data[fw][w]["cycle_mean_ms"] for w in workers]
        overheads = [fw_data[fw][w]["overhead_mean_ms"] for w in workers]

        ax1.plot(workers, cycles, color=style["color"], linestyle=style["linestyle"],
                 marker=style["marker"], label=f"{style['label']} (total)")
        ax1.plot(workers, overheads, color=style["color"], linestyle=":",
                 marker=style["marker"], alpha=0.6,
                 label=f"{style['label']} (overhead)")

    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Per-Cycle Timing")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: overhead percentage vs workers
    for fw in FRAMEWORKS:
        if fw not in fw_data:
            continue
        style = FW_STYLE[fw]
        workers = sorted(fw_data[fw].keys())
        pcts = [fw_data[fw][w]["overhead_pct"] for w in workers]
        ax2.plot(workers, pcts, color=style["color"], linestyle=style["linestyle"],
                 marker=style["marker"], label=style["label"])

    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Framework Overhead (% of cycle)")
    ax2.set_title("Overhead Fraction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figM4_iteration_loop.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figM4_iteration_loop.png")


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
    fig_dispatch(results_dir, out_dir)
    fig_broadcast(results_dir, out_dir)
    fig_straggler(results_dir, out_dir)
    fig_iteration_loop(results_dir, out_dir)
    print(f"\nAll figures saved under {out_dir}")


if __name__ == "__main__":
    main()
