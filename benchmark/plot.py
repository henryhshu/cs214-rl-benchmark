"""
Generate benchmark figures from results produced by run_all.py.

Usage:
    python benchmark/plot.py                         # uses benchmark/results/
    python benchmark/plot.py --results-dir path/to/results

Outputs six PNG figures to benchmark/results/figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
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
WORKERS = [1, 2, 4, 8]
HORIZONS = [128, 256, 512]
FRAMEWORKS = ["ray", "monarch"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(exp_dir: Path) -> list[dict]:
    p = exp_dir / "metrics.jsonl"
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_sysmon(exp_dir: Path) -> list[dict]:
    p = exp_dir / "sysmon.jsonl"
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_all(results_dir: Path) -> dict:
    """Return nested dict: data[framework][workers][horizon] = {metrics, sysmon, config}"""
    data = {}
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        fw = cfg["framework"]
        w = cfg["workers"]
        h = cfg["horizon"]
        data.setdefault(fw, {}).setdefault(w, {})[h] = {
            "metrics": load_metrics(exp_dir),
            "sysmon": load_sysmon(exp_dir),
            "config": cfg,
        }
    return data


def _mean_field(records: list[dict], field: str, skip_first: int = 2) -> float | None:
    """Mean of a field across records, skipping first N warm-up iterations."""
    vals = [r[field] for r in records[skip_first:] if field in r and r[field] is not None]
    return float(np.mean(vals)) if vals else None


def _final_field(records: list[dict], field: str) -> float | None:
    """Value from the last record that has this field."""
    for r in reversed(records):
        if field in r and r[field] is not None:
            return float(r[field])
    return None


def _sysmon_peak(records: list[dict], field: str) -> float | None:
    vals = [r[field] for r in records if field in r]
    return float(np.percentile(vals, 95)) if vals else None  # 95th pct avoids transients


def _sysmon_mean(records: list[dict], field: str) -> float | None:
    vals = [r[field] for r in records if field in r]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Figure 1: Throughput scaling (steps/sec vs workers)
# ---------------------------------------------------------------------------

def fig_throughput(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("Figure 1: Throughput Scaling (steps/sec vs. worker count)", y=1.02)

    for j, horizon in enumerate(HORIZONS):
        ax = axes[j]
        for fw in FRAMEWORKS:
            xs, ys = [], []
            for w in WORKERS:
                entry = data.get(fw, {}).get(w, {}).get(horizon)
                if entry:
                    v = _mean_field(entry["metrics"], "steps_per_sec")
                    if v is not None:
                        xs.append(w)
                        ys.append(v)
            if xs:
                style = FW_STYLE[fw]
                ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], label=style["label"])

        ax.set_title(f"Horizon = {horizon}")
        ax.set_xlabel("Workers")
        ax.set_ylabel("Steps / second")
        ax.set_xticks(WORKERS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_throughput.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_throughput.png")


# ---------------------------------------------------------------------------
# Figure 2: Iteration time breakdown (rollout vs update)
# ---------------------------------------------------------------------------

def fig_time_breakdown(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 2: Iteration Time Breakdown (rollout vs. update)", y=1.02)

    for i, fw in enumerate(FRAMEWORKS):
        ax = axes[i]
        # Group bars by worker count; within each group, one bar per horizon
        n_horizons = len(HORIZONS)
        x = np.arange(len(WORKERS))
        width = 0.25

        for j, horizon in enumerate(HORIZONS):
            rollouts, updates = [], []
            for w in WORKERS:
                entry = data.get(fw, {}).get(w, {}).get(horizon)
                if entry:
                    rt = _mean_field(entry["metrics"], "rollout_time_s") or 0
                    ut = _mean_field(entry["metrics"], "update_time_s") or 0
                else:
                    rt = ut = 0
                rollouts.append(rt)
                updates.append(ut)

            offset = (j - n_horizons / 2 + 0.5) * width
            color_rollout = plt.cm.Blues(0.4 + 0.3 * j / n_horizons)
            color_update = plt.cm.Oranges(0.4 + 0.3 * j / n_horizons)
            bars_r = ax.bar(x + offset, rollouts, width, color=color_rollout,
                            label=f"Rollout h={horizon}")
            bars_u = ax.bar(x + offset, updates, width, bottom=rollouts,
                            color=color_update, label=f"Update h={horizon}")

        ax.set_title(fw.capitalize())
        ax.set_xlabel("Workers")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKERS)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_time_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_time_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 3: Resource utilization (CPU, RAM, GPU util, VRAM) at horizon=256
# ---------------------------------------------------------------------------

def fig_resources(data: dict, out_dir: Path) -> None:
    HORIZON = 256
    metrics_cfg = [
        ("cpu_mean_pct",    "CPU Utilization (%)",   _sysmon_mean),
        ("ram_used_mb",     "RAM Used (MB)",          _sysmon_peak),
        ("gpu0_util_pct",   "GPU 0 Utilization (%)", _sysmon_mean),
        ("gpu0_vram_used_mb", "GPU 0 VRAM Used (MB)", _sysmon_peak),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Figure 3: Resource Utilization vs. Workers (horizon={HORIZON})", y=1.02)
    axes = axes.flatten()

    for k, (field, ylabel, agg_fn) in enumerate(metrics_cfg):
        ax = axes[k]
        for fw in FRAMEWORKS:
            xs, ys = [], []
            for w in WORKERS:
                entry = data.get(fw, {}).get(w, {}).get(HORIZON)
                if entry:
                    v = agg_fn(entry["sysmon"], field)
                    if v is not None:
                        xs.append(w)
                        ys.append(v)
            if xs:
                style = FW_STYLE[fw]
                ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], label=style["label"])

        ax.set_xlabel("Workers")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WORKERS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_resources.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_resources.png")


# ---------------------------------------------------------------------------
# Figure 4: Communication overhead at horizon=256
# ---------------------------------------------------------------------------

def fig_communication(data: dict, out_dir: Path) -> None:
    HORIZON = 256
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Figure 4: Communication Overhead vs. Workers (horizon={HORIZON})", y=1.02)

    for fw in FRAMEWORKS:
        style = FW_STYLE[fw]
        xs_mb, ys_mb = [], []
        xs_tp, ys_tp = [], []
        for w in WORKERS:
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            metrics = entry["metrics"]
            total_mb = _mean_field(metrics, "total_transfer_mb")
            throughput = _mean_field(metrics, "throughput_mb_s")
            if total_mb is not None:
                xs_mb.append(w)
                ys_mb.append(total_mb)
            if throughput is not None:
                xs_tp.append(w)
                ys_tp.append(throughput)

        if xs_mb:
            axes[0].plot(xs_mb, ys_mb, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])
        if xs_tp:
            axes[1].plot(xs_tp, ys_tp, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

    axes[0].set_title("Data transferred per iteration")
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("MB / iteration")
    axes[0].set_xticks(WORKERS)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Effective data throughput")
    axes[1].set_xlabel("Workers")
    axes[1].set_ylabel("MB / second")
    axes[1].set_xticks(WORKERS)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_communication.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_communication.png")


# ---------------------------------------------------------------------------
# Figure 5: Learning quality — return curves + final win rate
# ---------------------------------------------------------------------------

def fig_learning(data: dict, out_dir: Path) -> None:
    HORIZON = 256
    SELECTED_WORKERS = [1, 4]  # most informative pair

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 5: Learning Quality", y=1.02)

    # Left: return curves
    ax = axes[0]
    colors = {"ray": ["#aec7e8", "#1f77b4"], "monarch": ["#ffbb78", "#ff7f0e"]}
    for fw in FRAMEWORKS:
        for ci, w in enumerate(SELECTED_WORKERS):
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            records = entry["metrics"]
            iters = [r["iteration"] for r in records if "mean_worker_return" in r]
            returns = [r["mean_worker_return"] for r in records if "mean_worker_return" in r]
            if iters:
                ax.plot(iters, returns,
                        color=colors[fw][ci],
                        label=f"{FW_STYLE[fw]['label']} {w}w",
                        linewidth=1.5 if ci == 1 else 1.0)

    ax.set_title(f"Mean Return over Iterations (horizon={HORIZON})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")

    # Right: final win rate vs workers
    ax = axes[1]
    for fw in FRAMEWORKS:
        xs, ys = [], []
        for w in WORKERS:
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            # win_rate stored in the "final" record
            wr = _final_field(entry["metrics"], "win_rate")
            if wr is not None:
                xs.append(w)
                ys.append(wr * 100)
        if xs:
            style = FW_STYLE[fw]
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], label=style["label"])

    ax.set_title(f"Final Win Rate vs. Workers (horizon={HORIZON})")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(WORKERS)
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_learning.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_learning.png")


# ---------------------------------------------------------------------------
# Figure 6: Horizon effect (fixed workers=4)
# ---------------------------------------------------------------------------

def fig_horizon_effect(data: dict, out_dir: Path) -> None:
    WORKERS_FIXED = 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Figure 6: Horizon Effect (workers={WORKERS_FIXED})", y=1.02)

    for fw in FRAMEWORKS:
        style = FW_STYLE[fw]
        xs_tp, ys_tp = [], []
        xs_up, ys_up = [], []
        for h in HORIZONS:
            entry = data.get(fw, {}).get(WORKERS_FIXED, {}).get(h)
            if not entry:
                continue
            tp = _mean_field(entry["metrics"], "steps_per_sec")
            ut = _mean_field(entry["metrics"], "update_time_s")
            if tp is not None:
                xs_tp.append(h)
                ys_tp.append(tp)
            if ut is not None:
                xs_up.append(h)
                ys_up.append(ut * 1000)  # ms

        if xs_tp:
            axes[0].plot(xs_tp, ys_tp, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])
        if xs_up:
            axes[1].plot(xs_up, ys_up, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

    axes[0].set_title("Throughput vs. Horizon")
    axes[0].set_xlabel("Horizon (steps/rollout)")
    axes[0].set_ylabel("Steps / second")
    axes[0].set_xticks(HORIZONS)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Update Time vs. Horizon")
    axes[1].set_xlabel("Horizon (steps/rollout)")
    axes[1].set_ylabel("Update time (ms)")
    axes[1].set_xticks(HORIZONS)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_horizon_effect.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_horizon_effect.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    data = load_all(results_dir)

    fws = sorted(data.keys())
    total_exps = sum(len(ws) * len(hs) for ws in data.values() for hs in ws.values()
                     for _ in [hs])
    # simpler count:
    total_exps = sum(
        1
        for fw in data
        for w in data[fw]
        for h in data[fw][w]
    )
    print(f"Found {total_exps} experiments across frameworks: {fws}")
    print(f"Generating figures → {out_dir}\n")

    fig_throughput(data, out_dir)
    fig_time_breakdown(data, out_dir)
    fig_resources(data, out_dir)
    fig_communication(data, out_dir)
    fig_learning(data, out_dir)
    fig_horizon_effect(data, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
