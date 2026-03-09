"""
Generate benchmark figures from results produced by run_all.py.

Usage:
    python benchmark/plot.py                         # uses benchmark/results/
    python benchmark/plot.py --results-dir path/to/results

Figures are written to benchmark/results/figures/<env_mode>_<hidden_dim>d/
so that each (env_mode, hidden_dim) combination gets its own clean set.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Return nested dict: data[env_mode][hidden_dim][framework][workers][horizon]"""
    data = {}
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        fw  = cfg["framework"]
        w   = cfg["workers"]
        h   = cfg["horizon"]
        hd  = cfg.get("hidden_dim", 64)
        em  = cfg.get("env_mode", "websocket")
        (data
            .setdefault(em, {})
            .setdefault(hd, {})
            .setdefault(fw, {})
            .setdefault(w, {})[h]
        ) = {
            "metrics": load_metrics(exp_dir),
            "sysmon":  load_sysmon(exp_dir),
            "config":  cfg,
        }
    return data


def _mean_field(records: list[dict], field: str, skip_first: int = 2) -> float | None:
    vals = [r[field] for r in records[skip_first:] if field in r and r[field] is not None]
    return float(np.mean(vals)) if vals else None


def _final_field(records: list[dict], field: str) -> float | None:
    for r in reversed(records):
        if field in r and r[field] is not None:
            return float(r[field])
    return None


def _sysmon_peak(records: list[dict], field: str) -> float | None:
    vals = [r[field] for r in records if field in r]
    return float(np.percentile(vals, 95)) if vals else None


def _sysmon_mean(records: list[dict], field: str) -> float | None:
    vals = [r[field] for r in records if field in r]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Figure 1: Throughput scaling (steps/sec vs workers)
# ---------------------------------------------------------------------------

def fig_throughput(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        f"Throughput Scaling — {env_mode}, hidden={hidden_dim} "
        f"(steps/sec vs. worker count)",
        y=1.02,
    )

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

def fig_time_breakdown(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Iteration Time Breakdown — {env_mode}, hidden={hidden_dim} "
        f"(rollout vs. update)",
        y=1.02,
    )

    for i, fw in enumerate(FRAMEWORKS):
        ax = axes[i]
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
            color_update  = plt.cm.Oranges(0.4 + 0.3 * j / n_horizons)
            ax.bar(x + offset, rollouts, width, color=color_rollout,
                   label=f"Rollout h={horizon}")
            ax.bar(x + offset, updates, width, bottom=rollouts,
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
# Figure 3: Resource utilization at horizon=256
# ---------------------------------------------------------------------------

def fig_resources(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    metrics_cfg = [
        ("cpu_mean_pct",      "CPU Utilization (%)",    _sysmon_mean),
        ("ram_used_mb",       "RAM Used (MB)",           _sysmon_peak),
        ("gpu0_util_pct",     "GPU 0 Utilization (%)",  _sysmon_mean),
        ("gpu0_vram_used_mb", "GPU 0 VRAM Used (MB)",   _sysmon_peak),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Resource Utilization — {env_mode}, hidden={hidden_dim} "
        f"(horizon={HORIZON})",
        y=1.02,
    )
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

def fig_communication(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Communication Overhead — {env_mode}, hidden={hidden_dim} "
        f"(horizon={HORIZON})",
        y=1.02,
    )

    for fw in FRAMEWORKS:
        style = FW_STYLE[fw]
        xs_mb, ys_mb, xs_tp, ys_tp = [], [], [], []
        for w in WORKERS:
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            metrics = entry["metrics"]
            total_mb   = _mean_field(metrics, "total_transfer_mb")
            throughput = _mean_field(metrics, "throughput_mb_s")
            if total_mb is not None:
                xs_mb.append(w); ys_mb.append(total_mb)
            if throughput is not None:
                xs_tp.append(w); ys_tp.append(throughput)

        if xs_mb:
            axes[0].plot(xs_mb, ys_mb, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])
        if xs_tp:
            axes[1].plot(xs_tp, ys_tp, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

    axes[0].set_title("Data transferred per iteration")
    axes[0].set_xlabel("Workers"); axes[0].set_ylabel("MB / iteration")
    axes[0].set_xticks(WORKERS); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Effective data throughput")
    axes[1].set_xlabel("Workers"); axes[1].set_ylabel("MB / second")
    axes[1].set_xticks(WORKERS); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_communication.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_communication.png")


# ---------------------------------------------------------------------------
# Figure 5: Learning quality
# ---------------------------------------------------------------------------

def fig_learning(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    SELECTED_WORKERS = [1, 4]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Learning Quality — {env_mode}, hidden={hidden_dim}",
        y=1.02,
    )

    ax = axes[0]
    colors = {"ray": ["#aec7e8", "#1f77b4"], "monarch": ["#ffbb78", "#ff7f0e"]}
    for fw in FRAMEWORKS:
        for ci, w in enumerate(SELECTED_WORKERS):
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            records = entry["metrics"]
            iters   = [r["iteration"] for r in records if "mean_worker_return" in r]
            returns = [r["mean_worker_return"] for r in records if "mean_worker_return" in r]
            if iters:
                ax.plot(iters, returns, color=colors[fw][ci],
                        label=f"{FW_STYLE[fw]['label']} {w}w",
                        linewidth=1.5 if ci == 1 else 1.0)

    ax.set_title(f"Mean Return over Iterations (horizon={HORIZON})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Episode Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")

    ax = axes[1]
    for fw in FRAMEWORKS:
        xs, ys = [], []
        for w in WORKERS:
            entry = data.get(fw, {}).get(w, {}).get(HORIZON)
            if not entry:
                continue
            wr = _final_field(entry["metrics"], "win_rate")
            if wr is not None:
                xs.append(w); ys.append(wr * 100)
        if xs:
            style = FW_STYLE[fw]
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], label=style["label"])

    ax.set_title(f"Final Win Rate vs. Workers (horizon={HORIZON})")
    ax.set_xlabel("Workers"); ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(WORKERS); ax.set_ylim(0, 70)
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_learning.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_learning.png")


# ---------------------------------------------------------------------------
# Figure 6: Horizon effect (fixed workers=4)
# ---------------------------------------------------------------------------

def fig_horizon_effect(data: dict, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    WORKERS_FIXED = 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Horizon Effect — {env_mode}, hidden={hidden_dim} "
        f"(workers={WORKERS_FIXED})",
        y=1.02,
    )

    for fw in FRAMEWORKS:
        style = FW_STYLE[fw]
        xs_tp, ys_tp, xs_up, ys_up = [], [], [], []
        for h in HORIZONS:
            entry = data.get(fw, {}).get(WORKERS_FIXED, {}).get(h)
            if not entry:
                continue
            tp = _mean_field(entry["metrics"], "steps_per_sec")
            ut = _mean_field(entry["metrics"], "update_time_s")
            if tp is not None:
                xs_tp.append(h); ys_tp.append(tp)
            if ut is not None:
                xs_up.append(h); ys_up.append(ut * 1000)

        if xs_tp:
            axes[0].plot(xs_tp, ys_tp, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])
        if xs_up:
            axes[1].plot(xs_up, ys_up, color=style["color"], linestyle=style["linestyle"],
                         marker=style["marker"], label=style["label"])

    axes[0].set_title("Throughput vs. Horizon")
    axes[0].set_xlabel("Horizon (steps/rollout)"); axes[0].set_ylabel("Steps / second")
    axes[0].set_xticks(HORIZONS); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Update Time vs. Horizon")
    axes[1].set_xlabel("Horizon (steps/rollout)"); axes[1].set_ylabel("Update time (ms)")
    axes[1].set_xticks(HORIZONS); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_horizon_effect.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_horizon_effect.png")


# ---------------------------------------------------------------------------
# Figure 7: Small vs large model throughput comparison (local mode only)
# ---------------------------------------------------------------------------

def fig_model_size_comparison(all_data: dict, out_dir: Path) -> None:
    """Compare throughput at hidden=64 vs hidden=1024 in local mode."""
    local_data = all_data.get("local")
    if not local_data:
        return
    dims = sorted(local_data.keys())
    if len(dims) < 2:
        return

    HORIZON = 256
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    fig.suptitle("Model Size Effect on Throughput (local env, horizon=256)", y=1.02)

    for ax, hd in zip(axes, dims):
        fw_data = local_data[hd]
        for fw in FRAMEWORKS:
            xs, ys = [], []
            for w in WORKERS:
                entry = fw_data.get(fw, {}).get(w, {}).get(HORIZON)
                if entry:
                    v = _mean_field(entry["metrics"], "steps_per_sec")
                    if v is not None:
                        xs.append(w); ys.append(v)
            if xs:
                style = FW_STYLE[fw]
                ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], label=style["label"])

        ax.set_title(f"hidden_dim = {hd}")
        ax.set_xlabel("Workers"); ax.set_ylabel("Steps / second")
        ax.set_xticks(WORKERS); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig7_model_size.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7_model_size.png")


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

    base_out = results_dir / "figures"
    base_out.mkdir(exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    all_data = load_all(results_dir)

    total = sum(
        1
        for em in all_data
        for hd in all_data[em]
        for fw in all_data[em][hd]
        for w  in all_data[em][hd][fw]
        for h  in all_data[em][hd][fw][w]
    )
    print(f"Found {total} experiments across: {list(all_data.keys())}\n")

    # One figure set per (env_mode, hidden_dim) slice
    for env_mode, hd_data in sorted(all_data.items()):
        for hidden_dim, fw_data in sorted(hd_data.items()):
            slice_name = f"{env_mode}_{hidden_dim}d"
            out_dir = base_out / slice_name
            out_dir.mkdir(exist_ok=True)
            print(f"--- {slice_name} ---")
            fig_throughput(fw_data, out_dir, env_mode, hidden_dim)
            fig_time_breakdown(fw_data, out_dir, env_mode, hidden_dim)
            fig_resources(fw_data, out_dir, env_mode, hidden_dim)
            fig_communication(fw_data, out_dir, env_mode, hidden_dim)
            fig_learning(fw_data, out_dir, env_mode, hidden_dim)
            fig_horizon_effect(fw_data, out_dir, env_mode, hidden_dim)

    # Cross-slice comparison (model size)
    print("--- cross-slice ---")
    fig_model_size_comparison(all_data, base_out)

    print(f"\nAll figures saved under {base_out}")


if __name__ == "__main__":
    main()
