"""
Generate benchmark figures from results produced by run_all.py.

Usage:
    # Single run:
    python benchmark/plot.py --runs my-run

    # Compare two runs side-by-side:
    python benchmark/plot.py --runs baseline gpu-fix

    # Default: all runs found under benchmark/results/
    python benchmark/plot.py

Figures are written to benchmark/results/figures/<run_slug>/<env_mode>_<hidden_dim>d/
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

FW_COLOR  = {"ray": "#1f77b4", "monarch": "#ff7f0e"}
FW_MARKER = {"ray": "o",       "monarch": "s"}
RUN_LINESTYLES = ["-", "--", "-.", ":"]

WORKERS    = [1, 2, 4, 8]
HORIZONS   = [128, 256, 512]
FRAMEWORKS = ["ray", "monarch"]


def _series_style(fw: str, run_idx: int = 0, label: str | None = None) -> dict:
    return {
        "color":     FW_COLOR[fw],
        "linestyle": RUN_LINESTYLES[run_idx % len(RUN_LINESTYLES)],
        "marker":    FW_MARKER[fw],
        "label":     label or fw.capitalize(),
        "alpha":     1.0 if run_idx == 0 else 0.65,
    }


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


def load_run(run_dir: Path, run_name: str) -> dict:
    """Load one named run. Returns data[env_mode][hidden_dim][framework][workers][horizon]"""
    data = {}
    for exp_dir in run_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        fw = cfg["framework"]
        w  = cfg["workers"]
        h  = cfg["horizon"]
        hd = cfg.get("hidden_dim", 64)
        em = cfg.get("env_mode", "websocket")
        (data
            .setdefault(em, {})
            .setdefault(hd, {})
            .setdefault(fw, {})
            .setdefault(w, {})[h]
        ) = {
            "metrics":  load_metrics(exp_dir),
            "sysmon":   load_sysmon(exp_dir),
            "config":   cfg,
            "run_name": run_name,
        }
    return data


def load_runs(base_dir: Path, run_names: list[str]) -> list[tuple[str, dict]]:
    """Returns [(run_name, data), ...] for each requested run."""
    result = []
    for name in run_names:
        run_dir = base_dir / name
        if not run_dir.exists():
            print(f"  WARNING: run '{name}' not found at {run_dir}, skipping")
            continue
        result.append((name, load_run(run_dir, name)))
    return result


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
# Figure helpers
# ---------------------------------------------------------------------------

def _iter_series(runs: list, fw_data_fn):
    """
    Yield (run_idx, run_name, fw, series_label, fw_slice) for every
    (run, framework) combination that has data.

    fw_data_fn(data) -> fw_slice  (the slice of data keyed by fw)
    """
    multi = len(runs) > 1
    for run_idx, (run_name, data) in enumerate(runs):
        fw_slice = fw_data_fn(data)
        for fw in FRAMEWORKS:
            if fw not in fw_slice:
                continue
            label = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
            yield run_idx, run_name, fw, label, fw_slice[fw]


# ---------------------------------------------------------------------------
# Figure 1: Throughput scaling
# ---------------------------------------------------------------------------

def fig_throughput(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        f"Throughput Scaling — {env_mode}, hidden={hidden_dim} (steps/sec vs. workers)",
        y=1.02,
    )

    for j, horizon in enumerate(HORIZONS):
        ax = axes[j]
        for run_idx, run_name, fw, label, fw_data in _iter_series(
            runs, lambda d: d.get(fw, {}) if False else
            {f: d.get(f, {}) for f in FRAMEWORKS}
        ):
            # inline the loop manually since _iter_series needs a different shape here
            pass

        # Direct loop is cleaner for this figure
        multi = len(runs) > 1
        for run_idx, (run_name, data) in enumerate(runs):
            for fw in FRAMEWORKS:
                xs, ys = [], []
                for w in WORKERS:
                    entry = data.get(fw, {}).get(w, {}).get(horizon)
                    if entry:
                        v = _mean_field(entry["metrics"], "steps_per_sec")
                        if v is not None:
                            xs.append(w); ys.append(v)
                if xs:
                    lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
                    ax.plot(xs, ys, **_series_style(fw, run_idx, lbl))

        ax.set_title(f"Horizon = {horizon}")
        ax.set_xlabel("Workers"); ax.set_ylabel("Steps / second")
        ax.set_xticks(WORKERS); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_throughput.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_throughput.png")


# ---------------------------------------------------------------------------
# Figure 2: Iteration time breakdown (first run only — stacked bars)
# ---------------------------------------------------------------------------

def fig_time_breakdown(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    _, data = runs[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Iteration Time Breakdown — {env_mode}, hidden={hidden_dim} (rollout vs. update)",
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
                rollouts.append(rt); updates.append(ut)

            offset = (j - n_horizons / 2 + 0.5) * width
            ax.bar(x + offset, rollouts, width,
                   color=plt.cm.Blues(0.4 + 0.3 * j / n_horizons),
                   label=f"Rollout h={horizon}")
            ax.bar(x + offset, updates, width, bottom=rollouts,
                   color=plt.cm.Oranges(0.4 + 0.3 * j / n_horizons),
                   label=f"Update h={horizon}")

        ax.set_title(fw.capitalize())
        ax.set_xlabel("Workers"); ax.set_ylabel("Time (s)")
        ax.set_xticks(x); ax.set_xticklabels(WORKERS)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_time_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_time_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 3: Resource utilization
# ---------------------------------------------------------------------------

def fig_resources(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    metrics_cfg = [
        ("cpu_mean_pct",      "CPU Utilization (%)",   _sysmon_mean),
        ("ram_used_mb",       "RAM Used (MB)",          _sysmon_peak),
        ("gpu0_util_pct",     "GPU 0 Util (%)",        _sysmon_mean),
        ("gpu0_vram_used_mb", "GPU 0 VRAM (MB)",       _sysmon_peak),
    ]
    multi = len(runs) > 1

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Resource Utilization — {env_mode}, hidden={hidden_dim} (horizon={HORIZON})",
        y=1.02,
    )
    axes = axes.flatten()

    for k, (field, ylabel, agg_fn) in enumerate(metrics_cfg):
        ax = axes[k]
        for run_idx, (run_name, data) in enumerate(runs):
            for fw in FRAMEWORKS:
                xs, ys = [], []
                for w in WORKERS:
                    entry = data.get(fw, {}).get(w, {}).get(HORIZON)
                    if entry:
                        v = agg_fn(entry["sysmon"], field)
                        if v is not None:
                            xs.append(w); ys.append(v)
                if xs:
                    lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
                    ax.plot(xs, ys, **_series_style(fw, run_idx, lbl))

        ax.set_xlabel("Workers"); ax.set_ylabel(ylabel)
        ax.set_xticks(WORKERS); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_resources.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_resources.png")


# ---------------------------------------------------------------------------
# Figure 4: Communication overhead
# ---------------------------------------------------------------------------

def fig_communication(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    multi = len(runs) > 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Communication Overhead — {env_mode}, hidden={hidden_dim} (horizon={HORIZON})",
        y=1.02,
    )

    for run_idx, (run_name, data) in enumerate(runs):
        for fw in FRAMEWORKS:
            xs_mb, ys_mb, xs_tp, ys_tp = [], [], [], []
            for w in WORKERS:
                entry = data.get(fw, {}).get(w, {}).get(HORIZON)
                if not entry:
                    continue
                mb = _mean_field(entry["metrics"], "total_transfer_mb")
                tp = _mean_field(entry["metrics"], "throughput_mb_s")
                if mb is not None: xs_mb.append(w); ys_mb.append(mb)
                if tp is not None: xs_tp.append(w); ys_tp.append(tp)

            lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
            style = _series_style(fw, run_idx, lbl)
            if xs_mb: axes[0].plot(xs_mb, ys_mb, **style)
            if xs_tp: axes[1].plot(xs_tp, ys_tp, **style)

    axes[0].set_title("Data transferred / iteration")
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

def fig_learning(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    HORIZON = 256
    SELECTED_WORKERS = [1, 4]
    multi = len(runs) > 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Learning Quality — {env_mode}, hidden={hidden_dim}", y=1.02)

    ax = axes[0]
    for run_idx, (run_name, data) in enumerate(runs):
        colors = {
            "ray":     ["#aec7e8", "#1f77b4"],
            "monarch": ["#ffbb78", "#ff7f0e"],
        }
        for fw in FRAMEWORKS:
            for ci, w in enumerate(SELECTED_WORKERS):
                entry = data.get(fw, {}).get(w, {}).get(HORIZON)
                if not entry:
                    continue
                records = entry["metrics"]
                iters   = [r["iteration"] for r in records if "mean_worker_return" in r]
                returns = [r["mean_worker_return"] for r in records if "mean_worker_return" in r]
                if iters:
                    lbl = f"{fw.capitalize()} {w}w"
                    if multi: lbl += f" ({run_name})"
                    ls = RUN_LINESTYLES[run_idx % len(RUN_LINESTYLES)]
                    ax.plot(iters, returns, color=colors[fw][ci], linestyle=ls,
                            label=lbl, linewidth=1.5 if ci == 1 else 1.0)

    ax.set_title(f"Mean Return (horizon={HORIZON})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Episode Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")

    ax = axes[1]
    for run_idx, (run_name, data) in enumerate(runs):
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
                lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
                ax.plot(xs, ys, **_series_style(fw, run_idx, lbl))

    ax.set_title(f"Final Win Rate vs. Workers (horizon={HORIZON})")
    ax.set_xlabel("Workers"); ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(WORKERS); ax.set_ylim(0, 70)
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_learning.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_learning.png")


# ---------------------------------------------------------------------------
# Figure 6: Horizon effect
# ---------------------------------------------------------------------------

def fig_horizon_effect(runs: list, out_dir: Path, env_mode: str, hidden_dim: int) -> None:
    WORKERS_FIXED = 4
    multi = len(runs) > 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Horizon Effect — {env_mode}, hidden={hidden_dim} (workers={WORKERS_FIXED})",
        y=1.02,
    )

    for run_idx, (run_name, data) in enumerate(runs):
        for fw in FRAMEWORKS:
            xs_tp, ys_tp, xs_up, ys_up = [], [], [], []
            for h in HORIZONS:
                entry = data.get(fw, {}).get(WORKERS_FIXED, {}).get(h)
                if not entry:
                    continue
                tp = _mean_field(entry["metrics"], "steps_per_sec")
                ut = _mean_field(entry["metrics"], "update_time_s")
                if tp is not None: xs_tp.append(h); ys_tp.append(tp)
                if ut is not None: xs_up.append(h); ys_up.append(ut * 1000)

            lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
            style = _series_style(fw, run_idx, lbl)
            if xs_tp: axes[0].plot(xs_tp, ys_tp, **style)
            if xs_up: axes[1].plot(xs_up, ys_up, **style)

    axes[0].set_title("Throughput vs. Horizon")
    axes[0].set_xlabel("Horizon"); axes[0].set_ylabel("Steps / second")
    axes[0].set_xticks(HORIZONS); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Update Time vs. Horizon")
    axes[1].set_xlabel("Horizon"); axes[1].set_ylabel("Update time (ms)")
    axes[1].set_xticks(HORIZONS); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_horizon_effect.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_horizon_effect.png")


# ---------------------------------------------------------------------------
# Figure 7: Small vs large model (local mode, cross-run)
# ---------------------------------------------------------------------------

def fig_model_size_comparison(runs: list, out_dir: Path) -> None:
    """Compare throughput at hidden=64 vs hidden=1024 in local mode."""
    HORIZON = 256
    multi = len(runs) > 1

    # Collect all hidden dims available across runs
    all_dims = sorted({
        hd
        for _, data in runs
        for hd in data.get("local", {}).keys()
    })
    if len(all_dims) < 2:
        return

    fig, axes = plt.subplots(1, len(all_dims), figsize=(6 * len(all_dims), 4), sharey=False)
    if len(all_dims) == 1:
        axes = [axes]
    fig.suptitle("Model Size Effect on Throughput (local env, horizon=256)", y=1.02)

    for ax, hd in zip(axes, all_dims):
        for run_idx, (run_name, data) in enumerate(runs):
            for fw in FRAMEWORKS:
                xs, ys = [], []
                for w in WORKERS:
                    entry = data.get("local", {}).get(hd, {}).get(fw, {}).get(w, {}).get(HORIZON)
                    if entry:
                        v = _mean_field(entry["metrics"], "steps_per_sec")
                        if v is not None:
                            xs.append(w); ys.append(v)
                if xs:
                    lbl = f"{fw.capitalize()} ({run_name})" if multi else fw.capitalize()
                    ax.plot(xs, ys, **_series_style(fw, run_idx, lbl))

        ax.set_title(f"hidden_dim = {hd}")
        ax.set_xlabel("Workers"); ax.set_ylabel("Steps / second")
        ax.set_xticks(WORKERS); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig7_model_size.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_model_size.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent / "results",
                        help="Base directory containing named run folders")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Named run(s) to plot (subdirs of --results-dir). "
                             "Default: all runs found. Multiple runs are overlaid.")
    args = parser.parse_args()

    base_dir: Path = args.results_dir
    if not base_dir.exists():
        print(f"Results directory not found: {base_dir}")
        return

    # Discover run names if not specified
    if args.runs:
        run_names = args.runs
    else:
        run_names = sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir() and d.name != "figures" and (d / "manifest.json").exists()
        )
        if not run_names:
            print(f"No runs found under {base_dir} (looking for dirs with manifest.json)")
            return

    print(f"Loading {len(run_names)} run(s): {run_names}")
    runs = load_runs(base_dir, run_names)
    if not runs:
        print("No data loaded.")
        return

    # Output dir named after the run(s)
    slug = run_names[0] if len(run_names) == 1 else "compare_" + "_vs_".join(run_names)
    fig_base = base_dir / "figures" / slug
    fig_base.mkdir(parents=True, exist_ok=True)

    # Collect all (env_mode, hidden_dim) slices across all runs
    slices = sorted({
        (em, hd)
        for _, data in runs
        for em in data
        for hd in data[em]
    })

    for env_mode, hidden_dim in slices:
        slice_dir = fig_base / f"{env_mode}_{hidden_dim}d"
        slice_dir.mkdir(exist_ok=True)
        # Build a runs list sliced to this (env_mode, hidden_dim)
        sliced_runs = [
            (name, data.get(env_mode, {}).get(hidden_dim, {}))
            for name, data in runs
        ]
        print(f"\n--- {env_mode} / hidden={hidden_dim} ---")
        fig_throughput(sliced_runs, slice_dir, env_mode, hidden_dim)
        fig_time_breakdown(sliced_runs, slice_dir, env_mode, hidden_dim)
        fig_resources(sliced_runs, slice_dir, env_mode, hidden_dim)
        fig_communication(sliced_runs, slice_dir, env_mode, hidden_dim)
        fig_learning(sliced_runs, slice_dir, env_mode, hidden_dim)
        fig_horizon_effect(sliced_runs, slice_dir, env_mode, hidden_dim)

    print(f"\n--- cross-slice ---")
    fig_model_size_comparison(
        [(name, data) for name, data in runs],
        fig_base,
    )

    print(f"\nAll figures saved under {fig_base}")


if __name__ == "__main__":
    main()
