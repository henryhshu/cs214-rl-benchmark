"""
M3: Straggler Sensitivity
=========================
Workers do variable-duration work, simulating heterogeneous episode lengths
in RL rollouts. Some workers finish quickly, others take much longer.

Both frameworks must wait for the slowest worker before proceeding. The
interesting metric is the *framework overhead* beyond the slowest worker:
    overhead = wall_clock - max(worker_times)

This overhead comes from dispatch, serialization, and result collection.
We sweep the variance of worker durations to see if either framework handles
heterogeneity better.

Usage:
    python benchmark/tests/bench_straggler.py --framework ray
    python benchmark/tests/bench_straggler.py --framework monarch
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path


BASE_WORK_MS = 50
VARIANCE_LEVELS = [
    {"name": "uniform",   "spread": 0.0},   # All workers take exactly base_work
    {"name": "low_var",   "spread": 0.25},   # +/- 25% of base
    {"name": "med_var",   "spread": 0.5},    # +/- 50% of base
    {"name": "high_var",  "spread": 1.0},    # +/- 100% (some workers 2x slower)
    {"name": "extreme",   "spread": 2.0},    # +/- 200% (some workers 3x slower)
]
WORKER_COUNTS = [2, 4, 8]
NUM_BATCHES = 50
WARMUP = 5


def _work_duration_ms(base_ms: float, spread: float) -> float:
    if spread == 0:
        return base_ms
    lo = base_ms * max(0.1, 1.0 - spread)
    hi = base_ms * (1.0 + spread)
    return random.uniform(lo, hi)


def run_ray(worker_counts: list[int], num_batches: int) -> list[dict]:
    import ray

    @ray.remote
    class VariableWorker:
        def work(self, base_ms: float, spread: float) -> dict:
            duration_ms = _work_duration_ms(base_ms, spread)
            t0 = time.perf_counter()
            deadline = t0 + duration_ms / 1000.0
            while time.perf_counter() < deadline:
                pass
            return {"actual_s": time.perf_counter() - t0}

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        max_workers = max(worker_counts)
        pool = [VariableWorker.remote() for _ in range(max_workers)]
        ray.get([w.work.remote(BASE_WORK_MS, 0.0) for w in pool])

        for n_workers in worker_counts:
            workers = pool[:n_workers]

            # Warmup
            for _ in range(WARMUP):
                ray.get([w.work.remote(BASE_WORK_MS, 0.0) for w in workers])

            for var_cfg in VARIANCE_LEVELS:
                overheads = []
                wall_times = []
                max_worker_times = []

                for _ in range(num_batches):
                    t0 = time.perf_counter()
                    # Each worker independently randomizes its own duration
                    results_batch = ray.get([
                        w.work.remote(BASE_WORK_MS, var_cfg["spread"])
                        for w in workers
                    ])
                    wall = time.perf_counter() - t0

                    actuals = [r["actual_s"] for r in results_batch]
                    max_worker = max(actuals)
                    overhead = wall - max_worker

                    wall_times.append(wall)
                    max_worker_times.append(max_worker)
                    overheads.append(overhead)

                results.append({
                    "framework": "ray",
                    "workers": n_workers,
                    "variance": var_cfg["name"],
                    "spread": var_cfg["spread"],
                    "wall_mean_ms": statistics.mean(wall_times) * 1000,
                    "max_worker_mean_ms": statistics.mean(max_worker_times) * 1000,
                    "overhead_mean_ms": statistics.mean(overheads) * 1000,
                    "overhead_median_ms": statistics.median(overheads) * 1000,
                    "overhead_std_ms": statistics.stdev(overheads) * 1000 if len(overheads) > 1 else 0,
                    "overhead_pct": statistics.mean(overheads) / max(statistics.mean(wall_times), 1e-9) * 100,
                })
                print(f"  ray  {n_workers}w  {var_cfg['name']:>8s}  "
                      f"wall={results[-1]['wall_mean_ms']:.1f}ms  "
                      f"overhead={results[-1]['overhead_mean_ms']:.2f}ms "
                      f"({results[-1]['overhead_pct']:.1f}%)")
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_batches: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class VariableWorker(Actor):
        @endpoint
        async def work(self, base_ms: float, spread: float) -> dict:
            duration_ms = _work_duration_ms(base_ms, spread)
            t0 = time.perf_counter()
            deadline = t0 + duration_ms / 1000.0
            while time.perf_counter() < deadline:
                pass
            return {"actual_s": time.perf_counter() - t0}

    async def _run():
        results = []
        for n_workers in worker_counts:
            mesh = this_host().spawn_procs(per_host={"procs": n_workers})
            workers = mesh.spawn("straggler_workers", VariableWorker)

            # Warmup
            for _ in range(WARMUP):
                await workers.work.call(float(BASE_WORK_MS), 0.0)

            for var_cfg in VARIANCE_LEVELS:
                overheads = []
                wall_times = []
                max_worker_times = []

                for _ in range(num_batches):
                    # Monarch broadcasts same args; each worker randomizes internally
                    t0 = time.perf_counter()
                    result_mesh = await workers.work.call(
                        float(BASE_WORK_MS), var_cfg["spread"]
                    )
                    wall = time.perf_counter() - t0

                    actuals = [v["actual_s"] for v in result_mesh.values()]
                    max_worker = max(actuals)
                    overhead = wall - max_worker

                    wall_times.append(wall)
                    max_worker_times.append(max_worker)
                    overheads.append(overhead)

                results.append({
                    "framework": "monarch",
                    "workers": n_workers,
                    "variance": var_cfg["name"],
                    "spread": var_cfg["spread"],
                    "wall_mean_ms": statistics.mean(wall_times) * 1000,
                    "max_worker_mean_ms": statistics.mean(max_worker_times) * 1000,
                    "overhead_mean_ms": statistics.mean(overheads) * 1000,
                    "overhead_median_ms": statistics.median(overheads) * 1000,
                    "overhead_std_ms": statistics.stdev(overheads) * 1000 if len(overheads) > 1 else 0,
                    "overhead_pct": statistics.mean(overheads) / max(statistics.mean(wall_times), 1e-9) * 100,
                })
                print(f"  monarch  {n_workers}w  {var_cfg['name']:>8s}  "
                      f"wall={results[-1]['wall_mean_ms']:.1f}ms  "
                      f"overhead={results[-1]['overhead_mean_ms']:.2f}ms "
                      f"({results[-1]['overhead_pct']:.1f}%)")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="M3: Straggler sensitivity")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=WORKER_COUNTS)
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"M3 Straggler sensitivity: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_batches)
    else:
        results = run_monarch(args.worker_counts, args.num_batches)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
