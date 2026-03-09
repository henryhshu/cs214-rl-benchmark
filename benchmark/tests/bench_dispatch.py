"""
M1: Bulk Dispatch Overhead
==========================
Measures the cost of dispatching no-op tasks to N workers and collecting results.

Ray dispatches N separate .remote() calls through its scheduler, then waits on
N futures with ray.get(). Monarch dispatches a single .call() to the entire
mesh. This benchmark isolates that difference — no payload, no computation.

Usage:
    python benchmark/tests/bench_dispatch.py --framework ray --worker-counts 1 2 4 8 16
    python benchmark/tests/bench_dispatch.py --framework monarch --worker-counts 1 2 4 8 16
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


def run_ray(worker_counts: list[int], num_batches: int, warmup: int) -> list[dict]:
    import ray

    @ray.remote
    class NoOpWorker:
        def noop(self) -> int:
            return 1

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        # Create max pool once to avoid Ray actor lifecycle crashes
        max_workers = max(worker_counts)
        pool = [NoOpWorker.remote() for _ in range(max_workers)]
        ray.get([w.noop.remote() for w in pool])  # initial warmup

        for n_workers in worker_counts:
            workers = pool[:n_workers]

            # Warmup
            for _ in range(warmup):
                ray.get([w.noop.remote() for w in workers])

            dispatch_times = []
            for _ in range(num_batches):
                t0 = time.perf_counter()
                ray.get([w.noop.remote() for w in workers])
                dispatch_times.append(time.perf_counter() - t0)

            dt_ms = [t * 1000 for t in dispatch_times]
            results.append({
                "framework": "ray",
                "workers": n_workers,
                "num_batches": num_batches,
                "mean_ms": statistics.mean(dt_ms),
                "median_ms": statistics.median(dt_ms),
                "std_ms": statistics.stdev(dt_ms) if len(dt_ms) > 1 else 0,
                "p5_ms": sorted(dt_ms)[int(len(dt_ms) * 0.05)],
                "p95_ms": sorted(dt_ms)[int(len(dt_ms) * 0.95)],
                "min_ms": min(dt_ms),
                "max_ms": max(dt_ms),
            })
            print(f"  ray  workers={n_workers:2d}  "
                  f"mean={results[-1]['mean_ms']:.3f}ms  "
                  f"median={results[-1]['median_ms']:.3f}ms")
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_batches: int, warmup: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class NoOpWorker(Actor):
        @endpoint
        async def noop(self) -> int:
            return 1

    async def _run():
        results = []
        for n_workers in worker_counts:
            mesh = this_host().spawn_procs(per_host={"procs": n_workers})
            workers = mesh.spawn("dispatch_workers", NoOpWorker)

            # Warmup
            for _ in range(warmup):
                await workers.noop.call()

            dispatch_times = []
            for _ in range(num_batches):
                t0 = time.perf_counter()
                await workers.noop.call()
                dispatch_times.append(time.perf_counter() - t0)

            dt_ms = [t * 1000 for t in dispatch_times]
            results.append({
                "framework": "monarch",
                "workers": n_workers,
                "num_batches": num_batches,
                "mean_ms": statistics.mean(dt_ms),
                "median_ms": statistics.median(dt_ms),
                "std_ms": statistics.stdev(dt_ms) if len(dt_ms) > 1 else 0,
                "p5_ms": sorted(dt_ms)[int(len(dt_ms) * 0.05)],
                "p95_ms": sorted(dt_ms)[int(len(dt_ms) * 0.95)],
                "min_ms": min(dt_ms),
                "max_ms": max(dt_ms),
            })
            print(f"  monarch  workers={n_workers:2d}  "
                  f"mean={results[-1]['mean_ms']:.3f}ms  "
                  f"median={results[-1]['median_ms']:.3f}ms")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="M1: Bulk dispatch overhead")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"M1 Dispatch overhead: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_batches, args.warmup)
    else:
        results = run_monarch(args.worker_counts, args.num_batches, args.warmup)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
