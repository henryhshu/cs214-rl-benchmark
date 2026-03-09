"""
Measure pure dispatch-to-execution scheduling latency using no-op ping tasks.

This isolates the framework's task scheduling overhead from any actual
computation, giving a clean comparison of Ray's bottom-up scheduler
vs Monarch's top-down scheduler.

Usage:
    python benchmark/tests/bench_scheduling_latency.py --framework ray --worker-counts 1 2 4 8
    python benchmark/tests/bench_scheduling_latency.py --framework monarch --worker-counts 1 2 4 8
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


def run_ray(worker_counts: list[int], num_pings: int, warmup: int) -> list[dict]:
    import ray

    @ray.remote
    class TimingWorker:
        def ping(self, dispatch_time: float) -> float:
            return time.time() - dispatch_time

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        for n_workers in worker_counts:
            workers = [TimingWorker.remote() for _ in range(n_workers)]

            # Warmup
            for _ in range(warmup):
                t = time.time()
                ray.get([w.ping.remote(t) for w in workers])

            latencies = []
            for _ in range(num_pings):
                t = time.time()
                batch = ray.get([w.ping.remote(t) for w in workers])
                latencies.extend(batch)

            lat_ms = [l * 1000 for l in latencies]
            results.append({
                "framework": "ray",
                "workers": n_workers,
                "num_pings": num_pings,
                "total_samples": len(lat_ms),
                "mean_ms": statistics.mean(lat_ms),
                "median_ms": statistics.median(lat_ms),
                "p99_ms": sorted(lat_ms)[int(len(lat_ms) * 0.99)] if lat_ms else 0,
                "std_ms": statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0,
                "min_ms": min(lat_ms),
                "max_ms": max(lat_ms),
            })
            print(f"  ray  workers={n_workers:2d}  mean={results[-1]['mean_ms']:.3f}ms  "
                  f"p99={results[-1]['p99_ms']:.3f}ms")

            del workers
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_pings: int, warmup: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class TimingWorker(Actor):
        @endpoint
        async def ping(self, dispatch_time: float) -> float:
            return time.time() - dispatch_time

    async def _run():
        results = []
        for n_workers in worker_counts:
            mesh = this_host().spawn_procs(per_host={"procs": n_workers})
            workers = mesh.spawn("ping_workers", TimingWorker)

            # Warmup
            for _ in range(warmup):
                t = time.time()
                await workers.ping.call(t)

            latencies = []
            for _ in range(num_pings):
                t = time.time()
                result_mesh = await workers.ping.call(t)
                latencies.extend(result_mesh.values())

            lat_ms = [l * 1000 for l in latencies]
            results.append({
                "framework": "monarch",
                "workers": n_workers,
                "num_pings": num_pings,
                "total_samples": len(lat_ms),
                "mean_ms": statistics.mean(lat_ms),
                "median_ms": statistics.median(lat_ms),
                "p99_ms": sorted(lat_ms)[int(len(lat_ms) * 0.99)] if lat_ms else 0,
                "std_ms": statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0,
                "min_ms": min(lat_ms),
                "max_ms": max(lat_ms),
            })
            print(f"  monarch  workers={n_workers:2d}  mean={results[-1]['mean_ms']:.3f}ms  "
                  f"p99={results[-1]['p99_ms']:.3f}ms")
        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduling latency microbenchmark")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-pings", type=int, default=100,
                        help="Number of ping batches per worker count")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON results")
    args = parser.parse_args()

    print(f"Scheduling latency benchmark: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_pings, args.warmup)
    else:
        results = run_monarch(args.worker_counts, args.num_pings, args.warmup)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
