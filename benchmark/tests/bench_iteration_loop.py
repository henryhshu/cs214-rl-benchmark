"""
M4: Iteration Loop Overhead
============================
Simulates the PPO training loop: broadcast weights -> workers compute ->
gather results -> repeat. Measures the amortized per-cycle framework tax.

Each cycle:
  1. Learner sends model weights (broadcast)
  2. Workers do a fixed amount of work (simulating rollout)
  3. Workers return results (gather)

The work duration is fixed so we can isolate the framework coordination
overhead from computation. We vary the number of cycles and worker counts.

Usage:
    python benchmark/tests/bench_iteration_loop.py --framework ray
    python benchmark/tests/bench_iteration_loop.py --framework monarch
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import torch


WORKER_COUNTS = [1, 2, 4, 8]
NUM_CYCLES = 100
WARMUP = 10
WORK_MS = 20  # Simulated compute per worker
# Model weights payload (~200KB, similar to 256-dim model)
PAYLOAD_ELEMENTS = 50_000


def _make_payload() -> dict[str, torch.Tensor]:
    return {"weights": torch.randn(PAYLOAD_ELEMENTS)}


def run_ray(worker_counts: list[int], num_cycles: int) -> list[dict]:
    import ray

    @ray.remote
    class LoopWorker:
        def step(self, weights: dict, work_ms: float) -> dict:
            # Simulate receiving weights + doing work + returning results
            _ = weights["weights"].sum()  # Force deserialization
            t0 = time.perf_counter()
            deadline = t0 + work_ms / 1000.0
            while time.perf_counter() < deadline:
                pass
            compute_time = time.perf_counter() - t0
            # Return a result payload (simulating rollout data)
            return {
                "rollout": torch.randn(PAYLOAD_ELEMENTS // 10),
                "compute_s": compute_time,
            }

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        max_workers = max(worker_counts)
        pool = [LoopWorker.remote() for _ in range(max_workers)]
        payload = _make_payload()

        # Initial warmup for all workers
        ref = ray.put(payload)
        ray.get([w.step.remote(ref, WORK_MS) for w in pool])

        for n_workers in worker_counts:
            workers = pool[:n_workers]

            # Warmup
            for _ in range(WARMUP):
                ref = ray.put(payload)
                ray.get([w.step.remote(ref, WORK_MS) for w in workers])

            cycle_times = []
            overhead_times = []

            for _ in range(num_cycles):
                t0 = time.perf_counter()

                # 1. Broadcast (ray.put)
                ref = ray.put(payload)

                # 2+3. Dispatch + compute + gather
                batch = ray.get([w.step.remote(ref, WORK_MS) for w in workers])

                cycle_time = time.perf_counter() - t0
                cycle_times.append(cycle_time)

                max_compute = max(r["compute_s"] for r in batch)
                overhead_times.append(cycle_time - max_compute)

            ct_ms = [t * 1000 for t in cycle_times]
            oh_ms = [t * 1000 for t in overhead_times]
            results.append({
                "framework": "ray",
                "workers": n_workers,
                "num_cycles": num_cycles,
                "work_ms": WORK_MS,
                "cycle_mean_ms": statistics.mean(ct_ms),
                "cycle_median_ms": statistics.median(ct_ms),
                "cycle_std_ms": statistics.stdev(ct_ms) if len(ct_ms) > 1 else 0,
                "overhead_mean_ms": statistics.mean(oh_ms),
                "overhead_median_ms": statistics.median(oh_ms),
                "overhead_std_ms": statistics.stdev(oh_ms) if len(oh_ms) > 1 else 0,
                "overhead_pct": statistics.mean(overhead_times) / max(statistics.mean(cycle_times), 1e-9) * 100,
            })
            print(f"  ray  {n_workers}w  "
                  f"cycle={results[-1]['cycle_mean_ms']:.2f}ms  "
                  f"overhead={results[-1]['overhead_mean_ms']:.2f}ms "
                  f"({results[-1]['overhead_pct']:.1f}%)")
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_cycles: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class LoopWorker(Actor):
        @endpoint
        async def step(self, weights: dict, work_ms: float) -> dict:
            _ = weights["weights"].sum()
            t0 = time.perf_counter()
            deadline = t0 + work_ms / 1000.0
            while time.perf_counter() < deadline:
                pass
            compute_time = time.perf_counter() - t0
            return {
                "rollout": torch.randn(PAYLOAD_ELEMENTS // 10),
                "compute_s": compute_time,
            }

    async def _run():
        results = []
        for n_workers in worker_counts:
            mesh = this_host().spawn_procs(per_host={"procs": n_workers})
            workers = mesh.spawn("loop_workers", LoopWorker)
            payload = _make_payload()

            # Warmup
            for _ in range(WARMUP):
                await workers.step.call(payload, float(WORK_MS))

            cycle_times = []
            overhead_times = []

            for _ in range(num_cycles):
                t0 = time.perf_counter()
                result_mesh = await workers.step.call(payload, float(WORK_MS))
                cycle_time = time.perf_counter() - t0

                cycle_times.append(cycle_time)
                max_compute = max(v["compute_s"] for v in result_mesh.values())
                overhead_times.append(cycle_time - max_compute)

            ct_ms = [t * 1000 for t in cycle_times]
            oh_ms = [t * 1000 for t in overhead_times]
            results.append({
                "framework": "monarch",
                "workers": n_workers,
                "num_cycles": num_cycles,
                "work_ms": WORK_MS,
                "cycle_mean_ms": statistics.mean(ct_ms),
                "cycle_median_ms": statistics.median(ct_ms),
                "cycle_std_ms": statistics.stdev(ct_ms) if len(ct_ms) > 1 else 0,
                "overhead_mean_ms": statistics.mean(oh_ms),
                "overhead_median_ms": statistics.median(oh_ms),
                "overhead_std_ms": statistics.stdev(oh_ms) if len(oh_ms) > 1 else 0,
                "overhead_pct": statistics.mean(overhead_times) / max(statistics.mean(cycle_times), 1e-9) * 100,
            })
            print(f"  monarch  {n_workers}w  "
                  f"cycle={results[-1]['cycle_mean_ms']:.2f}ms  "
                  f"overhead={results[-1]['overhead_mean_ms']:.2f}ms "
                  f"({results[-1]['overhead_pct']:.1f}%)")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="M4: Iteration loop overhead")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=WORKER_COUNTS)
    parser.add_argument("--num-cycles", type=int, default=NUM_CYCLES)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"M4 Iteration loop overhead: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_cycles)
    else:
        results = run_monarch(args.worker_counts, args.num_cycles)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
