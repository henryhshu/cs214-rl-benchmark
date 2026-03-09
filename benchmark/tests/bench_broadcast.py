"""
M2: Broadcast Efficiency
========================
Sends the same tensor (model weights) to all N workers simultaneously.

Ray's object store puts data into shared memory once via ray.put(), then workers
read it zero-copy on the same node. Monarch serializes data through its
controller to each worker.

This is the model-weight broadcast pattern in PPO: every iteration, the learner
sends the same state dict to all rollout workers.

Usage:
    python benchmark/tests/bench_broadcast.py --framework ray
    python benchmark/tests/bench_broadcast.py --framework monarch
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import torch


# Payload sizes roughly matching: 64d model (~50KB), 256d (~200KB), 512d (~800KB), 1024d (~3.2MB)
PAYLOAD_CONFIGS = [
    {"name": "50KB",  "size": 50_000},
    {"name": "200KB", "size": 200_000},
    {"name": "800KB", "size": 800_000},
    {"name": "3.2MB", "size": 3_200_000},
]
WORKER_COUNTS = [1, 2, 4, 8]
NUM_TRIALS = 100
WARMUP = 10


def _make_payload(num_bytes: int) -> dict[str, torch.Tensor]:
    """Create a dict of tensors totaling approximately num_bytes."""
    # 4 bytes per float32 element
    n_elements = num_bytes // 4
    return {"weights": torch.randn(n_elements)}


def run_ray(worker_counts: list[int], num_trials: int) -> list[dict]:
    import ray

    @ray.remote
    class BroadcastWorker:
        def receive(self, data: dict) -> int:
            # Access the tensor to ensure deserialization actually happens
            return data["weights"].shape[0]

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        # Create max pool once to avoid SIGSEGV from repeated actor creation/destruction
        max_workers = max(worker_counts)
        pool = [BroadcastWorker.remote() for _ in range(max_workers)]

        # Quick warmup for all workers
        ref = ray.put(_make_payload(PAYLOAD_CONFIGS[0]["size"]))
        ray.get([w.receive.remote(ref) for w in pool])

        for cfg in PAYLOAD_CONFIGS:
            payload = _make_payload(cfg["size"])
            actual_bytes = payload["weights"].numel() * 4

            for n_workers in worker_counts:
                workers = pool[:n_workers]

                # Warmup
                for _ in range(WARMUP):
                    ref = ray.put(payload)
                    ray.get([w.receive.remote(ref) for w in workers])

                put_times = []
                total_times = []
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    ref = ray.put(payload)
                    t_put = time.perf_counter() - t0

                    t1 = time.perf_counter()
                    ray.get([w.receive.remote(ref) for w in workers])
                    t_get = time.perf_counter() - t1

                    put_times.append(t_put)
                    total_times.append(t_put + t_get)

                results.append({
                    "framework": "ray",
                    "payload_name": cfg["name"],
                    "payload_bytes": actual_bytes,
                    "workers": n_workers,
                    "put_mean_ms": statistics.mean(put_times) * 1000,
                    "total_mean_ms": statistics.mean(total_times) * 1000,
                    "total_median_ms": statistics.median(total_times) * 1000,
                    "total_std_ms": statistics.stdev(total_times) * 1000 if len(total_times) > 1 else 0,
                })
                print(f"  ray  {cfg['name']:>5s}  {n_workers}w  "
                      f"put={results[-1]['put_mean_ms']:.3f}ms  "
                      f"total={results[-1]['total_mean_ms']:.3f}ms")
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_trials: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class BroadcastWorker(Actor):
        @endpoint
        async def receive(self, data: dict) -> int:
            return data["weights"].shape[0]

    async def _run():
        results = []
        for cfg in PAYLOAD_CONFIGS:
            payload = _make_payload(cfg["size"])
            actual_bytes = payload["weights"].numel() * 4

            for n_workers in worker_counts:
                mesh = this_host().spawn_procs(per_host={"procs": n_workers})
                workers = mesh.spawn("bcast_workers", BroadcastWorker)

                # Warmup
                for _ in range(WARMUP):
                    await workers.receive.call(payload)

                total_times = []
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    await workers.receive.call(payload)
                    total_times.append(time.perf_counter() - t0)

                results.append({
                    "framework": "monarch",
                    "payload_name": cfg["name"],
                    "payload_bytes": actual_bytes,
                    "workers": n_workers,
                    "total_mean_ms": statistics.mean(total_times) * 1000,
                    "total_median_ms": statistics.median(total_times) * 1000,
                    "total_std_ms": statistics.stdev(total_times) * 1000 if len(total_times) > 1 else 0,
                })
                print(f"  monarch  {cfg['name']:>5s}  {n_workers}w  "
                      f"total={results[-1]['total_mean_ms']:.3f}ms")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="M2: Broadcast efficiency")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=WORKER_COUNTS)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"M2 Broadcast efficiency: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_trials)
    else:
        results = run_monarch(args.worker_counts, args.num_trials)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
