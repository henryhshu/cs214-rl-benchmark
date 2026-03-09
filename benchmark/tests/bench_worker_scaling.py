"""
Measure per-worker rollout time distributions as worker count scales.

Uses the real RolloutWorker and ActorCritic from the PPO scripts with
in-process (local) environments to measure actual per-worker rollout
variance and straggler effects.

Usage:
    python benchmark/tests/bench_worker_scaling.py --framework ray --worker-counts 1 2 4 8
    python benchmark/tests/bench_worker_scaling.py --framework monarch --worker-counts 1 2 4 8
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent

# Ensure imports work
sys.path.insert(0, str(ROOT / "benchmark"))

HORIZON = 256
HIDDEN_DIM = 64
GAME_NAME = "blackjack"
NUM_ROLLOUTS = 10


def _probe_env():
    from local_env import LocalEnv
    env = LocalEnv(GAME_NAME)
    result = env.reset()
    obs = list(result.observation.info_state)
    legal = list(result.observation.legal_actions)
    return len(obs), max(legal) + 1


def run_ray(worker_counts: list[int], num_rollouts: int) -> list[dict]:
    import ray
    sys.path.insert(0, str(ROOT / "ray_blackjack"))
    from ppo_ray import RolloutWorker, ActorCritic

    state_dim, action_dim = _probe_env()

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        model = ActorCritic(state_dim, HIDDEN_DIM, action_dim)
        model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for n_workers in worker_counts:
            workers = [
                RolloutWorker.options(num_cpus=0.5).remote(
                    HORIZON, state_dim, HIDDEN_DIM, action_dim,
                    "http://unused", GAME_NAME, True,
                )
                for _ in range(n_workers)
            ]

            # Warmup
            ref = ray.put(model_state)
            ray.get([w.collect.remote(ref, 0.0) for w in workers])

            all_worker_times = []
            wall_times = []
            for _ in range(num_rollouts):
                ref = ray.put(model_state)
                dispatch_time = time.time()
                t0 = time.perf_counter()
                rollouts = ray.get([w.collect.remote(ref, dispatch_time) for w in workers])
                wall_times.append(time.perf_counter() - t0)

                batch_worker_times = []
                for r in rollouts:
                    batch_worker_times.append(float(r["_worker_rollout_time_s"]))
                all_worker_times.append(batch_worker_times)

            # Flatten for per-worker stats
            flat_times = [t for batch in all_worker_times for t in batch]
            max_per_batch = [max(batch) for batch in all_worker_times]
            mean_per_batch = [statistics.mean(batch) for batch in all_worker_times]
            straggler_ratios = [mx / mn if mn > 0 else 1.0
                                for mx, mn in zip(max_per_batch, mean_per_batch)]

            results.append({
                "framework": "ray",
                "workers": n_workers,
                "horizon": HORIZON,
                "num_rollouts": num_rollouts,
                "wall_time_mean_s": statistics.mean(wall_times),
                "wall_time_std_s": statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                "worker_time_mean_s": statistics.mean(flat_times),
                "worker_time_std_s": statistics.stdev(flat_times) if len(flat_times) > 1 else 0,
                "worker_time_min_s": min(flat_times),
                "worker_time_max_s": max(flat_times),
                "straggler_ratio_mean": statistics.mean(straggler_ratios),
                "straggler_ratio_max": max(straggler_ratios),
                "per_batch_worker_times": all_worker_times,
            })
            print(f"  ray  workers={n_workers:2d}  wall={results[-1]['wall_time_mean_s']:.3f}s  "
                  f"straggler={results[-1]['straggler_ratio_mean']:.3f}")

            del workers
    finally:
        ray.shutdown()
    return results


def run_monarch(worker_counts: list[int], num_rollouts: int) -> list[dict]:
    from monarch.actor import this_host
    sys.path.insert(0, str(ROOT / "monarch_blackjack"))
    from ppo_monarch import RolloutWorker, ActorCritic

    state_dim, action_dim = _probe_env()

    async def _run():
        results = []
        model = ActorCritic(state_dim, HIDDEN_DIM, action_dim)
        model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for n_workers in worker_counts:
            mesh = this_host().spawn_procs(per_host={"procs": n_workers})
            workers = mesh.spawn(
                "scale_workers", RolloutWorker,
                HORIZON, state_dim, HIDDEN_DIM, action_dim,
                "http://unused", GAME_NAME, True,
            )

            # Warmup
            await workers.collect.call(model_state, 0.0)

            all_worker_times = []
            wall_times = []
            for _ in range(num_rollouts):
                dispatch_time = time.time()
                t0 = time.perf_counter()
                rollout_mesh = await workers.collect.call(model_state, dispatch_time)
                wall_times.append(time.perf_counter() - t0)

                rollouts = list(rollout_mesh.values())
                batch_worker_times = []
                for r in rollouts:
                    batch_worker_times.append(float(r["_worker_rollout_time_s"]))
                all_worker_times.append(batch_worker_times)

            flat_times = [t for batch in all_worker_times for t in batch]
            max_per_batch = [max(batch) for batch in all_worker_times]
            mean_per_batch = [statistics.mean(batch) for batch in all_worker_times]
            straggler_ratios = [mx / mn if mn > 0 else 1.0
                                for mx, mn in zip(max_per_batch, mean_per_batch)]

            results.append({
                "framework": "monarch",
                "workers": n_workers,
                "horizon": HORIZON,
                "num_rollouts": num_rollouts,
                "wall_time_mean_s": statistics.mean(wall_times),
                "wall_time_std_s": statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                "worker_time_mean_s": statistics.mean(flat_times),
                "worker_time_std_s": statistics.stdev(flat_times) if len(flat_times) > 1 else 0,
                "worker_time_min_s": min(flat_times),
                "worker_time_max_s": max(flat_times),
                "straggler_ratio_mean": statistics.mean(straggler_ratios),
                "straggler_ratio_max": max(straggler_ratios),
                "per_batch_worker_times": all_worker_times,
            })
            print(f"  monarch  workers={n_workers:2d}  wall={results[-1]['wall_time_mean_s']:.3f}s  "
                  f"straggler={results[-1]['straggler_ratio_mean']:.3f}")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker scaling microbenchmark")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Worker scaling benchmark: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.worker_counts, args.num_rollouts)
    else:
        results = run_monarch(args.worker_counts, args.num_rollouts)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
