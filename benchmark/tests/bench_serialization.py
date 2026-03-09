"""
Measure serialization/deserialization cost for model states and rollout tensors.

Isolates the data-transfer overhead from scheduling and computation, letting us
see how much time each framework spends marshalling tensors across processes.

Usage:
    python benchmark/tests/bench_serialization.py --framework ray
    python benchmark/tests/bench_serialization.py --framework monarch
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Reuse the actual model architecture
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ray_blackjack"))
from ppo_ray import ActorCritic


HIDDEN_DIMS = [64, 256, 1024]
HORIZONS = [128, 256, 512, 1024]
STATE_DIM = 52  # Blackjack observation size
ACTION_DIM = 4  # Blackjack action count
NUM_TRIALS = 50


def _make_model_state(hidden_dim: int) -> dict[str, torch.Tensor]:
    model = ActorCritic(STATE_DIM, hidden_dim, ACTION_DIM)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _make_rollout_tensor(horizon: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(horizon, STATE_DIM),
        "actions": torch.randint(0, ACTION_DIM, (horizon,)),
        "old_log_probs": torch.randn(horizon),
        "rewards": torch.randn(horizon),
        "values": torch.randn(horizon + 1),
        "dones": torch.zeros(horizon),
        "action_masks": torch.ones(horizon, ACTION_DIM),
        "mean_episode_return": torch.tensor(0.0),
    }


def _payload_bytes(d: dict[str, torch.Tensor]) -> int:
    return sum(v.element_size() * v.numel() for v in d.values() if hasattr(v, "numel"))


def run_ray(num_trials: int) -> list[dict]:
    import ray

    @ray.remote
    class EchoWorker:
        def echo(self, data: dict) -> dict:
            return data

    ray.init(ignore_reinit_error=True)
    results = []
    try:
        worker = EchoWorker.remote()
        # Warmup
        ray.get(worker.echo.remote({"x": torch.zeros(1)}))

        # Model state serialization
        for hd in HIDDEN_DIMS:
            state = _make_model_state(hd)
            nbytes = _payload_bytes(state)

            put_times, get_times, roundtrips = [], [], []
            for _ in range(num_trials):
                t0 = time.perf_counter()
                ref = ray.put(state)
                put_times.append(time.perf_counter() - t0)

                t1 = time.perf_counter()
                _ = ray.get(ref)
                get_times.append(time.perf_counter() - t1)

                t2 = time.perf_counter()
                ref2 = ray.put(state)
                ray.get(worker.echo.remote(ref2))
                roundtrips.append(time.perf_counter() - t2)

            results.append({
                "framework": "ray",
                "payload_type": "model_state",
                "param": f"hidden={hd}",
                "payload_bytes": nbytes,
                "put_mean_ms": statistics.mean(put_times) * 1000,
                "get_mean_ms": statistics.mean(get_times) * 1000,
                "roundtrip_mean_ms": statistics.mean(roundtrips) * 1000,
                "roundtrip_std_ms": statistics.stdev(roundtrips) * 1000 if len(roundtrips) > 1 else 0,
            })
            print(f"  ray  model hidden={hd:4d}  {nbytes:>8d}B  "
                  f"rt={results[-1]['roundtrip_mean_ms']:.3f}ms")

        # Rollout tensor serialization
        for horizon in HORIZONS:
            rollout = _make_rollout_tensor(horizon)
            nbytes = _payload_bytes(rollout)

            roundtrips = []
            for _ in range(num_trials):
                t0 = time.perf_counter()
                ref = ray.put(rollout)
                ray.get(worker.echo.remote(ref))
                roundtrips.append(time.perf_counter() - t0)

            results.append({
                "framework": "ray",
                "payload_type": "rollout",
                "param": f"horizon={horizon}",
                "payload_bytes": nbytes,
                "roundtrip_mean_ms": statistics.mean(roundtrips) * 1000,
                "roundtrip_std_ms": statistics.stdev(roundtrips) * 1000 if len(roundtrips) > 1 else 0,
            })
            print(f"  ray  rollout horizon={horizon:4d}  {nbytes:>8d}B  "
                  f"rt={results[-1]['roundtrip_mean_ms']:.3f}ms")

        del worker
    finally:
        ray.shutdown()
    return results


def run_monarch(num_trials: int) -> list[dict]:
    from monarch.actor import Actor, endpoint, this_host

    class EchoWorker(Actor):
        @endpoint
        async def echo(self, data: dict) -> dict:
            return data

    async def _run():
        results = []
        mesh = this_host().spawn_procs(per_host={"procs": 1})
        worker = mesh.spawn("echo", EchoWorker)

        # Warmup
        await worker.echo.call_one({"x": torch.zeros(1)})

        # Model state serialization
        for hd in HIDDEN_DIMS:
            state = _make_model_state(hd)
            nbytes = _payload_bytes(state)

            roundtrips = []
            for _ in range(num_trials):
                t0 = time.perf_counter()
                await worker.echo.call_one(state)
                roundtrips.append(time.perf_counter() - t0)

            results.append({
                "framework": "monarch",
                "payload_type": "model_state",
                "param": f"hidden={hd}",
                "payload_bytes": nbytes,
                "roundtrip_mean_ms": statistics.mean(roundtrips) * 1000,
                "roundtrip_std_ms": statistics.stdev(roundtrips) * 1000 if len(roundtrips) > 1 else 0,
            })
            print(f"  monarch  model hidden={hd:4d}  {nbytes:>8d}B  "
                  f"rt={results[-1]['roundtrip_mean_ms']:.3f}ms")

        # Rollout tensor serialization
        for horizon in HORIZONS:
            rollout = _make_rollout_tensor(horizon)
            nbytes = _payload_bytes(rollout)

            roundtrips = []
            for _ in range(num_trials):
                t0 = time.perf_counter()
                await worker.echo.call_one(rollout)
                roundtrips.append(time.perf_counter() - t0)

            results.append({
                "framework": "monarch",
                "payload_type": "rollout",
                "param": f"horizon={horizon}",
                "payload_bytes": nbytes,
                "roundtrip_mean_ms": statistics.mean(roundtrips) * 1000,
                "roundtrip_std_ms": statistics.stdev(roundtrips) * 1000 if len(roundtrips) > 1 else 0,
            })
            print(f"  monarch  rollout horizon={horizon:4d}  {nbytes:>8d}B  "
                  f"rt={results[-1]['roundtrip_mean_ms']:.3f}ms")

        return results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialization cost microbenchmark")
    parser.add_argument("--framework", required=True, choices=["ray", "monarch"])
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Serialization benchmark: framework={args.framework}")
    if args.framework == "ray":
        results = run_ray(args.num_trials)
    else:
        results = run_monarch(args.num_trials)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
