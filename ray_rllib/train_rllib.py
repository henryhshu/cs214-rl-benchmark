#!/usr/bin/env python3
"""
Train an RL agent on OpenEnv BlackJack using Ray RLlib.

This script is the main entry-point for the **Ray RLlib** training workflow.
It reads a YAML config, registers the Gymnasium environment with Ray, picks
the requested algorithm from the policy registry, runs the training loop with
periodic evaluation and checkpointing, and prints a summary at the end.

Usage
-----
    # Make sure the OpenEnv server is running first:
    #   OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app --port 8000

    python ray_rllib/train_rllib.py --config ray_rllib/rllib_blackjack.yaml

Architecture
------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    train_rllib.py                           │
    │  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   │
    │  │  YAML config │──▶│ PolicyRegistry│──▶│ Ray Algorithm │   │
    │  └──────────────┘   └──────────────┘   └───────┬───────┘   │
    │                                                │           │
    │        ┌───────────────────────────────────────┤           │
    │        ▼                                       ▼           │
    │  ┌──────────────┐                     ┌──────────────┐     │
    │  │  Gymnasium   │◀── WebSocket ──────▶│   OpenEnv    │     │
    │  │   Wrapper    │                     │   Server     │     │
    │  └──────────────┘                     └──────────────┘     │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Ensure project paths are available
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OPENENV_PATH = _PROJECT_ROOT / "external" / "OpenEnv"

# Add OpenEnv to sys.path so the Gymnasium wrapper can import the client.
for _p in [str(_OPENENV_PATH), str(_OPENENV_PATH / "envs")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load and return a YAML config as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def register_env_with_ray(env_config: Dict[str, Any]) -> str:
    """Register the OpenEnv Gymnasium wrapper with Ray and return the env id."""
    from ray.tune.registry import register_env

    # Import from local module (same directory as this script)
    rllib_dir = pathlib.Path(__file__).resolve().parent
    if str(rllib_dir) not in sys.path:
        sys.path.insert(0, str(rllib_dir))

    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env_id = "OpenEnvBlackjack-v0"

    def _env_creator(cfg: Dict[str, Any]):
        return OpenEnvGymnasiumWrapper(
            server_url=cfg.get("server_url", "http://localhost:8000"),
            game_name=cfg.get("game_name", "blackjack"),
            info_state_size=cfg.get("info_state_size", 111),
            max_actions=cfg.get("max_actions", 2),
            max_steps_per_episode=cfg.get("max_steps_per_episode", 10),
        )

    register_env(env_id, _env_creator)
    return env_id


def infer_action(algo, obs: np.ndarray) -> int:
    """Compute a single action using the new RLModule API (Ray >= 2.54).

    The old ``algo.compute_single_action(obs)`` is deprecated.
    This helper uses ``get_module().forward_inference()`` instead.
    """
    module = algo.get_module()
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    result = module.forward_inference({"obs": obs_tensor})
    # result["action_dist_inputs"] contains logits; pick argmax for greedy
    action_logits = result["action_dist_inputs"]
    return int(torch.argmax(action_logits, dim=-1).item())


def evaluate_policy(algo, env_config: Dict[str, Any], num_episodes: int = 50) -> Dict[str, Any]:
    """Manually run evaluation episodes and return statistics."""
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**env_config)
    wins = losses = pushes = 0
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = infer_action(algo, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            pushes += 1

    env.close()

    return {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / num_episodes if num_episodes else 0.0,
        "avg_reward": total_reward / num_episodes if num_episodes else 0.0,
    }


def play_random_baseline(env_config: Dict[str, Any], num_episodes: int = 100) -> Dict[str, Any]:
    """Play random games to establish a baseline."""
    import random
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**env_config)
    wins = losses = pushes = 0

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            pushes += 1

    env.close()

    return {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / num_episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent on OpenEnv BlackJack with Ray RLlib."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ray_rllib/rllib_blackjack.yaml",
        help="Path to training YAML config.",
    )
    args = parser.parse_args()

    # ---- Load config -------------------------------------------------------
    cfg = load_config(args.config)
    algo_name: str = cfg.get("algorithm", "PPO")
    env_cfg: Dict[str, Any] = cfg.get("environment", {})
    train_cfg: Dict[str, Any] = cfg.get("training", {})
    eval_cfg: Dict[str, Any] = cfg.get("evaluation", {})
    ckpt_cfg: Dict[str, Any] = cfg.get("checkpoint", {})
    ray_cfg: Dict[str, Any] = cfg.get("ray", {})

    num_iterations = train_cfg.pop("num_iterations", 100)
    eval_interval = eval_cfg.get("eval_interval", 10)
    num_eval_episodes = eval_cfg.get("num_eval_episodes", 50)
    save_dir = ckpt_cfg.get("save_dir", "checkpoints/rllib_blackjack")
    save_interval = ckpt_cfg.get("save_interval", 25)

    print("=" * 70)
    print(f"  Ray RLlib + OpenEnv BlackJack Training")
    print(f"  Algorithm : {algo_name}")
    print(f"  Server    : {env_cfg.get('server_url', 'http://localhost:8000')}")
    print(f"  Iterations: {num_iterations}")
    print("=" * 70)

    # ---- Initialise Ray ----------------------------------------------------
    import ray

    ray.init(
        num_cpus=ray_cfg.get("num_cpus", 4),
        num_gpus=ray_cfg.get("num_gpus", 0),
        ignore_reinit_error=True,
    )

    # ---- Register Gymnasium env with Ray -----------------------------------
    env_id = register_env_with_ray(env_cfg)

    # ---- Build the algorithm via the policy registry -----------------------
    from rllib_policies import get_policy

    policy = get_policy(algo_name)
    algo = policy.build(
        env_name=env_id,
        env_config=env_cfg,
        train_cfg=train_cfg,
    )

    print(f"\n✅ {algo_name} algorithm built successfully.\n")

    # ---- Random baseline ---------------------------------------------------
    print("🎲 Running random baseline …")
    baseline = play_random_baseline(env_cfg, num_episodes=100)
    print(f"   Random win rate: {baseline['win_rate']:.1%}  "
          f"(W {baseline['wins']} / L {baseline['losses']} / P {baseline['pushes']})\n")

    # ---- Training loop -----------------------------------------------------
    best_win_rate = 0.0
    training_history: list[Dict[str, Any]] = []
    start_time = time.time()

    for i in range(1, num_iterations + 1):
        result = algo.train()

        # Extract key metrics from training result
        env_runners = result.get("env_runners", {})
        mean_reward = env_runners.get("episode_reward_mean")
        eps_this_iter = env_runners.get("episodes_this_iter")

        if mean_reward is None:
            mean_reward = env_runners.get("episode_return_mean", float("nan"))
        if eps_this_iter is None:
            eps_this_iter = env_runners.get("num_episodes", 0)

        entry = {
            "iteration": i,
            "episode_reward_mean": mean_reward,
            "episodes_this_iter": eps_this_iter,
            "timestamp": time.time() - start_time,
        }

        # Periodic evaluation
        if i % eval_interval == 0 or i == num_iterations:
            eval_stats = evaluate_policy(algo, env_cfg, num_eval_episodes)
            entry["eval"] = eval_stats
            win_rate = eval_stats["win_rate"]

            symbol = "🏆" if win_rate > best_win_rate else "📊"
            best_win_rate = max(best_win_rate, win_rate)

            print(
                f"  [{i:>4}/{num_iterations}] reward_mean={mean_reward:+.3f}  "
                f"{symbol} eval win_rate={win_rate:.1%}  "
                f"(W {eval_stats['wins']} / L {eval_stats['losses']} / P {eval_stats['pushes']})"
            )
        else:
            print(
                f"  [{i:>4}/{num_iterations}] reward_mean={mean_reward:+.3f}  "
                f"episodes={eps_this_iter}"
            )

        training_history.append(entry)

        # Periodic checkpoint
        if i % save_interval == 0:
            ckpt_dir = pathlib.Path(save_dir) / f"iter_{i}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            algo.save(str(ckpt_dir.absolute()))
            print(f"  💾 Checkpoint saved → {ckpt_dir.absolute()}")

    elapsed = time.time() - start_time

    # ---- Final save --------------------------------------------------------
    final_dir = pathlib.Path(save_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    algo.save(str(final_dir.absolute()))

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Training Summary")
    print("=" * 70)
    print(f"  Algorithm        : {algo_name}")
    print(f"  Iterations       : {num_iterations}")
    print(f"  Wall time        : {elapsed:.1f}s")
    print(f"  Random baseline  : {baseline['win_rate']:.1%}")
    print(f"  Best eval win rate: {best_win_rate:.1%}")
    print(f"  Final checkpoint : {final_dir}")
    print("=" * 70)

    # Save training log
    log_path = pathlib.Path(save_dir) / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(
            {
                "algorithm": algo_name,
                "config": cfg,
                "baseline": baseline,
                "history": training_history,
                "best_win_rate": best_win_rate,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"  📄 Training log  : {log_path}\n")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
