"""
GRPO Utilities for OpenEnv Training (Lightweight Version)

Stripped-down module for running Parts 1 & 2 of the tutorial notebook locally
without GPU/Forge dependencies. Contains only the OpenEnv exploration and
baseline benchmarking functions.

For full GRPO training (Parts 4+), use the complete version from:
  external/OpenEnv/examples/grpo_blackjack/grpo_utils.py
which requires TorchForge + Linux + NVIDIA GPU.
"""

import asyncio
import random

from envs.openspiel_env import OpenSpielAction, OpenSpielEnv


# ============================================================================
# OpenEnv Helper Functions (Parts 1 & 2)
# ============================================================================


def show_openenv_observation(observation):
    """
    Pretty print an OpenEnv observation.

    Args:
        observation: OpenEnv observation object
    """
    print("📊 Observation:")
    print(f"  Game phase: {observation.game_phase}")
    print(f"  Legal actions: {observation.legal_actions}")
    print(f"  Info state shape: {len(observation.info_state)}")
    print(f"  Info state (first 10): {observation.info_state[:10]}")


async def play_random_policy_async(server_url: str, num_games: int = 100) -> dict:
    """
    Benchmark random policy on OpenEnv environment (async version).

    Args:
        server_url: OpenEnv server URL
        num_games: Number of games to play

    Returns:
        dict with statistics (wins, losses, pushes, win_rate, total_games)
    """
    env = OpenSpielEnv(base_url=server_url)
    wins = losses = pushes = 0

    for i in range(num_games):
        result = await env.reset()
        done = False
        step_count = 0

        while not done and step_count < 10:
            action = random.choice(result.observation.legal_actions)
            result = await env.step(
                OpenSpielAction(action_id=action, game_name="blackjack")
            )
            done = result.done
            step_count += 1

        if result.reward > 0:
            wins += 1
        elif result.reward < 0:
            losses += 1
        else:
            pushes += 1

    await env.close()

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / num_games,
        "total_games": num_games,
    }


def play_random_policy(server_url: str, num_games: int = 100) -> dict:
    """
    Benchmark random policy on OpenEnv environment.

    Sync wrapper around the async implementation. Works in both
    regular scripts and Jupyter notebooks (which have a running event loop).

    Args:
        server_url: OpenEnv server URL
        num_games: Number of games to play

    Returns:
        dict with statistics (wins, losses, pushes, win_rate, total_games)
    """
    try:
        loop = asyncio.get_running_loop()
        # We're inside an event loop (e.g. Jupyter) — use nest_asyncio or
        # return the coroutine for the caller to await
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(play_random_policy_async(server_url, num_games))
    except RuntimeError:
        # No running event loop — use asyncio.run()
        return asyncio.run(play_random_policy_async(server_url, num_games))


# ============================================================================
# Stubs for Parts 4+ (require Forge + GPU)
# ============================================================================


async def setup_forge_training(config_path: str):
    """
    Placeholder for Forge GRPO training setup.

    The full implementation requires TorchForge, torchstore, vLLM, and
    NVIDIA GPUs. See external/OpenEnv/examples/grpo_blackjack/grpo_utils.py
    for the complete version.
    """
    raise NotImplementedError(
        "GRPO training requires TorchForge + Linux + NVIDIA GPU.\n"
        "This lightweight grpo_utils.py only supports Parts 1 & 2.\n\n"
        "To run full training:\n"
        "  1. Use a Linux machine with an NVIDIA GPU\n"
        "  2. Install TorchForge: cd external/torchforge && ./scripts/install.sh\n"
        "  3. Use the full grpo_utils.py from:\n"
        "     external/OpenEnv/examples/grpo_blackjack/grpo_utils.py"
    )
