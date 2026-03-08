#!/usr/bin/env python3
"""
Test suite for the Ray RLlib + OpenEnv BlackJack pipeline.

Tests are numbered and run sequentially so that failures are easy to diagnose.
Each test prints a clear PASS / FAIL status.

Prerequisites
-------------
1. Activate the rl-benchmark conda environment:
       conda activate rl-benchmark

2. Start the OpenEnv BlackJack server (in a separate terminal):
       cd /Users/henry/cs214-rl-benchmark
       PYTHONPATH="external/OpenEnv:external/OpenEnv/envs:$PYTHONPATH" \
         OPENSPIEL_GAME=blackjack \
         python -m envs.openspiel_env.server.app --port 8004

3. Run the tests:
       python ray_rllib/test_rllib_pipeline.py

What is tested
--------------
  1. OpenEnv connectivity        – can we reach the server?
  2. Gymnasium wrapper spaces    – observation_space / action_space correct?
  3. Gymnasium wrapper reset     – does reset() return valid obs + info?
  4. Gymnasium wrapper step      – does step() return valid 5-tuple?
  5. Full episode rollout        – can we play a game to termination?
  6. Multiple episode rollouts   – stability over many consecutive games
  7. Policy registry             – can we list / instantiate policies?
  8. PPO algorithm build         – does PPO config build without error?
  9. PPO training (1 iteration)  – does algo.train() succeed?
 10. Policy inference            – does compute_single_action() work?
 11. Evaluation helper           – does evaluate_policy() return stats?
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Path setup (same as train_rllib.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_OPENENV_PATH = _PROJECT_ROOT / "external" / "OpenEnv"

for _p in [str(_OPENENV_PATH), str(_OPENENV_PATH / "envs"), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    """Run *fn*, capture pass/fail, print status."""
    print(f"\n{'─' * 60}")
    print(f"  TEST: {name}")
    print(f"{'─' * 60}")
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  ✅ PASS")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  ❌ FAIL: {e}")
        traceback.print_exc()


# ===================================================================
# Individual tests
# ===================================================================

SERVER_URL = "http://localhost:8000"
ENV_CONFIG = {
    "server_url": SERVER_URL,
    "game_name": "blackjack",
    "info_state_size": 111,
    "max_actions": 2,
    "max_steps_per_episode": 10,
}


def test_openenv_connectivity():
    """1. Verify the OpenEnv server is reachable."""
    from envs.openspiel_env import OpenSpielEnv
    import asyncio

    env = OpenSpielEnv(base_url=SERVER_URL)
    result = asyncio.run(env.reset())
    assert result is not None, "reset() returned None"
    assert hasattr(result, "observation"), "result has no 'observation'"
    assert len(result.observation.legal_actions) > 0, "no legal actions"
    asyncio.run(env.close())
    print(f"    Connected OK – legal_actions={result.observation.legal_actions}")


def test_gymnasium_wrapper_spaces():
    """2. Verify observation_space and action_space shapes."""
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**ENV_CONFIG)
    assert env.observation_space.shape == (111,), (
        f"Expected obs shape (111,), got {env.observation_space.shape}"
    )
    assert env.action_space.n == 2, (
        f"Expected 2 actions, got {env.action_space.n}"
    )
    print(f"    obs_space={env.observation_space}, action_space={env.action_space}")


def test_gymnasium_wrapper_reset():
    """3. Verify reset() returns a valid observation and info dict."""
    import numpy as np
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**ENV_CONFIG)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray), f"obs is {type(obs)}, expected ndarray"
    assert obs.shape == (111,), f"obs shape {obs.shape}, expected (111,)"
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}, expected float32"
    assert "legal_actions" in info, "info missing 'legal_actions'"
    env.close()
    print(f"    obs shape={obs.shape}, dtype={obs.dtype}, legal_actions={info['legal_actions']}")


def test_gymnasium_wrapper_step():
    """4. Verify step() returns the Gymnasium 5-tuple."""
    import numpy as np
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**ENV_CONFIG)
    obs, info = env.reset()

    action = info["legal_actions"][0]  # pick first legal action
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert isinstance(obs2, np.ndarray), f"obs2 is {type(obs2)}"
    assert obs2.shape == (111,), f"obs2 shape {obs2.shape}"
    assert isinstance(reward, float), f"reward is {type(reward)}"
    assert isinstance(terminated, bool), f"terminated is {type(terminated)}"
    assert isinstance(truncated, bool), f"truncated is {type(truncated)}"
    assert isinstance(info2, dict), f"info2 is {type(info2)}"
    env.close()
    print(f"    step returned: reward={reward}, terminated={terminated}, truncated={truncated}")


def test_full_episode():
    """5. Play a full episode to termination."""
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**ENV_CONFIG)
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()
    assert steps >= 1, "Episode ended with 0 steps"
    print(f"    Episode finished in {steps} steps, total_reward={total_reward}")


def test_multiple_episodes():
    """6. Play 10 consecutive episodes to verify stability."""
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper

    env = OpenEnvGymnasiumWrapper(**ENV_CONFIG)
    results = []

    for ep in range(10):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        results.append((steps, ep_reward))

    env.close()
    assert len(results) == 10, f"Expected 10 episodes, got {len(results)}"
    wins = sum(1 for _, r in results if r > 0)
    losses = sum(1 for _, r in results if r < 0)
    pushes = sum(1 for _, r in results if r == 0)
    print(f"    10 episodes played: W={wins} L={losses} P={pushes}")


def test_policy_registry():
    """7. Verify the policy registry lists and instantiates policies."""
    from rllib_policies import list_policies, get_policy

    policies = list_policies()
    assert "PPO" in policies, f"PPO not in registry: {policies}"
    assert "DQN" in policies, f"DQN not in registry: {policies}"

    ppo = get_policy("PPO")
    assert ppo.algorithm_name == "PPO"

    dqn = get_policy("DQN")
    assert dqn.algorithm_name == "DQN"

    # Test case-insensitive lookup
    ppo2 = get_policy("ppo")
    assert ppo2.algorithm_name == "PPO"

    print(f"    Registered policies: {policies}")


def test_ppo_build():
    """8. Build a PPO algorithm instance (no training yet)."""
    import ray
    from ray.tune.registry import register_env
    from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper
    from rllib_policies import get_policy

    ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

    env_id = "TestBlackjack-v0"
    register_env(env_id, lambda cfg: OpenEnvGymnasiumWrapper(**cfg))

    policy = get_policy("PPO")
    train_cfg = {
        "lr": 5e-4,
        "gamma": 0.99,
        "num_rollout_workers": 0,
        "train_batch_size": 128,
        "sgd_minibatch_size": 32,
        "num_sgd_iter": 2,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_clip_param": 10.0,
        "framework": "torch",
        "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},
    }

    global _algo  # keep alive for subsequent tests
    _algo = policy.build(env_name=env_id, env_config=ENV_CONFIG, train_cfg=train_cfg)
    print(f"    PPO algorithm built successfully")


def test_ppo_train_one_iteration():
    """9. Run a single PPO training iteration."""
    global _algo
    t0 = time.time()
    result = _algo.train()
    elapsed = time.time() - t0

    # Check that we got some metrics back
    env_runners = result.get("env_runners", {})
    
    ep_reward_mean = env_runners.get("episode_reward_mean")
    eps = env_runners.get("episodes_this_iter")
    
    if ep_reward_mean is None:
        ep_reward_mean = env_runners.get("episode_return_mean")
    if eps is None:
        eps = env_runners.get("num_episodes", 0)

    print(f"    1 iteration in {elapsed:.1f}s")
    print(f"    episode_reward_mean={ep_reward_mean}, episodes_this_iter={eps}")
    assert eps > 0 or ep_reward_mean is not None, "No episodes collected"


def test_policy_inference():
    """10. Verify forward_inference works with the trained algo."""
    import numpy as np
    global _algo

    from train_rllib import infer_action

    obs = np.zeros(111, dtype=np.float32)
    action = infer_action(_algo, obs)

    assert action in [0, 1], f"Unexpected action: {action}"
    print(f"    infer_action(zeros) → {action}")

    # Try with a random obs
    obs2 = np.random.randn(111).astype(np.float32)
    action2 = infer_action(_algo, obs2)
    assert action2 in [0, 1], f"Unexpected action: {action2}"
    print(f"    infer_action(random) → {action2}")
    
    # -----------------------------------------------------------------------
    # IMPORTANT: The single-player OpenEnv server only allows 1 session.
    # The _algo object holds a persistent session in its rollout worker.
    # We must stop the background workers before running local evaluation
    # so the evaluation script can acquire the session.
    # -----------------------------------------------------------------------
    _algo.stop()


def test_evaluation_helper():
    """11. Run the evaluate_policy helper from train_rllib."""
    global _algo

    # Import the helper directly from train_rllib
    sys.path.insert(0, str(_SCRIPT_DIR))
    from train_rllib import evaluate_policy

    stats = evaluate_policy(_algo, ENV_CONFIG, num_episodes=20)

    assert "win_rate" in stats, f"Missing 'win_rate' in stats: {stats}"
    assert "wins" in stats
    assert "losses" in stats
    assert "pushes" in stats
    assert stats["wins"] + stats["losses"] + stats["pushes"] == 20

    print(f"    20 eval episodes: W={stats['wins']} L={stats['losses']} "
          f"P={stats['pushes']}  win_rate={stats['win_rate']:.0%}")


# ===================================================================
# Runner
# ===================================================================

def main():
    print("=" * 60)
    print("  Ray RLlib + OpenEnv Pipeline Tests")
    print(f"  Server: {SERVER_URL}")
    print("=" * 60)

    t_start = time.time()

    run_test("1. OpenEnv connectivity", test_openenv_connectivity)
    run_test("2. Gymnasium wrapper spaces", test_gymnasium_wrapper_spaces)
    run_test("3. Gymnasium wrapper reset", test_gymnasium_wrapper_reset)
    run_test("4. Gymnasium wrapper step", test_gymnasium_wrapper_step)
    run_test("5. Full episode rollout", test_full_episode)
    run_test("6. Multiple episode rollouts", test_multiple_episodes)
    run_test("7. Policy registry", test_policy_registry)
    run_test("8. PPO algorithm build", test_ppo_build)
    run_test("9. PPO training (1 iteration)", test_ppo_train_one_iteration)
    run_test("10. Policy inference", test_policy_inference)
    run_test("11. Evaluation helper", test_evaluation_helper)

    # Cleanup
    try:
        import ray
        if "_algo" in globals():
            _algo.stop()
        ray.shutdown()
    except Exception:
        pass

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    for name, ok, err in _results:
        status = "✅ PASS" if ok else f"❌ FAIL: {err}"
        print(f"  {status}  {name}")

    print(f"\n  {passed}/{passed + failed} passed in {elapsed:.1f}s")

    if failed:
        print(f"\n  ⚠️  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n  🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
