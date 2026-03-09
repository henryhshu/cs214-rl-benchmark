import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(nn.Linear(hidden_dim, action_dim))
        self.value = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(x)
        policy_logits = self.policy(shared_out)
        value = self.value(shared_out)
        return policy_logits, value


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values[:-1]
    return advantages, returns


def configure_torch_allocator() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def resolve_device(device_env_var: str) -> torch.device:
    device_name = os.environ.get(
        device_env_var,
        "cuda" if torch.cuda.is_available() else "cpu",
    ).lower()

    if device_name == "cuda" and not torch.cuda.is_available():
        print(f"{device_env_var}=cuda requested but CUDA is unavailable; falling back to CPU")
        device_name = "cpu"

    return torch.device(device_name)


def configure_openenv_imports() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    openenv_path = project_root / "external" / "OpenEnv"
    if not openenv_path.exists():
        raise FileNotFoundError(f"OpenEnv directory not found at {openenv_path}")

    openenv_path_str = str(openenv_path)
    if openenv_path_str not in sys.path:
        sys.path.insert(0, openenv_path_str)


def extract_obs_legal(result: Any) -> tuple[List[float], List[int]]:
    obs_obj = result.observation
    return list(obs_obj.info_state), list(obs_obj.legal_actions)


def masked_dist(logits: torch.Tensor, legal_actions: List[int], action_dim: int) -> Categorical:
    mask = torch.zeros(action_dim, dtype=torch.bool, device=logits.device)
    mask[legal_actions] = True
    masked_logits = logits.masked_fill(~mask, -1e9)
    return Categorical(logits=masked_logits)


def is_retryable_openenv_error(exc: Exception) -> bool:
    exc_text = f"{type(exc).__name__}: {exc}"
    return (
        "ConnectionClosed" in exc_text
        or "CAPACITY_REACHED" in exc_text
        or "Server at capacity" in exc_text
    )


async def close_env_client_async(env: Any) -> None:
    close_fn = getattr(env, "close", None) or getattr(env, "disconnect", None)
    if close_fn is None:
        return

    try:
        maybe_result = close_fn()
        if asyncio.iscoroutine(maybe_result):
            await maybe_result
    except Exception:
        pass


def close_env_client_sync(env: Any) -> None:
    close_fn = getattr(env, "close", None) or getattr(env, "disconnect", None)
    if close_fn is None:
        return

    try:
        maybe_result = close_fn()
        if asyncio.iscoroutine(maybe_result):
            asyncio.run(maybe_result)
    except Exception:
        pass


def evaluate_policy_winrate(
    model_state: Dict[str, torch.Tensor],
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    server_url: str,
    game_name: str,
    device: torch.device,
    num_games: int = 20,
) -> Dict[str, float]:
    configure_openenv_imports()
    from envs.openspiel_env import OpenSpielAction, OpenSpielEnv

    eval_model = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
    eval_model.load_state_dict(model_state)
    eval_model.eval()

    wins = 0
    losses = 0
    draws = 0

    env = None
    try:
        env = OpenSpielEnv(base_url=server_url)
        for _ in range(num_games):
            result = env.reset()
            obs, legal_actions = extract_obs_legal(result)
            done = bool(result.done)
            final_reward = float(result.reward) if result.reward is not None else 0.0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, _ = eval_model(obs_t.unsqueeze(0))
                    logits = logits.squeeze(0)

                legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=device)
                legal_logits = logits[legal_idx]
                best_i = int(torch.argmax(legal_logits).item())
                action_id = int(legal_actions[best_i])

                step_result = env.step(OpenSpielAction(action_id=action_id, game_name=game_name))
                done = bool(step_result.done)
                if step_result.reward is not None:
                    final_reward = float(step_result.reward)

                if not done:
                    obs, legal_actions = extract_obs_legal(step_result)

            if final_reward > 0:
                wins += 1
            elif final_reward < 0:
                losses += 1
            else:
                draws += 1
    finally:
        if env is not None:
            close_env_client_sync(env)

    games_played = max(1, num_games)
    return {
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / games_played,
    }
