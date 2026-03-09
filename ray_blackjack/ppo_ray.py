import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import random

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
DEVICE_NAME = os.environ.get(
	"RAY_DEVICE",
	"cuda" if torch.cuda.is_available() else "cpu",
).lower()
if DEVICE_NAME == "cuda" and not torch.cuda.is_available():
	print("RAY_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU")
	DEVICE_NAME = "cpu"
DEVICE = torch.device(DEVICE_NAME)


def _configure_openenv_imports() -> None:
	project_root = Path(__file__).resolve().parent.parent
	openenv_path = project_root / "external" / "OpenEnv"
	if not openenv_path.exists():
		raise FileNotFoundError(f"OpenEnv directory not found at {openenv_path}")

	openenv_path_str = str(openenv_path)
	if openenv_path_str not in sys.path:
		sys.path.insert(0, openenv_path_str)


def _extract_obs_legal(result: Any) -> tuple[List[float], List[int]]:
	obs_obj = result.observation
	return list(obs_obj.info_state), list(obs_obj.legal_actions)


def _write_metrics(path: str | None, record: dict) -> None:
	if not path:
		return
	with open(path, "a") as f:
		f.write(json.dumps(record) + "\n")


def _tensor_bytes(d: Dict[str, Any]) -> int:
	return sum(v.element_size() * v.numel() for v in d.values() if hasattr(v, "numel"))


def _close_env_client_sync(env: Any) -> None:
	close_fn = getattr(env, "close", None) or getattr(env, "disconnect", None)
	if close_fn is None:
		return

	try:
		maybe_result = close_fn()
		if asyncio.iscoroutine(maybe_result):
			asyncio.run(maybe_result)
	except Exception:
		pass


@ray.remote
class RolloutWorker:
	def __init__(
		self,
		horizon: int,
		state_dim: int,
		hidden_dim: int,
		action_dim: int,
		base_url: str,
		game_name: str,
		local_env: bool = False,
	):
		self.horizon = horizon
		self.action_dim = action_dim
		self.base_url = base_url
		self.game_name = game_name
		self.local_env = local_env

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = ActorCritic(state_dim, hidden_dim, action_dim).to(self.device)
		self.model.eval()

		if local_env:
			# Import the local env here so it lives in the worker process
			import sys as _sys
			import pathlib as _pl
			_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "benchmark"))
			from local_env import LocalEnv
			self._LocalEnv = LocalEnv
		else:
			_configure_openenv_imports()
			from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
			self._OpenSpielAction = OpenSpielAction
			self._OpenSpielEnv = OpenSpielEnv

	def _masked_dist(self, logits: torch.Tensor, legal_actions: List[int]) -> Categorical:
		mask = torch.zeros(self.action_dim, dtype=torch.bool, device=logits.device)
		mask[legal_actions] = True
		masked_logits = logits.masked_fill(~mask, -1e9)
		return Categorical(logits=masked_logits)

	async def collect(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		self.model.load_state_dict(model_state)
		if self.local_env:
			return self._collect_local()
		max_retries = 12

		for attempt in range(max_retries):
			try:
				return await self._collect_async()
			except Exception as exc:
				exc_text = f"{type(exc).__name__}: {exc}"
				is_capacity = "CAPACITY_REACHED" in exc_text or "Server at capacity" in exc_text
				is_retryable = is_capacity or "ConnectionClosed" in exc_text
				if not is_retryable or attempt == max_retries - 1:
					raise
				# Exponential backoff with full jitter: [0, min(cap, base * 2^attempt)]
				cap = 5.0
				base = 0.25
				wait = random.uniform(0, min(cap, base * (2 ** attempt)))
				await asyncio.sleep(wait)

		raise RuntimeError("RolloutWorker.collect failed after retries")

	def _collect_local(self) -> Dict[str, torch.Tensor]:
		"""CPU-compute-bound rollout using in-process OpenSpiel (no network I/O)."""
		env = self._LocalEnv(self.game_name)
		obs_buf, actions_buf, logprob_buf = [], [], []
		rewards_buf, values_buf, dones_buf, masks_buf = [], [], [], []
		episode_returns, running_return = [], 0.0

		result = env.reset()
		obs, legal_actions = _extract_obs_legal(result)

		for _ in range(self.horizon):
			obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
			with torch.no_grad():
				logits, value = self.model(obs_t.unsqueeze(0))
				logits = logits.squeeze(0)
				value = value.squeeze(0).squeeze(-1)
				dist = self._masked_dist(logits, legal_actions)
				action = dist.sample()
				log_prob = dist.log_prob(action)

			next_result = env.step(int(action.item()))
			reward = float(next_result.reward or 0.0)
			done = bool(next_result.done)
			next_obs, next_legal = _extract_obs_legal(next_result)

			action_mask = torch.zeros(self.action_dim, dtype=torch.float32)
			action_mask[legal_actions] = 1.0
			obs_buf.append(obs_t.cpu())
			actions_buf.append(action.cpu())
			logprob_buf.append(log_prob.cpu())
			rewards_buf.append(reward)
			values_buf.append(value.cpu())
			dones_buf.append(float(done))
			masks_buf.append(action_mask)

			running_return += reward
			if done:
				episode_returns.append(running_return)
				running_return = 0.0
				result = env.reset()
				obs, legal_actions = _extract_obs_legal(result)
			else:
				obs, legal_actions = next_obs, next_legal

		with torch.no_grad():
			next_value = (
				self.model(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))[1]
				.squeeze(0).squeeze(-1).cpu()
			)

		values_plus_bootstrap = torch.cat([torch.stack(values_buf), next_value.view(1)], dim=0)
		return {
			"obs": torch.stack(obs_buf),
			"actions": torch.stack(actions_buf).long(),
			"old_log_probs": torch.stack(logprob_buf),
			"rewards": torch.tensor(rewards_buf, dtype=torch.float32),
			"values": values_plus_bootstrap,
			"dones": torch.tensor(dones_buf, dtype=torch.float32),
			"action_masks": torch.stack(masks_buf),
			"mean_episode_return": torch.tensor(
				episode_returns if episode_returns else [running_return], dtype=torch.float32
			).mean(),
		}

	async def _collect_async(self) -> Dict[str, torch.Tensor]:
		obs_buf: List[torch.Tensor] = []
		actions_buf: List[torch.Tensor] = []
		logprob_buf: List[torch.Tensor] = []
		rewards_buf: List[float] = []
		values_buf: List[torch.Tensor] = []
		dones_buf: List[float] = []
		masks_buf: List[torch.Tensor] = []
		episode_returns: List[float] = []
		running_return = 0.0

		# Stagger connection attempts so workers don't all hit the same
		# uvicorn process simultaneously (each process allows 1 session).
		await asyncio.sleep(random.uniform(0, 0.5))

		async with self._OpenSpielEnv(base_url=self.base_url) as env:
			result = await env.reset()
			obs, legal_actions = _extract_obs_legal(result)

			for _ in range(self.horizon):
				obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

				with torch.no_grad():
					logits, value = self.model(obs_t.unsqueeze(0))
					logits = logits.squeeze(0)
					value = value.squeeze(0).squeeze(-1)
					dist = self._masked_dist(logits, legal_actions)
					action = dist.sample()
					log_prob = dist.log_prob(action)

				next_result = await env.step(
					self._OpenSpielAction(action_id=int(action.item()), game_name=self.game_name)
				)

				reward = float(next_result.reward)
				done = bool(next_result.done)
				next_obs, next_legal = _extract_obs_legal(next_result)

				action_mask = torch.zeros(self.action_dim, dtype=torch.float32)
				action_mask[legal_actions] = 1.0

				obs_buf.append(obs_t.cpu())
				actions_buf.append(action.cpu())
				logprob_buf.append(log_prob.cpu())
				rewards_buf.append(reward)
				values_buf.append(value.cpu())
				dones_buf.append(float(done))
				masks_buf.append(action_mask)

				running_return += reward
				if done:
					episode_returns.append(running_return)
					running_return = 0.0
					reset_result = await env.reset()
					obs, legal_actions = _extract_obs_legal(reset_result)
				else:
					obs, legal_actions = next_obs, next_legal

			with torch.no_grad():
				next_value = (
					self.model(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))[1]
					.squeeze(0)
					.squeeze(-1)
					.cpu()
				)

		values_plus_bootstrap = torch.cat([torch.stack(values_buf), next_value.view(1)], dim=0)

		return {
			"obs": torch.stack(obs_buf),
			"actions": torch.stack(actions_buf).long(),
			"old_log_probs": torch.stack(logprob_buf),
			"rewards": torch.tensor(rewards_buf, dtype=torch.float32),
			"values": values_plus_bootstrap,
			"dones": torch.tensor(dones_buf, dtype=torch.float32),
			"action_masks": torch.stack(masks_buf),
			"mean_episode_return": torch.tensor(
				episode_returns if episode_returns else [running_return],
				dtype=torch.float32,
			).mean(),
		}


@ray.remote
class PPOLearner:
	def __init__(
		self,
		state_dim: int,
		hidden_dim: int,
		action_dim: int,
		lr: float,
		gamma: float,
		gae_lambda: float,
		clip_eps: float,
		vf_coef: float,
		ent_coef: float,
		max_grad_norm: float,
	):
		self.model = ActorCritic(state_dim, hidden_dim, action_dim).to(DEVICE)
		self.optim = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_eps = clip_eps
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.max_grad_norm = max_grad_norm

	def get_model_state(self) -> Dict[str, torch.Tensor]:
		return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

	def _evaluate_actions(
		self,
		states: torch.Tensor,
		actions: torch.Tensor,
		action_masks: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		logits, values = self.model(states)
		masked_logits = logits.masked_fill(action_masks <= 0, -1e9)

		dist = Categorical(logits=masked_logits)
		log_probs = dist.log_prob(actions)
		entropy = dist.entropy()
		return log_probs, entropy, values.squeeze(-1)

	def update(
		self,
		rollouts: List[Dict[str, torch.Tensor]],
		ppo_epochs: int,
		minibatch_size: int,
	) -> Dict[str, float]:
		states_list: List[torch.Tensor] = []
		actions_list: List[torch.Tensor] = []
		old_logp_list: List[torch.Tensor] = []
		returns_list: List[torch.Tensor] = []
		adv_list: List[torch.Tensor] = []
		masks_list: List[torch.Tensor] = []

		worker_returns: List[float] = []
		for rollout in rollouts:
			rewards = rollout["rewards"].to(DEVICE)
			values = rollout["values"].to(DEVICE)
			dones = rollout["dones"].to(DEVICE)

			advantages, returns = compute_gae(
				rewards,
				values,
				dones,
				gamma=self.gamma,
				lam=self.gae_lambda,
			)

			states_list.append(rollout["obs"].to(DEVICE))
			actions_list.append(rollout["actions"].to(DEVICE))
			old_logp_list.append(rollout["old_log_probs"].to(DEVICE))
			returns_list.append(returns)
			adv_list.append(advantages)
			masks_list.append(rollout["action_masks"].to(DEVICE))
			worker_returns.append(float(rollout["mean_episode_return"]))

		states = torch.cat(states_list, dim=0)
		actions = torch.cat(actions_list, dim=0)
		old_log_probs = torch.cat(old_logp_list, dim=0).detach()
		returns = torch.cat(returns_list, dim=0).detach()
		advantages = torch.cat(adv_list, dim=0).detach()
		action_masks = torch.cat(masks_list, dim=0)

		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		policy_losses: List[float] = []
		value_losses: List[float] = []
		entropies: List[float] = []

		batch_size = states.shape[0]
		minibatch_size = max(1, min(minibatch_size, batch_size))

		self.model.train()
		for _ in range(ppo_epochs):
			permutation = torch.randperm(batch_size, device=DEVICE)
			for start in range(0, batch_size, minibatch_size):
				mb_idx = permutation[start : start + minibatch_size]

				mb_states = states[mb_idx]
				mb_actions = actions[mb_idx]
				mb_old_log_probs = old_log_probs[mb_idx]
				mb_advantages = advantages[mb_idx]
				mb_returns = returns[mb_idx]
				mb_masks = action_masks[mb_idx]

				new_log_probs, entropy, values = self._evaluate_actions(
					mb_states,
					mb_actions,
					mb_masks,
				)

				ratio = torch.exp(new_log_probs - mb_old_log_probs)
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages

				policy_loss = -torch.min(surr1, surr2).mean()
				value_loss = F.mse_loss(values, mb_returns)
				entropy_bonus = entropy.mean()

				loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

				self.optim.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				self.optim.step()

				policy_losses.append(float(policy_loss.detach().cpu().item()))
				value_losses.append(float(value_loss.detach().cpu().item()))
				entropies.append(float(entropy_bonus.detach().cpu().item()))

		return {
			"policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
			"value_loss": sum(value_losses) / max(1, len(value_losses)),
			"entropy": sum(entropies) / max(1, len(entropies)),
			"mean_worker_return": sum(worker_returns) / max(1, len(worker_returns)),
		}


def evaluate_policy_winrate(
	model_state: Dict[str, torch.Tensor],
	state_dim: int,
	hidden_dim: int,
	action_dim: int,
	server_url: str,
	game_name: str,
	num_games: int = 20,
	local_env: bool = False,
) -> Dict[str, float]:
	if local_env:
		import sys as _sys
		_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
		from local_env import LocalEnv
		_configure_openenv_imports()
	from envs.openspiel_env import OpenSpielAction, OpenSpielEnv

	eval_model = ActorCritic(state_dim, hidden_dim, action_dim).to(DEVICE)
	eval_model.load_state_dict(model_state)
	eval_model.eval()

	if local_env:
		w = l = d = 0
		for _ in range(num_games):
			env = LocalEnv(game_name)
			result = env.reset()
			obs, legal_actions = _extract_obs_legal(result)
			done, final_reward = False, 0.0
			while not done:
				obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
				with torch.no_grad():
					logits, _ = eval_model(obs_t.unsqueeze(0))
					logits = logits.squeeze(0)
				legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=DEVICE)
				best_i = int(torch.argmax(logits[legal_idx]).item())
				step_result = env.step(int(legal_actions[best_i]))
				done = bool(step_result.done)
				final_reward = float(step_result.reward or final_reward)
				if not done:
					obs, legal_actions = _extract_obs_legal(step_result)
			if final_reward > 0: w += 1
			elif final_reward < 0: l += 1
			else: d += 1
		wins, losses, draws = w, l, d
	else:
		_configure_openenv_imports()
		from envs.openspiel_env import OpenSpielAction, OpenSpielEnv

		async def _run_eval() -> tuple[int, int, int]:
			w = l = d = 0
			async with OpenSpielEnv(base_url=server_url) as env:
				for _ in range(num_games):
					result = await env.reset()
					obs, legal_actions = _extract_obs_legal(result)
					done = bool(result.done)
					final_reward = float(result.reward) if result.reward is not None else 0.0

					while not done:
						obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
						with torch.no_grad():
							logits, _ = eval_model(obs_t.unsqueeze(0))
							logits = logits.squeeze(0)

						legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=DEVICE)
						legal_logits = logits[legal_idx]
						best_i = int(torch.argmax(legal_logits).item())
						action_id = int(legal_actions[best_i])

						step_result = await env.step(OpenSpielAction(action_id=action_id, game_name=game_name))
						done = bool(step_result.done)
						final_reward = float(step_result.reward) if step_result.reward is not None else final_reward

						if not done:
							obs, legal_actions = _extract_obs_legal(step_result)

					if final_reward > 0:
						w += 1
					elif final_reward < 0:
						l += 1
					else:
						d += 1
			return w, l, d

		wins, losses, draws = asyncio.run(_run_eval())

	games_played = max(1, num_games)
	return {
		"wins": float(wins),
		"losses": float(losses),
		"draws": float(draws),
		"win_rate": wins / games_played,
	}


def _actor_options(is_learner: bool) -> Dict[str, float]:
	if is_learner and DEVICE.type == "cuda":
		return {"num_gpus": 1.0}
	# Workers are I/O-bound (WebSocket); CPU inference is fine for this model size.
	return {"num_cpus": 1.0 if is_learner else 0.5}


def main() -> None:
	parser = argparse.ArgumentParser(description="Ray Core PPO with clipped objective")
	parser.add_argument("--server-url", type=str, default="http://localhost:8000")
	parser.add_argument("--game-name", type=str, default="blackjack")
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--horizon", type=int, default=256)
	parser.add_argument("--iterations", type=int, default=50)
	parser.add_argument("--hidden-dim", type=int, default=64)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--clip-eps", type=float, default=0.2)
	parser.add_argument("--vf-coef", type=float, default=0.5)
	parser.add_argument("--ent-coef", type=float, default=0.01)
	parser.add_argument("--ppo-epochs", type=int, default=4)
	parser.add_argument("--minibatch-size", type=int, default=256)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)
	parser.add_argument("--eval-games", type=int, default=20)
	parser.add_argument("--ray-address", type=str, default=None)
	parser.add_argument("--local-env", action="store_true",
		help="Run OpenSpiel in-process (CPU-bound) instead of via WebSocket server")
	parser.add_argument("--metrics-file", type=str, default=None,
		help="Path to write per-iteration JSONL metrics (for benchmarking)")
	args = parser.parse_args()

	init_kwargs = {"ignore_reinit_error": True}
	if args.ray_address:
		init_kwargs["address"] = args.ray_address
	ray.init(**init_kwargs)

	try:
		if args.local_env:
			import sys as _sys
			_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
			from local_env import LocalEnv as _LocalEnv
			_probe_env = _LocalEnv(args.game_name)
			obs, legal_actions = _extract_obs_legal(_probe_env.reset())
		else:
			_configure_openenv_imports()
			from envs.openspiel_env import OpenSpielEnv
			async def _probe():
				async with OpenSpielEnv(base_url=args.server_url) as env:
					return _extract_obs_legal(await env.reset())
			obs, legal_actions = asyncio.run(_probe())

		state_dim = len(obs)
		action_dim = max(legal_actions) + 1

		learner = PPOLearner.options(**_actor_options(is_learner=True)).remote(
			state_dim,
			args.hidden_dim,
			action_dim,
			args.lr,
			args.gamma,
			args.gae_lambda,
			args.clip_eps,
			args.vf_coef,
			args.ent_coef,
			args.max_grad_norm,
		)

		workers = [
			RolloutWorker.options(**_actor_options(is_learner=False)).remote(
				args.horizon,
				state_dim,
				args.hidden_dim,
				action_dim,
				args.server_url,
				args.game_name,
				args.local_env,
			)
			for _ in range(max(1, args.num_workers))
		]

		env_mode = "local" if args.local_env else "websocket"
		print(
			f"Starting PPO (Ray Core): device={DEVICE.type}, workers={len(workers)}, "
			f"horizon={args.horizon}, state_dim={state_dim}, action_dim={action_dim}, "
			f"env={env_mode}"
		)

		run_start = time.perf_counter()
		for iteration in range(args.iterations):
			iter_start = time.perf_counter()

			model_state = ray.get(learner.get_model_state.remote())
			model_state_ref = ray.put(model_state)

			t0 = time.perf_counter()
			rollout_refs = [worker.collect.remote(model_state_ref) for worker in workers]
			rollouts = ray.get(rollout_refs)
			rollout_time = time.perf_counter() - t0

			t1 = time.perf_counter()
			metrics = ray.get(
				learner.update.remote(
					rollouts,
					args.ppo_epochs,
					args.minibatch_size,
				)
			)
			update_time = time.perf_counter() - t1
			iter_time = time.perf_counter() - iter_start

			total_steps = len(workers) * args.horizon
			model_bytes = _tensor_bytes(model_state)
			rollout_bytes = sum(_tensor_bytes(r) for r in rollouts)
			total_mb = (model_bytes + rollout_bytes) / 1e6

			print(
				f"iter={iteration:03d} "
				f"return={metrics['mean_worker_return']:.3f} "
				f"pi_loss={metrics['policy_loss']:.4f} "
				f"v_loss={metrics['value_loss']:.4f} "
				f"entropy={metrics['entropy']:.4f}"
			)
			_write_metrics(args.metrics_file, {
				"iteration": iteration,
				"timestamp": time.time(),
				"rollout_time_s": rollout_time,
				"update_time_s": update_time,
				"iter_time_s": iter_time,
				"steps_collected": total_steps,
				"steps_per_sec": total_steps / max(rollout_time, 1e-9),
				"model_bytes": model_bytes,
				"rollout_bytes": rollout_bytes,
				"total_transfer_mb": total_mb,
				"throughput_mb_s": total_mb / max(iter_time, 1e-9),
				**metrics,
			})

		trained_state = ray.get(learner.get_model_state.remote())
		eval_metrics = evaluate_policy_winrate(
			model_state=trained_state,
			state_dim=state_dim,
			hidden_dim=args.hidden_dim,
			action_dim=action_dim,
			server_url=args.server_url,
			game_name=args.game_name,
			num_games=max(1, args.eval_games),
			local_env=args.local_env,
		)
		print(
			f"Evaluation over {int(args.eval_games)} games: "
			f"wins={int(eval_metrics['wins'])}, "
			f"losses={int(eval_metrics['losses'])}, "
			f"draws={int(eval_metrics['draws'])}, "
			f"win_rate={eval_metrics['win_rate']:.2%}"
		)
		_write_metrics(args.metrics_file, {
			"final": True,
			"total_time_s": time.perf_counter() - run_start,
			**eval_metrics,
		})
	finally:
		ray.shutdown()


if __name__ == "__main__":
	main()
