import argparse
import sys
import time
from typing import Dict, List

import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from benchmark import (
	ActorCritic,
	close_env_client_sync,
	compute_gae,
	configure_openenv_imports,
	configure_torch_allocator,
	evaluate_policy_winrate,
	extract_obs_legal,
	is_retryable_openenv_error,
	masked_dist,
	resolve_device,
)


configure_torch_allocator()
DEVICE = resolve_device("RAY_DEVICE")


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
	):
		self.horizon = horizon
		self.action_dim = action_dim
		self.base_url = base_url
		self.game_name = game_name

		self.model = ActorCritic(state_dim, hidden_dim, action_dim).to(DEVICE)
		self.model.eval()

		configure_openenv_imports()
		from envs.openspiel_env import OpenSpielAction, OpenSpielEnv

		self._OpenSpielAction = OpenSpielAction
		self._OpenSpielEnv = OpenSpielEnv

	def collect(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		self.model.load_state_dict(model_state)
		max_retries = 3

		for attempt in range(max_retries):
			env = None
			try:
				obs_buf: List[torch.Tensor] = []
				actions_buf: List[torch.Tensor] = []
				logprob_buf: List[torch.Tensor] = []
				rewards_buf: List[float] = []
				values_buf: List[torch.Tensor] = []
				dones_buf: List[float] = []
				masks_buf: List[torch.Tensor] = []
				episode_returns: List[float] = []
				running_return = 0.0

				env = self._OpenSpielEnv(base_url=self.base_url)
				result = env.reset()
				obs, legal_actions = extract_obs_legal(result)

				for _ in range(self.horizon):
					obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

					with torch.no_grad():
						logits, value = self.model(obs_t.unsqueeze(0))
						logits = logits.squeeze(0)
						value = value.squeeze(0).squeeze(-1)
						dist = masked_dist(logits, legal_actions, self.action_dim)
						action = dist.sample()
						log_prob = dist.log_prob(action)

					next_result = env.step(
						self._OpenSpielAction(action_id=int(action.item()), game_name=self.game_name)
					)

					reward = float(next_result.reward)
					done = bool(next_result.done)
					next_obs, next_legal = extract_obs_legal(next_result)

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
						reset_result = env.reset()
						obs, legal_actions = extract_obs_legal(reset_result)
					else:
						obs, legal_actions = next_obs, next_legal

				with torch.no_grad():
					next_value = (
						self.model(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))[1]
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
			except Exception as exc:
				if not is_retryable_openenv_error(exc) or attempt == max_retries - 1:
					raise
				time.sleep(0.3 * (attempt + 1))
			finally:
				if env is not None:
					close_env_client_sync(env)
					time.sleep(0.05)

		raise RuntimeError("RolloutWorker.collect failed after retries")


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


def _actor_options(is_learner: bool) -> Dict[str, float]:
	if DEVICE.type == "cuda":
		return {"num_gpus": 1.0}
	# Keep resource asks explicit so workers spread predictably.
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
	args = parser.parse_args()

	configure_openenv_imports()
	from envs.openspiel_env import OpenSpielEnv

	init_kwargs = {"ignore_reinit_error": True}
	if args.ray_address:
		init_kwargs["address"] = args.ray_address
	ray.init(**init_kwargs)

	try:
		env = None
		try:
			env = OpenSpielEnv(base_url=args.server_url)
			initial = env.reset()
			obs, legal_actions = extract_obs_legal(initial)
		finally:
			if env is not None:
				close_env_client_sync(env)

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
			)
			for _ in range(max(1, args.num_workers))
		]

		print(
			f"Starting PPO (Ray Core): device={DEVICE.type}, workers={len(workers)}, "
			f"horizon={args.horizon}, state_dim={state_dim}, action_dim={action_dim}"
		)

		for iteration in range(args.iterations):
			model_state = ray.get(learner.get_model_state.remote())
			model_state_ref = ray.put(model_state)

			rollout_refs = [worker.collect.remote(model_state_ref) for worker in workers]
			rollouts = ray.get(rollout_refs)

			metrics = ray.get(
				learner.update.remote(
					rollouts,
					args.ppo_epochs,
					args.minibatch_size,
				)
			)

			print(
				f"iter={iteration:03d} "
				f"return={metrics['mean_worker_return']:.3f} "
				f"pi_loss={metrics['policy_loss']:.4f} "
				f"v_loss={metrics['value_loss']:.4f} "
				f"entropy={metrics['entropy']:.4f}"
			)

		trained_state = ray.get(learner.get_model_state.remote())
		eval_metrics = evaluate_policy_winrate(
			model_state=trained_state,
			state_dim=state_dim,
			hidden_dim=args.hidden_dim,
			action_dim=action_dim,
			server_url=args.server_url,
			game_name=args.game_name,
			device=DEVICE,
			num_games=max(1, args.eval_games),
		)
		print(
			f"Evaluation over {int(args.eval_games)} games: "
			f"wins={int(eval_metrics['wins'])}, "
			f"losses={int(eval_metrics['losses'])}, "
			f"draws={int(eval_metrics['draws'])}, "
			f"win_rate={eval_metrics['win_rate']:.2%}"
		)
	finally:
		ray.shutdown()


if __name__ == "__main__":
	main()
