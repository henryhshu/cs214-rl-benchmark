import ray
from ray.tune.registry import register_env
from ray_rllib.rllib_policies import get_policy
from ray_rllib.openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper
import json

ray.init(num_cpus=2, ignore_reinit_error=True)

# Register env
ENV_CONFIG = {
    "server_url": "http://localhost:8000",
    "game_name": "blackjack",
    "info_state_size": 111,
    "max_actions": 2,
    "max_steps_per_episode": 10,
}
register_env("TestBlackjack-v0", lambda cfg: OpenEnvGymnasiumWrapper(**cfg))

policy = get_policy("PPO")
train_cfg = {
    "lr": 5e-5,
    "gamma": 0.99,
    "num_rollout_workers": 0,
    "train_batch_size": 128,
    "sgd_minibatch_size": 32,
}

algo = policy.build(env_name="TestBlackjack-v0", env_config=ENV_CONFIG, train_cfg=train_cfg)
result = algo.train()
print("KEYS IN RESULT:", list(result.keys()))
if "env_runners" in result:
    print("KEYS IN ENV_RUNNERS:", list(result["env_runners"].keys()))
    print("episodes_this_iter:", result["env_runners"].get("episodes_this_iter"))
    print("episode_reward_mean:", result["env_runners"].get("episode_reward_mean"))
