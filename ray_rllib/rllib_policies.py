"""
Modular RL Policy Definitions for Ray RLlib + OpenEnv

This module provides a *policy registry* pattern so that new algorithms
(PPO, DQN, DPO, …) can be added without touching the training loop.

Architecture
------------

    BaseRLPolicy          ← abstract interface
        ├── PPOPolicy     ← ships with this file
        ├── DQNPolicy     ← ships with this file
        └── (DPOPolicy)   ← add your own!

    POLICY_REGISTRY       ← ``str → BaseRLPolicy`` mapping

How to add a new policy
-----------------------
1.  Subclass ``BaseRLPolicy``.
2.  Implement ``algorithm_name``, ``build_config()``, ``customize_config()``.
3.  Register with ``@register_policy("my_algo")``.

Alternatively, call ``register_policy_class("my_algo", MyPolicy)`` at import
time.

Usage in training script::

    from rllib_policies import get_policy
    policy = get_policy("PPO")          # Returns a PPOPolicy instance
    algo = policy.build(env_name, env_config, train_cfg)
    for i in range(100):
        result = algo.train()
"""

from __future__ import annotations

import abc
import copy
import logging
from typing import Any, Dict, Optional, Type

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm

logger = logging.getLogger(__name__)


# ============================================================================
# POLICY REGISTRY
# ============================================================================

_POLICY_REGISTRY: Dict[str, Type["BaseRLPolicy"]] = {}


def register_policy(name: str):
    """Decorator that registers a policy class under *name* (case-insensitive)."""

    def _decorator(cls: Type[BaseRLPolicy]) -> Type[BaseRLPolicy]:
        _POLICY_REGISTRY[name.upper()] = cls
        return cls

    return _decorator


def register_policy_class(name: str, cls: Type["BaseRLPolicy"]) -> None:
    """Programmatic registration (useful when decorators are awkward)."""
    _POLICY_REGISTRY[name.upper()] = cls


def get_policy(name: str, **kwargs) -> "BaseRLPolicy":
    """Instantiate and return the policy registered under *name*."""
    key = name.upper()
    if key not in _POLICY_REGISTRY:
        available = ", ".join(sorted(_POLICY_REGISTRY.keys()))
        raise KeyError(
            f"Unknown policy '{name}'. Available policies: {available}"
        )
    return _POLICY_REGISTRY[key](**kwargs)


def list_policies() -> list[str]:
    """Return a sorted list of registered policy names."""
    return sorted(_POLICY_REGISTRY.keys())


# ============================================================================
# BASE POLICY
# ============================================================================


class BaseRLPolicy(abc.ABC):
    """Abstract base class for every RL algorithm adapter.

    Subclasses must implement three methods:
      * ``algorithm_name``  – returns the RLlib algorithm string.
      * ``build_config``    – returns a fresh ``AlgorithmConfig``.
      * ``customize_config``– applies user overrides to the config.
    """

    @property
    @abc.abstractmethod
    def algorithm_name(self) -> str:
        """Return the human-readable algorithm name (e.g. ``"PPO"``)."""
        ...

    @abc.abstractmethod
    def build_config(
        self,
        env_name: str,
        env_config: Dict[str, Any],
        train_cfg: Dict[str, Any],
    ) -> AlgorithmConfig:
        """Construct a complete RLlib ``AlgorithmConfig``.

        Parameters
        ----------
        env_name : str
            Gymnasium environment ID (or class name registered with Ray).
        env_config : dict
            Keyword arguments forwarded to the env constructor.
        train_cfg : dict
            User-supplied training hyper-parameters from the YAML file.
        """
        ...

    def customize_config(
        self, config: AlgorithmConfig, overrides: Dict[str, Any]
    ) -> AlgorithmConfig:
        """Apply free-form overrides to a built config.

        The default implementation does nothing.  Override in subclasses
        when the algorithm has knobs that don't map directly to
        ``build_config`` arguments.
        """
        return config

    # ------------------------------------------------------------------ #
    # Convenience entry-point
    # ------------------------------------------------------------------ #

    def build(
        self,
        env_name: str,
        env_config: Dict[str, Any],
        train_cfg: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Algorithm:
        """Build → customise → return a ready-to-train ``Algorithm``."""
        config = self.build_config(env_name, env_config, train_cfg)
        if overrides:
            config = self.customize_config(config, overrides)
        return config.build()


# ============================================================================
# PPO POLICY (default)
# ============================================================================


@register_policy("PPO")
class PPOPolicy(BaseRLPolicy):
    """Proximal Policy Optimization via Ray RLlib."""

    @property
    def algorithm_name(self) -> str:
        return "PPO"

    def build_config(
        self,
        env_name: str,
        env_config: Dict[str, Any],
        train_cfg: Dict[str, Any],
    ) -> AlgorithmConfig:
        from ray.rllib.algorithms.ppo import PPOConfig

        lr = float(train_cfg.get("lr", 5e-5))
        gamma = train_cfg.get("gamma", 0.99)
        num_workers = train_cfg.get("num_rollout_workers", 0)
        train_batch_size = train_cfg.get("train_batch_size", 512)
        sgd_minibatch_size = train_cfg.get("sgd_minibatch_size", 64)
        num_sgd_iter = train_cfg.get("num_sgd_iter", 10)
        clip_param = train_cfg.get("clip_param", 0.2)
        entropy_coeff = train_cfg.get("entropy_coeff", 0.01)
        vf_clip_param = train_cfg.get("vf_clip_param", 10.0)
        model_cfg = train_cfg.get("model", {})
        framework = train_cfg.get("framework", "torch")

        config = (
            PPOConfig()
            .environment(env=env_name, env_config=env_config)
            .framework(framework)
            .env_runners(num_env_runners=num_workers)
            .training(
                lr=lr,
                gamma=gamma,
                train_batch_size_per_learner=train_batch_size,
                minibatch_size=sgd_minibatch_size,
                num_epochs=num_sgd_iter,
                clip_param=clip_param,
                entropy_coeff=entropy_coeff,
                vf_clip_param=vf_clip_param,
            )
        )

        # Model architecture overrides
        if model_cfg:
            config = config.rl_module(
                model_config={
                    "fcnet_hiddens": model_cfg.get("fcnet_hiddens", [64, 64]),
                    "fcnet_activation": model_cfg.get("fcnet_activation", "relu"),
                }
            )

        return config


# ============================================================================
# DQN POLICY
# ============================================================================


@register_policy("DQN")
class DQNPolicy(BaseRLPolicy):
    """Deep Q-Network via Ray RLlib."""

    @property
    def algorithm_name(self) -> str:
        return "DQN"

    def build_config(
        self,
        env_name: str,
        env_config: Dict[str, Any],
        train_cfg: Dict[str, Any],
    ) -> AlgorithmConfig:
        from ray.rllib.algorithms.dqn import DQNConfig

        lr = float(train_cfg.get("lr", 5e-4))
        gamma = train_cfg.get("gamma", 0.99)
        num_workers = train_cfg.get("num_rollout_workers", 0)
        train_batch_size = train_cfg.get("train_batch_size", 256)
        epsilon_start = train_cfg.get("epsilon_start", 1.0)
        epsilon_end = train_cfg.get("epsilon_end", 0.02)
        epsilon_timesteps = train_cfg.get("epsilon_timesteps", 10000)
        buffer_size = train_cfg.get("replay_buffer_size", 50000)
        framework = train_cfg.get("framework", "torch")

        config = (
            DQNConfig()
            .environment(env=env_name, env_config=env_config)
            .framework(framework)
            .env_runners(num_env_runners=num_workers)
            .training(
                lr=lr,
                gamma=gamma,
                train_batch_size_per_learner=train_batch_size,
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": buffer_size,
                },
            )
        )

        return config


# ============================================================================
# Stub: DPO Policy (placeholder for future implementation)
# ============================================================================


# Uncomment and implement when DPO is ready:
#
# @register_policy("DPO")
# class DPOPolicy(BaseRLPolicy):
#     """Direct Preference Optimization – not yet implemented."""
#
#     @property
#     def algorithm_name(self) -> str:
#         return "DPO"
#
#     def build_config(self, env_name, env_config, train_cfg):
#         raise NotImplementedError("DPO is not yet supported by Ray RLlib out-of-the-box.")
