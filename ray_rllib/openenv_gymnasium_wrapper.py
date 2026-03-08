"""
OpenEnv Gymnasium Wrapper

Bridges the OpenEnv WebSocket-based environment (Blackjack, etc.) to the
standard Gymnasium interface that Ray RLlib expects.

This wrapper:
  1. Connects to a running OpenEnv server over WebSocket.
  2. Translates Gymnasium `reset()` / `step()` calls into OpenEnv messages.
  3. Exposes observation_space and action_space so RLlib can build policies.

Usage:
    >>> from openenv_gymnasium_wrapper import OpenEnvGymnasiumWrapper
    >>> env = OpenEnvGymnasiumWrapper(server_url="http://localhost:8000")
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(0)  # 0=HIT, 1=STAND
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop, or create a new one if none exists."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class OpenEnvGymnasiumWrapper(gym.Env):
    """
    Gymnasium wrapper around an OpenEnv WebSocket server.

    Parameters
    ----------
    server_url : str
        Base URL of the OpenEnv server (e.g. ``http://localhost:8000``).
    game_name : str
        OpenSpiel game name (default ``"blackjack"``).
    info_state_size : int
        Length of the ``info_state`` vector returned by the server.
        For Blackjack this is typically 111.
    max_actions : int
        Number of discrete actions (for Blackjack: 2 – HIT / STAND).
    max_steps_per_episode : int
        Safety limit to prevent infinite episodes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        game_name: str = "blackjack",
        info_state_size: int = 111,
        max_actions: int = 2,
        max_steps_per_episode: int = 10,
    ):
        super().__init__()

        self.server_url = server_url
        self.game_name = game_name
        self.info_state_size = info_state_size
        self.max_steps_per_episode = max_steps_per_episode

        # --- Gymnasium spaces ------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(info_state_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max_actions)

        # --- Internal state ---------------------------------------------------
        self._env: Any = None  # OpenSpielEnv client for current episode
        self._last_legal_actions: list[int] = []
        self._step_count = 0

        # Dedicated event loop running in a background thread.
        # All async calls go through this loop so the WebSocket futures
        # are never orphaned by asyncio.run() creating throwaway loops.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._loop_thread.start()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_env(self):
        """Create a fresh OpenEnv client (one WebSocket session per episode)."""
        from envs.openspiel_env import OpenSpielEnv
        return OpenSpielEnv(base_url=self.server_url)

    def _close_env(self) -> None:
        """Close the current WebSocket session, if any."""
        if self._env is not None:
            try:
                self._run_async(self._env.close())
            except Exception:
                pass
            self._env = None

    def _run_async(self, coro):
        """Schedule *coro* on the persistent background event loop.

        This avoids the problem of ``asyncio.run()`` creating a new loop
        each time, which would orphan the WebSocket connection's futures.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def _obs_to_array(self, observation) -> np.ndarray:
        """Convert an OpenEnv observation to a fixed-size numpy array."""
        info_state = list(observation.info_state)

        # Pad or truncate to the declared size
        if len(info_state) < self.info_state_size:
            info_state += [0.0] * (self.info_state_size - len(info_state))
        elif len(info_state) > self.info_state_size:
            info_state = info_state[: self.info_state_size]

        return np.array(info_state, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Each reset creates a fresh WebSocket connection because each
        OpenEnv session corresponds to exactly one episode.
        """
        super().reset(seed=seed)
        self._step_count = 0

        # Close stale connection and open a fresh one
        self._close_env()
        self._env = self._create_env()

        result = self._run_async(self._env.reset())

        obs = self._obs_to_array(result.observation)
        self._last_legal_actions = list(result.observation.legal_actions)

        info: Dict[str, Any] = {
            "legal_actions": self._last_legal_actions,
            "game_phase": result.observation.game_phase,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute *action* and return the Gymnasium 5-tuple."""
        self._step_count += 1

        from envs.openspiel_env import OpenSpielAction

        # Clamp to legal actions
        if action not in self._last_legal_actions and self._last_legal_actions:
            action = self._last_legal_actions[0]

        result = self._run_async(
            self._env.step(
                OpenSpielAction(action_id=int(action), game_name=self.game_name)
            )
        )

        obs = self._obs_to_array(result.observation)
        reward = float(result.reward) if result.reward is not None else 0.0
        terminated = bool(result.done)
        truncated = self._step_count >= self.max_steps_per_episode and not terminated

        self._last_legal_actions = list(result.observation.legal_actions)

        info: Dict[str, Any] = {
            "legal_actions": self._last_legal_actions,
            "game_phase": result.observation.game_phase,
            "raw_reward": result.reward,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Tear down the WebSocket connection and stop the event loop."""
        self._close_env()
        self._loop.call_soon_threadsafe(self._loop.stop)

