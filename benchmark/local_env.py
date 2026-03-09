"""
Local (in-process) OpenSpiel environment with the same interface as the
WebSocket-based OpenSpielEnv client, so rollout workers can switch between
modes without changing their core logic.

This eliminates all network I/O, making rollout collection CPU-compute-bound
(game logic + model inference) rather than I/O-bound (WebSocket round trips).
That's necessary to actually test Ray and Monarch's scheduling architectures.

Usage in a worker:
    from benchmark.local_env import LocalEnv
    env = LocalEnv("blackjack")
    result = env.reset()
    obs, legal = result.observation.info_state, result.observation.legal_actions
    result = env.step(0)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pyspiel
except ImportError as e:
    raise ImportError(
        "open-spiel is required for local env mode. "
        "It should already be installed via uv sync."
    ) from e


@dataclass
class _Obs:
    info_state: List[float]
    legal_actions: List[int]


@dataclass
class _Result:
    observation: _Obs
    reward: Optional[float]
    done: bool


class LocalEnv:
    """
    Thin wrapper around a pyspiel game that presents the same reset/step
    interface as the OpenEnv WebSocket client.

    Handles chance nodes (card dealing) transparently: after any player
    action or reset, all consecutive chance nodes are resolved automatically
    before returning control to the caller.
    """

    def __init__(self, game_name: str = "blackjack"):
        self._game = pyspiel.load_game(game_name)
        self._state = None
        self._info_state_size: Optional[int] = None
        # Probe once at construction time to avoid repeated C++ stderr noise per step.
        self._use_information_state = self._probe_information_state()

    def _probe_information_state(self) -> bool:
        """Return True if this game supports information_state_tensor."""
        state = self._game.new_initial_state()
        # Advance past any chance nodes to reach a player decision point.
        while not state.is_terminal() and state.is_chance_node():
            action = state.chance_outcomes()[0][0]
            state.apply_action(action)
        if state.is_terminal():
            return False
        try:
            state.information_state_tensor(0)
            return True
        except pyspiel.SpielError:
            return False

    # ------------------------------------------------------------------
    # Public API (mirrors OpenEnv client)
    # ------------------------------------------------------------------

    def reset(self) -> _Result:
        self._state = self._game.new_initial_state()
        self._resolve_chance()
        return self._make_result(reward=None, done=False)

    def step(self, action_id: int) -> _Result:
        assert self._state is not None, "Call reset() before step()"
        self._state.apply_action(action_id)
        self._resolve_chance()
        done = self._state.is_terminal()
        reward = float(self._state.returns()[0]) if done else None
        return self._make_result(reward=reward, done=done)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_chance(self) -> None:
        """Apply chance node actions (card deals) until it's the player's turn."""
        while not self._state.is_terminal() and self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            # Sample according to the chance distribution
            actions, probs = zip(*outcomes)
            action = random.choices(actions, weights=probs, k=1)[0]
            self._state.apply_action(action)

    def _info_state(self) -> List[float]:
        if self._state.is_terminal():
            if self._info_state_size is None:
                return [0.0]
            return [0.0] * self._info_state_size

        player = 0

        if self._use_information_state:
            tensor = self._state.information_state_tensor(player)
        else:
            tensor = self._state.observation_tensor(player)

        if self._info_state_size is None:
            self._info_state_size = len(tensor)

        return list(tensor)

    def _legal_actions(self) -> List[int]:
        if self._state.is_terminal():
            return [0]  # dummy; won't be sampled
        return list(self._state.legal_actions(0))

    def _make_result(self, reward: Optional[float], done: bool) -> _Result:
        obs = _Obs(
            info_state=self._info_state(),
            legal_actions=self._legal_actions(),
        )
        return _Result(observation=obs, reward=reward, done=done)
