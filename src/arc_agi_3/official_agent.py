"""Adapter for running local solver policies inside ARC-AGI-3 Agents."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Any

from arcengine import GameAction, GameState

from .baseline import ActionDecision, EffectPriorPolicy, state_hash
from .effects import infer_primary_object_motion
from .features import frame_diff_summary


@dataclass
class RuntimeBudget:
    max_actions: int = 80
    max_seconds: float = 60.0
    started_at: float = 0.0

    def start(self) -> None:
        if self.started_at <= 0:
            self.started_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self.started_at <= 0:
            return 0.0
        return time.monotonic() - self.started_at

    def exceeded(self, action_counter: int) -> bool:
        return action_counter >= self.max_actions or self.elapsed >= self.max_seconds


class ObjectModelAgentCore:
    """Stateful bridge from official Agent calls to the current policy code."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_actions: int = 80,
        max_seconds: float = 60.0,
        allow_reset: bool = False,
    ) -> None:
        self.budget = RuntimeBudget(max_actions=max_actions, max_seconds=max_seconds)
        self.policy = EffectPriorPolicy(
            seed=seed,
            allow_reset=allow_reset,
            coordinate_samples=8,
            min_effect_samples=2,
        )
        self.observed_initial = False
        self.pending_before_frame: Any | None = None
        self.pending_before_hash: str | None = None
        self.pending_decision: ActionDecision | None = None

    def observe_initial_once(self, latest_frame: Any | None) -> None:
        if self.observed_initial:
            return
        self.policy.observe_initial(latest_frame)
        self.observed_initial = True

    def is_done(self, latest_frame: Any | None, action_counter: int) -> bool:
        self.budget.start()
        state = getattr(latest_frame, "state", None)
        if state is GameState.WIN:
            return True
        return self.budget.exceeded(action_counter)

    def observe_pending_result(self, latest_frame: Any | None) -> None:
        if self.pending_decision is None:
            return
        diff_summary = frame_diff_summary(self.pending_before_frame, latest_frame)
        motion = infer_primary_object_motion(self.pending_before_frame, latest_frame)
        self.policy.observe_result(
            self.pending_before_hash,
            self.pending_decision,
            state_hash(latest_frame),
            motion,
            diff_summary,
        )
        self.policy.observe_frame(latest_frame)
        self.pending_before_frame = None
        self.pending_before_hash = None
        self.pending_decision = None

    def remember_pending_action(
        self,
        latest_frame: Any | None,
        decision: ActionDecision,
    ) -> None:
        self.pending_before_frame = latest_frame
        self.pending_before_hash = state_hash(latest_frame)
        self.pending_decision = decision

    def choose_decision(
        self,
        latest_frame: Any | None,
        action_counter: int,
    ) -> ActionDecision:
        self.budget.start()
        self.observe_initial_once(latest_frame)
        self.observe_pending_result(latest_frame)

        state = getattr(latest_frame, "state", None)
        if state in {GameState.NOT_PLAYED, GameState.GAME_OVER}:
            decision = ActionDecision(
                action=GameAction.RESET,
                action_data={},
                reason=f"reset_for_state:{getattr(state, 'name', state)}",
            )
            self.remember_pending_action(latest_frame, decision)
            return decision

        decision = self.policy.select_action(latest_frame, action_counter)
        if decision is None:
            decision = ActionDecision(
                action=GameAction.RESET,
                action_data={},
                reason="fallback_no_available_action",
            )
            self.remember_pending_action(latest_frame, decision)
            return decision
        self.remember_pending_action(latest_frame, decision)
        return decision


def _attach_action_data(
    action: GameAction,
    action_data: dict[str, int],
    *,
    game_id: str,
) -> GameAction:
    payload: dict[str, Any] = {"game_id": game_id}
    payload.update(action_data)
    action.set_data(payload)
    return action


def build_object_model_agent_class(agent_base: type[Any] | None = None) -> type[Any]:
    """Build an official ARC-AGI-3 Agent subclass when the package is present."""
    if agent_base is None:
        from agents.agent import Agent as agent_base

    class ObjectModelAgent(agent_base):  # type: ignore[valid-type, misc]
        MAX_ACTIONS = 80

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seed = abs(hash(getattr(self, "game_id", ""))) % 1_000_000
            self._object_model_core = ObjectModelAgentCore(
                seed=seed,
                max_actions=self.MAX_ACTIONS,
                max_seconds=60.0,
                allow_reset=False,
            )

        def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
            return self._object_model_core.is_done(
                latest_frame,
                getattr(self, "action_counter", 0),
            )

        def choose_action(self, frames: list[Any], latest_frame: Any) -> GameAction:
            decision = self._object_model_core.choose_decision(
                latest_frame,
                getattr(self, "action_counter", 0),
            )
            action = decision.action
            if action.is_complex() or decision.action_data:
                _attach_action_data(
                    action,
                    decision.action_data,
                    game_id=getattr(self, "game_id", ""),
                )
            else:
                _attach_action_data(action, {}, game_id=getattr(self, "game_id", ""))
            action.reasoning = {
                "agent": "object_model",
                "reason": decision.reason,
                "action_counter": getattr(self, "action_counter", 0),
            }
            return action

    ObjectModelAgent.__name__ = "ObjectModelAgent"
    ObjectModelAgent.__qualname__ = "ObjectModelAgent"
    return ObjectModelAgent


def register_with_official_agents(agent_name: str = "objectmodel") -> type[Any]:
    """Register ObjectModelAgent in ARC-AGI-3-Agents when imported in Kaggle."""
    from agents import AVAILABLE_AGENTS

    agent_class = build_object_model_agent_class()
    AVAILABLE_AGENTS[agent_name] = agent_class
    return agent_class


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_minimal_official_agents_package(
    agents_repo: str | Path,
) -> tuple[type[Any], type[Any], dict[str, type[Any]]]:
    """Load ARC-AGI-3-Agents core modules without importing LLM templates."""
    repo_path = Path(agents_repo)
    package_path = repo_path / "agents"
    if not package_path.is_dir():
        raise FileNotFoundError(f"Missing agents package at {package_path}")

    package = ModuleType("agents")
    package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    package.AVAILABLE_AGENTS = {}
    sys.modules["agents"] = package

    recorder_module = _load_module("agents.recorder", package_path / "recorder.py")
    tracing_module = _load_module("agents.tracing", package_path / "tracing.py")
    agent_module = _load_module("agents.agent", package_path / "agent.py")
    swarm_module = _load_module("agents.swarm", package_path / "swarm.py")

    package.Recorder = recorder_module.Recorder
    package.Agent = agent_module.Agent
    package.Playback = agent_module.Playback
    package.Swarm = swarm_module.Swarm
    package.AVAILABLE_AGENTS = {}
    package.tracing = tracing_module
    return agent_module.Agent, swarm_module.Swarm, package.AVAILABLE_AGENTS


def register_with_minimal_official_agents(
    agents_repo: str | Path,
    agent_name: str = "objectmodel",
) -> tuple[type[Any], type[Any]]:
    """Register ObjectModelAgent after minimal official package loading."""
    agent_base, swarm_class, available_agents = load_minimal_official_agents_package(
        agents_repo
    )
    agent_class = build_object_model_agent_class(agent_base)
    available_agents[agent_name] = agent_class
    return swarm_class, agent_class
