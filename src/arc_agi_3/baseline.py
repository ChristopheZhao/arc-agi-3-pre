"""Baseline agent helpers for ARC-AGI-3 environments."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from arcengine import FrameDataRaw, GameAction, GameState

from .effects import ObjectMotion
from .features import (
    FrameDiffSummary,
    extract_color_objects,
    object_coordinate_candidates,
)


@dataclass(frozen=True)
class FrameSummary:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    frame_layers: int
    height: int
    width: int


@dataclass(frozen=True)
class RunSummary:
    agent: str
    game_id: str
    seed: int
    steps_attempted: int
    steps_completed: int
    state: str
    levels_completed: int
    win_levels: int
    stopped_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    agent: str
    game_id: str
    seed: int
    step: int
    action: str
    action_id: int
    action_data: dict[str, int]
    before_hash: str | None
    after_hash: str | None
    state: str
    levels_completed: int
    win_levels: int
    changed_cells: int = 0
    changed_bbox: tuple[int, int, int, int] | None = None
    motion_dx: int = 0
    motion_dy: int = 0
    motion_direction: str = ""
    motion_color: int | None = None
    decision_reason: str = ""
    level_delta: int = 0
    state_changed: bool = False
    action_utility: float = 0.0
    object_event_count: int = 0
    object_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionDecision:
    action: GameAction
    action_data: dict[str, int]
    reason: str

    @property
    def action_key(self) -> str:
        if self.action_data:
            payload = ",".join(
                f"{key}={self.action_data[key]}" for key in sorted(self.action_data)
            )
            return f"{self.action.name}:{payload}"
        return self.action.name


def summarize_frame(frame_data: FrameDataRaw | None) -> FrameSummary | None:
    """Return a compact, JSON-serializable description of an observation."""
    if frame_data is None:
        return None

    height = 0
    width = 0
    if frame_data.frame:
        first_layer = frame_data.frame[0]
        shape = getattr(first_layer, "shape", None)
        if shape is not None and len(shape) >= 2:
            height = int(shape[0])
            width = int(shape[1])
        else:
            height = len(first_layer)
            width = len(first_layer[0]) if height else 0

    state = (
        frame_data.state.name
        if hasattr(frame_data.state, "name")
        else str(frame_data.state)
    )
    return FrameSummary(
        game_id=frame_data.game_id,
        state=state,
        levels_completed=frame_data.levels_completed,
        win_levels=frame_data.win_levels,
        available_actions=list(frame_data.available_actions),
        frame_layers=len(frame_data.frame),
        height=height,
        width=width,
    )


def state_hash(frame_data: FrameDataRaw | None) -> str | None:
    """Hash frame content and coarse progress state for replay comparison."""
    if frame_data is None:
        return None

    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(frame_data.game_id).encode("utf-8"))
    digest.update(str(frame_data.state).encode("utf-8"))
    digest.update(str(frame_data.levels_completed).encode("ascii"))
    digest.update(str(frame_data.win_levels).encode("ascii"))
    for layer in frame_data.frame:
        shape = getattr(layer, "shape", None)
        dtype = getattr(layer, "dtype", None)
        digest.update(str(shape).encode("ascii"))
        digest.update(str(dtype).encode("ascii"))
        if hasattr(layer, "tobytes"):
            digest.update(layer.tobytes())
        else:
            digest.update(repr(layer).encode("utf-8"))
    return digest.hexdigest()


def available_game_actions(
    action_ids: Sequence[int],
    *,
    allow_reset: bool = False,
) -> list[GameAction]:
    """Convert integer action IDs to GameAction values, optionally masking reset."""
    actions: list[GameAction] = []
    for action_id in action_ids:
        action = GameAction.from_id(int(action_id))
        if action is GameAction.RESET and not allow_reset:
            continue
        actions.append(action)
    return actions


def frame_dimensions(frame_data: FrameDataRaw | None) -> tuple[int, int]:
    summary = summarize_frame(frame_data)
    if summary is None or summary.width <= 0 or summary.height <= 0:
        return 64, 64
    return summary.width, summary.height


def random_action_data(
    action: GameAction,
    frame_data: FrameDataRaw | None,
    rng: random.Random,
) -> dict[str, int]:
    """Build action payload for complex ARC-AGI-3 actions."""
    if not action.is_complex():
        return {}

    width, height = frame_dimensions(frame_data)
    return {
        "x": rng.randrange(max(width, 1)),
        "y": rng.randrange(max(height, 1)),
    }


def coordinate_candidates(
    frame_data: FrameDataRaw | None,
    rng: random.Random,
    *,
    samples: int = 8,
) -> list[dict[str, int]]:
    """Return deterministic-prior plus random coordinate payloads for ACTION6."""
    width, height = frame_dimensions(frame_data)
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)

    points: list[tuple[int, int]] = [
        (width // 2, height // 2),
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (width // 2, 0),
        (width // 2, max_y),
        (0, height // 2),
        (max_x, height // 2),
    ]
    for candidate in object_coordinate_candidates(frame_data):
        points.append((candidate["x"], candidate["y"]))
    for _ in range(max(samples, 0)):
        points.append((rng.randrange(max(width, 1)), rng.randrange(max(height, 1))))

    deduped: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        point = (min(max(x, 0), max_x), min(max(y, 0), max_y))
        if point in seen:
            continue
        seen.add(point)
        deduped.append({"x": point[0], "y": point[1]})
    return deduped


def is_terminal_state(frame_data: FrameDataRaw | None) -> bool:
    if frame_data is None:
        return True
    return frame_data.state in {GameState.WIN, GameState.GAME_OVER}


def primary_object_position(frame_data: FrameDataRaw | None) -> tuple[int, int] | None:
    objects = extract_color_objects(frame_data, max_objects=1)
    if not objects:
        return None
    obj = objects[0]
    return obj.centroid_x, obj.centroid_y


def direction_to_target(
    origin: tuple[int, int],
    target: tuple[int, int],
) -> str:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if dx == 0 and dy == 0:
        return "static"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def changed_bbox_center(
    bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int] | None:
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max + 1) // 2, (y_min + y_max + 1) // 2


def opposite_direction(direction: str) -> str:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(direction, "")


class RandomPolicy:
    name = "random"

    def __init__(self, *, seed: int, allow_reset: bool) -> None:
        self.rng = random.Random(seed)
        self.allow_reset = allow_reset

    def select_action(
        self,
        frame_data: FrameDataRaw | None,
        step_idx: int,
    ) -> ActionDecision | None:
        actions = available_game_actions(
            frame_data.available_actions if frame_data is not None else [],
            allow_reset=self.allow_reset,
        )
        if not actions:
            return None
        action = self.rng.choice(actions)
        return ActionDecision(
            action=action,
            action_data=random_action_data(action, frame_data, self.rng),
            reason="uniform_available_action",
        )

    def observe_result(
        self,
        before_hash: str | None,
        decision: ActionDecision,
        after_hash: str | None,
        motion: ObjectMotion | None = None,
        diff_summary: FrameDiffSummary | None = None,
    ) -> None:
        return None


class NoveltyPolicy:
    name = "novelty"

    def __init__(
        self,
        *,
        seed: int,
        allow_reset: bool,
        coordinate_samples: int = 8,
    ) -> None:
        self.rng = random.Random(seed)
        self.allow_reset = allow_reset
        self.coordinate_samples = coordinate_samples
        self.state_visits: Counter[str] = Counter()
        self.transition_counts: Counter[tuple[str, str]] = Counter()

    def observe_initial(self, frame_data: FrameDataRaw | None) -> None:
        current_hash = state_hash(frame_data)
        if current_hash is not None:
            self.state_visits[current_hash] += 1

    def select_action(
        self,
        frame_data: FrameDataRaw | None,
        step_idx: int,
    ) -> ActionDecision | None:
        current_hash = state_hash(frame_data)
        actions = available_game_actions(
            frame_data.available_actions if frame_data is not None else [],
            allow_reset=self.allow_reset,
        )
        if not actions:
            return None

        decisions: list[ActionDecision] = []
        for action in actions:
            if action.is_complex():
                for payload in coordinate_candidates(
                    frame_data,
                    self.rng,
                    samples=self.coordinate_samples,
                ):
                    decisions.append(
                        ActionDecision(
                            action=action,
                            action_data=payload,
                            reason="least_tried_state_action",
                        )
                    )
            else:
                decisions.append(
                    ActionDecision(
                        action=action,
                        action_data={},
                        reason="least_tried_state_action",
                    )
                )

        if current_hash is None:
            return self.rng.choice(decisions)

        def score(decision: ActionDecision) -> tuple[int, float]:
            return (
                self.transition_counts[(current_hash, decision.action_key)],
                self.rng.random(),
            )

        return min(decisions, key=score)

    def observe_result(
        self,
        before_hash: str | None,
        decision: ActionDecision,
        after_hash: str | None,
        motion: ObjectMotion | None = None,
        diff_summary: FrameDiffSummary | None = None,
    ) -> None:
        if before_hash is not None:
            self.transition_counts[(before_hash, decision.action_key)] += 1
        if after_hash is not None:
            self.state_visits[after_hash] += 1


class EffectPriorPolicy(NoveltyPolicy):
    name = "effect-prior"

    def __init__(
        self,
        *,
        seed: int,
        allow_reset: bool,
        coordinate_samples: int = 8,
        min_effect_samples: int = 2,
    ) -> None:
        super().__init__(
            seed=seed,
            allow_reset=allow_reset,
            coordinate_samples=coordinate_samples,
        )
        self.min_effect_samples = min_effect_samples
        self.action_direction_counts: Counter[tuple[str, str]] = Counter()
        self.action_trials: Counter[str] = Counter()
        self.action_no_motion_streak: Counter[str] = Counter()
        self.visited_positions: Counter[tuple[int, int]] = Counter()
        self.frontier_positions: Counter[tuple[int, int]] = Counter()
        self.min_frontier_changed_cells = 4
        self.last_desired_direction = ""
        self.last_target_position: tuple[int, int] | None = None

    def observe_initial(self, frame_data: FrameDataRaw | None) -> None:
        super().observe_initial(frame_data)
        self.observe_frame(frame_data)

    def observe_frame(self, frame_data: FrameDataRaw | None) -> None:
        position = primary_object_position(frame_data)
        if position is not None:
            self.visited_positions[position] += 1

    def learned_direction(self, action: GameAction) -> str:
        counts = {
            direction: count
            for (action_name, direction), count in self.action_direction_counts.items()
            if action_name == action.name
        }
        if not counts:
            return ""
        direction, count = max(counts.items(), key=lambda item: item[1])
        return direction if count >= self.min_effect_samples else ""

    def nearby_visit_count(self, target: tuple[int, int], *, radius: int = 1) -> int:
        return sum(
            count
            for position, count in self.visited_positions.items()
            if abs(position[0] - target[0]) + abs(position[1] - target[1]) <= radius
        )

    def frontier_targets(
        self,
        *,
        width: int,
        height: int,
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        targets: list[tuple[int, int]] = []
        for target in self.frontier_positions:
            x, y = target
            if target == current or x < 0 or x >= width or y < 0 or y >= height:
                continue
            targets.append(target)
        return targets

    def desired_direction(self, frame_data: FrameDataRaw | None) -> str:
        current = primary_object_position(frame_data)
        width, height = frame_dimensions(frame_data)
        if current is None or width <= 0 or height <= 0:
            self.last_target_position = None
            return ""

        frontier_targets = self.frontier_targets(
            width=width,
            height=height,
            current=current,
        )
        if frontier_targets:
            def frontier_score(
                target: tuple[int, int],
            ) -> tuple[int, int, int, int, float]:
                nearest_seen = min(
                    abs(target[0] - seen[0]) + abs(target[1] - seen[1])
                    for seen in self.visited_positions
                )
                target_distance = abs(target[0] - current[0]) + abs(
                    target[1] - current[1]
                )
                return (
                    -self.nearby_visit_count(target),
                    self.frontier_positions[target],
                    nearest_seen,
                    target_distance,
                    self.rng.random(),
                )

            target = max(frontier_targets, key=frontier_score)
            self.last_target_position = target
            return direction_to_target(current, target)

        targets = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, 0),
            (width // 2, height - 1),
            (0, height // 2),
            (width - 1, height // 2),
        ]
        if not self.visited_positions:
            target = targets[0]
            self.last_target_position = target
            return direction_to_target(current, target)

        def score(target: tuple[int, int]) -> tuple[int, int, float]:
            nearest_seen = min(
                abs(target[0] - seen[0]) + abs(target[1] - seen[1])
                for seen in self.visited_positions
            )
            target_distance = abs(target[0] - current[0]) + abs(target[1] - current[1])
            return (nearest_seen, target_distance, self.rng.random())

        target = max(targets, key=score)
        self.last_target_position = target
        return direction_to_target(current, target)

    def direction_penalty(self, decision: ActionDecision, desired: str) -> int:
        if not desired or desired == "static":
            return 1
        learned = self.learned_direction(decision.action)
        if not learned:
            return 1
        if learned == desired:
            return 0
        if learned == opposite_direction(desired):
            return 3
        return 2

    def stall_penalty(self, decision: ActionDecision) -> int:
        return min(self.action_no_motion_streak[decision.action.name], 3)

    def select_action(
        self,
        frame_data: FrameDataRaw | None,
        step_idx: int,
    ) -> ActionDecision | None:
        current_hash = state_hash(frame_data)
        actions = available_game_actions(
            frame_data.available_actions if frame_data is not None else [],
            allow_reset=self.allow_reset,
        )
        if not actions:
            return None

        decisions: list[ActionDecision] = []
        for action in actions:
            if action.is_complex():
                for payload in coordinate_candidates(
                    frame_data,
                    self.rng,
                    samples=self.coordinate_samples,
                ):
                    decisions.append(
                        ActionDecision(
                            action=action,
                            action_data=payload,
                            reason="effect_prior_novelty",
                        )
                    )
            else:
                decisions.append(
                    ActionDecision(
                        action=action,
                        action_data={},
                        reason="effect_prior_novelty",
                    )
                )

        if current_hash is None:
            return self.rng.choice(decisions)

        desired = self.desired_direction(frame_data)
        self.last_desired_direction = desired
        trial_floor = min(
            self.action_trials[decision.action.name] for decision in decisions
        )
        needs_bootstrap = trial_floor < self.min_effect_samples

        def score(decision: ActionDecision) -> tuple[int, int, int, int, float]:
            if needs_bootstrap:
                return (
                    self.action_trials[decision.action.name],
                    self.stall_penalty(decision),
                    self.transition_counts[(current_hash, decision.action_key)],
                    self.direction_penalty(decision, desired),
                    self.rng.random(),
                )
            return (
                self.stall_penalty(decision),
                self.direction_penalty(decision, desired),
                self.transition_counts[(current_hash, decision.action_key)],
                self.action_trials[decision.action.name],
                self.rng.random(),
            )

        return min(decisions, key=score)

    def observe_result(
        self,
        before_hash: str | None,
        decision: ActionDecision,
        after_hash: str | None,
        motion: ObjectMotion | None = None,
        diff_summary: FrameDiffSummary | None = None,
    ) -> None:
        super().observe_result(before_hash, decision, after_hash, motion, diff_summary)
        self.action_trials[decision.action.name] += 1
        if (
            diff_summary is not None
            and diff_summary.changed_cells >= self.min_frontier_changed_cells
        ):
            frontier = changed_bbox_center(diff_summary.changed_bbox)
            if frontier is not None:
                self.frontier_positions[frontier] += 1
        if motion is None or motion.direction in ("", "static"):
            self.action_no_motion_streak[decision.action.name] += 1
            return
        self.action_no_motion_streak[decision.action.name] = 0
        self.action_direction_counts[(decision.action.name, motion.direction)] += 1
