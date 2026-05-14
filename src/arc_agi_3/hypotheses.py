"""Action-level hypotheses learned from object event traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ActionHypothesis:
    action: str
    action_id: int
    samples: int
    avg_utility: float
    progress_count: int
    state_change_count: int
    event_count: int
    no_event_count: int
    event_type_counts: dict[str, int]
    move_direction_counts: dict[str, int]
    predicted_direction: str
    direction_confidence: float
    event_rate: float
    priority_score: float
    recommended_use: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recommended_use(
    *,
    samples: int,
    avg_utility: float,
    progress_count: int,
    event_rate: float,
    direction_confidence: float,
    min_samples: int,
) -> str:
    if progress_count > 0:
        return "exploit_progress"
    if samples < min_samples:
        return "probe_under_sampled"
    if avg_utility >= 1.5 and direction_confidence >= 0.5:
        return "planner_primitive"
    if avg_utility <= 0.25 and event_rate <= 0.2:
        return "deprioritize"
    return "probe_selectively"


def _priority_score(
    *,
    samples: int,
    avg_utility: float,
    progress_count: int,
    event_rate: float,
    direction_confidence: float,
) -> float:
    progress_bonus = 25.0 * progress_count / samples
    confidence_bonus = 0.5 * direction_confidence
    event_bonus = 0.25 * event_rate
    return avg_utility + progress_bonus + confidence_bonus + event_bonus


def infer_action_hypotheses(
    step_rows: Iterable[dict[str, Any]],
    *,
    min_samples: int = 2,
) -> list[ActionHypothesis]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        action = str(row.get("action", "UNKNOWN"))
        action_id = int(row.get("action_id", -1))
        groups.setdefault((action, action_id), []).append(row)

    hypotheses: list[ActionHypothesis] = []
    for (action, action_id), rows in sorted(groups.items(), key=lambda item: item[0]):
        samples = len(rows)
        utility_total = sum(float(row.get("action_utility") or 0.0) for row in rows)
        progress_count = sum(1 for row in rows if int(row.get("level_delta") or 0) > 0)
        state_change_count = sum(1 for row in rows if bool(row.get("state_changed")))
        no_event_count = sum(1 for row in rows if not row.get("object_events"))
        event_type_counts: Counter[str] = Counter()
        move_direction_counts: Counter[str] = Counter()

        for row in rows:
            for event in row.get("object_events") or []:
                event_type = str(event.get("event_type", "unknown"))
                event_type_counts[event_type] += 1
                direction = str(event.get("direction", ""))
                if event_type in {"move", "transform"} and direction not in {
                    "",
                    "static",
                }:
                    move_direction_counts[direction] += 1

        event_count = sum(event_type_counts.values())
        directional_total = sum(move_direction_counts.values())
        if move_direction_counts:
            predicted_direction, direction_samples = move_direction_counts.most_common(1)[0]
            direction_confidence = direction_samples / directional_total
        else:
            predicted_direction = "none"
            direction_confidence = 0.0

        avg_utility = utility_total / samples
        event_rate = (samples - no_event_count) / samples
        priority_score = _priority_score(
            samples=samples,
            avg_utility=avg_utility,
            progress_count=progress_count,
            event_rate=event_rate,
            direction_confidence=direction_confidence,
        )
        hypotheses.append(
            ActionHypothesis(
                action=action,
                action_id=action_id,
                samples=samples,
                avg_utility=avg_utility,
                progress_count=progress_count,
                state_change_count=state_change_count,
                event_count=event_count,
                no_event_count=no_event_count,
                event_type_counts=dict(sorted(event_type_counts.items())),
                move_direction_counts=dict(sorted(move_direction_counts.items())),
                predicted_direction=predicted_direction,
                direction_confidence=direction_confidence,
                event_rate=event_rate,
                priority_score=priority_score,
                recommended_use=_recommended_use(
                    samples=samples,
                    avg_utility=avg_utility,
                    progress_count=progress_count,
                    event_rate=event_rate,
                    direction_confidence=direction_confidence,
                    min_samples=min_samples,
                ),
            )
        )

    hypotheses.sort(key=lambda item: (-item.priority_score, item.action_id, item.action))
    return hypotheses
