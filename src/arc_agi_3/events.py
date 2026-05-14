"""Object-centric event extraction for ARC-AGI-3 frame transitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from arcengine import FrameDataRaw

from .effects import direction_from_delta
from .features import ColorObject, FrameDiffSummary, extract_color_objects
from .features import frame_diff_summary as summarize_frame_diff


@dataclass(frozen=True)
class ObjectState:
    color: int
    area: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    centroid_x: int
    centroid_y: int
    width: int
    height: int

    @classmethod
    def from_color_object(cls, obj: ColorObject) -> ObjectState:
        return cls(
            color=obj.color,
            area=obj.area,
            x_min=obj.x_min,
            y_min=obj.y_min,
            x_max=obj.x_max,
            y_max=obj.y_max,
            centroid_x=obj.centroid_x,
            centroid_y=obj.centroid_y,
            width=obj.x_max - obj.x_min + 1,
            height=obj.y_max - obj.y_min + 1,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectEvent:
    event_type: str
    color: int
    before: ObjectState | None
    after: ObjectState | None
    dx: int
    dy: int
    direction: str
    area_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameTransitionSummary:
    changed_cells: int
    changed_bbox: tuple[int, int, int, int] | None
    changed_colors: dict[int, int]
    level_delta: int
    win_level_delta: int
    state_changed: bool
    object_events: list[ObjectEvent]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionObjectEventAggregate:
    action: str
    action_id: int
    count: int
    event_count: int
    move_count: int
    appear_count: int
    disappear_count: int
    reshape_count: int
    transform_count: int
    avg_utility: float
    common_move_direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position_key(obj: ObjectState) -> tuple[int, int, int, int]:
    return obj.y_min, obj.x_min, obj.area, obj.color


def _event_position(event: ObjectEvent) -> tuple[int, int, int]:
    obj = event.before if event.before is not None else event.after
    if obj is None:
        return (0, 0, event.color)
    return obj.y_min, obj.x_min, event.color


def _match_score(before: ObjectState, after: ObjectState) -> tuple[int, int, int, int]:
    area_delta = abs(after.area - before.area)
    shape_delta = abs(after.width - before.width) + abs(after.height - before.height)
    centroid_delta = abs(after.centroid_x - before.centroid_x) + abs(
        after.centroid_y - before.centroid_y
    )
    bbox_delta = abs(after.x_min - before.x_min) + abs(after.y_min - before.y_min)
    return area_delta, shape_delta, centroid_delta, bbox_delta


def _build_event(
    before: ObjectState | None,
    after: ObjectState | None,
) -> ObjectEvent:
    color = before.color if before is not None else after.color if after else -1
    before_x = before.centroid_x if before is not None else after.centroid_x if after else 0
    before_y = before.centroid_y if before is not None else after.centroid_y if after else 0
    after_x = after.centroid_x if after is not None else before_x
    after_y = after.centroid_y if after is not None else before_y
    dx = after_x - before_x
    dy = after_y - before_y
    before_area = before.area if before is not None else 0
    after_area = after.area if after is not None else 0
    area_delta = after_area - before_area

    if before is None:
        event_type = "appear"
    elif after is None:
        event_type = "disappear"
    else:
        moved = dx != 0 or dy != 0
        shape_changed = (
            area_delta != 0
            or before.width != after.width
            or before.height != after.height
        )
        if moved and shape_changed:
            event_type = "transform"
        elif moved:
            event_type = "move"
        elif shape_changed:
            event_type = "reshape"
        else:
            event_type = "static"

    return ObjectEvent(
        event_type=event_type,
        color=color,
        before=before,
        after=after,
        dx=dx,
        dy=dy,
        direction=direction_from_delta(dx, dy),
        area_delta=area_delta,
    )


def _group_by_color(objects: list[ObjectState]) -> dict[int, list[ObjectState]]:
    grouped: dict[int, list[ObjectState]] = {}
    for obj in objects:
        grouped.setdefault(obj.color, []).append(obj)
    for values in grouped.values():
        values.sort(key=_position_key)
    return grouped


def _match_color_group(
    before_objects: list[ObjectState],
    after_objects: list[ObjectState],
) -> list[tuple[ObjectState | None, ObjectState | None]]:
    candidates: list[tuple[tuple[int, int, int, int], int, int]] = []
    for before_idx, before in enumerate(before_objects):
        for after_idx, after in enumerate(after_objects):
            candidates.append((_match_score(before, after), before_idx, after_idx))

    matched_before: set[int] = set()
    matched_after: set[int] = set()
    pairs: list[tuple[ObjectState | None, ObjectState | None]] = []
    for _, before_idx, after_idx in sorted(candidates):
        if before_idx in matched_before or after_idx in matched_after:
            continue
        matched_before.add(before_idx)
        matched_after.add(after_idx)
        pairs.append((before_objects[before_idx], after_objects[after_idx]))

    for before_idx, before in enumerate(before_objects):
        if before_idx not in matched_before:
            pairs.append((before, None))
    for after_idx, after in enumerate(after_objects):
        if after_idx not in matched_after:
            pairs.append((None, after))
    return pairs


def infer_object_events(
    before: FrameDataRaw | None,
    after: FrameDataRaw | None,
    *,
    max_objects: int = 64,
    include_static: bool = False,
) -> list[ObjectEvent]:
    """Infer object-level event candidates by matching same-color components."""
    before_objects = [
        ObjectState.from_color_object(obj)
        for obj in extract_color_objects(before, max_objects=max_objects)
    ]
    after_objects = [
        ObjectState.from_color_object(obj)
        for obj in extract_color_objects(after, max_objects=max_objects)
    ]
    before_by_color = _group_by_color(before_objects)
    after_by_color = _group_by_color(after_objects)

    events: list[ObjectEvent] = []
    for color in sorted(set(before_by_color) | set(after_by_color)):
        pairs = _match_color_group(
            before_by_color.get(color, []),
            after_by_color.get(color, []),
        )
        for before_obj, after_obj in pairs:
            event = _build_event(before_obj, after_obj)
            if event.event_type == "static" and not include_static:
                continue
            events.append(event)

    events.sort(key=lambda event: (event.color, _event_position(event), event.event_type))
    return events


def frame_transition_summary(
    before: FrameDataRaw | None,
    after: FrameDataRaw | None,
    *,
    diff_summary: FrameDiffSummary | None = None,
) -> FrameTransitionSummary:
    diff = diff_summary or summarize_frame_diff(before, after)
    before_levels = int(getattr(before, "levels_completed", 0) or 0)
    after_levels = int(getattr(after, "levels_completed", 0) or 0)
    before_win_levels = int(getattr(before, "win_levels", 0) or 0)
    after_win_levels = int(getattr(after, "win_levels", 0) or 0)
    before_state = getattr(before, "state", None)
    after_state = getattr(after, "state", None)
    return FrameTransitionSummary(
        changed_cells=diff.changed_cells,
        changed_bbox=diff.changed_bbox,
        changed_colors=diff.changed_colors,
        level_delta=after_levels - before_levels,
        win_level_delta=after_win_levels - before_win_levels,
        state_changed=before_state != after_state,
        object_events=infer_object_events(before, after),
    )


def transition_utility(summary: FrameTransitionSummary) -> float:
    if summary.level_delta > 0:
        return 100.0 + summary.level_delta
    if summary.state_changed:
        return 5.0
    event_types = {event.event_type for event in summary.object_events}
    utility = 0.0
    if event_types & {"appear", "disappear", "transform"}:
        utility += 2.0
    if event_types & {"move", "reshape"}:
        utility += 1.0
    if summary.changed_cells > 0 and not event_types:
        utility += 0.25
    return utility


def aggregate_object_events(
    step_rows: Iterable[dict[str, Any]],
) -> list[ActionObjectEventAggregate]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        action = str(row.get("action", "UNKNOWN"))
        action_id = int(row.get("action_id", -1))
        groups.setdefault((action, action_id), []).append(row)

    aggregates: list[ActionObjectEventAggregate] = []
    for (action, action_id), rows in sorted(groups.items(), key=lambda item: item[0]):
        event_counts: Counter[str] = Counter()
        move_directions: Counter[str] = Counter()
        event_count = 0
        utility_total = 0.0
        for row in rows:
            utility_total += float(row.get("action_utility") or 0.0)
            events = row.get("object_events") or []
            for event in events:
                event_type = str(event.get("event_type", "unknown"))
                event_counts[event_type] += 1
                event_count += 1
                direction = str(event.get("direction", ""))
                if event_type in {"move", "transform"} and direction not in {
                    "",
                    "static",
                }:
                    move_directions[direction] += 1
        aggregates.append(
            ActionObjectEventAggregate(
                action=action,
                action_id=action_id,
                count=len(rows),
                event_count=event_count,
                move_count=event_counts["move"],
                appear_count=event_counts["appear"],
                disappear_count=event_counts["disappear"],
                reshape_count=event_counts["reshape"],
                transform_count=event_counts["transform"],
                avg_utility=utility_total / len(rows),
                common_move_direction=move_directions.most_common(1)[0][0]
                if move_directions
                else "none",
            )
        )
    return aggregates
