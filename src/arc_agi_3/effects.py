"""Action-effect inference from ARC-AGI-3 frame pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from arcengine import FrameDataRaw

from .features import primary_layer_grid


@dataclass(frozen=True)
class ObjectMotion:
    color: int
    area_before: int
    area_after: int
    before_x: int
    before_y: int
    after_x: int
    after_y: int
    dx: int
    dy: int
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionEffectAggregate:
    action: str
    action_id: int
    count: int
    avg_changed_cells: float
    motion_count: int
    avg_motion_dx: float
    avg_motion_dy: float
    common_direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def direction_from_delta(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "static"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def _centroid(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return (
        (sum(cell[0] for cell in cells) + len(cells) // 2) // len(cells),
        (sum(cell[1] for cell in cells) + len(cells) // 2) // len(cells),
    )


def _infer_changed_pixel_motion(
    before: FrameDataRaw | None,
    after: FrameDataRaw | None,
) -> ObjectMotion | None:
    before_grid = primary_layer_grid(before)
    after_grid = primary_layer_grid(after)
    if not before_grid or not after_grid:
        return None

    height = min(len(before_grid), len(after_grid))
    width = min(len(before_grid[0]), len(after_grid[0])) if height else 0
    before_by_color: dict[int, list[tuple[int, int]]] = {}
    after_by_color: dict[int, list[tuple[int, int]]] = {}
    before_counts: Counter[int] = Counter()
    after_counts: Counter[int] = Counter()

    for y in range(height):
        before_counts.update(before_grid[y][:width])
        after_counts.update(after_grid[y][:width])

    background_colors = {
        before_counts.most_common(1)[0][0] if before_counts else None,
        after_counts.most_common(1)[0][0] if after_counts else None,
    }

    for y in range(height):
        for x in range(width):
            before_color = before_grid[y][x]
            after_color = after_grid[y][x]
            if before_color == after_color:
                continue
            before_by_color.setdefault(before_color, []).append((x, y))
            after_by_color.setdefault(after_color, []).append((x, y))

    best: ObjectMotion | None = None
    best_score: tuple[int, int] | None = None
    for color in set(before_by_color) & set(after_by_color):
        if color in background_colors:
            continue
        before_cells = before_by_color[color]
        after_cells = after_by_color[color]
        shared_area = min(len(before_cells), len(after_cells))
        if shared_area < 2:
            continue
        before_x, before_y = _centroid(before_cells)
        after_x, after_y = _centroid(after_cells)
        dx = after_x - before_x
        dy = after_y - before_y
        moved_bonus = 1 if dx or dy else 0
        score = (moved_bonus, shared_area)
        if best_score is not None and score <= best_score:
            continue
        best_score = score
        best = ObjectMotion(
            color=color,
            area_before=len(before_cells),
            area_after=len(after_cells),
            before_x=before_x,
            before_y=before_y,
            after_x=after_x,
            after_y=after_y,
            dx=dx,
            dy=dy,
            direction=direction_from_delta(dx, dy),
        )

    return best


def infer_primary_object_motion(
    before: FrameDataRaw | None,
    after: FrameDataRaw | None,
) -> ObjectMotion | None:
    """Return the strongest same-color motion in changed pixels."""
    return _infer_changed_pixel_motion(before, after)


def aggregate_action_effects(
    step_rows: Iterable[dict[str, Any]],
) -> list[ActionEffectAggregate]:
    """Aggregate action-effect statistics from step JSONL rows."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        action = str(row.get("action", "UNKNOWN"))
        action_id = int(row.get("action_id", -1))
        groups.setdefault((action, action_id), []).append(row)

    aggregates: list[ActionEffectAggregate] = []
    for (action, action_id), rows in sorted(groups.items(), key=lambda item: item[0]):
        changed_total = sum(int(row.get("changed_cells") or 0) for row in rows)
        motion_rows = [
            row
            for row in rows
            if row.get("motion_direction") not in (None, "", "static")
        ]
        direction_counts: Counter[str] = Counter(
            str(row.get("motion_direction")) for row in motion_rows
        )
        motion_count = len(motion_rows)
        dx_total = sum(int(row.get("motion_dx") or 0) for row in motion_rows)
        dy_total = sum(int(row.get("motion_dy") or 0) for row in motion_rows)
        aggregates.append(
            ActionEffectAggregate(
                action=action,
                action_id=action_id,
                count=len(rows),
                avg_changed_cells=changed_total / len(rows),
                motion_count=motion_count,
                avg_motion_dx=(dx_total / motion_count) if motion_count else 0.0,
                avg_motion_dy=(dy_total / motion_count) if motion_count else 0.0,
                common_direction=direction_counts.most_common(1)[0][0]
                if direction_counts
                else "none",
            )
        )
    return aggregates
