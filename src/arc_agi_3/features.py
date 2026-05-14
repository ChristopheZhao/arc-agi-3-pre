"""Frame feature extraction helpers for ARC-AGI-3 agents."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any

from arcengine import FrameDataRaw


def _round_half_up(total: int, count: int) -> int:
    return (total + count // 2) // count


@dataclass(frozen=True)
class ColorObject:
    color: int
    area: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    centroid_x: int
    centroid_y: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FrameFeatures:
    width: int
    height: int
    background_color: int | None
    color_counts: dict[int, int]
    objects: list[ColorObject]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "background_color": self.background_color,
            "color_counts": self.color_counts,
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass(frozen=True)
class FrameDiffSummary:
    changed_cells: int
    changed_bbox: tuple[int, int, int, int] | None
    changed_colors: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_cells": self.changed_cells,
            "changed_bbox": self.changed_bbox,
            "changed_colors": self.changed_colors,
        }


def primary_layer_grid(frame_data: FrameDataRaw | None) -> list[list[int]]:
    """Return the first visual layer as a plain Python int grid."""
    if frame_data is None or not frame_data.frame:
        return []

    layer = frame_data.frame[0]
    if hasattr(layer, "tolist"):
        raw_grid = layer.tolist()
    else:
        raw_grid = layer

    return [[int(value) for value in row] for row in raw_grid]


def color_counts(frame_data: FrameDataRaw | None) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in primary_layer_grid(frame_data):
        counts.update(row)
    return counts


def dominant_color(frame_data: FrameDataRaw | None) -> int | None:
    counts = color_counts(frame_data)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def extract_color_objects(
    frame_data: FrameDataRaw | None,
    *,
    ignore_background: bool = True,
    min_area: int = 1,
    max_objects: int = 64,
) -> list[ColorObject]:
    """Find 4-connected same-color components in the first frame layer."""
    grid = primary_layer_grid(frame_data)
    if not grid:
        return []

    height = len(grid)
    width = len(grid[0]) if height else 0
    if width == 0:
        return []

    background = dominant_color(frame_data) if ignore_background else None
    visited: set[tuple[int, int]] = set()
    objects: list[ColorObject] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in visited:
                continue
            color = grid[y][x]
            if ignore_background and color == background:
                visited.add((x, y))
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in (
                    (cx + 1, cy),
                    (cx - 1, cy),
                    (cx, cy + 1),
                    (cx, cy - 1),
                ):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if (nx, ny) in visited or grid[ny][nx] != color:
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))

            area = len(cells)
            if area < min_area:
                continue

            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            objects.append(
                ColorObject(
                    color=color,
                    area=area,
                    x_min=min(xs),
                    y_min=min(ys),
                    x_max=max(xs),
                    y_max=max(ys),
                    centroid_x=_round_half_up(sum(xs), area),
                    centroid_y=_round_half_up(sum(ys), area),
                )
            )

    objects.sort(key=lambda obj: (-obj.area, obj.color, obj.y_min, obj.x_min))
    return objects[:max_objects]


def extract_frame_features(frame_data: FrameDataRaw | None) -> FrameFeatures:
    grid = primary_layer_grid(frame_data)
    height = len(grid)
    width = len(grid[0]) if height else 0
    counts = color_counts(frame_data)
    background = counts.most_common(1)[0][0] if counts else None
    return FrameFeatures(
        width=width,
        height=height,
        background_color=background,
        color_counts=dict(sorted(counts.items())),
        objects=extract_color_objects(frame_data),
    )


def object_coordinate_candidates(
    frame_data: FrameDataRaw | None,
    *,
    max_objects: int = 16,
) -> list[dict[str, int]]:
    """Use object centroids and boxes as candidate coordinates for ACTION6."""
    candidates: list[tuple[int, int]] = []
    for obj in extract_color_objects(frame_data, max_objects=max_objects):
        candidates.extend(
            [
                (obj.centroid_x, obj.centroid_y),
                (obj.x_min, obj.y_min),
                (obj.x_max, obj.y_max),
            ]
        )

    deduped: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in candidates:
        if point in seen:
            continue
        seen.add(point)
        deduped.append({"x": point[0], "y": point[1]})
    return deduped


def frame_diff_summary(
    before: FrameDataRaw | None,
    after: FrameDataRaw | None,
) -> FrameDiffSummary:
    before_grid = primary_layer_grid(before)
    after_grid = primary_layer_grid(after)
    if not before_grid or not after_grid:
        return FrameDiffSummary(changed_cells=0, changed_bbox=None, changed_colors={})

    height = min(len(before_grid), len(after_grid))
    width = min(len(before_grid[0]), len(after_grid[0])) if height else 0
    changed: list[tuple[int, int]] = []
    changed_colors: Counter[int] = Counter()

    for y in range(height):
        for x in range(width):
            if before_grid[y][x] == after_grid[y][x]:
                continue
            changed.append((x, y))
            changed_colors[after_grid[y][x]] += 1

    if not changed:
        return FrameDiffSummary(changed_cells=0, changed_bbox=None, changed_colors={})

    xs = [point[0] for point in changed]
    ys = [point[1] for point in changed]
    return FrameDiffSummary(
        changed_cells=len(changed),
        changed_bbox=(min(xs), min(ys), max(xs), max(ys)),
        changed_colors=dict(sorted(changed_colors.items())),
    )
