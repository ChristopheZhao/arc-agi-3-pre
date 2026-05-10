"""ACTION6 click prior built from frame statistics.

Two layers, both produced as a `(64, 64)` `[0, 1]` multiplier on sigmoid coord
probabilities before sampling:

  1. **Static / background mask** (`StaticPriorBuilder`):
     - cells that never change after `mask_after_frames` observations → 0
       (UI / status-bar / inert borders)
     - cells holding the modal color → ×`background_downweight` (walls, floor)

  2. **Segment equalization** (optional, `enable_segment_prior=True`):
     4-connected same-color flood-fill on the current frame, drop tiny
     specks (< `min_segment_size` pixels), then per-pixel weight = 1/segment_size
     so each *segment* — i.e. each visually distinct "button" — gets the same
     total click mass regardless of its area. This is the cheap version of
     the dolphin-in-a-coma 17/25 priority-tier idea; it doesn't pick *which*
     segment to favor (no learned saliency), it just refuses to spend the
     budget on big background blobs.

The two layers compose by multiplication:
    prior = static_mask × segment_equalize
so the final per-pixel prob is also pushed toward 0 for static cells, soft-
suppressed on background, and renormalized so each foreground segment has
equal expected weight.
"""

from __future__ import annotations

import numpy as np

GRID = 64
NUM_COLOURS = 16


def label_4conn_same_color(arr_2d: np.ndarray) -> tuple[np.ndarray, int]:
    """Flood-fill 4-connected same-color regions.

    Returns:
        labels: int32 array, same shape as input, label ids in [1, n_labels].
                (Label 0 is unused; this matches scipy.ndimage.label convention.)
        n_labels: number of distinct segments.
    """
    H, W = arr_2d.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 0
    stack: list[tuple[int, int]] = []
    for sy in range(H):
        for sx in range(W):
            if labels[sy, sx]:
                continue
            next_label += 1
            color = arr_2d[sy, sx]
            labels[sy, sx] = next_label
            stack.append((sy, sx))
            while stack:
                cy, cx = stack.pop()
                # 4-connected neighbours
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not labels[ny, nx] and arr_2d[ny, nx] == color:
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
    return labels, next_label


class StaticPriorBuilder:
    """Per-game running statistics over observed frames + per-frame click prior."""

    def __init__(
        self,
        grid_size: int = GRID,
        num_colours: int = NUM_COLOURS,
        mask_after_frames: int = 20,
        background_downweight: float = 0.1,
        enable_segment_prior: bool = False,
        min_segment_size: int = 3,
    ) -> None:
        """
        Args:
            mask_after_frames: only start blocking static cells after this many
                observations — early frames may legitimately be static while
                the agent hasn't tried much yet.
            background_downweight: multiplier applied to the cells holding the
                modal color across all observed frames. 0.1 means a click on
                background is 10x less likely than on non-background.
            enable_segment_prior: if True, layer in 4-connected segment
                equalization (each segment gets equal total mass; tiny specks
                under `min_segment_size` get zero).
            min_segment_size: drop segments smaller than this (likely noise /
                antialiasing). Only used when `enable_segment_prior=True`.
        """
        self.grid_size = grid_size
        self.num_colours = num_colours
        self.mask_after_frames = mask_after_frames
        self.background_downweight = background_downweight
        self.enable_segment_prior = enable_segment_prior
        self.min_segment_size = min_segment_size
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.change_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.color_count = np.zeros(self.num_colours, dtype=np.int64)
        self.last_frame: np.ndarray | None = None
        self.n_frames = 0

    def reset(self) -> None:
        """Wipe stats, e.g. on level transition."""
        self._reset_stats()

    def observe(self, frame_2d: np.ndarray) -> None:
        """Update stats from a single 2D palette-index frame."""
        if frame_2d.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"expected ({self.grid_size},{self.grid_size}); got {frame_2d.shape}")
        if self.last_frame is not None:
            self.change_count += (frame_2d != self.last_frame).astype(np.int32)
        self.last_frame = frame_2d.copy()
        self.n_frames += 1
        # vectorized color tally
        bins = np.bincount(frame_2d.ravel(), minlength=self.num_colours)
        self.color_count += bins.astype(np.int64)

    # ---- prior layers ----

    def _static_layer(self, frame_2d: np.ndarray) -> np.ndarray:
        prior = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        if self.n_frames >= self.mask_after_frames:
            static = self.change_count == 0
            prior[static] = 0.0
        if self.color_count.sum() > 0 and self.background_downweight < 1.0:
            bg_color = int(self.color_count.argmax())
            prior[frame_2d == bg_color] *= self.background_downweight
        return prior

    def _segment_layer(self, frame_2d: np.ndarray) -> np.ndarray:
        """Per-pixel weight = 1/segment_size; tiny segments dropped to 0."""
        labels, n = label_4conn_same_color(frame_2d)
        if n == 0:
            return np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        sizes = np.bincount(labels.ravel(), minlength=n + 1)  # sizes[0] is unused
        # per-pixel weight: 1/size for valid segments, 0 for too-small ones
        per_label_w = np.zeros(n + 1, dtype=np.float32)
        valid = sizes >= self.min_segment_size
        per_label_w[valid] = 1.0 / sizes[valid]
        return per_label_w[labels]  # gather: (H, W) of weights

    def click_prior(self, frame_2d: np.ndarray) -> np.ndarray:
        """Return (64, 64) float32 — multiplier on sigmoid coord probabilities.

        Composition: static_layer × segment_layer (when segment prior enabled).
        """
        prior = self._static_layer(frame_2d)
        if self.enable_segment_prior:
            prior = prior * self._segment_layer(frame_2d)
        return prior

    def fraction_active(self) -> float:
        """Fraction of cells with prior > 0 (how much the prior is biting)."""
        if self.last_frame is None:
            return 1.0
        prior = self.click_prior(self.last_frame)
        return float((prior > 0).mean())

    def n_segments(self) -> int:
        """Diagnostic: number of segments (>= min_segment_size) in the last frame."""
        if self.last_frame is None or not self.enable_segment_prior:
            return 0
        labels, n = label_4conn_same_color(self.last_frame)
        if n == 0:
            return 0
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        return int((sizes[1:] >= self.min_segment_size).sum())
