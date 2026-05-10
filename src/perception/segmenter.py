"""Lightweight ACTION6 click prior built from frame statistics.

The dolphin-in-a-coma 17/25 paper showed that just throwing all 4096 click
positions into a CNN coord head wastes a huge amount of the agent's action
budget on inert UI / status-bar pixels. Their fix:

  1. mask cells that never change (UI / static borders / status display)
  2. downweight the modal background color (walls, empty floor)
  3. group remaining cells into connected-component segments,
     weight each segment so total mass per *segment* (not per *pixel*) is fair

This module implements (1) and (2) — the cheap, no-deps wins. (3) requires
proper labeling and is left for a follow-up if the simpler prior turns out to
help on benchmark games.

The output is a `(64, 64)` `[0, 1]` multiplier that the SG agent multiplies
into its sigmoid coord probabilities before sampling. Static cells get 0
(blocked); background-color cells get a heavy downweight; everything else
keeps full weight.
"""

from __future__ import annotations

import numpy as np

GRID = 64
NUM_COLOURS = 16


class StaticPriorBuilder:
    """Per-game running statistics over observed frames + per-frame click prior."""

    def __init__(
        self,
        grid_size: int = GRID,
        num_colours: int = NUM_COLOURS,
        mask_after_frames: int = 20,
        background_downweight: float = 0.1,
    ) -> None:
        """
        Args:
            mask_after_frames: only start blocking static cells after this many
                observations — early frames may legitimately be static while
                the agent hasn't tried much yet.
            background_downweight: multiplier applied to the cells holding the
                modal color across all observed frames. 0.1 means a click on
                background is 10x less likely than on non-background.
        """
        self.grid_size = grid_size
        self.num_colours = num_colours
        self.mask_after_frames = mask_after_frames
        self.background_downweight = background_downweight
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

    def click_prior(self, frame_2d: np.ndarray) -> np.ndarray:
        """Return (64, 64) float32 in [0, 1] — multiplier on coord probabilities."""
        prior = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        if self.n_frames >= self.mask_after_frames:
            static = self.change_count == 0
            prior[static] = 0.0

        if self.color_count.sum() > 0 and self.background_downweight < 1.0:
            bg_color = int(self.color_count.argmax())
            prior[frame_2d == bg_color] *= self.background_downweight

        return prior

    def fraction_active(self) -> float:
        """Fraction of cells with prior > 0 (rough indicator of how much the prior is biting)."""
        if self.last_frame is None:
            return 1.0
        prior = self.click_prior(self.last_frame)
        return float((prior > 0).mean())
