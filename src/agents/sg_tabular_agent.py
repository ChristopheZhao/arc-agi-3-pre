"""Sprint B Kaggle agent: tabular novelty + frontier + segment-prior coord pruning.

Single-file by design so sg_submit.py can `cp` this file verbatim into
/kaggle/working/my_agent.py without needing src/perception imports.

Algorithm summary:
  - State hashing: blake2b(game_id + state + levels + win_levels + frame_bytes)
  - Decision scoring (mirrors 0.15 EffectPriorPolicy):
      bootstrap-phase tuple: (trials, stall, transitions, dir_penalty, rng)
      mature-phase tuple:   (stall, dir_penalty, transitions, trials, rng)
  - Coord candidate generation: geometric points + object centroids +
    frontier_positions + random samples → segment-prior-weighted top-K
    (this is the project-specific signal absent from 0.15).
  - Cross-level: counters persist (per 0.15 design).
  - Per-game: fresh agent instance per Kaggle Swarm convention.

Budget: MAX_ACTIONS=80, per-game 60s — same as 0.15 archive.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
import traceback
from collections import Counter
from typing import Any, Optional

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

# ============================================================================
# Constants
# ============================================================================

GRID = 64
NUM_COLOURS = 16
MAX_OBJECTS = 64

# ============================================================================
# Frame extraction helpers
# ============================================================================


def primary_layer(frame_data: FrameData) -> Optional[np.ndarray]:
    """Return the last visual layer as a (H, W) int array, or None."""
    if frame_data is None or not getattr(frame_data, "frame", None):
        return None
    arr = np.array(frame_data.frame, dtype=np.int64)
    if arr.ndim < 2:
        return None
    return arr[-1]


def state_hash(frame_data: FrameData) -> Optional[str]:
    """Stable hash of game state for tabular indexing."""
    if frame_data is None:
        return None
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(getattr(frame_data, "game_id", "")).encode("utf-8"))
    digest.update(str(getattr(frame_data, "state", "")).encode("utf-8"))
    digest.update(str(getattr(frame_data, "levels_completed", 0)).encode("ascii"))
    digest.update(str(getattr(frame_data, "win_levels", 0)).encode("ascii"))
    for layer in frame_data.frame:
        arr = np.asarray(layer)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


# ============================================================================
# 4-conn segmentation + object extraction
# ============================================================================


def label_4conn_same_color(arr_2d: np.ndarray) -> tuple[np.ndarray, int]:
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
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and not labels[ny, nx]
                        and arr_2d[ny, nx] == color
                    ):
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
    return labels, next_label


def extract_objects(
    arr_2d: np.ndarray,
    bg_color: Optional[int],
    max_objects: int = MAX_OBJECTS,
) -> list[dict[str, int]]:
    """Return list of object dicts (color/area/bbox/centroid), ignoring bg."""
    labels, n = label_4conn_same_color(arr_2d)
    if n == 0:
        return []
    out: list[dict[str, int]] = []
    for label_id in range(1, n + 1):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        color = int(arr_2d[ys[0], xs[0]])
        if bg_color is not None and color == bg_color:
            continue
        area = int(len(xs))
        out.append(
            {
                "color": color,
                "area": area,
                "xmin": int(xs.min()),
                "xmax": int(xs.max()),
                "ymin": int(ys.min()),
                "ymax": int(ys.max()),
                "cx": int(round(float(xs.mean()))),
                "cy": int(round(float(ys.mean()))),
            }
        )
    out.sort(key=lambda o: (-o["area"], o["color"], o["ymin"], o["xmin"]))
    return out[:max_objects]


def primary_object_centroid(
    arr_2d: np.ndarray, bg_color: Optional[int]
) -> Optional[tuple[int, int]]:
    objs = extract_objects(arr_2d, bg_color, max_objects=1)
    return (objs[0]["cx"], objs[0]["cy"]) if objs else None


# ============================================================================
# Segment prior (StaticPriorBuilder, inlined from sg_kaggle_agent.py)
# ============================================================================


class StaticPriorBuilder:
    """Per-game running stats + per-frame click prior."""

    def __init__(
        self,
        mask_after_frames: int = 20,
        background_downweight: float = 0.1,
        enable_segment_prior: bool = True,
        min_segment_size: int = 3,
    ) -> None:
        self.mask_after_frames = mask_after_frames
        self.background_downweight = background_downweight
        self.enable_segment_prior = enable_segment_prior
        self.min_segment_size = min_segment_size
        self.reset()

    def reset(self) -> None:
        self.change_count = np.zeros((GRID, GRID), dtype=np.int32)
        self.color_count = np.zeros(NUM_COLOURS, dtype=np.int64)
        self.last_frame: Optional[np.ndarray] = None
        self.n_frames = 0

    def observe(self, frame_2d: np.ndarray) -> None:
        if frame_2d.shape != (GRID, GRID):
            return
        if self.last_frame is not None:
            self.change_count += (frame_2d != self.last_frame).astype(np.int32)
        self.last_frame = frame_2d.copy()
        self.n_frames += 1
        self.color_count += np.bincount(
            frame_2d.ravel(), minlength=NUM_COLOURS
        ).astype(np.int64)

    def bg_color(self) -> Optional[int]:
        if self.color_count.sum() == 0:
            return None
        return int(self.color_count.argmax())

    def click_prior(self, frame_2d: np.ndarray) -> np.ndarray:
        prior = np.ones((GRID, GRID), dtype=np.float32)
        if self.n_frames >= self.mask_after_frames:
            prior[self.change_count == 0] = 0.0
        bg = self.bg_color()
        if bg is not None and self.background_downweight < 1.0:
            prior[frame_2d == bg] *= self.background_downweight
        if self.enable_segment_prior:
            labels, n = label_4conn_same_color(frame_2d)
            if n > 0:
                sizes = np.bincount(labels.ravel(), minlength=n + 1)
                per_label_w = np.zeros(n + 1, dtype=np.float32)
                valid = sizes >= self.min_segment_size
                per_label_w[valid] = 1.0 / sizes[valid]
                prior = prior * per_label_w[labels]
        return prior


# ============================================================================
# Direction helpers
# ============================================================================


def direction_to_target(
    origin: tuple[int, int], target: tuple[int, int]
) -> str:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if dx == 0 and dy == 0:
        return "static"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def opposite_direction(d: str) -> str:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(d, "")


def infer_motion(
    prev_centroid: Optional[tuple[int, int]],
    cur_centroid: Optional[tuple[int, int]],
) -> str:
    if prev_centroid is None or cur_centroid is None:
        return ""
    dx = cur_centroid[0] - prev_centroid[0]
    dy = cur_centroid[1] - prev_centroid[1]
    if dx == 0 and dy == 0:
        return "static"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def changed_bbox_center(
    bbox: Optional[tuple[int, int, int, int]]
) -> Optional[tuple[int, int]]:
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return ((xmin + xmax + 1) // 2, (ymin + ymax + 1) // 2)


# ============================================================================
# Decision record
# ============================================================================


class Decision:
    __slots__ = ("action_id", "action_data", "reason")

    def __init__(self, action_id: int, action_data: dict[str, int], reason: str = ""):
        self.action_id = action_id  # 1..6
        self.action_data = action_data
        self.reason = reason

    @property
    def action_key(self) -> str:
        if self.action_data:
            payload = ",".join(f"{k}={self.action_data[k]}" for k in sorted(self.action_data))
            return f"{self.action_id}:{payload}"
        return str(self.action_id)


# ============================================================================
# Policy
# ============================================================================


class TabularPolicy:
    """Tabular novelty + direction prior + frontier + segment-prior coord pruning.

    Mirrors 0.15 EffectPriorPolicy.score() shape literally. Differs from 0.15
    in coord generation: 0.15 uses geometric + object-centroid + random; we
    use the same set then prune by segment_prior weight (top-K).
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        allow_reset: bool = False,
        coord_samples: int = 32,
        top_k_coords: int = 24,
        min_effect_samples: int = 4,
    ) -> None:
        self.rng = random.Random(seed)
        self.allow_reset = allow_reset
        self.coord_samples = coord_samples
        self.top_k_coords = top_k_coords
        self.min_effect_samples = min_effect_samples

        self.state_visits: Counter = Counter()
        self.transition_counts: Counter = Counter()
        self.action_trials: Counter = Counter()
        self.action_direction_counts: Counter = Counter()
        self.action_no_motion_streak: Counter = Counter()
        self.visited_positions: Counter = Counter()
        self.frontier_positions: Counter = Counter()
        self.min_frontier_changed_cells = 4
        self.last_target_position: Optional[tuple[int, int]] = None

    # ---- direction / frontier helpers ------------------------------------

    def learned_direction(self, action_id: int) -> str:
        counts = {
            d: c
            for (aid, d), c in self.action_direction_counts.items()
            if aid == action_id
        }
        if not counts:
            return ""
        direction, count = max(counts.items(), key=lambda kv: kv[1])
        return direction if count >= self.min_effect_samples else ""

    def nearby_visit_count(
        self, target: tuple[int, int], *, radius: int = 1
    ) -> int:
        return sum(
            count
            for pos, count in self.visited_positions.items()
            if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= radius
        )

    def desired_direction(
        self,
        cur: Optional[tuple[int, int]],
        width: int,
        height: int,
    ) -> str:
        if cur is None or width <= 0 or height <= 0:
            self.last_target_position = None
            return ""

        frontier_targets = [
            t
            for t in self.frontier_positions
            if t != cur and 0 <= t[0] < width and 0 <= t[1] < height
        ]
        if frontier_targets:
            def f_score(t: tuple[int, int]) -> tuple[int, int, int, int, float]:
                nearest_seen = (
                    min(
                        abs(t[0] - s[0]) + abs(t[1] - s[1])
                        for s in self.visited_positions
                    )
                    if self.visited_positions
                    else 0
                )
                td = abs(t[0] - cur[0]) + abs(t[1] - cur[1])
                return (
                    -self.nearby_visit_count(t),
                    self.frontier_positions[t],
                    nearest_seen,
                    td,
                    self.rng.random(),
                )

            target = max(frontier_targets, key=f_score)
            self.last_target_position = target
            return direction_to_target(cur, target)

        # Fall back to corners/edges
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
            self.last_target_position = targets[0]
            return direction_to_target(cur, targets[0])

        def g_score(t: tuple[int, int]) -> tuple[int, int, float]:
            nearest_seen = min(
                abs(t[0] - s[0]) + abs(t[1] - s[1])
                for s in self.visited_positions
            )
            td = abs(t[0] - cur[0]) + abs(t[1] - cur[1])
            return (nearest_seen, td, self.rng.random())

        target = max(targets, key=g_score)
        self.last_target_position = target
        return direction_to_target(cur, target)

    def direction_penalty(self, decision: Decision, desired: str) -> int:
        if not desired or desired == "static":
            return 1
        learned = self.learned_direction(decision.action_id)
        if not learned:
            return 1
        if learned == desired:
            return 0
        if learned == opposite_direction(desired):
            return 3
        return 2

    def stall_penalty(self, decision: Decision) -> int:
        return min(self.action_no_motion_streak[decision.action_id], 3)

    # ---- coord candidate generation --------------------------------------

    def generate_coord_candidates(
        self,
        frame_2d: np.ndarray,
        prior: np.ndarray,
        objects: list[dict[str, int]],
    ) -> list[tuple[int, int]]:
        h, w = frame_2d.shape
        max_x = max(w - 1, 0)
        max_y = max(h - 1, 0)

        # Seed candidates: geometric points + object centroids + frontier + random
        points: list[tuple[int, int]] = [
            (w // 2, h // 2),
            (0, 0),
            (max_x, 0),
            (0, max_y),
            (max_x, max_y),
            (w // 2, 0),
            (w // 2, max_y),
            (0, h // 2),
            (max_x, h // 2),
        ]
        for obj in objects:
            points.append((obj["cx"], obj["cy"]))
            points.append((obj["xmin"], obj["ymin"]))
            points.append((obj["xmax"], obj["ymax"]))
        for fpt in list(self.frontier_positions):
            points.append(fpt)
        for _ in range(max(self.coord_samples, 0)):
            points.append(
                (self.rng.randrange(max(w, 1)), self.rng.randrange(max(h, 1)))
            )

        # Clamp + dedupe
        seen: set[tuple[int, int]] = set()
        valid: list[tuple[int, int]] = []
        for x, y in points:
            cx = min(max(int(x), 0), max_x)
            cy = min(max(int(y), 0), max_y)
            p = (cx, cy)
            if p in seen:
                continue
            seen.add(p)
            valid.append(p)

        if not valid:
            return []

        # Weight by segment prior; pick top-K.
        # Tie-break: any positive prior beats zero; among ties keep insertion order.
        weighted = sorted(
            valid,
            key=lambda p: (-float(prior[p[1], p[0]]), valid.index(p)),
        )
        return weighted[: max(self.top_k_coords, 1)]

    # ---- main selection --------------------------------------------------

    def select(
        self,
        frame_data: FrameData,
        frame_2d: Optional[np.ndarray],
        prior: np.ndarray,
        primary_cent: Optional[tuple[int, int]],
        objects: list[dict[str, int]],
        cur_hash: Optional[str],
    ) -> Optional[Decision]:
        if frame_data is None:
            return None
        avail = getattr(frame_data, "available_actions", None) or []
        action_ids: list[int] = []
        for a in avail:
            aid = a.value if hasattr(a, "value") else int(a)
            if 1 <= aid <= 6 and (aid != 0 or self.allow_reset):
                action_ids.append(aid)
        if not action_ids:
            return None

        decisions: list[Decision] = []
        for aid in action_ids:
            if aid == 6 and frame_2d is not None:
                for cx, cy in self.generate_coord_candidates(frame_2d, prior, objects):
                    decisions.append(Decision(6, {"x": cx, "y": cy}, "tabular_novelty_segmentprior"))
            else:
                decisions.append(Decision(aid, {}, "tabular_novelty"))

        if not decisions:
            return None
        if cur_hash is None:
            return self.rng.choice(decisions)

        h = frame_2d.shape[0] if frame_2d is not None else 64
        w = frame_2d.shape[1] if frame_2d is not None else 64
        desired = self.desired_direction(primary_cent, w, h)

        trial_floor = min(self.action_trials[d.action_id] for d in decisions)
        bootstrap = trial_floor < self.min_effect_samples

        def score(d: Decision) -> tuple[int, int, int, int, float]:
            if bootstrap:
                return (
                    self.action_trials[d.action_id],
                    self.stall_penalty(d),
                    self.transition_counts[(cur_hash, d.action_key)],
                    self.direction_penalty(d, desired),
                    self.rng.random(),
                )
            return (
                self.stall_penalty(d),
                self.direction_penalty(d, desired),
                self.transition_counts[(cur_hash, d.action_key)],
                self.action_trials[d.action_id],
                self.rng.random(),
            )

        return min(decisions, key=score)

    # ---- observation -----------------------------------------------------

    def observe_initial(self, frame_data: FrameData) -> None:
        h = state_hash(frame_data)
        if h is not None:
            self.state_visits[h] += 1

    def observe_frame_position(self, primary_cent: Optional[tuple[int, int]]) -> None:
        if primary_cent is not None:
            self.visited_positions[primary_cent] += 1

    def observe_result(
        self,
        before_hash: Optional[str],
        decision: Decision,
        after_hash: Optional[str],
        motion_direction: str,
        changed_cells: int,
        changed_bbox: Optional[tuple[int, int, int, int]],
    ) -> None:
        if before_hash is not None:
            self.transition_counts[(before_hash, decision.action_key)] += 1
        if after_hash is not None:
            self.state_visits[after_hash] += 1
        self.action_trials[decision.action_id] += 1
        if changed_cells >= self.min_frontier_changed_cells:
            fr = changed_bbox_center(changed_bbox)
            if fr is not None:
                self.frontier_positions[fr] += 1
        if motion_direction in ("", "static"):
            self.action_no_motion_streak[decision.action_id] += 1
            return
        self.action_no_motion_streak[decision.action_id] = 0
        self.action_direction_counts[(decision.action_id, motion_direction)] += 1


# ============================================================================
# Agent (Kaggle gateway interface)
# ============================================================================


class MyAgent(Agent):
    """Tabular agent for Kaggle submission. Same Agent contract as random sample."""

    MAX_ACTIONS = 80
    _MAX_FRAMES = 10
    PER_GAME_SECONDS = 60.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(getattr(self, "game_id", "")) % 1000000
        seed = seed % (2**31 - 1)
        self.start_time = time.time()
        self.logger = logging.getLogger(f"sg_tabular_{getattr(self, 'game_id', '?')}")

        self.policy = TabularPolicy(
            seed=seed,
            allow_reset=False,
            coord_samples=32,
            top_k_coords=24,
            min_effect_samples=4,
        )
        self.prior = StaticPriorBuilder(enable_segment_prior=True)

        self.current_level = -1
        self.observed_initial = False
        self.pending_decision: Optional[Decision] = None
        self.pending_before_hash: Optional[str] = None
        self.pending_before_centroid: Optional[tuple[int, int]] = None
        self.pending_before_frame: Optional[np.ndarray] = None

        print(f"[{getattr(self, 'game_id', '?')}] sg_tabular_agent ready")

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not getattr(self, "is_playback", False):
            import json
            self.recorder.record(json.loads(frame.model_dump_json()))

    def _budget_exceeded(self) -> bool:
        return (time.time() - self.start_time) >= self.PER_GAME_SECONDS

    def is_done(self, frames, latest_frame) -> bool:
        try:
            return any(
                [
                    latest_frame.state is GameState.WIN,
                    self.action_counter >= self.MAX_ACTIONS,
                    self._budget_exceeded(),
                ]
            )
        except Exception as e:
            print(f"[{self.game_id}] is_done crashed: {e}")
            traceback.print_exc()
            return True

    def _level_of(self, frame: FrameData) -> int:
        return getattr(frame, "score", None) or frame.levels_completed

    def _observe_pending(self, latest_frame: FrameData, cur_frame_2d: Optional[np.ndarray]) -> None:
        if self.pending_decision is None:
            return
        after = state_hash(latest_frame)
        # Motion
        bg = self.prior.bg_color()
        cur_cent = primary_object_centroid(cur_frame_2d, bg) if cur_frame_2d is not None else None
        motion_dir = infer_motion(self.pending_before_centroid, cur_cent)
        # Diff summary
        changed = 0
        bbox: Optional[tuple[int, int, int, int]] = None
        if self.pending_before_frame is not None and cur_frame_2d is not None and self.pending_before_frame.shape == cur_frame_2d.shape:
            diff_mask = self.pending_before_frame != cur_frame_2d
            changed = int(diff_mask.sum())
            if changed > 0:
                ys, xs = np.where(diff_mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        self.policy.observe_result(
            self.pending_before_hash,
            self.pending_decision,
            after,
            motion_dir,
            changed,
            bbox,
        )
        self.policy.observe_frame_position(cur_cent)

        self.pending_decision = None
        self.pending_before_hash = None
        self.pending_before_centroid = None
        self.pending_before_frame = None

    def choose_action(self, frames, latest_frame) -> GameAction:
        try:
            # Level transition: clear prior buffer (per-game) but keep policy counters
            level = self._level_of(latest_frame)
            if level != self.current_level:
                self.prior.reset()
                self.current_level = level
                print(f"[{self.game_id}] level transition → prior reset; lvl={level}")

            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self.pending_decision = None
                action = GameAction.RESET
                action.reasoning = "game needs reset"
                return action

            cur_frame_2d = primary_layer(latest_frame)
            if cur_frame_2d is not None and cur_frame_2d.shape == (GRID, GRID):
                self.prior.observe(cur_frame_2d)

            # Observe outcome of previous action
            self._observe_pending(latest_frame, cur_frame_2d)

            # Observe initial (after prior has its first frame)
            if not self.observed_initial:
                self.policy.observe_initial(latest_frame)
                self.observed_initial = True

            # Build current frame derived state
            bg = self.prior.bg_color()
            primary_cent = (
                primary_object_centroid(cur_frame_2d, bg)
                if cur_frame_2d is not None
                else None
            )
            objects = (
                extract_objects(cur_frame_2d, bg, max_objects=MAX_OBJECTS)
                if cur_frame_2d is not None
                else []
            )
            prior_arr = (
                self.prior.click_prior(cur_frame_2d)
                if cur_frame_2d is not None
                and cur_frame_2d.shape == (GRID, GRID)
                else np.ones((GRID, GRID), dtype=np.float32)
            )

            cur_hash = state_hash(latest_frame)
            decision = self.policy.select(
                latest_frame, cur_frame_2d, prior_arr, primary_cent, objects, cur_hash
            )
            if decision is None:
                # Fallback: random simple action
                action = GameAction.ACTION1
                action.reasoning = "fallback no candidates"
                return action

            # Remember pending so we can observe outcome on next call
            self.pending_decision = decision
            self.pending_before_hash = cur_hash
            self.pending_before_centroid = primary_cent
            self.pending_before_frame = (
                cur_frame_2d.copy() if cur_frame_2d is not None else None
            )

            # Build GameAction
            game_action = GameAction.from_id(decision.action_id)
            if decision.action_data:
                payload = {"game_id": getattr(self, "game_id", "")}
                payload.update(decision.action_data)
                game_action.set_data(payload)
            else:
                game_action.set_data({"game_id": getattr(self, "game_id", "")})
            game_action.reasoning = {
                "agent": "sg_tabular",
                "reason": decision.reason,
                "action_counter": getattr(self, "action_counter", 0),
            }
            return game_action

        except Exception as e:
            print(
                f"[{self.game_id}] choose_action crashed at step "
                f"{getattr(self, 'action_counter', '?')}: {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            action = GameAction.ACTION1
            action.reasoning = f"fallback after error: {e}"
            return action
