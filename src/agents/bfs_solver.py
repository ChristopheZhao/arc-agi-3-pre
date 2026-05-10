"""Offline BFS solver — instantiate game class from source, search via deepcopy.

The decisive insight from the FORGE notebook (LB 0.39, 137 votes):
ARC-AGI-3 game source files (.py) are downloaded as data; the contest is NOT
a black-box API contest. We can `importlib`-load each game's class, then
deepcopy the live game object and call `perform_action` directly to step
through state space — no server roundtrips.

This is the CPU-only path. A 64×64 grid with ~10 effective actions and
~30 reachable states per level finds the optimal action sequence in
milliseconds-to-seconds with plain BFS.

Compared to the SG bandit, this is qualitatively different: we are not
*learning* a policy from sparse reward, we are *searching* for the perfect
action sequence in a fully-known deterministic MDP and replaying it.

Implements minimum FORGE-style BFS first; advanced tricks (hidden-field
probing, cross-level transfer, warm-up unlock) are TODO for v2.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cloudpickle
import numpy as np

from arcengine import ActionInput, GameAction

LOG = logging.getLogger(__name__)


def find_game_source_and_class(
    game_id: str,
    environment_files_dir: str | Path = "environment_files",
) -> tuple[Optional[Path], Optional[str]]:
    """Locate game .py and resolve its `class X(ARCBaseGame)` name.

    `game_id` may be either short (e.g. "cn04") or "<short>-<version>"
    (e.g. "cn04-2fe56bfb"). The Arcade SDK lays files under
    `environment_files/<short>/<version>/<short>.py`.

    Returns (path, class_name). Either may be None if not found.
    """
    short = game_id.split("-")[0]
    base = Path(environment_files_dir)

    # Try direct: environment_files/<short>/<version>/<short>.py
    versions = sorted((base / short).glob("*")) if (base / short).is_dir() else []
    candidate_paths = [v / f"{short}.py" for v in versions if (v / f"{short}.py").is_file()]
    # Fallback: any */<short>.py under base (recursive, shallow)
    if not candidate_paths:
        candidate_paths = list(base.glob(f"**/{short}.py"))

    if not candidate_paths:
        return None, None

    src_path = candidate_paths[0]
    # Some game files are huge (lp85.py is 21k lines). Don't truncate before
    # the regex or we'll miss the class declaration.
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^class\s+(\w+)\s*\(\s*ARCBaseGame", text, flags=re.MULTILINE)
    cls_name = m.group(1) if m else short.capitalize()
    return src_path, cls_name


@dataclass
class BFSResult:
    actions: Optional[list[tuple[int, Optional[dict]]]]   # action sequence; None on failure
    explored: int                                          # nodes expanded
    visited: int                                           # unique states seen
    elapsed_s: float
    reason: str                                            # "solved" / "timeout" / "max_states" / "no_actions"


class BFSSolver:
    """Per-game BFS solver. Loads the game class once, solves levels on demand."""

    def __init__(
        self,
        game_path: str | Path,
        class_name: str,
        scan_timeout: float = 3.0,
        bfs_timeout: float = 60.0,
        max_states: int = 500_000,
        max_depth: int = 30,
        click_step: int = 2,
        hash_mode: str = "frame",     # "frame" | "full"
    ) -> None:
        self.game_path = Path(game_path)
        self.class_name = class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.max_states = max_states
        self.max_depth = max_depth
        self.click_step = click_step
        if hash_mode not in ("frame", "full"):
            raise ValueError(f"hash_mode must be 'frame' or 'full', got {hash_mode!r}")
        self.hash_mode = hash_mode
        self.game_cls = None
        self.solutions: dict[int, list[tuple[int, Optional[dict]]]] = {}

    def load(self) -> bool:
        """importlib-load the game module. Returns True on success."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"_arc_game_{self.class_name}", str(self.game_path)
            )
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            LOG.warning("BFS load failed: %s", e)
            return False

    # ---- helpers ----

    @staticmethod
    def _frame_hash(frame: np.ndarray) -> str:
        return hashlib.md5(frame.tobytes()).hexdigest()[:16]

    @staticmethod
    def _full_hash(game, frame: np.ndarray) -> str:
        """cloudpickle the whole game object — slow (~3 ms) but captures sprite
        positions and any non-scalar internal state that frame hashing misses."""
        return hashlib.md5(cloudpickle.dumps(game)).hexdigest()[:16]

    def _hash(self, game, frame: np.ndarray) -> str:
        return self._full_hash(game, frame) if self.hash_mode == "full" else self._frame_hash(frame)

    @staticmethod
    def _to_2d(raw_frame) -> np.ndarray:
        """Convert FrameDataRaw.frame[-1] to (64, 64) int64 grid."""
        arr = np.array(raw_frame, dtype=np.int64)
        return arr

    def _fresh_game_at_level(self, level_idx: int):
        """Instantiate a fresh game object positioned at `level_idx` after one RESET."""
        g = self.game_cls()
        g.set_level(level_idx)
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        # In FORGE, two RESETs are issued. The first is a no-op pass-through; the
        # second is the canonical level start. We follow that convention.
        r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        return g, r

    def _scan_actions(self, game, f0: np.ndarray, bg_color: int) -> list[tuple[int, Optional[dict]]]:
        """Probe each available simple action and step-2 click positions. Return actions
        that visibly change the frame. Click effects are de-duplicated by post-effect
        frame hash so 'every pixel of the same button' counts once."""
        actions: list[tuple[int, Optional[dict]]] = []
        avail = list(game._available_actions)

        # Simple actions ACTION1..5
        for a in [a for a in avail if 1 <= a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(self._to_2d(r.frame[-1]) != f0) > 0:
                    actions.append((a, None))
            except Exception:
                pass

        # ACTION6 click probing
        if 6 in avail:
            t0 = time.time()
            seen_effects: set[str] = set()
            for y in range(0, 64, self.click_step):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, self.click_step):
                    if f0[y, x] == bg_color:
                        continue          # skip background pixels for speed
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={"x": x, "y": y, "game_id": "bfs"}),
                            raw=True,
                        )
                        if not r.frame:
                            continue
                        f_after = self._to_2d(r.frame[-1])
                        if np.sum(f_after != f0) == 0:
                            continue
                        h = self._frame_hash(f_after)
                        if h in seen_effects:
                            continue
                        seen_effects.add(h)
                        actions.append((6, {"x": x, "y": y, "game_id": "bfs"}))
                    except Exception:
                        pass

        return actions

    def solve_level(self, level_idx: int) -> BFSResult:
        """Plain frame-hash BFS. Returns BFSResult with action list on success."""
        if not self.game_cls:
            return BFSResult(None, 0, 0, 0.0, "not_loaded")

        t_total = time.time()

        try:
            game, r0 = self._fresh_game_at_level(level_idx)
        except Exception as e:
            LOG.warning("L%d: fresh-game setup failed: %s", level_idx, e)
            return BFSResult(None, 0, 0, time.time() - t_total, "setup_failed")
        if not r0.frame:
            return BFSResult(None, 0, 0, time.time() - t_total, "no_initial_frame")

        f0 = self._to_2d(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Phase 1: scan actions
        actions = self._scan_actions(game, f0, bg)
        LOG.info("L%d: %d effective actions (after %.2fs scan)", level_idx, len(actions), time.time() - t_total)
        if not actions:
            # Warmup unlock (FORGE v18 trick): some levels start in a locked state
            # where no action visibly does anything. Try each ACTION1..5 once, see
            # if it unlocks a state where actions become visible.
            avail = list(game._available_actions)
            for a in [a for a in avail if 1 <= a <= 4]:
                g_warm = copy.deepcopy(game)
                try:
                    g_warm.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                    f_after = self._to_2d(g_warm.get_pixels(0, 0, 64, 64))
                    bg2 = int(np.bincount(f_after.flatten(), minlength=16).argmax())
                    warm_actions = self._scan_actions(g_warm, f_after, bg2)
                    if warm_actions:
                        LOG.info("L%d: UNLOCKED with ACTION%d → %d actions",
                                 level_idx, a, len(warm_actions))
                        # Treat ACTION%d as the first move; restart BFS from g_warm
                        game, f0, actions = g_warm, f_after, warm_actions
                        break
                except Exception:
                    pass
            if not actions:
                return BFSResult(None, 0, 0, time.time() - t_total, "no_actions")

        # Phase 2: BFS
        visited = {self._hash(game, f0)}
        queue: deque = deque()
        queue.append((copy.deepcopy(game), [], 0))
        explored = 0
        t_bfs = time.time()

        while queue and explored < self.max_states and (time.time() - t_bfs) < self.bfs_timeout:
            g, hist, depth = queue.popleft()

            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data \
                         else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except Exception:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = self._to_2d(r.frame[-1])
                h = self._hash(g2, f)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]

                # Win detection: either response field bumped, or game's internal
                # level index advanced (FORGE checks both — sometimes only one fires
                # depending on whether the action triggered an explicit next_level()).
                won = r.levels_completed > level_idx
                if not won:
                    cli = getattr(g2, "_current_level_index", None)
                    if cli is not None and cli > level_idx:
                        won = True
                if won:
                    elapsed = time.time() - t_total
                    LOG.info("L%d: SOLVED in %d actions (%d explored, %d visited, %.1fs)",
                             level_idx, len(new_hist), explored, len(visited), elapsed)
                    self.solutions[level_idx] = new_hist
                    return BFSResult(new_hist, explored, len(visited), elapsed, "solved")

                if depth + 1 < self.max_depth:
                    queue.append((g2, new_hist, depth + 1))

        elapsed = time.time() - t_total
        reason = ("max_states" if explored >= self.max_states
                  else "timeout" if (time.time() - t_bfs) >= self.bfs_timeout
                  else "queue_empty")
        LOG.info("L%d: NOT solved (%s) — explored=%d visited=%d %.1fs",
                 level_idx, reason, explored, len(visited), elapsed)
        return BFSResult(None, explored, len(visited), elapsed, reason)
