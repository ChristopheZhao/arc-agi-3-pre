"""Benchmark the offline BFS solver across many ARC-AGI-3 games.

Per game:
  1. arc.make(game_id) + env.reset() (downloads game .py if not cached)
  2. find_game_source_and_class on the now-cached file
  3. BFSSolver.load() + solve_level(L) for each L in [0, win_levels)
  4. replay the discovered action sequence through env.step
  5. record per-level outcome + final scorecard

Per-game wall-clock cap: max_levels * (scan_timeout + bfs_timeout) + slack.

Resumable: re-run skips games already present in runs/<tag>/results.json.

Usage:
    uv run python scripts/bfs_benchmark.py --tag bfs-25demo \\
        --max-levels 5 --scan-timeout 5 --bfs-timeout 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from src.agents.bfs_solver import BFSSolver, find_game_source_and_class

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger("bfs_benchmark")


def solve_one(
    arc: Arcade,
    game_id: str,
    card_id: str,
    scan_to: float,
    bfs_to: float,
    max_levels: int,
    hash_mode: str,
    dense_scan: bool,
) -> dict:
    """Solve and replay one game. Returns a summary dict."""
    summary: dict = {"game_id": game_id, "started_at": time.time()}
    t_total = time.time()

    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        summary["error"] = "arc.make returned None"
        return summary
    frame = env.reset()
    summary["win_levels"] = int(getattr(frame, "win_levels", 0))
    summary["initial_state"] = frame.state.name

    src, cls = find_game_source_and_class(game_id)
    if src is None or cls is None:
        summary["error"] = "source/class not found after make+reset"
        summary["wallclock_s"] = round(time.time() - t_total, 1)
        return summary
    summary["source"] = str(src)
    summary["class"] = cls

    solver = BFSSolver(
        src, cls,
        scan_timeout=scan_to,
        bfs_timeout=bfs_to,
        hash_mode=hash_mode,
        dense_scan=dense_scan,
    )
    if not solver.load():
        summary["error"] = "solver load failed"
        summary["wallclock_s"] = round(time.time() - t_total, 1)
        return summary

    levels: list[dict] = []
    n_lvls = min(summary["win_levels"] or 1, max_levels)
    for L in range(n_lvls):
        bfs_res = solver.solve_level(L)
        lvl: dict = {
            "level": L,
            "reason": bfs_res.reason,
            "explored": bfs_res.explored,
            "visited": bfs_res.visited,
            "elapsed_s": round(bfs_res.elapsed_s, 1),
            "bfs_steps": None if bfs_res.actions is None else len(bfs_res.actions),
        }
        if bfs_res.actions is None:
            lvl["replay"] = "skipped"
            levels.append(lvl)
            break

        # replay through live env
        replay_steps = 0
        replay_status = "ok"
        for act_id, data in bfs_res.actions:
            try:
                new_frame = env.step(
                    GameAction.from_id(act_id),
                    data=data,
                    reasoning={"src": "bfs"},
                )
            except Exception as e:
                replay_status = f"step_exc:{type(e).__name__}"
                break
            if new_frame is None:
                replay_status = "env_returned_none"
                break
            replay_steps += 1
            frame = new_frame
            if frame.state in (GameState.WIN, GameState.GAME_OVER):
                break
        lvl["replay_steps"] = replay_steps
        lvl["replay_status"] = replay_status
        lvl["post_state"] = frame.state.name
        lvl["post_levels_completed"] = int(frame.levels_completed)
        if frame.levels_completed <= L and replay_status == "ok":
            lvl["replay_status"] = "did_not_advance"
        LOG.info(
            "[%s] L%d: bfs=%s/%s replay=%s post=%s/L=%d",
            game_id, L, bfs_res.reason, lvl["bfs_steps"],
            lvl["replay_status"], frame.state.name, frame.levels_completed,
        )
        levels.append(lvl)
        if lvl["replay_status"] != "ok":
            break
        if frame.state == GameState.WIN:
            break

    summary["levels"] = levels
    summary["max_levels_solved"] = int(frame.levels_completed)
    summary["final_state"] = frame.state.name
    summary["wallclock_s"] = round(time.time() - t_total, 1)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=str, default=None, help="comma-separated game_id prefixes; default = all")
    p.add_argument("--tag", type=str, default="bfs-bench", help="run tag (output dir)")
    p.add_argument("--max-levels", type=int, default=5, help="max levels to attempt per game")
    p.add_argument("--scan-timeout", type=float, default=5.0)
    p.add_argument("--bfs-timeout", type=float, default=60.0)
    p.add_argument("--hash-mode", type=str, default="frame", choices=["frame", "full"])
    p.add_argument("--no-dense-scan", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="stop after N games (0 = no limit)")
    args = p.parse_args()

    if not os.getenv("ARC_API_KEY"):
        LOG.error("ARC_API_KEY not set"); sys.exit(2)

    out_dir = Path("runs") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    done: dict[str, dict] = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text()):
            done[r["game_id"]] = r
        LOG.info("found %d existing results in %s; will skip those", len(done), results_path)

    # NORMAL constructor lets env vars (OPERATION_MODE, ARC_BASE_URL) win — Kaggle
    # sets OPERATION_MODE=online + ARC_BASE_URL=http://gateway:8001/ at submission time.
    arc = Arcade(operation_mode=OperationMode.NORMAL)
    LOG.info("arcade mode=%s base_url=%s", arc.operation_mode.value, arc.arc_base_url)
    envs = arc.get_environments()
    LOG.info("catalog: %d games", len(envs))

    if args.games:
        prefixes = [s.strip() for s in args.games.split(",")]
        selected = [e for e in envs if any(e.game_id.startswith(p) for p in prefixes)]
    else:
        selected = list(envs)
    if args.limit:
        selected = selected[: args.limit]
    LOG.info("will play %d games (after filter/limit)", len(selected))

    card_id = arc.create_scorecard(tags=[args.tag, "bfs"])
    LOG.info("scorecard: %s", card_id)

    results: list[dict] = list(done.values())
    for env_info in selected:
        gid = env_info.game_id
        if gid in done:
            LOG.info("[%s] skip (already in %s)", gid, results_path); continue
        LOG.info("=== %s (baseline=%s) ===", gid, getattr(env_info, "baseline_actions", None))
        try:
            summary = solve_one(
                arc, gid, card_id,
                scan_to=args.scan_timeout, bfs_to=args.bfs_timeout,
                max_levels=args.max_levels, hash_mode=args.hash_mode,
                dense_scan=not args.no_dense_scan,
            )
        except Exception as e:
            LOG.exception("[%s] crashed", gid)
            summary = {"game_id": gid, "error": f"{type(e).__name__}: {e}"}
        results.append(summary)
        results_path.write_text(json.dumps(results, indent=2))
        LOG.info("[%s] wrote summary -> %s", gid, results_path)

    card = arc.close_scorecard(card_id)
    if card is not None:
        (out_dir / "scorecard.json").write_text(card.model_dump_json(indent=2))
        LOG.info("=== final scorecard total = %s ===", card.score)
        for env_entry in card.environments:
            for run in env_entry.runs:
                LOG.info(
                    "%s state=%s actions=%d levels=%d scores=%s",
                    env_entry.id, run.state.name, run.actions, run.levels_completed,
                    [round(s, 3) for s in run.level_scores],
                )


if __name__ == "__main__":
    main()
