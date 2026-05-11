"""Benchmark SG baseline across many ARC-AGI-3 games with per-game time budgets.

Designed for cloud GPU runs:
  uv run python scripts/benchmark.py --device cuda --budget-min 30 --tag run-v1

Iterates games in catalog order (or --games filter). Per game:
  - resets the agent (fresh model + fresh segmenter)
  - plays until time budget exhausted OR scorecard.completed=True
  - dumps per-game JSON + appends a top-level results.json

Resumable: if results.json already has an entry for game_id (with the same tag),
that game is skipped.

Outputs to runs/<tag>/:
  results.json          — list of per-game summaries
  <game_id>.scorecard.json — full scorecard
  <game_id>.steps.jsonl   — one line per step (sampled)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from src.agents.sg_baseline import SGBaselineAgent

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger("benchmark")


def play_one(arc: Arcade, agent: SGBaselineAgent, game_id: str, budget_s: float,
             card_id: str, log_every: int, steps_path: Path) -> dict:
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {"game_id": game_id, "error": "make returned None"}

    frame = env.reset()
    LOG.info("[%s] reset state=%s available=%s baseline=?", game_id, frame.state.name, frame.available_actions)

    prev_state_t = None
    prev_action_idx = -1
    t0 = time.time()
    step = 0
    levels_max = frame.levels_completed

    with steps_path.open("w") as steps_f:
        while True:
            elapsed = time.time() - t0
            if elapsed >= budget_s:
                LOG.info("[%s] budget %.0fs exhausted after %d steps", game_id, budget_s, step)
                break
            if frame.state == GameState.WIN:
                LOG.info("[%s] WIN at step %d", game_id, step)
                break

            agent.maybe_reset_for_new_level(frame, reset_model=False)  # cross-level keep model

            if prev_state_t is not None and prev_action_idx >= 0:
                agent.observe(prev_state_t, prev_action_idx, frame)

            cur_state_t = agent.frame_to_tensor(frame) if frame.state == GameState.NOT_FINISHED else None
            action, data, action_idx = agent.choose_action(frame)
            new_frame = env.step(action, data=data, reasoning={"src": "benchmark"})
            if new_frame is None:
                LOG.warning("[%s] env returned None at step %d", game_id, step); break

            stats = None
            if step > 0 and step % agent.train_frequency == 0:
                stats = agent.train_step()

            if step % log_every == 0:
                LOG.info(
                    "[%s] step %4d %5.0fs L=%d/%d buf=%d act=%s loss=%s",
                    game_id, step, elapsed, new_frame.levels_completed, new_frame.win_levels,
                    len(agent.experience_buffer),
                    action.name + (f"({data['x']},{data['y']})" if data else ""),
                    f"{stats.main_loss:.3f}" if stats else "-",
                )

            steps_f.write(json.dumps({
                "step": step, "elapsed": round(elapsed, 2),
                "action": action.name, "data": data,
                "state": new_frame.state.name,
                "levels_completed": new_frame.levels_completed,
                "loss": stats.main_loss if stats else None,
            }) + "\n")

            prev_state_t = cur_state_t
            prev_action_idx = action_idx
            frame = new_frame
            levels_max = max(levels_max, frame.levels_completed)
            step += 1

    return {
        "game_id": game_id, "steps": step, "wallclock_s": round(time.time() - t0, 1),
        "final_state": frame.state.name, "levels_completed": levels_max,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=str, default=None, help="comma-separated game_id prefixes; default = all")
    p.add_argument("--budget-min", type=float, default=30.0, help="minutes per game")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="benchmark", help="run tag (used for output dir)")
    p.add_argument("--no-coord-prior", action="store_true")
    p.add_argument("--segment-prior", action="store_true",
                   help="layer in 4-connected segment-equalization (dolphin-style action-space compression)")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    if not os.getenv("ARC_API_KEY"):
        LOG.error("ARC_API_KEY not set"); sys.exit(2)

    out_dir = Path("runs") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    done = {}
    if results_path.exists():
        done = {r["game_id"]: r for r in json.loads(results_path.read_text())}
        LOG.info("found %d existing results in %s; will skip those", len(done), results_path)

    arc = Arcade(operation_mode=OperationMode.NORMAL)
    LOG.info("arcade mode=%s base_url=%s", arc.operation_mode.value, arc.arc_base_url)
    envs = arc.get_environments()
    LOG.info("catalog: %d games", len(envs))

    selected = []
    if args.games:
        prefixes = [s.strip() for s in args.games.split(",")]
        selected = [e for e in envs if any(e.game_id.startswith(p) for p in prefixes)]
    else:
        selected = list(envs)
    LOG.info("will play %d games (after filter), budget %.1f min each", len(selected), args.budget_min)

    card_id = arc.create_scorecard(tags=[args.tag, args.device, "sg-bench"])
    LOG.info("scorecard: %s", card_id)

    results = list(done.values())
    for env_info in selected:
        gid = env_info.game_id
        if gid in done:
            LOG.info("[%s] skip (already done in %s)", gid, results_path); continue
        LOG.info("=== %s (baseline=%s) ===", gid, getattr(env_info, "baseline_actions", None))
        agent = SGBaselineAgent(
            device=args.device,
            use_coord_prior=not args.no_coord_prior,
            enable_segment_prior=args.segment_prior,
            seed=args.seed,
        )
        steps_path = out_dir / f"{gid}.steps.jsonl"
        summary = play_one(arc, agent, gid, args.budget_min * 60, card_id, args.log_every, steps_path)
        results.append(summary)
        results_path.write_text(json.dumps(results, indent=2))
        LOG.info("[%s] -> %s", gid, summary)

    card = arc.close_scorecard(card_id)
    if card is not None:
        (out_dir / "scorecard.json").write_text(card.model_dump_json(indent=2))
        LOG.info("=== final scorecard total = %s ===", card.score)
        for env_entry in card.environments:
            for run in env_entry.runs:
                LOG.info("%s  state=%s actions=%d levels=%d  scores=%s",
                         env_entry.id, run.state.name, run.actions, run.levels_completed,
                         [round(s, 3) for s in run.level_scores])


if __name__ == "__main__":
    main()
