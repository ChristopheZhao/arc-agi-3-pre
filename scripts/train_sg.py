"""Sanity-train SG baseline against a single ARC-AGI-3 game.

Not a benchmark run: this is to verify the port works end-to-end on CPU.
For real numbers we need a GPU.

Usage:
    uv run python scripts/train_sg.py                         # default cn04, 500 steps
    uv run python scripts/train_sg.py --game ft09 --steps 2000
    uv run python scripts/train_sg.py --no-reset-on-level     # don't wipe model on level-up
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make our src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from src.agents.sg_baseline import SGBaselineAgent

load_dotenv()

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger("train_sg")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", type=str, default=None, help="game_id prefix (e.g. cn04); default first available")
    p.add_argument("--steps", type=int, default=500, help="max env steps after first reset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--train-freq", type=int, default=5)
    p.add_argument("--log-every", type=int, default=25, help="log progress every N steps")
    p.add_argument("--no-reset-on-level", action="store_true",
                   help="don't wipe model+buffer on level-up (vs SG default)")
    p.add_argument("--no-coord-prior", action="store_true",
                   help="disable the static-pixel + background-color click prior")
    p.add_argument("--tag", type=str, default="", help="extra scorecard tag")
    args = p.parse_args()

    if not os.getenv("ARC_API_KEY"):
        LOG.error("ARC_API_KEY not set"); sys.exit(2)

    arc = Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    if not envs:
        LOG.error("No envs available"); sys.exit(2)
    target = next((e for e in envs if (args.game or "") in e.game_id), envs[0])
    LOG.info("game = %s  (baseline_actions per level = %s)",
             target.game_id, getattr(target, "baseline_actions", None))

    tags = ["sg-baseline-sanity", args.device]
    if not args.no_coord_prior:
        tags.append("coord-prior")
    if args.no_reset_on_level:
        tags.append("no-level-reset")
    if args.tag:
        tags.append(args.tag)
    card_id = arc.create_scorecard(tags=tags)
    env = arc.make(target.game_id, seed=args.seed, scorecard_id=card_id)
    if env is None:
        LOG.error("arcade.make returned None"); sys.exit(2)

    agent = SGBaselineAgent(
        device=args.device,
        train_frequency=args.train_freq,
        use_coord_prior=not args.no_coord_prior,
        seed=args.seed,
    )
    LOG.info("agent on device=%s, model params=%d",
             agent.device, sum(p.numel() for p in agent.model.parameters()))

    frame = env.reset()
    LOG.info("reset: state=%s  levels=%d/%d  available=%s",
             frame.state.name, frame.levels_completed, frame.win_levels, frame.available_actions)

    prev_state_t = None
    prev_action_idx = -1
    t0 = time.time()
    last_stats = None
    pos_seen = 0
    neg_seen = 0

    for step in range(args.steps):
        agent.maybe_reset_for_new_level(frame, reset_model=not args.no_reset_on_level)

        # observe transition for the previous action (if any)
        if prev_state_t is not None and prev_action_idx >= 0:
            agent.observe(prev_state_t, prev_action_idx, frame)
            r = agent.experience_buffer[-1]["reward"] if agent.experience_buffer else 0
            pos_seen += int(r > 0)
            neg_seen += int(r == 0)

        # snapshot current state before stepping
        cur_state_t = agent.frame_to_tensor(frame) if frame.state == GameState.NOT_FINISHED else None

        action, data, action_idx = agent.choose_action(frame)
        new_frame = env.step(action, data=data, reasoning={"src": "sg-baseline"})
        if new_frame is None:
            LOG.warning("env returned None at step %d", step); break

        # train periodically
        if step > 0 and step % args.train_freq == 0:
            stats = agent.train_step()
            if stats is not None:
                last_stats = stats

        if step % args.log_every == 0 or new_frame.state in (GameState.WIN, GameState.GAME_OVER):
            elapsed = time.time() - t0
            fps = (step + 1) / max(elapsed, 1e-3)
            ls = last_stats
            buf = len(agent.experience_buffer)
            prior_frac = (agent.segmenter.fraction_active() if agent.segmenter else 1.0)
            LOG.info(
                "step %4d %4.1fs %5.1f fps  L=%d/%d  buf=%d  pos=%d neg=%d  "
                "prior_active=%.2f  act=%s  loss=%s  acc=%s",
                step, elapsed, fps,
                new_frame.levels_completed, new_frame.win_levels, buf,
                pos_seen, neg_seen, prior_frac,
                action.name + (f"({data['x']},{data['y']})" if data else ""),
                f"{ls.main_loss:.3f}" if ls else "-",
                f"{ls.accuracy:.2f}" if ls else "-",
            )

        prev_state_t = cur_state_t
        prev_action_idx = action_idx
        frame = new_frame

        if frame.state == GameState.WIN:
            LOG.info("WIN at step %d, breaking", step)
            break
        # GAME_OVER: agent will issue RESET on next choose_action and keep training

    card = arc.close_scorecard(card_id)
    LOG.info("=== final scorecard ===")
    if card is not None:
        LOG.info("total score=%s", card.score)
        for env_entry in card.environments:
            LOG.info("env %s — %d run(s)", env_entry.id, len(env_entry.runs))
            for i, run in enumerate(env_entry.runs):
                LOG.info(
                    "  run %d  state=%s  actions=%d  resets=%d  levels=%d  "
                    "lvl_actions=%s  lvl_scores=%s  baseline=%s",
                    i, run.state.name, run.actions, run.resets, run.levels_completed,
                    run.level_actions,
                    [round(s, 3) for s in run.level_scores],
                    run.level_baseline_actions,
                )


if __name__ == "__main__":
    main()
