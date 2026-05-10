"""Minimal random-action episode against a public ARC-AGI-3 game.

Requires ARC_API_KEY in the environment (or .env). Falls back to a clear
error message if no games are available.

Usage:
    uv run python scripts/quickstart.py                # picks first game
    uv run python scripts/quickstart.py --game ls20    # specific game
    uv run python scripts/quickstart.py --steps 50
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from dotenv import load_dotenv

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

load_dotenv()


def random_action() -> tuple[GameAction, dict | None]:
    a = random.choice([x for x in GameAction if x is not GameAction.RESET])
    if a.is_complex():
        return a, {"x": random.randint(0, 63), "y": random.randint(0, 63)}
    return a, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=str, default=None, help="game_id; default = first available")
    parser.add_argument("--steps", type=int, default=20, help="max steps to take after reset")
    parser.add_argument("--seed", type=int, default=0, help="env seed")
    args = parser.parse_args()

    if not os.getenv("ARC_API_KEY"):
        print("ARC_API_KEY not set — copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(2)

    random.seed(args.seed)

    arcade = Arcade(operation_mode=OperationMode.NORMAL)
    envs = arcade.get_environments()
    if not envs:
        print("No environments available. Check API key.", file=sys.stderr)
        sys.exit(2)

    game_ids = [e.game_id for e in envs]
    print(f"Available games ({len(game_ids)}):")
    for gid in game_ids[:10]:
        print(" -", gid)
    if len(game_ids) > 10:
        print(f"   ... +{len(game_ids) - 10} more")

    target = args.game or game_ids[0]
    if target not in game_ids:
        print(f"\nGame {target!r} not in catalog; pick one of the above.", file=sys.stderr)
        sys.exit(2)

    print(f"\n>>> Playing {target!r} for {args.steps} random steps")
    card_id = arcade.create_scorecard(tags=["quickstart", "random"])
    env = arcade.make(target, seed=args.seed, scorecard_id=card_id)
    if env is None:
        print("arcade.make returned None — game not loadable.", file=sys.stderr)
        sys.exit(2)

    frame = env.reset()
    print(f"  reset: state={frame.state.name}  levels={frame.levels_completed}/{frame.win_levels}"
          f"  available={frame.available_actions}")

    for i in range(args.steps):
        action, data = random_action()
        frame = env.step(action, data=data, reasoning={"src": "quickstart-random"})
        if frame is None:
            print(f"  step {i}: env returned None, stopping")
            break
        n_subframes = len(frame.frame) if hasattr(frame, "frame") else 0
        print(f"  step {i:3d}: {action.name:8s} data={data} -> "
              f"state={frame.state.name:13s} L={frame.levels_completed}/{frame.win_levels} "
              f"frames={n_subframes}")
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            print(f"  -> terminal state {frame.state.name}, breaking")
            break

    card = arcade.close_scorecard(card_id)
    if card is not None:
        print(f"\nScorecard {card.card_id}:  total score = {card.score}")
        for env_score in card.environments[:5]:
            print(" ", env_score)


if __name__ == "__main__":
    main()
