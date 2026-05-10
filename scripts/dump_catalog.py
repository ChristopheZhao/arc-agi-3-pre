"""Print the public env catalog: id, tags, level count, human baseline actions.

Doesn't play any game — just lists what the API exposes via Arcade.get_environments().
"""

from __future__ import annotations

from dotenv import load_dotenv

from arc_agi import Arcade, OperationMode

load_dotenv()


def main() -> None:
    arc = Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    print(f"Total environments: {len(envs)}\n")
    print(f"{'game_id':30s}  {'levels':>6s}  {'tags':30s}  baseline_actions (per level)")
    print("-" * 110)
    for e in envs:
        d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
        gid = d.get("game_id", "?")
        tags = ",".join(d.get("tags") or []) or "-"
        ba = d.get("baseline_actions") or []
        print(f"{gid:30s}  {len(ba):>6d}  {tags:30s}  {ba}")


if __name__ == "__main__":
    main()
