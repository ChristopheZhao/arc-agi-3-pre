"""Print the SDK surface — works without ARC_API_KEY."""

from __future__ import annotations

import inspect

import arc_agi
import arcengine
from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState


def main() -> None:
    print("=" * 60)
    print("arc_agi public:", sorted(n for n in dir(arc_agi) if not n.startswith("_")))
    print("arcengine public:", sorted(n for n in dir(arcengine) if not n.startswith("_")))

    print("\n=== OperationMode ===")
    for m in OperationMode:
        print(" ", m.name, "=", repr(m.value))

    print("\n=== GameAction ===")
    for a in GameAction:
        print(f"  {a.name:7s} id={a.value}  type={'complex' if a.is_complex() else 'simple'}")

    print("\n=== GameState ===")
    for s in GameState:
        print(" ", s.name, "=", repr(s.value))

    print("\n=== FrameData fields ===")
    for name, field in FrameData.model_fields.items():
        print(f"  {name:20s} {field.annotation}  default={field.default!r}")

    print("\n=== Arcade.make signature ===")
    print(" ", inspect.signature(Arcade.make))

    print("\n=== Trying OFFLINE mode (no API key needed) ===")
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    envs = arc.get_environments()
    print(f"  Found {len(envs)} local environments under {arc.environments_dir!r}")
    for e in envs[:10]:
        print("   -", e)

    if not envs:
        print("\n  (Empty — drop game folders into ./environment_files/<game_id>/<version>/")
        print("   to play offline. Otherwise set ARC_API_KEY and use NORMAL/ONLINE mode.)")


if __name__ == "__main__":
    main()
