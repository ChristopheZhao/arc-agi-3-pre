"""Run StochasticGoose verbatim from ref/ARC3-solution.

Path α reproduction: clones, applies the two README monkey-patches if missing,
then `make install && make action`. Stops on first failure with a hint pointing
at docs/reproduce-sg.md path β.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "ref" / "ARC3-solution"
SUBMODULE = REPO / "ARC-AGI-3-Agents"
INIT_PATCH = """\
# --- StochasticGoose patch (added by scripts/run_sg.py) ---
import sys as _sg_sys
import os as _sg_os
_sg_sys.path.append(
    _sg_os.path.dirname(
        _sg_os.path.dirname(
            _sg_os.path.dirname(_sg_os.path.abspath(__file__))
        )
    )
)
from custom_agent import *  # noqa: F401,F403,E402
# --- end patch ---
"""

STRUCTS_PATCH_LINE = "    available_actions: list[GameAction] = Field(default_factory=list)"


def patch_init() -> None:
    p = SUBMODULE / "agents" / "__init__.py"
    text = p.read_text()
    if "StochasticGoose patch" in text:
        return
    p.write_text(INIT_PATCH + text)
    print(f"  patched: {p}")


def patch_structs() -> None:
    p = SUBMODULE / "agents" / "structs.py"
    text = p.read_text()
    if "available_actions" in text:
        return
    # Insert after `full_reset: bool = False` line in FrameData
    needle = "    full_reset: bool = False"
    if needle not in text:
        print(f"  WARN: could not find anchor in {p}; you may need to patch structs.py manually")
        return
    text = text.replace(needle, needle + "\n" + STRUCTS_PATCH_LINE)
    p.write_text(text)
    print(f"  patched: {p}")


def write_env() -> None:
    key = os.getenv("ARC_API_KEY") or _read_dotenv()
    if not key:
        print("ARC_API_KEY not set in env or ../.env -- aborting")
        print("Get one at https://three.arcprize.org/user")
        sys.exit(2)
    env_file = SUBMODULE / ".env"
    env_file.write_text(f"ARC_API_KEY={key}\n")
    print(f"  wrote: {env_file}")


def _read_dotenv() -> str | None:
    p = REPO.parent.parent / ".env"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("ARC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}  (in {cwd})")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"\n[!] {' '.join(cmd)} failed (rc={r.returncode}).")
        print("    See docs/reproduce-sg.md path β if this is a schema-drift error.")
        sys.exit(r.returncode)


def main() -> None:
    if not REPO.exists():
        print(f"missing {REPO}; clone it first:")
        print("  git clone --recurse-submodules https://github.com/DriesSmit/ARC3-solution ref/ARC3-solution")
        sys.exit(1)
    if not (SUBMODULE / "agents").exists():
        print("submodule not initialized; run:")
        print(f"  cd {REPO} && git submodule update --init --recursive")
        sys.exit(1)
    if shutil.which("uv") is None:
        print("uv not found in PATH"); sys.exit(1)
    if shutil.which("make") is None:
        print("make not found in PATH"); sys.exit(1)

    print("=== applying SG monkey-patches ===")
    patch_init()
    patch_structs()
    write_env()

    print("\n=== make install ===")
    run(["make", "install"], cwd=REPO)

    print("\n=== make action (Ctrl+C to stop training) ===")
    run(["make", "action"], cwd=REPO)


if __name__ == "__main__":
    main()
