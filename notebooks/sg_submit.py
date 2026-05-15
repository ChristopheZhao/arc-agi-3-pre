"""Kaggle submission scaffold for sg_kaggle_agent.

Mirrors notebooks/arc_agi_3_submission.py from the 0.15 reference: this
self-contained script handles wheels install (when invoked with that flag),
dataset/runner discovery, gateway readiness wait, runner repo setup, agent
registration, and main.py launch. Edit-mode writes a dummy submission.parquet.

The companion notebook (submit_sg.ipynb) is intentionally a 3-cell thin shell
that installs wheels and delegates here via runpy.run_path so logic updates
ship through dataset version bumps without touching notebook cell content.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMPETITION_BUNDLE = "arc-prize-2026-arc-agi-3"
DEFAULT_GATEWAY_URL = "http://gateway:8001"
AGENT_MARKER = "src/agents/sg_kaggle_agent.py"
RUNNER_MARKER = "agents/agent.py"


def find_repo_src() -> Path | None:
    """Locate attached project dataset root via env var, fixed candidates, or recursive glob."""
    explicit = os.environ.get("ARC_AGI_3_PRE_ROOT")
    if explicit:
        p = Path(explicit)
        if (p / AGENT_MARKER).exists():
            return p
    input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    candidates = [
        input_root / "arc-agi-3-pre",
        input_root / "arc-agi-3-pre-v1",
        input_root / "arc-agi-3-pre-repo",
    ]
    for cand in candidates:
        if (cand / AGENT_MARKER).exists():
            return cand
    # Scheme-agnostic fallback: handles /kaggle/input/datasets/<user>/<slug>/
    # and any other layout that exposes <repo>/src/agents/sg_kaggle_agent.py
    for marker in input_root.glob(f"**/{AGENT_MARKER}"):
        return marker.parent.parent.parent
    return None


def find_agents_repo() -> Path | None:
    """Locate the official ARC-AGI-3-Agents runner."""
    explicit = os.environ.get("ARC_AGI_3_AGENTS_ROOT")
    if explicit:
        p = Path(explicit)
        if (p / RUNNER_MARKER).exists():
            return p
    input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    candidates = [
        input_root / "competitions" / COMPETITION_BUNDLE / "ARC-AGI-3-Agents",
        input_root / "arc-agi-3-agents",
    ]
    for cand in candidates:
        if (cand / RUNNER_MARKER).exists():
            return cand
    for marker in input_root.glob(f"**/{RUNNER_MARKER}"):
        return marker.parent.parent
    return None


def find_wheels_dir() -> Path | None:
    explicit = os.environ.get("ARC_AGI_3_WHEELS_DIR")
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p
    input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    cand = input_root / "competitions" / COMPETITION_BUNDLE / "arc_agi_3_wheels"
    return cand if cand.is_dir() else None


def install_competition_wheels() -> None:
    wheels_dir = find_wheels_dir()
    if wheels_dir is None:
        raise RuntimeError(
            "Could not find official arc-agi-3 wheels under "
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
        )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index", "--find-links", str(wheels_dir),
            "arc-agi", "python-dotenv",
        ],
        check=True,
    )


def in_kaggle_competition_rerun() -> bool:
    return bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))


def wait_for_gateway(timeout_seconds: int = 600) -> None:
    if not in_kaggle_competition_rerun():
        return
    url = f"{DEFAULT_GATEWAY_URL.rstrip('/')}/api/games"
    deadline = time.monotonic() + timeout_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
        time.sleep(5)
    raise RuntimeError(f"ARC gateway not ready at {url}: {last_err}")


def setup_and_run(repo_src: Path, agents_repo: Path) -> None:
    """Copy runner, install our agent, rewrite __init__, write .env, launch main.py."""
    working = Path("/kaggle/working")
    working.mkdir(parents=True, exist_ok=True)

    runner_dst = working / "ARC-AGI-3-Agents"
    if not runner_dst.is_dir():
        subprocess.run(["cp", "-r", str(agents_repo), str(runner_dst)], check=True)

    my_agent_src = repo_src / AGENT_MARKER
    my_agent_dst = runner_dst / "agents" / "templates" / "my_agent.py"
    subprocess.run(["cp", str(my_agent_src), str(my_agent_dst)], check=True)

    # Rewrite agents/__init__.py to skip eager LLM-template imports that would
    # fail in Kaggle (no langgraph/smolagents installed). Register only what we use.
    (runner_dst / "agents" / "__init__.py").write_text(
        'from typing import Type\n'
        'from dotenv import load_dotenv\n'
        'from .agent import Agent, Playback\n'
        'from .swarm import Swarm\n'
        'from .templates.random_agent import Random\n'
        'from .templates.my_agent import MyAgent\n'
        '\n'
        'load_dotenv()\n'
        '\n'
        'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n'
        '    "random": Random,\n'
        '    "myagent": MyAgent,\n'
        '}\n'
    )

    # Point runner at the in-pod gateway
    (runner_dst / ".env").write_text(
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    )

    env = os.environ.copy()
    env["MPLBACKEND"] = "agg"
    subprocess.run(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=str(runner_dst),
        env=env,
        check=True,
    )


def run_submission() -> None:
    repo_src = find_repo_src()
    if repo_src is None:
        input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
        print("Mounted /kaggle/input entries:")
        if input_root.exists():
            for child in sorted(input_root.iterdir()):
                print(" -", child)
        raise RuntimeError(
            f"Could not find {AGENT_MARKER} under /kaggle/input. "
            f"Attach the arc-agi-3-pre dataset to this notebook."
        )
    print(f"Using repo source: {repo_src}")

    agents_repo = find_agents_repo()
    if agents_repo is None:
        raise RuntimeError("Could not find ARC-AGI-3-Agents runner under /kaggle/input.")
    print(f"Using agents repo: {agents_repo}")

    wait_for_gateway()
    setup_and_run(repo_src, agents_repo)


def write_dummy_submission() -> None:
    import pandas as pd

    output_path = Path(
        os.environ.get(
            "SUBMISSION_OUTPUT",
            "/kaggle/working/submission.parquet"
            if Path("/kaggle/working").exists()
            else "submission.parquet",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        data=[["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    )
    submission.to_parquet(output_path, index=False)
    print(f"Wrote dummy submission to {output_path}")


def dry_run_imports() -> None:
    repo_src = find_repo_src()
    if repo_src is None:
        raise RuntimeError(
            f"Could not find {AGENT_MARKER} under /kaggle/input."
        )
    print(f"Dry-run: repo_src = {repo_src}")
    agents_repo = find_agents_repo()
    print(f"Dry-run: agents_repo = {agents_repo}")
    wheels_dir = find_wheels_dir()
    print(f"Dry-run: wheels_dir = {wheels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 sg_kaggle_agent Kaggle submission")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve paths only; do not install, wait, or run")
    parser.add_argument("--install-competition-wheels", action="store_true",
                        help="Install arc-agi + python-dotenv from competition wheel bundle")
    args = parser.parse_args()

    if args.install_competition_wheels:
        install_competition_wheels()

    if args.dry_run:
        dry_run_imports()
        return

    if not in_kaggle_competition_rerun():
        write_dummy_submission()
        return

    run_submission()


if __name__ == "__main__":
    main()
