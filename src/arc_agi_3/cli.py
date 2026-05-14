"""Command-line tools for local ARC-AGI-3 experiments."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable


def _prepare_runtime_dirs() -> None:
    cache_dir = Path(".cache/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_prepare_runtime_dirs()

from arc_agi import Arcade, OperationMode  # noqa: E402

from .baseline import (  # noqa: E402
    EffectPriorPolicy,
    NoveltyPolicy,
    RandomPolicy,
    RunSummary,
    StepRecord,
    is_terminal_state,
    summarize_frame,
    state_hash,
)
from .effects import aggregate_action_effects, infer_primary_object_motion  # noqa: E402
from .events import (  # noqa: E402
    aggregate_object_events,
    frame_transition_summary,
    transition_utility,
)
from .features import frame_diff_summary  # noqa: E402
from .hypotheses import infer_action_hypotheses  # noqa: E402


def _operation_mode(raw: str) -> OperationMode:
    try:
        return OperationMode(raw)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in OperationMode)
        raise argparse.ArgumentTypeError(f"mode must be one of: {choices}") from exc


def _arcade_from_args(args: argparse.Namespace) -> Arcade:
    return Arcade(
        operation_mode=args.mode,
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
    )


def _write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    append: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def list_games(args: argparse.Namespace) -> int:
    arcade = _arcade_from_args(args)
    games = arcade.get_environments()
    rows: list[dict[str, Any]] = []
    for env in games[: args.limit if args.limit else None]:
        rows.append(
            {
                "game_id": env.game_id,
                "title": env.title,
                "default_fps": env.default_fps,
                "tags": env.tags or [],
                "local": env.local_dir is not None,
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("No environments found.")
        return 0

    for row in rows:
        local_flag = "local" if row["local"] else "remote"
        title = f" - {row['title']}" if row["title"] else ""
        print(f"{row['game_id']}\t{local_flag}{title}")
    return 0


def execute_policy_episode(
    arcade: Arcade,
    *,
    policy: Any,
    game_id: str,
    seed: int,
    max_steps: int,
    save_recording: bool,
    include_frame_data: bool,
    render_mode: str | None,
) -> tuple[RunSummary, list[StepRecord], Any | None]:
    env = arcade.make(
        game_id,
        seed=seed,
        save_recording=save_recording,
        include_frame_data=include_frame_data,
        render_mode=render_mode,
    )
    if env is None:
        raise RuntimeError(f"Failed to create environment for game_id={game_id!r}")

    frame_data = env.observation_space or env.reset()
    initial = summarize_frame(frame_data)
    observe_initial = getattr(policy, "observe_initial", None)
    if observe_initial is not None:
        observe_initial(frame_data)

    records: list[StepRecord] = []
    completed_steps = 0
    stopped_reason = "max_steps"
    for step_idx in range(max_steps):
        if is_terminal_state(frame_data):
            stopped_reason = "terminal_state"
            break

        before_frame = frame_data
        before_hash = state_hash(frame_data)
        decision = policy.select_action(frame_data, step_idx)
        if decision is None:
            stopped_reason = "no_available_actions"
            break

        reasoning = {
            "agent": policy.name,
            "seed": seed,
            "step": step_idx,
            "reason": decision.reason,
        }
        frame_data = env.step(
            decision.action,
            data=decision.action_data,
            reasoning=reasoning,
        )
        after_hash = state_hash(frame_data)
        diff_summary = frame_diff_summary(before_frame, frame_data)
        transition_summary = frame_transition_summary(
            before_frame,
            frame_data,
            diff_summary=diff_summary,
        )
        motion = infer_primary_object_motion(before_frame, frame_data)
        policy.observe_result(before_hash, decision, after_hash, motion, diff_summary)
        observe_frame = getattr(policy, "observe_frame", None)
        if observe_frame is not None:
            observe_frame(frame_data)
        if frame_data is None:
            stopped_reason = "step_failed"
            break

        step_summary = summarize_frame(frame_data)
        if step_summary is not None:
            records.append(
                StepRecord(
                    agent=policy.name,
                    game_id=step_summary.game_id,
                    seed=seed,
                    step=step_idx,
                    action=decision.action.name,
                    action_id=int(decision.action.value),
                    action_data=decision.action_data,
                    before_hash=before_hash,
                    after_hash=after_hash,
                    state=step_summary.state,
                    levels_completed=step_summary.levels_completed,
                    win_levels=step_summary.win_levels,
                    changed_cells=diff_summary.changed_cells,
                    changed_bbox=diff_summary.changed_bbox,
                    motion_dx=motion.dx if motion is not None else 0,
                    motion_dy=motion.dy if motion is not None else 0,
                    motion_direction=motion.direction if motion is not None else "",
                    motion_color=motion.color if motion is not None else None,
                    decision_reason=decision.reason,
                    level_delta=transition_summary.level_delta,
                    state_changed=transition_summary.state_changed,
                    action_utility=transition_utility(transition_summary),
                    object_event_count=len(transition_summary.object_events),
                    object_events=[
                        event.to_dict()
                        for event in transition_summary.object_events
                    ],
                )
            )
        completed_steps += 1

    if is_terminal_state(frame_data) and stopped_reason == "max_steps":
        stopped_reason = "terminal_state"

    final = summarize_frame(frame_data)
    summary = RunSummary(
        agent=policy.name,
        game_id=game_id,
        seed=seed,
        steps_attempted=max_steps,
        steps_completed=completed_steps,
        state=final.state if final else "NONE",
        levels_completed=final.levels_completed if final else 0,
        win_levels=final.win_levels if final else 0,
        stopped_reason=stopped_reason,
    )
    return summary, records, initial


def execute_random_episode(
    arcade: Arcade,
    *,
    game_id: str,
    seed: int,
    max_steps: int,
    allow_reset: bool,
    save_recording: bool,
    include_frame_data: bool,
    render_mode: str | None,
) -> tuple[RunSummary, list[StepRecord], Any | None]:
    return execute_policy_episode(
        arcade,
        policy=RandomPolicy(seed=seed, allow_reset=allow_reset),
        game_id=game_id,
        seed=seed,
        max_steps=max_steps,
        save_recording=save_recording,
        include_frame_data=include_frame_data,
        render_mode=render_mode,
    )


def execute_novelty_episode(
    arcade: Arcade,
    *,
    game_id: str,
    seed: int,
    max_steps: int,
    allow_reset: bool,
    coordinate_samples: int,
    save_recording: bool,
    include_frame_data: bool,
    render_mode: str | None,
) -> tuple[RunSummary, list[StepRecord], Any | None]:
    return execute_policy_episode(
        arcade,
        policy=NoveltyPolicy(
            seed=seed,
            allow_reset=allow_reset,
            coordinate_samples=coordinate_samples,
        ),
        game_id=game_id,
        seed=seed,
        max_steps=max_steps,
        save_recording=save_recording,
        include_frame_data=include_frame_data,
        render_mode=render_mode,
    )


def execute_effect_prior_episode(
    arcade: Arcade,
    *,
    game_id: str,
    seed: int,
    max_steps: int,
    allow_reset: bool,
    coordinate_samples: int,
    min_effect_samples: int,
    save_recording: bool,
    include_frame_data: bool,
    render_mode: str | None,
) -> tuple[RunSummary, list[StepRecord], Any | None]:
    return execute_policy_episode(
        arcade,
        policy=EffectPriorPolicy(
            seed=seed,
            allow_reset=allow_reset,
            coordinate_samples=coordinate_samples,
            min_effect_samples=min_effect_samples,
        ),
        game_id=game_id,
        seed=seed,
        max_steps=max_steps,
        save_recording=save_recording,
        include_frame_data=include_frame_data,
        render_mode=render_mode,
    )


def run_random(args: argparse.Namespace) -> int:
    arcade = _arcade_from_args(args)
    try:
        summary, records, initial = execute_random_episode(
            arcade,
            game_id=args.game_id,
            seed=args.seed,
            max_steps=args.max_steps,
            allow_reset=args.allow_reset,
            save_recording=args.save_recording,
            include_frame_data=args.include_frame_data,
            render_mode=args.render_mode,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2

    if args.print_initial and initial is not None:
        print(json.dumps(initial.__dict__, indent=2, sort_keys=True))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    if args.summary_output:
        _write_jsonl(args.summary_output, [summary.to_dict()], append=args.append)
    if args.steps_output:
        _write_jsonl(
            args.steps_output,
            (record.to_dict() for record in records),
            append=args.append,
        )

    scorecard = arcade.close_scorecard()
    if args.print_scorecard and scorecard is not None:
        print(scorecard.model_dump_json(indent=2))

    return 0 if summary.stopped_reason != "step_failed" else 3


def run_explore(args: argparse.Namespace) -> int:
    arcade = _arcade_from_args(args)
    try:
        summary, records, initial = execute_novelty_episode(
            arcade,
            game_id=args.game_id,
            seed=args.seed,
            max_steps=args.max_steps,
            allow_reset=args.allow_reset,
            coordinate_samples=args.coordinate_samples,
            save_recording=args.save_recording,
            include_frame_data=args.include_frame_data,
            render_mode=args.render_mode,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2

    if args.print_initial and initial is not None:
        print(json.dumps(initial.__dict__, indent=2, sort_keys=True))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    if args.summary_output:
        _write_jsonl(args.summary_output, [summary.to_dict()], append=args.append)
    if args.steps_output:
        _write_jsonl(
            args.steps_output,
            (record.to_dict() for record in records),
            append=args.append,
        )

    scorecard = arcade.close_scorecard()
    if args.print_scorecard and scorecard is not None:
        print(scorecard.model_dump_json(indent=2))

    return 0 if summary.stopped_reason != "step_failed" else 3


def run_effect_explore(args: argparse.Namespace) -> int:
    arcade = _arcade_from_args(args)
    try:
        summary, records, initial = execute_effect_prior_episode(
            arcade,
            game_id=args.game_id,
            seed=args.seed,
            max_steps=args.max_steps,
            allow_reset=args.allow_reset,
            coordinate_samples=args.coordinate_samples,
            min_effect_samples=args.min_effect_samples,
            save_recording=args.save_recording,
            include_frame_data=args.include_frame_data,
            render_mode=args.render_mode,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2

    if args.print_initial and initial is not None:
        print(json.dumps(initial.__dict__, indent=2, sort_keys=True))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    if args.summary_output:
        _write_jsonl(args.summary_output, [summary.to_dict()], append=args.append)
    if args.steps_output:
        _write_jsonl(
            args.steps_output,
            (record.to_dict() for record in records),
            append=args.append,
        )

    scorecard = arcade.close_scorecard()
    if args.print_scorecard and scorecard is not None:
        print(scorecard.model_dump_json(indent=2))

    return 0 if summary.stopped_reason != "step_failed" else 3


def run_batch(
    args: argparse.Namespace,
    *,
    episode_runner: Any,
) -> int:
    arcade = _arcade_from_args(args)
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    failures = 0

    for game_id in args.game_id:
        for seed in args.seed:
            try:
                runner_kwargs: dict[str, Any] = {}
                if hasattr(args, "coordinate_samples"):
                    runner_kwargs["coordinate_samples"] = args.coordinate_samples
                if hasattr(args, "min_effect_samples"):
                    runner_kwargs["min_effect_samples"] = args.min_effect_samples
                summary, records, _ = episode_runner(
                    arcade,
                    game_id=game_id,
                    seed=seed,
                    max_steps=args.max_steps,
                    allow_reset=args.allow_reset,
                    save_recording=args.save_recording,
                    include_frame_data=args.include_frame_data,
                    render_mode=None,
                    **runner_kwargs,
                )
                summaries.append(summary.to_dict())
                step_rows.extend(record.to_dict() for record in records)
                if summary.stopped_reason == "step_failed":
                    failures += 1
            except RuntimeError as exc:
                failures += 1
                summaries.append(
                    {
                        "agent": getattr(args, "agent_name", "unknown"),
                        "game_id": game_id,
                        "seed": seed,
                        "steps_attempted": args.max_steps,
                        "steps_completed": 0,
                        "state": "NONE",
                        "levels_completed": 0,
                        "win_levels": 0,
                        "stopped_reason": str(exc),
                    }
                )
            finally:
                arcade.close_scorecard()

    _write_jsonl(args.output, summaries, append=args.append)
    if args.steps_output:
        _write_jsonl(args.steps_output, step_rows, append=args.append)

    print(
        json.dumps(
            {
                "episodes": len(summaries),
                "failures": failures,
                "output": args.output,
                "steps_output": args.steps_output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failures == 0 else 4


def run_batch_random(args: argparse.Namespace) -> int:
    args.agent_name = "random"
    return run_batch(args, episode_runner=execute_random_episode)


def run_batch_explore(args: argparse.Namespace) -> int:
    args.agent_name = "novelty"
    return run_batch(args, episode_runner=execute_novelty_episode)


def run_batch_effect_explore(args: argparse.Namespace) -> int:
    args.agent_name = "effect-prior"
    return run_batch(args, episode_runner=execute_effect_prior_episode)


def analyze_steps(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.steps_jsonl:
        rows.extend(_read_jsonl(path))
    aggregates = aggregate_action_effects(rows)

    if args.json:
        print(json.dumps([item.to_dict() for item in aggregates], indent=2))
        return 0

    if not aggregates:
        print("No step rows found.")
        return 0

    for item in aggregates:
        print(
            "\t".join(
                [
                    item.action,
                    f"count={item.count}",
                    f"avg_changed={item.avg_changed_cells:.2f}",
                    f"motion={item.motion_count}",
                    f"avg_dx={item.avg_motion_dx:.2f}",
                    f"avg_dy={item.avg_motion_dy:.2f}",
                    f"dir={item.common_direction}",
                ]
            )
        )
    return 0


def analyze_events(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.steps_jsonl:
        rows.extend(_read_jsonl(path))
    aggregates = aggregate_object_events(rows)

    if args.json:
        print(json.dumps([item.to_dict() for item in aggregates], indent=2))
        return 0

    if not aggregates:
        print("No step rows found.")
        return 0

    for item in aggregates:
        print(
            "\t".join(
                [
                    item.action,
                    f"count={item.count}",
                    f"events={item.event_count}",
                    f"move={item.move_count}",
                    f"appear={item.appear_count}",
                    f"disappear={item.disappear_count}",
                    f"reshape={item.reshape_count}",
                    f"transform={item.transform_count}",
                    f"avg_utility={item.avg_utility:.2f}",
                    f"dir={item.common_move_direction}",
                ]
            )
        )
    return 0


def analyze_hypotheses(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.steps_jsonl:
        rows.extend(_read_jsonl(path))
    hypotheses = infer_action_hypotheses(rows, min_samples=args.min_samples)

    if args.json:
        print(json.dumps([item.to_dict() for item in hypotheses], indent=2))
        return 0

    if not hypotheses:
        print("No step rows found.")
        return 0

    for item in hypotheses:
        print(
            "\t".join(
                [
                    item.action,
                    f"samples={item.samples}",
                    f"avg_utility={item.avg_utility:.2f}",
                    f"progress={item.progress_count}",
                    f"events={item.event_count}",
                    f"event_rate={item.event_rate:.2f}",
                    f"dir={item.predicted_direction}",
                    f"dir_conf={item.direction_confidence:.2f}",
                    f"score={item.priority_score:.2f}",
                    f"use={item.recommended_use}",
                ]
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 experiment utilities")
    parser.add_argument(
        "--mode",
        type=_operation_mode,
        default=OperationMode.OFFLINE,
        help="Arcade operation mode: normal, online, offline, competition",
    )
    parser.add_argument(
        "--environments-dir",
        default="environment_files",
        help="Directory containing downloaded local environments",
    )
    parser.add_argument(
        "--recordings-dir",
        default="recordings",
        help="Directory for ARC scorecard recordings",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-games", help="List visible environments")
    list_parser.add_argument("--limit", type=int, default=0)
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_games)

    random_parser = subparsers.add_parser("random", help="Run a random baseline")
    random_parser.add_argument("game_id")
    random_parser.add_argument("--seed", type=int, default=0)
    random_parser.add_argument("--max-steps", type=int, default=100)
    random_parser.add_argument("--allow-reset", action="store_true")
    random_parser.add_argument("--save-recording", action="store_true")
    random_parser.add_argument("--include-frame-data", action="store_true")
    random_parser.add_argument(
        "--render-mode",
        choices=["terminal", "terminal-fast", "human"],
        default=None,
    )
    random_parser.add_argument("--print-initial", action="store_true")
    random_parser.add_argument("--print-scorecard", action="store_true")
    random_parser.add_argument("--summary-output", default=None)
    random_parser.add_argument("--steps-output", default=None)
    random_parser.add_argument("--append", action="store_true")
    random_parser.set_defaults(func=run_random)

    explore_parser = subparsers.add_parser(
        "explore",
        help="Run a state-novelty exploration baseline",
    )
    explore_parser.add_argument("game_id")
    explore_parser.add_argument("--seed", type=int, default=0)
    explore_parser.add_argument("--max-steps", type=int, default=100)
    explore_parser.add_argument("--allow-reset", action="store_true")
    explore_parser.add_argument("--coordinate-samples", type=int, default=8)
    explore_parser.add_argument("--save-recording", action="store_true")
    explore_parser.add_argument("--include-frame-data", action="store_true")
    explore_parser.add_argument(
        "--render-mode",
        choices=["terminal", "terminal-fast", "human"],
        default=None,
    )
    explore_parser.add_argument("--print-initial", action="store_true")
    explore_parser.add_argument("--print-scorecard", action="store_true")
    explore_parser.add_argument("--summary-output", default=None)
    explore_parser.add_argument("--steps-output", default=None)
    explore_parser.add_argument("--append", action="store_true")
    explore_parser.set_defaults(func=run_explore)

    effect_parser = subparsers.add_parser(
        "effect-explore",
        help="Run novelty exploration with online action-effect priors",
    )
    effect_parser.add_argument("game_id")
    effect_parser.add_argument("--seed", type=int, default=0)
    effect_parser.add_argument("--max-steps", type=int, default=100)
    effect_parser.add_argument("--allow-reset", action="store_true")
    effect_parser.add_argument("--coordinate-samples", type=int, default=8)
    effect_parser.add_argument("--min-effect-samples", type=int, default=2)
    effect_parser.add_argument("--save-recording", action="store_true")
    effect_parser.add_argument("--include-frame-data", action="store_true")
    effect_parser.add_argument(
        "--render-mode",
        choices=["terminal", "terminal-fast", "human"],
        default=None,
    )
    effect_parser.add_argument("--print-initial", action="store_true")
    effect_parser.add_argument("--print-scorecard", action="store_true")
    effect_parser.add_argument("--summary-output", default=None)
    effect_parser.add_argument("--steps-output", default=None)
    effect_parser.add_argument("--append", action="store_true")
    effect_parser.set_defaults(func=run_effect_explore)

    batch_parser = subparsers.add_parser(
        "batch-random",
        help="Run random baseline episodes and write JSONL summaries",
    )
    batch_parser.add_argument("game_id", nargs="+")
    batch_parser.add_argument("--seed", type=int, nargs="+", default=[0])
    batch_parser.add_argument("--max-steps", type=int, default=100)
    batch_parser.add_argument("--allow-reset", action="store_true")
    batch_parser.add_argument("--save-recording", action="store_true")
    batch_parser.add_argument("--include-frame-data", action="store_true")
    batch_parser.add_argument(
        "--output",
        default="experiments/runs/random-baseline.jsonl",
        help="JSONL file for episode summaries",
    )
    batch_parser.add_argument(
        "--steps-output",
        default=None,
        help="Optional JSONL file for per-step replay summaries",
    )
    batch_parser.add_argument("--append", action="store_true")
    batch_parser.set_defaults(func=run_batch_random)

    batch_explore_parser = subparsers.add_parser(
        "batch-explore",
        help="Run novelty exploration episodes and write JSONL summaries",
    )
    batch_explore_parser.add_argument("game_id", nargs="+")
    batch_explore_parser.add_argument("--seed", type=int, nargs="+", default=[0])
    batch_explore_parser.add_argument("--max-steps", type=int, default=100)
    batch_explore_parser.add_argument("--allow-reset", action="store_true")
    batch_explore_parser.add_argument("--coordinate-samples", type=int, default=8)
    batch_explore_parser.add_argument("--save-recording", action="store_true")
    batch_explore_parser.add_argument("--include-frame-data", action="store_true")
    batch_explore_parser.add_argument(
        "--output",
        default="experiments/runs/novelty-baseline.jsonl",
        help="JSONL file for episode summaries",
    )
    batch_explore_parser.add_argument(
        "--steps-output",
        default=None,
        help="Optional JSONL file for per-step replay summaries",
    )
    batch_explore_parser.add_argument("--append", action="store_true")
    batch_explore_parser.set_defaults(func=run_batch_explore)

    batch_effect_parser = subparsers.add_parser(
        "batch-effect-explore",
        help="Run effect-prior exploration episodes and write JSONL summaries",
    )
    batch_effect_parser.add_argument("game_id", nargs="+")
    batch_effect_parser.add_argument("--seed", type=int, nargs="+", default=[0])
    batch_effect_parser.add_argument("--max-steps", type=int, default=100)
    batch_effect_parser.add_argument("--allow-reset", action="store_true")
    batch_effect_parser.add_argument("--coordinate-samples", type=int, default=8)
    batch_effect_parser.add_argument("--min-effect-samples", type=int, default=2)
    batch_effect_parser.add_argument("--save-recording", action="store_true")
    batch_effect_parser.add_argument("--include-frame-data", action="store_true")
    batch_effect_parser.add_argument(
        "--output",
        default="experiments/runs/effect-prior-baseline.jsonl",
        help="JSONL file for episode summaries",
    )
    batch_effect_parser.add_argument(
        "--steps-output",
        default=None,
        help="Optional JSONL file for per-step replay summaries",
    )
    batch_effect_parser.add_argument("--append", action="store_true")
    batch_effect_parser.set_defaults(func=run_batch_effect_explore)

    analyze_parser = subparsers.add_parser(
        "analyze-steps",
        help="Aggregate action-effect statistics from step JSONL files",
    )
    analyze_parser.add_argument("steps_jsonl", nargs="+")
    analyze_parser.add_argument("--json", action="store_true")
    analyze_parser.set_defaults(func=analyze_steps)

    analyze_events_parser = subparsers.add_parser(
        "analyze-events",
        help="Aggregate object event statistics from step JSONL files",
    )
    analyze_events_parser.add_argument("steps_jsonl", nargs="+")
    analyze_events_parser.add_argument("--json", action="store_true")
    analyze_events_parser.set_defaults(func=analyze_events)

    analyze_hypotheses_parser = subparsers.add_parser(
        "analyze-hypotheses",
        help="Infer action hypotheses from object event step JSONL files",
    )
    analyze_hypotheses_parser.add_argument("steps_jsonl", nargs="+")
    analyze_hypotheses_parser.add_argument("--json", action="store_true")
    analyze_hypotheses_parser.add_argument("--min-samples", type=int, default=2)
    analyze_hypotheses_parser.set_defaults(func=analyze_hypotheses)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
