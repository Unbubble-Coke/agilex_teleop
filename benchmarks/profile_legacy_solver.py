#!/usr/bin/env python3
"""Profile the legacy Nero analytic ``Solver`` without changing solver math."""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import functools
import io
import json
import math
import pstats
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.kinematics.debug_tools import (  # noqa: E402
    DEFAULT_NERO_JOINT_LIMITS,
    clip_to_joint_limits,
    sample_random_q,
    scalar_stats,
)
from nero.kinematics.solver_debug_adapter import make_debug_solver  # noqa: E402


@dataclass
class ReachableSample:
    q_target: np.ndarray
    target_pose: np.ndarray
    q_init: np.ndarray


class StageProfiler:
    """Small exclusive-time profiler for coarse legacy IK stages."""

    def __init__(self) -> None:
        self.stage_ms: Dict[str, List[float]] = defaultdict(list)
        self._active = False
        self._stack: List[Tuple[str, float]] = []
        self._current_s: Dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def call(self):
        if self._active:
            yield
            return
        self._active = True
        self._stack = []
        self._current_s = defaultdict(float)
        try:
            yield
        finally:
            while self._stack:
                name, start = self._stack.pop()
                self._current_s[name] += time.perf_counter() - start
            self._active = False

    @contextlib.contextmanager
    def stage(self, name: str):
        if not self._active:
            yield
            return

        now = time.perf_counter()
        if self._stack:
            parent_name, parent_start = self._stack[-1]
            self._current_s[parent_name] += now - parent_start
            self._stack[-1] = (parent_name, now)

        self._stack.append((name, now))
        try:
            yield
        finally:
            end = time.perf_counter()
            stage_name, stage_start = self._stack.pop()
            self._current_s[stage_name] += end - stage_start
            if self._stack:
                parent_name, _ = self._stack[-1]
                self._stack[-1] = (parent_name, end)

    def finish_call(self, total_ms: float) -> None:
        accounted_ms = 0.0
        for name, elapsed_s in self._current_s.items():
            elapsed_ms = elapsed_s * 1000.0
            self.stage_ms[name].append(elapsed_ms)
            accounted_ms += elapsed_ms
        self.stage_ms["unattributed_python_overhead"].append(max(0.0, total_ms - accounted_ms))


@contextlib.contextmanager
def instrument_legacy_stages(stage_profiler: StageProfiler):
    """Temporarily wrap coarse functions so stage timing stays outside production code."""

    from nero.kinematics import analytic_IK_solver as analytic
    from nero.kinematics.nero_kinematics.nero_ik import ik_solver as legacy

    patches = [
        (analytic.Solver, "_pose_to_matrix", "parse_input"),
        (analytic.Solver, "_clamp_joints", "joint_limit_check"),
        (analytic.Solver, "_detect_and_guard_output", "output_guard"),
        (analytic, "solve_pose_continuous_with_state", "core_solver_overhead"),
        (legacy, "ik_arm_angle_with_report", "global_fallback_overhead"),
        (legacy, "_scan_theta0_solutions", "candidate_generation"),
        (legacy, "_collect_unique_solutions_for_theta0_grid", "candidate_generation"),
        (legacy, "_best_weighted_from_cached", "choose_solution"),
        (legacy, "_optimize_q_with_1d_qp", "qp_refinement"),
        (legacy, "_within_limits", "candidate_filtering"),
        (legacy, "fk", "fk_validation"),
        (legacy, "pose_error", "fk_validation"),
    ]

    originals = []
    for owner, attr, stage_name in patches:
        original = getattr(owner, attr)

        @functools.wraps(original)
        def wrapped(*args, __original=original, __stage_name=stage_name, **kwargs):
            with stage_profiler.stage(__stage_name):
                return __original(*args, **kwargs)

        originals.append((owner, attr, original))
        setattr(owner, attr, wrapped)

    try:
        yield
    finally:
        for owner, attr, original in reversed(originals):
            setattr(owner, attr, original)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-psi", type=int, default=61)
    parser.add_argument(
        "--sort-by",
        default="cumulative",
        help="pstats sort key, for example cumulative, tottime, calls, or name.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/legacy_solver_profile.txt"),
        help="Text report path. A JSON sidecar with the same stem is also written.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to the text output path with .json suffix.",
    )
    return parser.parse_args()


def make_original_solver(n_psi: int):
    return make_debug_solver(
        "original",
        joint_limits=DEFAULT_NERO_JOINT_LIMITS,
        dt=0.05,
        n_psi=n_psi,
        max_iterations=80,
        tol_pos=1e-5,
        tol_rot=1e-4,
    )


def try_make_pinocchio_solver(n_psi: int):
    try:
        solver = make_debug_solver(
            "pinocchio",
            joint_limits=DEFAULT_NERO_JOINT_LIMITS,
            dt=0.05,
            n_psi=n_psi,
            max_iterations=80,
            tol_pos=1e-5,
            tol_rot=1e-4,
        )
        return solver, None
    except Exception as exc:  # Pinocchio is optional for this legacy-only profile.
        return None, f"{type(exc).__name__}: {exc}"


def timed_solver_init(n_psi: int):
    start = time.perf_counter()
    solver = make_original_solver(n_psi)
    return solver, (time.perf_counter() - start) * 1000.0


def build_reachable_samples(solver, rng: np.random.Generator, num_samples: int) -> List[ReachableSample]:
    samples: List[ReachableSample] = []
    q_targets = sample_random_q(rng, DEFAULT_NERO_JOINT_LIMITS, num_samples=num_samples, margin=0.08)
    for q_target in q_targets:
        target_pose = solver.fk_pose(q_target)
        q_init = clip_to_joint_limits(
            q_target + rng.normal(0.0, 0.03, size=q_target.shape),
            DEFAULT_NERO_JOINT_LIMITS,
        )
        samples.append(
            ReachableSample(
                q_target=np.asarray(q_target, dtype=float),
                target_pose=np.asarray(target_pose, dtype=float),
                q_init=np.asarray(q_init, dtype=float),
            )
        )
    return samples


def run_legacy_solves(
    solver,
    samples: Iterable[ReachableSample],
    profiler: Optional[cProfile.Profile] = None,
    stage_profiler: Optional[StageProfiler] = None,
):
    latencies_ms: List[float] = []
    reports = []
    success_count = 0

    for sample in samples:
        solver.init_state(sample.q_init)
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            if stage_profiler is None:
                q_solution = solver.solve(sample.target_pose, limit_output_step=False)
            else:
                with stage_profiler.call():
                    q_solution = solver.solve(sample.target_pose, limit_output_step=False)
        finally:
            if profiler is not None:
                profiler.disable()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if stage_profiler is not None:
            stage_profiler.finish_call(latency_ms)

        latencies_ms.append(latency_ms)
        report = dict(solver.last_report or {})
        reports.append(report)
        if q_solution is not None:
            success_count += 1

    return {
        "latencies_ms": latencies_ms,
        "reports": reports,
        "success_count": success_count,
    }


def summarize_stage_breakdown(stage_profiler: StageProfiler, total_latencies_ms: List[float]):
    mean_total = float(np.mean(total_latencies_ms)) if total_latencies_ms else float("nan")
    breakdown = {}
    for name, values in sorted(stage_profiler.stage_ms.items()):
        stats = scalar_stats(values)
        mean_ms = stats["mean"]
        breakdown[name] = {
            "mean_ms": mean_ms,
            "median_ms": stats["median"],
            "p95_ms": stats["p95"],
            "p99_ms": stats["p99"],
            "max_ms": stats["max"],
            "percent_of_total": (
                float(100.0 * mean_ms / mean_total)
                if mean_total and math.isfinite(mean_total) and math.isfinite(mean_ms)
                else float("nan")
            ),
        }
    return breakdown


def format_pstats(profile: cProfile.Profile, sort_by: str, top_k: int) -> str:
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(sort_by)
    stats.print_stats(top_k)
    return stream.getvalue()


def extract_hotspots(profile: cProfile.Profile, sort_by: str, top_k: int):
    stats = pstats.Stats(profile).strip_dirs()
    entries = []
    for (filename, line, func_name), stat in stats.stats.items():
        primitive_calls, total_calls, total_time_s, cumulative_time_s, _ = stat
        if sort_by in {"time", "tottime"}:
            sort_value = total_time_s
        elif sort_by in {"calls", "ncalls"}:
            sort_value = total_calls
        else:
            sort_value = cumulative_time_s
        entries.append(
            {
                "function": f"{filename}:{line}({func_name})",
                "primitive_calls": int(primitive_calls),
                "call_count": int(total_calls),
                "tottime_ms": float(total_time_s * 1000.0),
                "cumtime_ms": float(cumulative_time_s * 1000.0),
                "percall_ms": (
                    float(cumulative_time_s * 1000.0 / total_calls)
                    if total_calls
                    else float("nan")
                ),
                "_sort_value": sort_value,
            }
        )
    entries.sort(key=lambda item: item["_sort_value"], reverse=True)
    for item in entries:
        item.pop("_sort_value", None)
    return entries[:top_k]


def summarize_reports(reports):
    method_counts = Counter(str(report.get("method", "unknown")) for report in reports)
    candidate_counts = [
        float(report["candidate_count"])
        for report in reports
        if report.get("candidate_count") is not None
    ]
    return {
        "method_counts": dict(method_counts),
        "candidate_count_stats": scalar_stats(candidate_counts),
    }


def make_text_report(results, pstats_text: str) -> str:
    latency = results["latency_stats"]
    stage_lines = []
    for name, item in sorted(
        results["stage_breakdown"].items(),
        key=lambda kv: kv[1]["mean_ms"],
        reverse=True,
    ):
        stage_lines.append(
            f"{name:30s} mean={item['mean_ms']:.6f} ms "
            f"p95={item['p95_ms']:.6f} ms "
            f"p99={item['p99_ms']:.6f} ms "
            f"max={item['max_ms']:.6f} ms "
            f"percent={item['percent_of_total']:.2f}%"
        )
    stage_block = "\n".join(stage_lines) if stage_lines else "(no stage timings)"

    return "\n".join(
        [
            "# Legacy Solver Profiling Report",
            "",
            f"solver: {results['solver']}",
            f"num_samples: {results['num_samples']}",
            f"python: {results['python_executable']} ({results['python_version']})",
            f"pinocchio_solver_constructed: {results['pinocchio_solver_constructed']}",
            f"pinocchio_solver_error: {results['pinocchio_solver_error']}",
            "",
            "## Latency",
            f"init_time_ms: {results['init_time_ms']:.6f}",
            f"cold_import_and_init_time_ms: {results['cold_import_and_init_time_ms']:.6f}",
            f"mean_total_ms: {results['mean_total_ms']:.6f}",
            f"median_total_ms: {latency['median']:.6f}",
            f"p95_total_ms: {latency['p95']:.6f}",
            f"p99_total_ms: {latency['p99']:.6f}",
            f"max_total_ms: {latency['max']:.6f}",
            f"success_rate: {results['success_rate']:.6f}",
            "",
            "## Report Summary",
            json.dumps(results["report_summary"], indent=2, sort_keys=True),
            "",
            "## Stage Timing",
            stage_block,
            "",
            "## cProfile Top Functions",
            pstats_text.rstrip(),
            "",
            "## JSON Summary",
            json.dumps(results, indent=2, sort_keys=True),
            "",
        ]
    )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    init_solver, cold_import_and_init_time_ms = timed_solver_init(args.n_psi)
    warm_init_times_ms = []
    for _ in range(5):
        _, warm_init_time_ms = timed_solver_init(args.n_psi)
        warm_init_times_ms.append(warm_init_time_ms)
    pin_solver, pin_error = try_make_pinocchio_solver(args.n_psi)
    samples = build_reachable_samples(init_solver, rng, args.num_samples)

    baseline_solver = make_original_solver(args.n_psi)
    baseline_run = run_legacy_solves(baseline_solver, samples)

    stage_solver = make_original_solver(args.n_psi)
    stage_profiler = StageProfiler()
    with instrument_legacy_stages(stage_profiler):
        stage_run = run_legacy_solves(
            stage_solver,
            samples,
            stage_profiler=stage_profiler,
        )

    profile_solver = make_original_solver(args.n_psi)
    profile = cProfile.Profile()
    profile_run = run_legacy_solves(profile_solver, samples, profiler=profile)

    pstats_text = format_pstats(profile, args.sort_by, args.top_k)
    latency_stats = scalar_stats(baseline_run["latencies_ms"])
    results = {
        "solver": "Solver",
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "n_psi": int(args.n_psi),
        "init_time_ms": float(np.mean(warm_init_times_ms)),
        "init_time_stats": scalar_stats(warm_init_times_ms),
        "cold_import_and_init_time_ms": float(cold_import_and_init_time_ms),
        "mean_total_ms": latency_stats["mean"],
        "latency_stats": latency_stats,
        "instrumented_mean_total_ms": float(np.mean(stage_run["latencies_ms"])),
        "instrumented_latency_stats": scalar_stats(stage_run["latencies_ms"]),
        "cprofile_latency_stats": scalar_stats(profile_run["latencies_ms"]),
        "success_count": int(baseline_run["success_count"]),
        "success_rate": float(baseline_run["success_count"] / max(1, args.num_samples)),
        "stage_breakdown": summarize_stage_breakdown(
            stage_profiler,
            stage_run["latencies_ms"],
        ),
        "hotspots": extract_hotspots(profile, args.sort_by, args.top_k),
        "report_summary": summarize_reports(baseline_run["reports"]),
        "pinocchio_solver_constructed": pin_solver is not None,
        "pinocchio_solver_error": pin_error,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
    }

    output_path = args.output
    json_output = args.json_output or output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(make_text_report(results, pstats_text), encoding="utf-8")
    json_output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
