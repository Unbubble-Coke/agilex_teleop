#!/usr/bin/env python3
"""Debug FK/IK for one reachable Nero target."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.kinematics.debug_tools import (
    DEFAULT_NERO_JOINT_LIMITS,
    clip_to_joint_limits,
    joint_limit_violation,
    pose_errors,
    sample_random_q,
)
from nero.kinematics.solver_debug_adapter import make_debug_solver


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=("pinocchio", "original"),
        default="pinocchio",
        help="Solver to debug: Pinocchio_Solver or original Solver.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q", type=float, nargs=7, default=None, help="Target joint vector")
    parser.add_argument("--q-init", type=float, nargs=7, default=None, help="IK initial joint vector")
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--n-psi", type=int, default=181, help="Arm-angle grid size for original Solver.")
    parser.add_argument("--pos-tol", type=float, default=1e-3)
    parser.add_argument("--ori-tol", type=float, default=1e-2)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    try:
        solver = make_debug_solver(
            args.solver,
            joint_limits=DEFAULT_NERO_JOINT_LIMITS,
            dt=0.05,
            n_psi=args.n_psi,
            max_iterations=args.max_iters,
            tol_pos=1e-5,
            tol_rot=1e-4,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{args.solver} solver dependencies are missing. "
            "For Pinocchio_Solver install: pip install -e '.[dynamics]'."
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    q_target = (
        np.asarray(args.q, dtype=float)
        if args.q is not None
        else sample_random_q(rng, DEFAULT_NERO_JOINT_LIMITS, num_samples=1, margin=0.1)[0]
    )
    q_target = clip_to_joint_limits(q_target, DEFAULT_NERO_JOINT_LIMITS)
    q_init = (
        np.asarray(args.q_init, dtype=float)
        if args.q_init is not None
        else clip_to_joint_limits(q_target + rng.normal(0.0, 0.03, size=7), DEFAULT_NERO_JOINT_LIMITS)
    )

    target_T = solver.fk_matrix(q_target)
    target_pose = solver.fk_pose(q_target)

    solver.init_state(q_init)
    start = time.perf_counter()
    q_solution = solver.solve(target_pose, limit_output_step=False)
    latency_ms = (time.perf_counter() - start) * 1000.0

    pos_err = None
    ori_err = None
    if q_solution is not None:
        pos_err, ori_err = pose_errors(solver.fk_matrix(q_solution), target_T)

    result = {
        "solver": "Pinocchio_Solver" if args.solver == "pinocchio" else "Solver",
        "urdf_path": solver.urdf_path,
        "ee_frame": solver.ee_frame_name,
        "joint_names": solver.joint_names,
        "q_target": q_target.tolist(),
        "target_pose_xyz_rpy": target_pose.tolist(),
        "q_init": q_init.tolist(),
        "q_solution": None if q_solution is None else q_solution.tolist(),
        "position_error": pos_err,
        "orientation_error": ori_err,
        "within_position_tolerance": None if pos_err is None else bool(pos_err <= args.pos_tol),
        "within_orientation_tolerance": None if ori_err is None else bool(ori_err <= args.ori_tol),
        "joint_limit_violation": joint_limit_violation(q_solution, DEFAULT_NERO_JOINT_LIMITS),
        "latency_ms": latency_ms,
        "solver_report": solver.last_report,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
