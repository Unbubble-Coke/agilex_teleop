import numpy as np
from nero.kinematics.debug_tools import (
    DEFAULT_NERO_JOINT_LIMITS,
    joint_limit_violation,
    pose_errors,
    sample_random_q,
)


def test_joint_limit_helper_detects_violations():
    assert not joint_limit_violation(np.mean(DEFAULT_NERO_JOINT_LIMITS, axis=1), DEFAULT_NERO_JOINT_LIMITS)

    too_low = np.mean(DEFAULT_NERO_JOINT_LIMITS, axis=1)
    too_low[0] = DEFAULT_NERO_JOINT_LIMITS[0, 0] - 0.01
    assert joint_limit_violation(too_low, DEFAULT_NERO_JOINT_LIMITS)

    too_high = np.mean(DEFAULT_NERO_JOINT_LIMITS, axis=1)
    too_high[-1] = DEFAULT_NERO_JOINT_LIMITS[-1, 1] + 0.01
    assert joint_limit_violation(too_high, DEFAULT_NERO_JOINT_LIMITS)


def test_solver_clamps_and_returns_solutions_inside_joint_limits(solver_factory):
    rng = np.random.default_rng(13)
    solver = solver_factory(max_iterations=120, n_psi=181)

    below_limits = DEFAULT_NERO_JOINT_LIMITS[:, 0] - 1.0
    above_limits = DEFAULT_NERO_JOINT_LIMITS[:, 1] + 1.0
    mixed = np.where(np.arange(7) % 2 == 0, below_limits, above_limits)
    clamped = solver.clamp_joints(mixed)
    assert not joint_limit_violation(clamped, DEFAULT_NERO_JOINT_LIMITS)

    q_target = sample_random_q(rng, DEFAULT_NERO_JOINT_LIMITS, num_samples=1, margin=0.1)[0]
    target_T = solver.fk_matrix(q_target)
    target_pose = solver.fk_pose(q_target)
    solver.init_state(q_target)

    q_solution = solver.solve(target_pose, limit_output_step=False)

    assert q_solution is not None, solver.last_report
    assert not joint_limit_violation(q_solution, DEFAULT_NERO_JOINT_LIMITS)
    pos_err, ori_err = pose_errors(solver.fk_matrix(q_solution), target_T)
    assert pos_err < 1e-3
    assert ori_err < 1e-2
