import numpy as np
from nero.kinematics.debug_tools import (
    DEFAULT_NERO_EE_FRAME,
    DEFAULT_NERO_JOINT_LIMITS,
    DEFAULT_NERO_JOINT_NAMES,
    clip_to_joint_limits,
    pose_errors,
    sample_random_q,
)


def test_fk_output_contract(solver_factory):
    solver = solver_factory()
    q = np.mean(DEFAULT_NERO_JOINT_LIMITS, axis=1)

    T = solver.fk_matrix(q)
    pose = solver.fk_pose(q)

    assert T.shape == (4, 4)
    assert pose.shape == (6,)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(pose))
    assert np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-8
    assert solver.ee_frame_name == DEFAULT_NERO_EE_FRAME
    assert solver.joint_names == DEFAULT_NERO_JOINT_NAMES


def test_fk_ik_consistency_reachable_targets(solver_factory, solver_name):
    rng = np.random.default_rng(7)
    solver = solver_factory(max_iterations=120, n_psi=181)
    sample_count = 3 if solver_name == "original" else 8
    samples = sample_random_q(rng, DEFAULT_NERO_JOINT_LIMITS, num_samples=sample_count, margin=0.08)

    for q_target in samples:
        target_T = solver.fk_matrix(q_target)
        target_pose = solver.fk_pose(q_target)
        q_init = clip_to_joint_limits(
            q_target + rng.normal(0.0, 0.02, size=q_target.shape),
            DEFAULT_NERO_JOINT_LIMITS,
        )

        solver.init_state(q_init)
        q_solution = solver.solve(target_pose, limit_output_step=False)

        assert q_solution is not None, solver.last_report
        actual_T = solver.fk_matrix(q_solution)
        pos_err, ori_err = pose_errors(actual_T, target_T)
        assert pos_err < 1e-3, solver.last_report
        assert ori_err < 1e-2, solver.last_report
