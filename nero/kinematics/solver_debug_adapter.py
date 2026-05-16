"""Small adapter used by debug tests and benchmarks to select a solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nero.kinematics.debug_tools import (
    DEFAULT_NERO_EE_FRAME,
    DEFAULT_NERO_JOINT_NAMES,
    matrix_to_pose6,
)


@dataclass
class SolverDebugAdapter:
    name: str
    solver: object
    supports_jacobian: bool
    supports_max_iterations: bool

    @property
    def last_report(self):
        return getattr(self.solver, "last_report", None)

    @property
    def state(self):
        return getattr(self.solver, "state", None)

    @property
    def joint_names(self):
        return getattr(self.solver, "joint_names", DEFAULT_NERO_JOINT_NAMES)

    @property
    def frame_names(self):
        return getattr(self.solver, "frame_names", [])

    @property
    def ee_frame_name(self):
        return getattr(self.solver, "ee_frame_name", DEFAULT_NERO_EE_FRAME)

    @property
    def urdf_path(self):
        return getattr(self.solver, "urdf_path", None)

    def init_state(self, q):
        return self.solver.init_state(q)

    def solve(self, target_pose, limit_output_step: bool = False):
        return self.solver.solve(target_pose, limit_output_step=limit_output_step)

    def fk_matrix(self, q):
        if self.name == "pinocchio":
            return self.solver.fk_matrix(q)

        from nero.kinematics.nero_kinematics.nero_ik.ik_solver import fk

        return fk(np.asarray(q, dtype=float), self.solver.nero_params)

    def fk_pose(self, q):
        if self.name == "pinocchio":
            return self.solver.fk_pose(q)
        return matrix_to_pose6(self.fk_matrix(q))

    def jacobian_matrix(self, q):
        if not self.supports_jacobian:
            raise NotImplementedError(f"{self.name} solver does not expose an analytic Jacobian")
        return self.solver.jacobian_matrix(q)

    def clamp_joints(self, q):
        return self.solver._clamp_joints(q)


def make_debug_solver(
    solver_name: str,
    joint_limits,
    dt: float = 0.05,
    n_psi: int = 61,
    max_iterations: int = 80,
    tol_pos: float = 1e-5,
    tol_rot: float = 1e-4,
    urdf_path=None,
    tcp_offset=None,
) -> SolverDebugAdapter:
    normalized = solver_name.lower().replace("-", "_")
    if normalized in {"pinocchio", "pinocchio_solver"}:
        from nero.kinematics.analytic_IK_solver import Pinocchio_Solver

        return SolverDebugAdapter(
            name="pinocchio",
            solver=Pinocchio_Solver(
                joint_limits=joint_limits,
                dt=dt,
                n_psi=n_psi,
                urdf_path=urdf_path,
                tcp_offset=tcp_offset,
                max_iterations=max_iterations,
                tol_pos=tol_pos,
                tol_rot=tol_rot,
            ),
            supports_jacobian=True,
            supports_max_iterations=True,
        )

    if normalized in {"original", "solver", "analytic"}:
        from nero.kinematics.analytic_IK_solver import Solver

        solver = Solver(joint_limits=joint_limits, dt=dt, n_psi=n_psi)
        # The production wrapper delays global fallback for servo-loop latency.
        # Offline tests need the first frame to be solvable without theta0 history.
        solver._fallback_after_failures = 0
        solver.continuity.enable_global_fallback = True
        return SolverDebugAdapter(
            name="original",
            solver=solver,
            supports_jacobian=False,
            supports_max_iterations=False,
        )

    raise ValueError(f"Unsupported solver: {solver_name!r}")
