# Dependency Audit: Nero Kinematics

Audit date: 2026-05-16

## 1. Current Dependency List

Declared runtime dependencies in `pyproject.toml`:

- `numpy`
- `scipy`
- `zerorpc`
- `python-can`
- `typing-extensions`
- `PyYAML`

Optional dependency groups:

- `sim`: `pybullet`, `ikpy`, `meshcat`
- `dynamics`: `pinocchio`, `placo`, `eigenpy`, `eiquadprog`
- `test`: `pytest`

The repository also vendors a `pyAgxArm/` package, including CAN protocol code,
Nero/Piper configs, demos, and the `pyAgxArm/asserts/agx_arm_urdf/` robot model
assets.

## 2. Core Kinematics Dependencies

`nero/kinematics/analytic_IK_solver.py` contains two solver entry points:

- `Solver`: legacy analytic-wrapper path using the in-repo
  `nero.kinematics.nero_kinematics.nero_ik.ik_solver`.
- `Pinocchio_Solver`: iterative damped least-squares IK using Pinocchio.

After this audit, the kinematics core imports only generic numerical/robotics
libraries:

- `numpy`
- `scipy.optimize.minimize_scalar` through the legacy analytic IK module
- `pinocchio` for `Pinocchio_Solver`
- Python stdlib modules such as `math`, `os`, `time`, `dataclasses`

Core solver inputs/outputs are plain robot data:

- `numpy` arrays or array-like lists
- joint vectors
- 4x4 transforms
- `[x, y, z, roll, pitch, yaw]` pose vectors in meters/radians
- URDF path, frame name, joint limits

## 3. Specialized Dependencies Found

| Location | Specialized dependency | Use | Class |
| --- | --- | --- | --- |
| `nero/kinematics/analytic_IK_solver.py` before this change | `from pyAgxArm.utiles.tf import rpy_to_rot, rot_to_rpy` | Generic RPY/matrix conversion | B |
| `nero/kinematics/analytic_IK_solver.py` | fallback path to `pyAgxArm/asserts/agx_arm_urdf/nero/urdf/nero_description.urdf` | Default Nero URDF discovery | A/B |
| `nero/kinematics/analytic_IK_solver.py` | frame name `link7`, joint names `joint1`...`joint7` | Nero robot model convention | A |
| `nero/kinematics/nero_kinematics/nero_ik/ik_solver.py` | hard-coded Nero DH parameters and joint limits | Analytic Nero model | A for Nero analytic solver |
| `pyAgxArm/configs/nero.json` | Nero joint limits and joint names | SDK/hardware config source | C |
| `nero/teleop/interface/nero_interface_server.py` | `pyAgxArm`, `AgxArmFactory`, `create_agx_arm_config` | Hardware connection and robot factory | C |
| `nero/teleop/interface/nero_interface_server.py` | `robot.get_tcp_pose()`, `robot.get_joint_angles()`, `.msg` | SDK message adapters | C |
| `nero/teleop/interface/nero_interface_server.py` | `robot.move_j()`, `robot.move_js()` | Hardware command dispatch | C |
| `nero/tests/*.py`, `pyAgxArm/demos/*` | `pyAgxArm` APIs and CAN channels | Hardware tests and demos | C |
| `nero/kinematics/nero_kinematics/nero_ik/ik_joint_state_publisher.py`, `interactive_target_marker.py` | ROS message packages | ROS integration/debug tooling | C |

Class definitions:

- A: necessary for this Nero kinematics model, such as robot geometry,
  joint limits, and frame names.
- B: implementation detail that can be replaced by generic code.
- C: SDK/integration/example concern that should remain outside the core solver.

## 4. Replacements Completed

- Removed the core import of `pyAgxArm.utiles.tf` from
  `nero/kinematics/analytic_IK_solver.py`.
- Added local generic `rpy_to_rot()` and `rot_to_rpy()` helpers with the same
  ZYX convention, preserving compatibility for existing imports from this module.
- Changed `nero/__init__.py` to lazy-load `NeroDualArmServer`. Importing
  `nero.kinematics...` no longer requires `zerorpc` or teleop-side dependencies.
- Added `Pinocchio_Solver.jacobian_matrix()` / `jacobian()` for explicit
  6xN Jacobian debug checks.
- Added richer `Pinocchio_Solver.last_report` fields:
  `reason`, `timed_out`, `last_q`, `best_q`, and `solution_q`.
- Added a debug/benchmark solver adapter so offline tests can select either
  the original `Solver` or `Pinocchio_Solver` without adding SDK coupling.
- Fixed the legacy analytic IK optimizer path so 1D QP is skipped when both QP
  weights are set to zero; this preserves the existing wrapper's "QP disabled"
  behavior during tests and benchmarks.

## 5. Dependencies That Remain

The Pinocchio solver still has a fallback URDF path under
`pyAgxArm/asserts/agx_arm_urdf/...`. This is not an SDK API call, but it is a
repository-layout dependency on robot model assets bundled under the SDK tree.
It remains for backward compatibility and because the URDF/mesh package has not
yet been moved to a neutral asset location.

Mitigation already available:

- Pass `urdf_path=...` to `Pinocchio_Solver`.
- Or set `NERO_URDF_PATH=/path/to/nero_description.urdf`.

Recommended future cleanup:

- Move/copy Nero URDF and meshes into a neutral package location such as
  `nero/assets/agx_arm_urdf/`.
- Keep the current `pyAgxArm/asserts/...` fallback for one release as a
  compatibility path, then deprecate it.

Teleop/server files intentionally keep SDK dependencies because they are adapter
and integration code. The solver should continue receiving only numeric
`target_pose`, `joint_limits`, `tcp_offset`, and current joint seed values from
that layer.

## 6. Remaining Risks and TODO

- The legacy analytic solver encodes Nero-specific DH parameters and joint limits
  directly in `NeroParams.default()`. That is acceptable for a Nero-specific
  analytic solver, but should not be reused as a generic arm model.
- `nero/teleop/interface/nero_interface_server.py` still imports
  `pyAgxArm.utiles.tf` in teleop-specific pose conversion paths. This is class C
  adapter code and was not moved.
- Some old files under `nero/tests/` are hardware scripts, not hermetic pytest
  tests. They still call real CAN/SDK APIs and should be kept separate from CI.
- The default project dependency group keeps Pinocchio under `dynamics`, not
  mandatory runtime dependencies. Kinematics tests/benchmarks that use
  `Pinocchio_Solver` require installing that extra.
