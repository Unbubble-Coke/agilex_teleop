# Kinematics Debug Tests and Benchmarks

This document covers the hermetic Nero kinematics checks added for the
Pinocchio-based solver. These tests do not connect to CAN, pyAgxArm, or real
hardware.

## Setup

Install the dynamics and test extras:

```bash
pip install -e ".[dynamics,test]"
```

If the default bundled URDF is not available, point the solver at a Nero URDF:

```bash
export NERO_URDF_PATH=/path/to/nero_description.urdf
```

The solver defaults to:

- joint order: `joint1` ... `joint7`
- end-effector frame: `link7`
- units: meters and radians
- pose vector format: `[x, y, z, roll, pitch, yaw]`

## Tests

FK/IK consistency and FK output contract:

```bash
pytest tests/test_fk_ik_consistency.py
pytest tests/test_fk_ik_consistency.py --solver=original
```

Jacobian numerical consistency:

```bash
pytest tests/test_jacobian_consistency.py
```

`original` solver does not expose an analytic Jacobian, so this test is skipped
when run with:

```bash
pytest tests/test_jacobian_consistency.py --solver=original
```

Joint limit checks:

```bash
pytest tests/test_joint_limits.py
pytest tests/test_joint_limits.py --solver=original
```

Run all hermetic kinematics tests:

```bash
pytest tests
```

Notes:

- IK success is judged by FK pose error, not by element-wise equality to the
  sampled joint vector. Nero is a 7-DoF redundant arm, so multiple valid joint
  solutions may exist.
- Orientation error is computed as an SO(3) geodesic angle in radians.
- The Jacobian test compares both translational and rotational rows against a
  central finite-difference Jacobian. Angular velocity is expressed in the
  world frame.

## Single Target Debug

Generate one reachable target from FK and solve it:

```bash
python examples/debug_single_target.py --seed 0
python examples/debug_single_target.py --solver original --seed 0
```

Use explicit target and seed joints:

```bash
python examples/debug_single_target.py \
  --q 0 0.2 0 -0.3 0.1 0.2 0 \
  --q-init 0 0.18 0 -0.28 0.08 0.18 0
```

The script prints JSON containing:

- URDF path
- end-effector frame
- joint names
- target pose
- solved joint vector
- position/orientation error
- joint-limit violation flag
- solver report and latency

## IK Benchmark

Run reachable-target IK benchmark:

```bash
python benchmarks/benchmark_ik.py \
  --solver pinocchio \
  --num-samples 1000 \
  --output results/ik_benchmark.json \
  --log-failures results/ik_failures.jsonl

python benchmarks/benchmark_ik.py \
  --solver original \
  --num-samples 1000 \
  --n-psi 181 \
  --output results/ik_benchmark_original.json \
  --log-failures results/ik_failures_original.jsonl
```

Useful options:

- `--solver`: `pinocchio` or `original`, default `pinocchio`
- `--num-samples`: default `1000`
- `--seed`: default `0`
- `--max-iters`: default `80`; applies to `Pinocchio_Solver`
- `--n-psi`: default `181`; applies to original `Solver`
- `--pos-tol`: default `1e-3`
- `--ori-tol`: default `1e-2`
- `--output`: optional JSON summary path
- `--log-failures`: optional JSONL failure log path
- `--seed-trials`: optional seed-sensitivity trials per target

Summary fields include:

- `num_samples`
- `success_count`
- `success_rate`
- `mean_position_error`, `median_position_error`, `max_position_error`
- `mean_orientation_error`, `median_orientation_error`, `max_orientation_error`
- `iterations_available`
- `mean_iterations`, `max_iterations`
- `mean_latency_ms`, `median_latency_ms`, `p90_latency_ms`,
  `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`
- `timeout_rate`
- `joint_limit_violation_rate`
- `seed_sensitivity`

Failure records include `target_pose`, `q_init`, `best_q`, `last_q`, error,
iterations, reason, solver report, and solve time.

## Trajectory Continuity Benchmark

Run warm-start trajectory continuity benchmark:

```bash
python benchmarks/benchmark_trajectory_continuity.py \
  --solver pinocchio \
  --num-samples 300 \
  --output results/trajectory_benchmark.json \
  --log-failures results/trajectory_failures.jsonl

python benchmarks/benchmark_trajectory_continuity.py \
  --solver original \
  --num-samples 300 \
  --n-psi 181 \
  --output results/trajectory_benchmark_original.json \
  --log-failures results/trajectory_failures_original.jsonl
```

This benchmark creates a continuous reachable joint trajectory, maps each frame
through FK, then solves IK frame-by-frame using the previous IK result as the
next initial state. In addition to the standard IK summary fields, it reports:

- `mean_joint_step_norm`
- `median_joint_step_norm`
- `p95_joint_step_norm`
- `max_joint_step_norm`
- `configuration_jump_count`
- `configuration_jump_rate`
- `jump_threshold`

This approximates the warm-start behavior used in realtime teleoperation.
