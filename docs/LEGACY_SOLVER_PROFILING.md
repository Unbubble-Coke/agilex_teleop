# Legacy Solver Profiling

Date: 2026-05-17

This note profiles the legacy analytic `Solver` path. It does not change solver
math or benchmark conclusions.

## Solver Location And Call Chain

- Public legacy class: `nero/kinematics/analytic_IK_solver.py:49`, `class Solver`.
- Public solve entry used by benchmarks: `Solver.solve()` at
  `nero/kinematics/analytic_IK_solver.py:169`.
- Analytic core: `nero/kinematics/nero_kinematics/nero_ik/ik_solver.py`.
- Benchmark factory:
  - `benchmarks/benchmark_ik.py:120` constructs one debug adapter.
  - `nero/kinematics/solver_debug_adapter.py:106` selects the original solver.
  - `solver_debug_adapter.py:109` constructs `Solver(...)`.
  - `solver_debug_adapter.py:112-113` sets `_fallback_after_failures = 0` and
    `continuity.enable_global_fallback = True` for offline debug/benchmark use.
- Single-target benchmark call:
  - `benchmarks/benchmark_ik.py:91-97` calls `solver.init_state(q_init)` before
    every timed solve, then times `solver.solve(target_pose, limit_output_step=False)`.
  - This does not create a new solver instance per sample, but it resets the
    continuity runtime state so `theta0_prev` is `None`.
- Trajectory benchmark call:
  - `benchmarks/benchmark_trajectory_continuity.py:119` constructs one solver.
  - `benchmark_trajectory_continuity.py:134` calls `init_state()` once before the
    loop.
  - `benchmark_trajectory_continuity.py:141-143` times `solve()` for each step,
    preserving `theta0_prev` after successful solves.

Call path for single-target legacy benchmark:

```text
benchmark_ik.solve_once
  -> SolverDebugAdapter.solve
    -> Solver.solve
      -> Solver._pose_to_matrix
      -> solve_pose_continuous_with_state
        -> ik_arm_angle_with_report          # because theta0_prev is None
          -> _scan_theta0_solutions
            -> _ik_one_arm_angle             # repeated theta0 grid scan
              -> _compute_swe_from_target
              -> _elbow_from_arm_angle
              -> _solve_q123_from_swe
              -> _dh_A / _invert_rigid_transform / _extract_567_from_T47_paper
              -> _within_limits
          -> _best_weighted_from_cached
          -> _optimize_q_with_1d_qp          # fallback uses default QP weights
        -> score candidate with fk/pose_error
        -> _optimize_q_with_1d_qp            # returns immediately when weights are zero
        -> fk/pose_error for final pose_best
      -> _clamp_joints
```

## Initialization And Wrapper Findings

- `benchmark_ik.py` creates the solver once, not once per target.
- The single-target benchmark does call `init_state(q_init)` for every target.
  This is cheap directly, but important behaviorally: it removes `theta0_prev`,
  so each timed solve starts from the global fallback path.
- The legacy `Solver` does not load a URDF or model during each IK call. It uses
  `NeroParams.default()` in `Solver.__init__`.
- The legacy path has no Pinocchio model, no kinematic-chain parser, and no URDF
  file I/O.
- `Pinocchio_Solver` loads URDF in its own constructor, but this is not on the
  legacy timed path.
- The adapter overhead is negligible. cProfile shows
  `SolverDebugAdapter.solve` has near-zero own time and only delegates.
- Pose target generation (`fk_matrix`, `fk_pose`) happens outside the timed
  single-target solve. Inside `Solver.solve`, input pose conversion is
  `_pose_to_matrix`, measured at about 0.02 ms per call in this environment.
- No ROS, hardware, publisher, marker, RViz, sleep, or visualization code is on
  the benchmark solve path.
- `print()` exists in failure and jump-detection branches, but successful
  benchmark calls with `limit_output_step=False` do not print inside the timed
  solve.

## Profiling Environment

The repository default `python` is 3.8.10, while `pyproject.toml` requires
Python `>=3.10` and the code uses Python 3.10 type syntax. Profiling was run
with:

```text
/home/keyz/miniconda3/envs/RoboTwin/bin/python
Python 3.10.20
```

Pinocchio is not installed in this environment, so the profiling script records
that `Pinocchio_Solver` construction fails and continues profiling only the
legacy `Solver`.

Absolute latency in this environment is slower than the user-provided benchmark
numbers. The useful evidence here is the call path, call counts, and relative
time split.

## Latency Summary From New Script

Command:

```bash
/home/keyz/miniconda3/envs/RoboTwin/bin/python benchmarks/profile_legacy_solver.py \
  --num-samples 1000 \
  --top-k 40 \
  --output results/legacy_solver_profile_1000.txt
```

Output artifacts:

- `results/legacy_solver_profile_1000.txt`
- `results/legacy_solver_profile_1000.json`

Baseline, uninstrumented solve latency from this environment:

| samples | success_rate | mean_ms | median_ms | p95_ms | p99_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 100% | 58.930 | 57.929 | 66.052 | 73.654 | 88.977 |
| 1000 | 100% | 55.496 | 54.993 | 71.586 | 84.880 | 100.496 |

All 1000 single-target samples reported:

```json
{"continuous_global_fallback+1DQP": 1000}
```

This confirms the benchmarked single-target path is the global fallback path,
not the local trajectory-tracking path.

## Stage Timing Breakdown

Manual stage timing is collected with temporary wrappers in
`benchmarks/profile_legacy_solver.py`; production solver code and public API are
not changed. Percentages below use the instrumented total for the 1000-sample
run.

| stage | mean_ms | p95_ms | p99_ms | percent |
| --- | ---: | ---: | ---: | ---: |
| candidate_generation | 47.146 | 63.026 | 72.589 | 67.99% |
| fk_validation | 11.226 | 14.638 | 18.606 | 16.19% |
| candidate_filtering | 7.276 | 9.651 | 11.718 | 10.49% |
| qp_refinement | 1.981 | 2.835 | 3.916 | 2.86% |
| choose_solution | 1.424 | 2.720 | 3.851 | 2.05% |
| global_fallback_overhead | 0.123 | 0.209 | 0.295 | 0.18% |
| core_solver_overhead | 0.085 | 0.125 | 0.173 | 0.12% |
| parse_input | 0.019 | 0.027 | 0.048 | 0.03% |
| joint_limit_check | 0.021 | 0.033 | 0.055 | 0.03% |

Main interpretation:

- The dominant cost is the theta0 candidate-generation scan.
- Repeated FK/pose validation is the next major cost.
- Candidate filtering via joint-limit checks inside the branch enumeration is
  also measurable.
- Input parsing, wrapper overhead, final clamping, and return formatting are not
  meaningful contributors.

## cProfile Hotspots

Top cumulative-time functions from the 1000-sample cProfile run:

| function | calls | cumulative_ms | per_call_ms |
| --- | ---: | ---: | ---: |
| `solver_debug_adapter.py:50(solve)` | 1000 | 84509.489 | 84.509 |
| `analytic_IK_solver.py:169(solve)` | 1000 | 84506.989 | 84.507 |
| `ik_solver.py:884(solve_pose_continuous_with_state)` | 1000 | 84436.226 | 84.436 |
| `ik_solver.py:698(ik_arm_angle_with_report)` | 1000 | 83673.495 | 83.673 |
| `ik_solver.py:415(_scan_theta0_solutions)` | 1000 | 65072.574 | 65.073 |
| `ik_solver.py:601(_ik_one_arm_angle)` | 62000 | 64847.525 | 1.046 |
| `ik_solver.py:35(_dh_A)` | 1307000 | 22027.771 | 0.017 |
| `ik_solver.py:528(_optimize_q_with_1d_qp)` | 2000 | 16196.343 | 8.098 |
| `scipy/_minimize.py:735(minimize_scalar)` | 7000 | 16082.858 | 2.298 |
| `scipy/_optimize.py:2171(_minimize_scalar_bounded)` | 7000 | 16040.356 | 2.291 |
| `ik_solver.py:488(_qp_1d_objective)` | 42000 | 15285.624 | 0.364 |
| `numpy/numeric.py:1468(cross)` | 259000 | 13826.548 | 0.053 |
| `ik_solver.py:319(_elbow_from_arm_angle)` | 62000 | 10628.642 | 0.171 |
| `ik_solver.py:13(pose_error)` | 45000 | 7869.778 | 0.175 |
| `ik_solver.py:157(fk)` | 45000 | 7608.637 | 0.169 |

Additional high-volume calls:

- `_within_limits`: 496000 calls, 7164.294 ms cumulative.
- `wrap_to_pi`: 546000 calls, 2818.463 ms cumulative.
- built-in `numpy.array`: 5657074 calls, 5677.195 ms total.
- `fk_all`: 45000 calls, 7260.430 ms cumulative.

The call counts line up with the code:

- `n_psi=61` gives a full `theta0` sweep of about 62 samples due
  `np.arange(-pi, pi, step)`.
- 1000 solves therefore call `_ik_one_arm_angle` 62000 times.
- Each `_ik_one_arm_angle` does branch enumeration, small matrix construction,
  four `_dh_A` multiplications for each q123/q4 branch, wrist extraction, limit
  checks, and array concatenation.

## Hypothesis Checks

### Hypothesis A

Legacy `Solver` is slow because each IK enumerates multiple 7-DoF redundancy
candidates or free-joint samples.

Verdict: verified.

Evidence:

- Single-target benchmark resets state with `init_state(q_init)` for every
  target, so `theta0_prev` is `None`.
- `solve_pose_continuous_with_state` therefore falls into
  `ik_arm_angle_with_report`.
- `_scan_theta0_solutions` is called once per solve.
- `_ik_one_arm_angle` is called 62000 times for 1000 solves.
- Stage timing attributes about 68% of instrumented time to candidate
  generation.

### Hypothesis B

Legacy `Solver` is slow because it does many Python-level matrix calculations
and candidate filtering, rather than C++/Pinocchio work.

Verdict: verified.

Evidence:

- Hotspots are Python functions in `ik_solver.py`: `_scan_theta0_solutions`,
  `_ik_one_arm_angle`, `_dh_A`, `_within_limits`, `fk`, `fk_all`.
- There is no Pinocchio call on the legacy path.
- cProfile shows millions of small NumPy array constructions and hundreds of
  thousands of Python-level branch/filter calls.

### Hypothesis C

Legacy `Solver` is slow because it uses a stricter analytic/optimization flow
to reach near-machine precision.

Verdict: partially verified.

Evidence:

- The analytic branch enumeration is the largest cost and is part of the
  exact/reference-style solution strategy.
- Global fallback also runs `_optimize_q_with_1d_qp` with default
  `ContinuityParams()` inside `ik_arm_angle_with_report`, even though the outer
  `Solver` continuity config sets QP weights to zero.
- cProfile shows 7000 `minimize_scalar` calls for 1000 solves, one bounded
  scalar optimization for each of 7 joints per global-fallback solve.
- QP/refinement plus FK objective evaluation is significant, but the full
  theta0 scan remains the primary cost.

### Hypothesis D

Legacy `Solver` is slow because the benchmark wrapper reinitializes the solver
or model each time.

Verdict: mostly rejected, with an important nuance.

Evidence:

- The benchmark constructs one solver before the loop.
- The legacy solver does not load URDF/model per IK.
- Warm `Solver` construction is about 0.02 ms in this environment.
- However, the single-target benchmark does reset continuity state per sample
  with `init_state(q_init)`. This is not expensive by itself, but it forces the
  solve path into global fallback every time.

### Hypothesis E

Legacy `Solver` is slow because ROS, visualization, logging, or hardware
wrappers are mixed in.

Verdict: rejected for the benchmarked path.

Evidence:

- The timed call path uses `benchmark_ik.py`,
  `solver_debug_adapter.py`, `analytic_IK_solver.py`, and `ik_solver.py`.
- No ROS publisher, hardware interface, marker, RViz, sleep, or file I/O is
  called during successful timed solves.
- Failure/jump `print()` branches exist, but successful benchmark solves with
  `limit_output_step=False` do not execute them.

### Hypothesis F

Legacy `Solver` is slow because return format conversion or FK validation is
repeated many times.

Verdict: partially verified.

Evidence:

- Return-format conversion is negligible.
- FK/pose validation is significant: 45000 `fk` calls and 45000 `pose_error`
  calls for 1000 solves.
- Stage timing attributes about 16% of instrumented time to `fk_validation`.
- These FK calls come from scoring/final validation and the QP objective, not
  from the benchmark wrapper return conversion.

## Conclusion

Legacy `Solver` latency is primarily algorithmic plus Python implementation
cost:

1. The algorithmic shape is a full redundancy/arm-angle sweep in the
   single-target benchmark path.
2. The implementation executes that sweep in Python with many small NumPy
   allocations, matrix constructions, branch loops, and per-candidate
   filtering.
3. The benchmark does not recreate the solver/model per IK, and no ROS,
   visualization, hardware, or file I/O explains the per-call latency.
4. The single-target benchmark's per-sample `init_state()` reset indirectly
   increases latency by removing continuity history and causing global fallback
   on every timed solve.
5. The global fallback path currently runs the 1D QP refinement with default
   weights, despite the outer production `Solver` config intending QP to be
   disabled for latency. This is an implementation/configuration mismatch, but
   changing it could alter final precision and should be validated carefully.

The legacy solver is therefore not slow because of repeated model loading or
wrapper I/O. It is slow because the reference analytic search is implemented as
a Python-level theta0 scan plus branch filtering and repeated FK/QP validation.

## Minimal Optimization Suggestions

These suggestions avoid changing the solver's mathematical intent. They should
be applied one at a time with before/after benchmark and accuracy checks.

| suggestion | expected benefit | risk | math/result impact |
| --- | --- | --- | --- |
| Preserve continuity state when benchmarking trajectory-like usage, or report single-target global-fallback latency separately from warm-start latency. | Clarifies benchmark meaning; can avoid forcing full global fallback in warm-start use cases. | Low for reporting; medium if API semantics change. | Does not change math if only benchmark mode changes. |
| Thread the existing outer QP weights into `ik_arm_angle_with_report`, or add an explicit `enable_qp_refinement` flag for global fallback. | Potentially removes 7 `minimize_scalar` calls per global fallback solve; cProfile shows this path is significant. | Medium: QP may improve pose error or branch stability. | May change final numeric result; must verify 100% success and machine precision before enabling by default. |
| Cache constant MDH pieces used by `_dh_A`, especially constant `RotX(alpha_prev)` and fixed translation terms. | High within candidate generation; `_dh_A` is called 1,307,000 times for 1000 solves. | Low if implemented with exact equivalent formulas and tests. | Should not change math; floating-point roundoff may shift at last-bit level. |
| Reduce small array allocation in `_ik_one_arm_angle`, `_dh_A`, `wrap_to_pi`, and debug `extra` construction. | Moderate to high; `numpy.array` is called over 5.6M times in 1000 profiled solves. | Low to medium; array-shape bugs are possible. | Should not change math if shapes and dtype stay identical. |
| Batch or vectorize candidate limit checks where candidates are already materialized. | Moderate; `_within_limits` is called 496,000 times for 1000 solves. | Low to medium. | Should not change math if tolerance remains identical. |
| Avoid repeated FK in QP/scoring when the same candidate has already been evaluated. | Moderate; `fk`/`pose_error` is about 16% of stage time. | Medium because cached values must correspond exactly to the post-QP candidate. | Should not change math if cache invalidation is correct. |
| Keep debug geometry (`S`, `W`, `E`) construction optional when not requested. | Small to moderate; each accepted candidate concatenates debug extras. | Low if report fields are preserved when debug is enabled. | Could change report payload only, not IK math. |
| Add an explicit `reference_solver` or `debug_reference` mode for the legacy solver. | Operational clarity; avoids accidental servo-loop use of the slow path. | Low. | No math change. |

Avoid large rewrites until these smaller changes have been validated. In
particular, do not change `n_psi`, early-exit the scan, or skip validation in
the default reference path unless success rate and precision are remeasured.

## Reproduction Commands

The default `python` in this workspace is Python 3.8. Use the Python 3.10
environment shown below or another environment with NumPy and SciPy installed.

```bash
/home/keyz/miniconda3/envs/RoboTwin/bin/python -m compileall -q tests benchmarks examples nero pyAgxArm

/home/keyz/miniconda3/envs/RoboTwin/bin/python benchmarks/profile_legacy_solver.py \
  --num-samples 100 \
  --top-k 40 \
  --output results/legacy_solver_profile_100.txt

/home/keyz/miniconda3/envs/RoboTwin/bin/python benchmarks/profile_legacy_solver.py \
  --num-samples 1000 \
  --top-k 40 \
  --output results/legacy_solver_profile_1000.txt
```

The exact user-requested compile command was also attempted:

```bash
python -m compileall -q src tests benchmarks examples
```

It printed `Can't list 'src'` because this repository has no `src/` directory.
