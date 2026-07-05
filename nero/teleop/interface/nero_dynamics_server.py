"""
Standalone Nero dynamics data-collection server.

This server is intentionally independent from nero_interface_server.py so the
teleoperation stack can remain a stable hardware baseline. It reuses the same
pyAgxArm SDK access pattern, but exposes RPCs focused on dynamics-learning
experiments: raw state snapshots, autonomous joint-space excitation, and raw
episode saving.
"""

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zerorpc

_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from nero_interface_server import (
    DEFAULT_NERO_END_EFFECTOR_FRAME,
    DEFAULT_NERO_JOINT_NAMES,
    PinocchioKinematicsServoAdapter,
    get_robot_urdf_path,
    quat_multiply,
)


log = logging.getLogger(__name__)


DEFAULT_LEFT_HOME = [0.0, -0.2, 0.0, 1.87, -0.7, 0.0, 1.1]
DEFAULT_RIGHT_HOME = [0.0, -0.2, 0.0, 1.87, 0.7, 0.0, 1.1]
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "nero_dynamics", "raw")
)
JOINT_TRAJECTORIES = {"static_zero", "single_joint_sine", "multisine", "random_waypoints", "prbs"}
CARTESIAN_TRAJECTORIES = {"cartesian_sine", "cartesian_random_waypoints", "cartesian_prbs"}


def _plain(value: Any) -> Any:
    """Best-effort conversion of SDK message objects into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(k): _plain(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class NeroDynamicsServer:
    """RPC server dedicated to autonomous dynamics data collection."""

    def __init__(
        self,
        *,
        connect_left: bool = True,
        connect_right: bool = True,
        auto_enable: bool = True,
        auto_home: bool = False,
    ):
        self.left_robot = None
        self.right_robot = None
        self.left_cfg = None
        self.right_cfg = None
        self.left_ik_solver = None
        self.right_ik_solver = None

        self.tcp_offset = [0.0] * 6
        self.limit_z = 0.07
        self.ik_urdf_path = os.getenv("NERO_URDF_PATH")

        self.track_freq = 50.0
        self.dt = 1.0 / self.track_freq
        self.max_cart_step_m = 0.03
        self.max_rot_step_rad = 0.35
        self.max_ik_solve_ms = 30.0

        self._collection_lock = threading.RLock()
        self._collection_thread = None
        self._stop_event = threading.Event()
        self._latest_sample: Dict[str, Any] = {}
        self._status: Dict[str, Any] = self._idle_status()

        if connect_left:
            self.left_robot, self.left_cfg = self._connect_robot(
                channel="can_left",
                name="left_robot",
                auto_enable=auto_enable,
                home=DEFAULT_LEFT_HOME if auto_home else None,
            )
        if connect_right:
            self.right_robot, self.right_cfg = self._connect_robot(
                channel="can_right",
                name="right_robot",
                auto_enable=auto_enable,
                home=DEFAULT_RIGHT_HOME if auto_home else None,
            )

        try:
            if self.left_robot is not None and self.left_cfg is not None:
                self.left_ik_solver = self._setup_ik_solver(self.left_robot, self.left_cfg, "left_robot")
            if self.right_robot is not None and self.right_cfg is not None:
                self.right_ik_solver = self._setup_ik_solver(self.right_robot, self.right_cfg, "right_robot")
        except Exception as exc:
            log.error("[DYNAMICS] IK solver init failed: %s", exc)

        log.info("=" * 50)
        log.info("Nero Dynamics Server Ready")
        log.info("=" * 50)

    def _idle_status(self) -> Dict[str, Any]:
        return {
            "running": False,
            "state": "idle",
            "message": "",
            "samples": 0,
            "invalid_samples": 0,
            "unsafe_stop": False,
            "unsafe_reason": "",
            "output_path": "",
            "metadata_path": "",
            "jsonl_path": "",
            "config": {},
            "started_at": None,
            "finished_at": None,
        }

    def _connect_robot(self, *, channel: str, name: str, auto_enable: bool, home: Optional[list]):
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            cfg = create_agx_arm_config(robot="nero", comm="can", channel=channel)
            robot = AgxArmFactory.create_arm(cfg)
            robot.connect()
            time.sleep(0.3)
            robot.set_normal_mode()
            time.sleep(0.3)

            if auto_enable:
                start_t = time.monotonic()
                while not robot.enable(255):
                    if time.monotonic() - start_t > 5.0:
                        log.warning("[%s] enable timeout", name)
                        break
                    time.sleep(0.01)

            if home is not None:
                robot.set_speed_percent(30)
                robot.move_j(home)
                self._wait_for_motion_complete(robot, home, timeout=20.0)

            log.info("[%s] connected on %s", name, channel)
            return robot, cfg
        except Exception as exc:
            log.error("[%s] connection failed: %s", name, exc)
            return None, None

    # ==================== RPC: basic API ====================

    def get_dynamics_api_version(self) -> Dict[str, Any]:
        return {
            "api": "nero_dynamics",
            "version": 1,
            "methods": [
                "get_dynamics_sample",
                "get_dynamics_static_info",
                "start_dynamics_collection",
                "stop_dynamics_collection",
                "get_collection_status",
                "get_latest_dynamics_sample",
            ],
        }

    def get_dynamics_sample(self, robot_arm: str, include_driver: bool = True) -> Dict[str, Any]:
        """Read one full raw dynamics snapshot without commanding motion."""
        robot = self._select_robot(robot_arm)
        sample = self._read_dynamics_sample(
            robot_arm=robot_arm,
            robot=robot,
            include_driver=include_driver,
            command=None,
            trajectory="manual_sample",
            elapsed=0.0,
        )
        with self._collection_lock:
            self._latest_sample = sample
        return sample

    def get_latest_dynamics_sample(self) -> Dict[str, Any]:
        with self._collection_lock:
            return dict(self._latest_sample)

    def get_dynamics_static_info(self, robot_arm: str) -> Dict[str, Any]:
        robot = self._select_robot(robot_arm)
        if robot is None:
            return {"arm": robot_arm, "ok": False, "reason": "robot_not_connected"}

        get_joint_angle_vel_limits = getattr(robot, "get_joint_angle_vel_limits", None)
        get_joint_acc_limits = getattr(robot, "get_joint_acc_limits", None)
        has_joint_angle_vel_limits = callable(get_joint_angle_vel_limits)
        has_joint_acc_limits = callable(get_joint_acc_limits)
        joint_angle_vel_limits = []
        joint_acc_limits = []
        for joint_index in range(1, 8):
            if has_joint_angle_vel_limits:
                joint_angle_vel_limits.append(_plain(self._safe_call(get_joint_angle_vel_limits, joint_index)))
            else:
                joint_angle_vel_limits.append(None)
            if has_joint_acc_limits:
                joint_acc_limits.append(_plain(self._safe_call(get_joint_acc_limits, joint_index)))
            else:
                joint_acc_limits.append(None)

        return {
            "arm": robot_arm,
            "ok": True,
            "joint_nums": int(getattr(robot, "joint_nums", 7)),
            "static_limit_api_available": {
                "joint_angle_vel_limits": has_joint_angle_vel_limits,
                "joint_acc_limits": has_joint_acc_limits,
            },
            "joint_angle_vel_limits": joint_angle_vel_limits,
            "joint_acc_limits": joint_acc_limits,
            "enable_status": _plain(self._safe_call(robot.get_joints_enable_status_list)),
            "crash_protection": _plain(self._safe_call(robot.get_crash_protection_rating)),
            "arm_status": self._read_arm_status(robot),
        }

    # ==================== RPC: collection control ====================

    def start_dynamics_collection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start autonomous joint-space excitation and raw data collection."""
        with self._collection_lock:
            if self._collection_thread is not None and self._collection_thread.is_alive():
                return {"ok": False, "reason": "collection_already_running", "status": dict(self._status)}

            cfg = self._normalize_collection_config(config)
            self._stop_event.clear()
            self._status = self._idle_status()
            self._status.update(
                {
                    "running": True,
                    "state": "starting",
                    "message": "collection thread starting",
                    "config": cfg,
                    "started_at": time.time(),
                }
            )
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                args=(cfg,),
                name="nero_dynamics_collection",
                daemon=True,
            )
            self._collection_thread.start()
            return {"ok": True, "status": dict(self._status)}

    def stop_dynamics_collection(self) -> Dict[str, Any]:
        self._stop_event.set()
        with self._collection_lock:
            self._status["message"] = "stop requested"
        return {"ok": True, "status": self.get_collection_status()}

    def get_collection_status(self) -> Dict[str, Any]:
        with self._collection_lock:
            return dict(self._status)

    # ==================== Collection implementation ====================

    def _normalize_collection_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = dict(config or {})
        output_dir = os.path.abspath(str(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
        sample_hz = float(config.get("sample_hz", 100.0))
        if sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive")

        return {
            "arm": str(config.get("arm", "left_robot")),
            "trajectory": str(config.get("trajectory", "static_zero")),
            "duration": float(config.get("duration", 60.0)),
            "sample_hz": sample_hz,
            "include_driver": bool(config.get("include_driver", True)),
            "output_dir": output_dir,
            "episode_name": str(config.get("episode_name", "")),
            "max_motor_time_skew_s": float(config.get("max_motor_time_skew_s", 0.005)),
            "move_to_home_first": bool(config.get("move_to_home_first", False)),
            "return_home_after": bool(config.get("return_home_after", False)),
            "max_joint_step_rad": float(config.get("max_joint_step_rad", 0.05)),
            "amplitudes": config.get("amplitudes", [0.08] * 7),
            "frequencies": config.get("frequencies", [0.07, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29]),
            "phases": config.get("phases", [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2]),
            "single_joint_index": int(config.get("single_joint_index", 1)),
            "single_joint_amplitude": float(config.get("single_joint_amplitude", 0.08)),
            "single_joint_frequency": float(config.get("single_joint_frequency", 0.08)),
            "joint_waypoint_interval": float(config.get("joint_waypoint_interval", 2.0)),
            "joint_transition_fraction": float(config.get("joint_transition_fraction", 0.35)),
            "joint_random_seed": int(config.get("joint_random_seed", 42)),
            "cartesian_position_amplitudes": config.get("cartesian_position_amplitudes", [0.015, 0.015, 0.01]),
            "cartesian_rotation_amplitudes": config.get("cartesian_rotation_amplitudes", [0.03, 0.03, 0.03]),
            "cartesian_frequencies": config.get("cartesian_frequencies", [0.05, 0.07, 0.09, 0.04, 0.06, 0.08]),
            "cartesian_phases": config.get("cartesian_phases", [0.0, 0.7, 1.4, 2.1, 2.8, 3.5]),
            "cartesian_axes": config.get("cartesian_axes", [1, 1, 1, 0, 0, 0]),
            "cartesian_workspace_min": config.get("cartesian_workspace_min", [-0.8, -0.8, 0.07]),
            "cartesian_workspace_max": config.get("cartesian_workspace_max", [0.8, 0.8, 1.2]),
            "cartesian_waypoint_interval": float(config.get("cartesian_waypoint_interval", 2.0)),
            "cartesian_transition_fraction": float(config.get("cartesian_transition_fraction", 0.35)),
            "cartesian_random_seed": int(config.get("cartesian_random_seed", 42)),
        }

    def _collection_loop(self, config: Dict[str, Any]) -> None:
        robot_arm = config["arm"]
        robot = self._select_robot(robot_arm)
        samples = []
        episode_paths = self._make_episode_paths(config)
        jsonl_path = episode_paths["jsonl_path"]
        metadata_path = episode_paths["metadata_path"]
        npz_path = episode_paths["npz_path"]

        try:
            if robot is None:
                raise RuntimeError(f"{robot_arm} is not connected")

            if config["move_to_home_first"]:
                self._go_home(robot_arm, robot)

            q0 = self._read_current_joint_angles(robot)
            if q0 is None:
                raise RuntimeError("failed to read initial joint angles")
            q_prev_cmd = np.asarray(q0, dtype=float)
            runtime = self._prepare_trajectory_runtime(config, robot_arm, robot, q0)

            os.makedirs(config["output_dir"], exist_ok=True)
            metadata = {
                "schema_version": 1,
                "server": "nero_dynamics_server",
                "config": config,
                "initial_q": q0.tolist(),
                "trajectory_runtime": _plain(runtime.get("metadata", {})),
                "created_at": time.time(),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=True)

            with self._collection_lock:
                self._status.update(
                    {
                        "state": "running",
                        "message": "collection running",
                        "output_path": npz_path,
                        "metadata_path": metadata_path,
                        "jsonl_path": jsonl_path,
                    }
                )

            start_perf = time.perf_counter()
            next_tick = start_perf
            period = 1.0 / config["sample_hz"]

            with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                while not self._stop_event.is_set():
                    now = time.perf_counter()
                    elapsed = now - start_perf
                    if elapsed >= config["duration"]:
                        break

                    command = self._command_for_elapsed(config, elapsed, q0, runtime)
                    if command is not None:
                        q_cmd = self._limit_command_step(
                            np.asarray(command["q_cmd"], dtype=float),
                            q_prev_cmd,
                            config["max_joint_step_rad"],
                        )
                        q_prev_cmd = q_cmd
                        command["q_cmd"] = q_cmd.tolist()
                        if command.get("control_mode") == "cartesian_servo_OL":
                            runtime["ik_solver"].init_state(q_cmd)
                        if command.get("ik_success", True):
                            robot.move_js(command["q_cmd"])

                    sample = self._read_dynamics_sample(
                        robot_arm=robot_arm,
                        robot=robot,
                        include_driver=config["include_driver"],
                        command=command,
                        trajectory=config["trajectory"],
                        elapsed=elapsed,
                    )
                    sample_valid, invalid_reason = self._validate_sample(sample, config)
                    sample["valid"] = bool(sample_valid)
                    sample["invalid_reason"] = invalid_reason

                    samples.append(sample)
                    f_jsonl.write(json.dumps(_plain(sample), ensure_ascii=True) + "\n")

                    with self._collection_lock:
                        self._latest_sample = sample
                        self._status["samples"] += 1
                        if not sample_valid:
                            self._status["invalid_samples"] += 1

                    next_tick += period
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            self._save_npz(npz_path, samples)

            if config["return_home_after"]:
                self._go_home(robot_arm, robot)

            with self._collection_lock:
                self._status.update(
                    {
                        "running": False,
                        "state": "finished",
                        "message": "collection finished",
                        "finished_at": time.time(),
                    }
                )
        except Exception as exc:
            log.exception("[DYNAMICS] collection failed")
            with self._collection_lock:
                self._status.update(
                    {
                        "running": False,
                        "state": "failed",
                        "message": str(exc),
                        "unsafe_stop": True,
                        "unsafe_reason": str(exc),
                        "finished_at": time.time(),
                    }
                )

    def _command_for_elapsed(
        self,
        config: Dict[str, Any],
        elapsed: float,
        q0: np.ndarray,
        runtime: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        trajectory = config["trajectory"]
        if trajectory == "static_zero":
            return None
        if trajectory == "single_joint_sine":
            joint_idx = max(0, min(6, int(config["single_joint_index"]) - 1))
            amp = float(config["single_joint_amplitude"])
            freq = float(config["single_joint_frequency"])
            omega = 2.0 * math.pi * freq
            q_cmd = np.asarray(q0, dtype=float).copy()
            qd_cmd = np.zeros(7, dtype=float)
            qdd_cmd = np.zeros(7, dtype=float)
            q_cmd[joint_idx] += amp * math.sin(omega * elapsed)
            qd_cmd[joint_idx] = amp * omega * math.cos(omega * elapsed)
            qdd_cmd[joint_idx] = -amp * omega * omega * math.sin(omega * elapsed)
            return {
                "control_mode": "move_js",
                "q_cmd": q_cmd.tolist(),
                "qd_cmd": qd_cmd.tolist(),
                "qdd_cmd": qdd_cmd.tolist(),
            }
        if trajectory == "multisine":
            amps = np.asarray(config["amplitudes"], dtype=float).reshape(7)
            freqs = np.asarray(config["frequencies"], dtype=float).reshape(7)
            phases = np.asarray(config["phases"], dtype=float).reshape(7)
            omega = 2.0 * math.pi * freqs
            arg = omega * elapsed + phases
            q_cmd = np.asarray(q0, dtype=float) + amps * np.sin(arg)
            qd_cmd = amps * omega * np.cos(arg)
            qdd_cmd = -amps * omega * omega * np.sin(arg)
            return {
                "control_mode": "move_js",
                "q_cmd": q_cmd.tolist(),
                "qd_cmd": qd_cmd.tolist(),
                "qdd_cmd": qdd_cmd.tolist(),
            }
        if trajectory == "random_waypoints":
            q_delta, qd_cmd, qdd_cmd = self._joint_waypoint_delta(config, elapsed, runtime)
            q_cmd = np.asarray(q0, dtype=float) + q_delta
            return {
                "control_mode": "move_js",
                "joint_mode": "random_waypoints",
                "q_cmd": q_cmd.astype(float).tolist(),
                "qd_cmd": qd_cmd.astype(float).tolist(),
                "qdd_cmd": qdd_cmd.astype(float).tolist(),
            }
        if trajectory == "prbs":
            q_delta, qd_cmd, qdd_cmd = self._joint_prbs_delta(config, elapsed, runtime)
            q_cmd = np.asarray(q0, dtype=float) + q_delta
            return {
                "control_mode": "move_js",
                "joint_mode": "prbs",
                "q_cmd": q_cmd.astype(float).tolist(),
                "qd_cmd": qd_cmd.astype(float).tolist(),
                "qdd_cmd": qdd_cmd.astype(float).tolist(),
            }
        if trajectory in CARTESIAN_TRAJECTORIES:
            return self._cartesian_command_for_elapsed(config, elapsed, runtime)
        raise ValueError(f"unknown trajectory: {trajectory}")

    def _joint_waypoint_delta(
        self, config: Dict[str, Any], elapsed: float, runtime: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = max(1e-6, float(config["joint_waypoint_interval"]))
        deltas = np.asarray(runtime["joint_deltas"], dtype=float)
        idx = min(int(elapsed // interval), len(deltas) - 2)
        tau = min(max((elapsed - idx * interval) / interval, 0.0), 1.0)
        return self._quintic_interp(deltas[idx], deltas[idx + 1], tau, interval)

    def _joint_prbs_delta(
        self, config: Dict[str, Any], elapsed: float, runtime: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = max(1e-6, float(config["joint_waypoint_interval"]))
        transition = min(max(float(config["joint_transition_fraction"]), 0.01), 1.0) * interval
        deltas = np.asarray(runtime["joint_deltas"], dtype=float)
        idx = min(int(elapsed // interval), len(deltas) - 2)
        local_t = elapsed - idx * interval
        if local_t <= transition:
            tau = local_t / transition
            return self._quintic_interp(deltas[idx], deltas[idx + 1], tau, transition)
        return deltas[idx + 1].copy(), np.zeros(7), np.zeros(7)

    def _prepare_trajectory_runtime(
        self,
        config: Dict[str, Any],
        robot_arm: str,
        robot,
        q0: np.ndarray,
    ) -> Dict[str, Any]:
        trajectory = config["trajectory"]
        if trajectory in {"random_waypoints", "prbs"}:
            duration = float(config["duration"])
            interval = max(1e-6, float(config["joint_waypoint_interval"]))
            count = int(math.ceil(duration / interval)) + 2
            rng = np.random.default_rng(int(config["joint_random_seed"]))
            amps = np.asarray(config["amplitudes"], dtype=float).reshape(7)
            if trajectory == "random_waypoints":
                deltas = rng.uniform(-amps, amps, size=(count, 7))
                deltas[0] = 0.0
            else:
                signs = rng.choice([-1.0, 1.0], size=(count, 7))
                deltas = signs * amps
                deltas[0] = 0.0
            return {
                "type": "joint_space_random",
                "joint_deltas": deltas,
                "metadata": {
                    "type": "joint_space_random",
                    "joint_deltas": deltas.tolist(),
                },
            }
        if trajectory not in CARTESIAN_TRAJECTORIES:
            return {"metadata": {"type": "joint_space"}}

        ik_solver = self._select_ik_solver(robot_arm)
        if ik_solver is None:
            raise RuntimeError(f"IK solver for {robot_arm} is not ready")
        ik_solver.init_state(q0)
        x0 = np.asarray(ik_solver.fk_pose(q0), dtype=float)

        runtime: Dict[str, Any] = {
            "type": "cartesian",
            "ik_solver": ik_solver,
            "x0": x0,
            "metadata": {
                "type": "cartesian",
                "x0": x0.tolist(),
                "ik_ee_frame": getattr(ik_solver, "ee_frame_name", ""),
            },
        }

        if trajectory in {"cartesian_random_waypoints", "cartesian_prbs"}:
            duration = float(config["duration"])
            interval = max(1e-6, float(config["cartesian_waypoint_interval"]))
            count = int(math.ceil(duration / interval)) + 2
            rng = np.random.default_rng(int(config["cartesian_random_seed"]))
            amps = self._cartesian_amplitudes(config)
            axes = np.asarray(config["cartesian_axes"], dtype=float).reshape(6)
            if trajectory == "cartesian_random_waypoints":
                deltas = rng.uniform(-amps, amps, size=(count, 6)) * axes
                deltas[0] = 0.0
            else:
                signs = rng.choice([-1.0, 1.0], size=(count, 6))
                deltas = signs * amps * axes
                deltas[0] = 0.0
            runtime["cartesian_deltas"] = deltas
            runtime["metadata"]["cartesian_deltas"] = deltas.tolist()

        return runtime

    def _cartesian_command_for_elapsed(
        self, config: Dict[str, Any], elapsed: float, runtime: Dict[str, Any]
    ) -> Dict[str, Any]:
        ik_solver = runtime["ik_solver"]
        x0 = np.asarray(runtime["x0"], dtype=float)
        trajectory = config["trajectory"]

        if trajectory == "cartesian_sine":
            delta, xd_cmd, xdd_cmd = self._cartesian_sine_delta(config, elapsed)
        elif trajectory == "cartesian_random_waypoints":
            delta, xd_cmd, xdd_cmd = self._cartesian_waypoint_delta(config, elapsed, runtime)
        elif trajectory == "cartesian_prbs":
            delta, xd_cmd, xdd_cmd = self._cartesian_prbs_delta(config, elapsed, runtime)
        else:
            raise ValueError(f"unknown cartesian trajectory: {trajectory}")

        delta = self._limit_pose_delta(delta)
        target_pose = x0 + delta
        target_pose[3:] = (target_pose[3:] + np.pi) % (2.0 * np.pi) - np.pi

        workspace_ok, workspace_reason = self._check_cartesian_workspace(target_pose, config)
        q_cmd = None
        ik_success = False
        ik_report = {"workspace_ok": workspace_ok, "workspace_reason": workspace_reason}
        solve_time_ms = np.nan

        if workspace_ok:
            t0 = time.perf_counter()
            q_cmd = ik_solver.solve(target_pose, limit_output_step=True)
            solve_time_ms = (time.perf_counter() - t0) * 1000.0
            ik_success = q_cmd is not None and len(q_cmd) == 7 and solve_time_ms <= self.max_ik_solve_ms
            ik_report.update(getattr(ik_solver, "last_report", {}) or {})
            ik_report["solve_time_ms_outer"] = solve_time_ms
            ik_report["jump_report"] = getattr(ik_solver, "last_jump_report", None)
            if q_cmd is None:
                q_cmd = ik_solver.state.q_prev if ik_solver.state is not None else np.zeros(7)
        else:
            q_cmd = ik_solver.state.q_prev if ik_solver.state is not None else np.zeros(7)

        q_cmd = np.asarray(q_cmd, dtype=float).reshape(7)
        return {
            "control_mode": "cartesian_servo_OL",
            "cartesian_mode": trajectory,
            "x0": x0.tolist(),
            "x_delta_cmd": delta.astype(float).tolist(),
            "xd_cmd": xd_cmd.astype(float).tolist(),
            "xdd_cmd": xdd_cmd.astype(float).tolist(),
            "x_cmd": target_pose.astype(float).tolist(),
            "q_cmd": q_cmd.astype(float).tolist(),
            "qd_cmd": [np.nan] * 7,
            "qdd_cmd": [np.nan] * 7,
            "ik_success": bool(ik_success),
            "ik_solve_time_ms": float(solve_time_ms),
            "ik_report": _plain(ik_report),
        }

    def _cartesian_amplitudes(self, config: Dict[str, Any]) -> np.ndarray:
        pos = np.asarray(config["cartesian_position_amplitudes"], dtype=float).reshape(3)
        rot = np.asarray(config["cartesian_rotation_amplitudes"], dtype=float).reshape(3)
        return np.concatenate([pos, rot])

    def _cartesian_sine_delta(
        self, config: Dict[str, Any], elapsed: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        amps = self._cartesian_amplitudes(config)
        freqs = np.asarray(config["cartesian_frequencies"], dtype=float).reshape(6)
        phases = np.asarray(config["cartesian_phases"], dtype=float).reshape(6)
        axes = np.asarray(config["cartesian_axes"], dtype=float).reshape(6)
        omega = 2.0 * math.pi * freqs
        arg = omega * elapsed + phases
        delta = amps * axes * np.sin(arg)
        xd = amps * axes * omega * np.cos(arg)
        xdd = -amps * axes * omega * omega * np.sin(arg)
        return delta, xd, xdd

    def _cartesian_waypoint_delta(
        self, config: Dict[str, Any], elapsed: float, runtime: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = max(1e-6, float(config["cartesian_waypoint_interval"]))
        deltas = np.asarray(runtime["cartesian_deltas"], dtype=float)
        idx = min(int(elapsed // interval), len(deltas) - 2)
        tau = min(max((elapsed - idx * interval) / interval, 0.0), 1.0)
        return self._quintic_interp(deltas[idx], deltas[idx + 1], tau, interval)

    def _cartesian_prbs_delta(
        self, config: Dict[str, Any], elapsed: float, runtime: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = max(1e-6, float(config["cartesian_waypoint_interval"]))
        transition = min(max(float(config["cartesian_transition_fraction"]), 0.01), 1.0) * interval
        deltas = np.asarray(runtime["cartesian_deltas"], dtype=float)
        idx = min(int(elapsed // interval), len(deltas) - 2)
        local_t = elapsed - idx * interval
        if local_t <= transition:
            tau = local_t / transition
            return self._quintic_interp(deltas[idx], deltas[idx + 1], tau, transition)
        return deltas[idx + 1].copy(), np.zeros(6), np.zeros(6)

    def _quintic_interp(
        self, start: np.ndarray, target: np.ndarray, tau: float, duration: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = min(max(float(tau), 0.0), 1.0)
        duration = max(float(duration), 1e-6)
        dq = np.asarray(target, dtype=float) - np.asarray(start, dtype=float)
        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
        dds = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (duration**2)
        return np.asarray(start, dtype=float) + s * dq, ds * dq, dds * dq

    def _read_dynamics_sample(
        self,
        *,
        robot_arm: str,
        robot,
        include_driver: bool,
        command: Optional[Dict[str, Any]],
        trajectory: str,
        elapsed: float,
    ) -> Dict[str, Any]:
        host_start = time.time()
        if robot is None:
            return {
                "schema_version": 1,
                "arm": robot_arm,
                "trajectory": trajectory,
                "elapsed": float(elapsed),
                "host_timestamp_start": host_start,
                "host_timestamp_end": time.time(),
                "ok": False,
                "reason": "robot_not_connected",
                "joint_angles": {"q": [np.nan] * 7, "timestamp": np.nan, "hz": np.nan, "valid": False},
                "motor": {
                    "position": [np.nan] * 7,
                    "velocity": [np.nan] * 7,
                    "torque": [np.nan] * 7,
                    "current": [np.nan] * 7,
                    "timestamp": [np.nan] * 7,
                    "hz": [np.nan] * 7,
                    "valid": [False] * 7,
                    "time_skew": np.nan,
                },
                "driver": {},
                "command": command or {},
                "arm_status": {"valid": False},
                "enable_status": [],
            }
        joint_angles = self._read_joint_angles_message(robot)
        motor = self._read_motor_states(robot)
        driver = self._read_driver_states(robot) if include_driver else {}
        host_end = time.time()
        return {
            "schema_version": 1,
            "arm": robot_arm,
            "trajectory": trajectory,
            "elapsed": float(elapsed),
            "host_timestamp_start": host_start,
            "host_timestamp_end": host_end,
            "joint_angles": joint_angles,
            "motor": motor,
            "driver": driver,
            "command": command or {},
            "arm_status": self._read_arm_status(robot),
            "enable_status": _plain(self._safe_call(robot.get_joints_enable_status_list)),
        }

    def _read_motor_states(self, robot) -> Dict[str, Any]:
        position, velocity, torque, current = [], [], [], []
        timestamps, hz, valid = [], [], []
        for joint_index in range(1, 8):
            result = self._safe_call(robot.get_motor_states, joint_index)
            if result is None:
                position.append(np.nan)
                velocity.append(np.nan)
                torque.append(np.nan)
                current.append(np.nan)
                timestamps.append(np.nan)
                hz.append(np.nan)
                valid.append(False)
                continue
            msg = result.msg
            position.append(_float_or_none(getattr(msg, "position", np.nan)))
            velocity.append(_float_or_none(getattr(msg, "velocity", np.nan)))
            torque.append(_float_or_none(getattr(msg, "torque", np.nan)))
            current.append(_float_or_none(getattr(msg, "current", np.nan)))
            timestamps.append(_float_or_none(getattr(result, "timestamp", np.nan)))
            hz.append(_float_or_none(getattr(result, "hz", np.nan)))
            valid.append(True)

        ts = np.asarray(timestamps, dtype=float)
        finite_ts = ts[np.isfinite(ts)]
        time_skew = float(np.max(finite_ts) - np.min(finite_ts)) if finite_ts.size else np.nan
        return {
            "position": position,
            "velocity": velocity,
            "torque": torque,
            "current": current,
            "timestamp": timestamps,
            "hz": hz,
            "valid": valid,
            "time_skew": time_skew,
        }

    def _read_driver_states(self, robot) -> Dict[str, Any]:
        vol, foc_temp, motor_temp, bus_current = [], [], [], []
        timestamps, hz, valid, foc_status = [], [], [], []
        for joint_index in range(1, 8):
            result = self._safe_call(robot.get_driver_states, joint_index)
            if result is None:
                vol.append(np.nan)
                foc_temp.append(np.nan)
                motor_temp.append(np.nan)
                bus_current.append(np.nan)
                timestamps.append(np.nan)
                hz.append(np.nan)
                valid.append(False)
                foc_status.append({})
                continue
            msg = result.msg
            vol.append(_float_or_none(getattr(msg, "vol", np.nan)))
            foc_temp.append(_float_or_none(getattr(msg, "foc_temp", np.nan)))
            motor_temp.append(_float_or_none(getattr(msg, "motor_temp", np.nan)))
            bus_current.append(_float_or_none(getattr(msg, "bus_current", np.nan)))
            timestamps.append(_float_or_none(getattr(result, "timestamp", np.nan)))
            hz.append(_float_or_none(getattr(result, "hz", np.nan)))
            valid.append(True)
            foc_status.append(_plain(getattr(msg, "foc_status", None)))
        return {
            "vol": vol,
            "foc_temp": foc_temp,
            "motor_temp": motor_temp,
            "bus_current": bus_current,
            "timestamp": timestamps,
            "hz": hz,
            "valid": valid,
            "foc_status": foc_status,
        }

    def _read_joint_angles_message(self, robot) -> Dict[str, Any]:
        result = self._safe_call(robot.get_joint_angles)
        if result is None:
            return {"q": [np.nan] * 7, "timestamp": np.nan, "hz": np.nan, "valid": False}
        return {
            "q": _plain(result.msg),
            "timestamp": _float_or_none(getattr(result, "timestamp", np.nan)),
            "hz": _float_or_none(getattr(result, "hz", np.nan)),
            "valid": True,
        }

    def _read_current_joint_angles(self, robot, timeout: float = 2.0) -> Optional[np.ndarray]:
        start_t = time.monotonic()
        while time.monotonic() - start_t < timeout:
            result = self._safe_call(robot.get_joint_angles)
            if result is not None:
                return np.asarray(result.msg, dtype=float)
            time.sleep(0.005)
        return None

    def _read_arm_status(self, robot) -> Dict[str, Any]:
        result = self._safe_call(robot.get_arm_status)
        if result is None:
            return {"valid": False}
        msg = result.msg
        return {
            "valid": True,
            "ctrl_mode": _plain(getattr(msg, "ctrl_mode", None)),
            "arm_status": _plain(getattr(msg, "arm_status", None)),
            "motion_status": _plain(getattr(msg, "motion_status", None)),
            "mode_feedback": _plain(getattr(msg, "mode_feedback", None)),
            "trajectory_num": _plain(getattr(msg, "trajectory_num", None)),
            "err_status": _plain(getattr(msg, "err_status", None)),
            "timestamp": _float_or_none(getattr(result, "timestamp", np.nan)),
            "hz": _float_or_none(getattr(result, "hz", np.nan)),
        }

    def _validate_sample(self, sample: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
        motor = sample.get("motor", {})
        if not all(motor.get("valid", [])):
            return False, "missing_motor_feedback"
        time_skew = motor.get("time_skew", np.nan)
        if np.isfinite(time_skew) and time_skew > config["max_motor_time_skew_s"]:
            return False, "motor_timestamp_skew"
        enable = sample.get("enable_status")
        if isinstance(enable, list) and len(enable) >= 7 and not all(enable[:7]):
            return False, "joint_not_enabled"
        if not sample.get("arm_status", {}).get("valid", False):
            return False, "missing_arm_status"
        command = sample.get("command", {})
        if "ik_success" in command and not command.get("ik_success", False):
            reason = command.get("ik_report", {}).get("workspace_reason", "") or "ik_failed"
            return False, reason
        return True, ""

    def _save_npz(self, path: str, samples: list) -> None:
        if not samples:
            np.savez_compressed(path, empty=np.asarray([True]))
            return

        def arr(section: str, key: str, default=np.nan):
            return np.asarray(
                [s.get(section, {}).get(key, default) for s in samples],
                dtype=float,
            )

        command_q = []
        command_qd = []
        command_qdd = []
        command_x = []
        command_x_delta = []
        command_xd = []
        command_xdd = []
        ik_success = []
        ik_solve_time_ms = []
        for sample in samples:
            command = sample.get("command", {})
            command_q.append(command.get("q_cmd", [np.nan] * 7))
            command_qd.append(command.get("qd_cmd", [np.nan] * 7))
            command_qdd.append(command.get("qdd_cmd", [np.nan] * 7))
            command_x.append(command.get("x_cmd", [np.nan] * 6))
            command_x_delta.append(command.get("x_delta_cmd", [np.nan] * 6))
            command_xd.append(command.get("xd_cmd", [np.nan] * 6))
            command_xdd.append(command.get("xdd_cmd", [np.nan] * 6))
            ik_success.append(bool(command.get("ik_success", False)))
            ik_solve_time_ms.append(float(command.get("ik_solve_time_ms", np.nan)))

        np.savez_compressed(
            path,
            host_timestamp_start=np.asarray([s["host_timestamp_start"] for s in samples], dtype=float),
            host_timestamp_end=np.asarray([s["host_timestamp_end"] for s in samples], dtype=float),
            elapsed=np.asarray([s["elapsed"] for s in samples], dtype=float),
            valid=np.asarray([bool(s.get("valid", False)) for s in samples], dtype=bool),
            joint_angle_q=arr("joint_angles", "q"),
            joint_angle_timestamp=arr("joint_angles", "timestamp"),
            motor_position=arr("motor", "position"),
            motor_velocity=arr("motor", "velocity"),
            motor_torque=arr("motor", "torque"),
            motor_current=arr("motor", "current"),
            motor_timestamp=arr("motor", "timestamp"),
            motor_hz=arr("motor", "hz"),
            motor_time_skew=arr("motor", "time_skew"),
            driver_vol=arr("driver", "vol"),
            driver_foc_temp=arr("driver", "foc_temp"),
            driver_motor_temp=arr("driver", "motor_temp"),
            driver_bus_current=arr("driver", "bus_current"),
            q_cmd=np.asarray(command_q, dtype=float),
            qd_cmd=np.asarray(command_qd, dtype=float),
            qdd_cmd=np.asarray(command_qdd, dtype=float),
            x_cmd=np.asarray(command_x, dtype=float),
            x_delta_cmd=np.asarray(command_x_delta, dtype=float),
            xd_cmd=np.asarray(command_xd, dtype=float),
            xdd_cmd=np.asarray(command_xdd, dtype=float),
            ik_success=np.asarray(ik_success, dtype=bool),
            ik_solve_time_ms=np.asarray(ik_solve_time_ms, dtype=float),
        )

    # ==================== Utilities ====================

    def _setup_ik_solver(self, robot, cfg, robot_name: str, timeout_sec: float = 2.0):
        """Set up the same Pinocchio IK adapter used by nero_interface_server.py."""
        current_joints = self._read_current_joint_angles(robot, timeout=timeout_sec)
        if current_joints is None:
            current_joints = np.zeros(7, dtype=float)
            log.warning("[%s] get_joint_angles timeout, using zeros for IK init", robot_name)

        joint_limits = []
        for i in range(1, 8):
            lo, hi = cfg["joint_limits"][f"joint{i}"]
            joint_limits.append((lo, hi))

        urdf_path = get_robot_urdf_path("nero", self.ik_urdf_path)
        ik_solver = PinocchioKinematicsServoAdapter(
            urdf_path=urdf_path,
            end_effector_frame=DEFAULT_NERO_END_EFFECTOR_FRAME,
            active_joint_names=DEFAULT_NERO_JOINT_NAMES,
            joint_limits=joint_limits,
            dt=self.dt,
            tcp_offset=self.tcp_offset,
        )
        ik_solver.init_state(current_joints)
        log.info(
            "[%s] IK initialized: URDF=%s, ee_frame=%s, q=%s",
            robot_name,
            urdf_path,
            DEFAULT_NERO_END_EFFECTOR_FRAME,
            np.asarray(current_joints, dtype=float).round(3),
        )
        return ik_solver

    def _select_ik_solver(self, robot_arm: str):
        if robot_arm == "left_robot":
            return self.left_ik_solver
        if robot_arm == "right_robot":
            return self.right_ik_solver
        raise ValueError("robot_arm must be 'left_robot' or 'right_robot'")

    def _limit_pose_delta(self, pose_delta: np.ndarray) -> np.ndarray:
        pose_delta = np.asarray(pose_delta, dtype=float).reshape(6)
        out = pose_delta.copy()
        out[:3] = np.clip(out[:3], -self.max_cart_step_m, self.max_cart_step_m)
        out[3:] = np.clip(out[3:], -self.max_rot_step_rad, self.max_rot_step_rad)
        return out

    def _check_cartesian_workspace(
        self, target_pose: np.ndarray, config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        xyz = target_pose[:3]
        workspace_min = np.asarray(config["cartesian_workspace_min"], dtype=float).reshape(3)
        workspace_max = np.asarray(config["cartesian_workspace_max"], dtype=float).reshape(3)
        if xyz[2] < self.limit_z:
            return False, f"z_below_limit:{xyz[2]:.4f}<{self.limit_z:.4f}"
        if np.any(xyz < workspace_min) or np.any(xyz > workspace_max):
            return False, "outside_workspace_box"
        return True, ""

    def _select_robot(self, robot_arm: str):
        if robot_arm == "left_robot":
            return self.left_robot
        if robot_arm == "right_robot":
            return self.right_robot
        raise ValueError("robot_arm must be 'left_robot' or 'right_robot'")

    def _safe_call(self, fn, *args):
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception as exc:
            log.warning("[DYNAMICS] SDK call failed: %s", exc)
            return None

    def _limit_command_step(self, q_cmd: np.ndarray, q_prev: np.ndarray, max_step: float) -> np.ndarray:
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(7)
        q_prev = np.asarray(q_prev, dtype=float).reshape(7)
        dq = np.clip(q_cmd - q_prev, -abs(max_step), abs(max_step))
        return q_prev + dq

    def _make_episode_paths(self, config: Dict[str, Any]) -> Dict[str, str]:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        episode_name = config["episode_name"] or f"{config['arm']}_{config['trajectory']}_{stamp}"
        base = os.path.join(config["output_dir"], episode_name)
        return {
            "jsonl_path": base + ".jsonl",
            "metadata_path": base + ".metadata.json",
            "npz_path": base + ".npz",
        }

    def _go_home(self, robot_arm: str, robot) -> bool:
        home = DEFAULT_LEFT_HOME if robot_arm == "left_robot" else DEFAULT_RIGHT_HOME
        robot.set_speed_percent(30)
        robot.move_j(home)
        return self._wait_for_motion_complete(robot, home, timeout=20.0)

    def _wait_for_motion_complete(
        self, robot, target_joints: list, timeout: float = 10.0, tolerance: float = 0.01
    ) -> bool:
        start_t = time.monotonic()
        target = np.asarray(target_joints, dtype=float)
        while time.monotonic() - start_t < timeout:
            result = self._safe_call(robot.get_joint_angles)
            if result is None:
                time.sleep(0.05)
                continue
            current = np.asarray(result.msg, dtype=float)
            if np.allclose(current, target, atol=tolerance):
                return True
            status = self._safe_call(robot.get_arm_status)
            if status is not None and getattr(status.msg, "motion_status", None) == 0:
                return bool(np.allclose(current, target, atol=tolerance))
            time.sleep(0.1)
        return False

    def robot_stop(self, robot_arm: str) -> bool:
        try:
            robot = self._select_robot(robot_arm)
            if robot is not None:
                robot.electronic_emergency_stop()
            return True
        except Exception as exc:
            log.error("[DYNAMICS] stop failed: %s", exc)
            return False


def start_server(
    ip: str,
    port: int = 4243,
    connect_left: bool = True,
    connect_right: bool = True,
    auto_enable: bool = True,
    auto_home: bool = False,
):
    server = zerorpc.Server(
        NeroDynamicsServer(
            connect_left=connect_left,
            connect_right=connect_right,
            auto_enable=auto_enable,
            auto_home=auto_home,
        )
    )
    server.bind(f"tcp://{ip}:{port}")
    log.info("[DYNAMICS] Listening on tcp://%s:%s", ip, port)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--no-auto-enable", action="store_true")
    parser.add_argument("--auto-home", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    start_server(
        ip=args.ip,
        port=args.port,
        connect_left=not args.right_only,
        connect_right=not args.left_only,
        auto_enable=not args.no_auto_enable,
        auto_home=args.auto_home,
    )
