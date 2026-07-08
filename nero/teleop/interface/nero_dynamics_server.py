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
            "attempt_samples": 0,
            "attempt_index": 0,
            "accepted_episodes": 0,
            "failed_attempts": 0,
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
                "finalize_dynamics_dataset",
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
        get_enable_status = getattr(robot, "get_joints_enable_status_list", None)
        get_crash_protection = getattr(robot, "get_crash_protection_rating", None)
        has_joint_angle_vel_limits = callable(get_joint_angle_vel_limits)
        has_joint_acc_limits = callable(get_joint_acc_limits)
        has_enable_status = callable(get_enable_status)
        has_crash_protection = callable(get_crash_protection)
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
                "enable_status": has_enable_status,
                "crash_protection": has_crash_protection,
            },
            "joint_angle_vel_limits": joint_angle_vel_limits,
            "joint_acc_limits": joint_acc_limits,
            "enable_status": _plain(self._safe_call(get_enable_status)) if has_enable_status else None,
            "crash_protection": _plain(self._safe_call(get_crash_protection)) if has_crash_protection else None,
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

    def finalize_dynamics_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the single-dataset NPZ from the appended JSONL file."""
        cfg = self._normalize_collection_config(config)
        paths = self._make_episode_paths(cfg)
        if cfg["save_layout"] != "single_dataset":
            return {"ok": False, "reason": "save_layout_is_not_single_dataset"}
        samples = self._load_jsonl_samples(paths["jsonl_path"])
        if not samples:
            return {"ok": False, "reason": "dataset_jsonl_empty", "paths": paths}
        self._save_npz(paths["npz_path"], samples)
        metadata = self._load_dataset_metadata(paths["metadata_path"])
        metadata.update(
            {
                "finalized": True,
                "finalized_at": time.time(),
                "total_samples": len(samples),
                "valid_samples": int(sum(bool(s.get("valid", False)) for s in samples)),
                "npz_path": paths["npz_path"],
            }
        )
        self._write_json(paths["metadata_path"], metadata)
        return {
            "ok": True,
            "samples": len(samples),
            "valid_samples": metadata["valid_samples"],
            "paths": paths,
        }

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
            "save_layout": str(config.get("save_layout", "episode_files")),
            "dataset_name": str(config.get("dataset_name", "")),
            "num_episodes": int(config.get("num_episodes", 1)),
            "episode_index": int(config.get("episode_index", 0)),
            "episode_name": str(config.get("episode_name", "")),
            "reset_dataset": bool(config.get("reset_dataset", False)),
            "max_motor_time_skew_s": float(config.get("max_motor_time_skew_s", 0.005)),
            "move_to_home_first": bool(config.get("move_to_home_first", False)),
            "return_home_after": bool(config.get("return_home_after", False)),
            "max_joint_step_rad": float(config.get("max_joint_step_rad", 0.05)),
            "joint_limit_margin": float(config.get("joint_limit_margin", 0.05)),
            "random_start_enabled": bool(config.get("random_start_enabled", False)),
            "random_start_seed": int(config.get("random_start_seed", 42)),
            "random_start_center": config.get("random_start_center", []),
            "random_start_range": config.get("random_start_range", [0.0] * 7),
            "random_start_min": config.get("random_start_min", []),
            "random_start_max": config.get("random_start_max", []),
            "random_start_move_speed_percent": float(config.get("random_start_move_speed_percent", 20.0)),
            "random_start_timeout": float(config.get("random_start_timeout", 20.0)),
            "random_start_tolerance": float(config.get("random_start_tolerance", 0.05)),
            "random_start_startup_grace_s": float(config.get("random_start_startup_grace_s", 1.0)),
            "random_start_stationary_checks": int(config.get("random_start_stationary_checks", 5)),
            "random_start_settle_time": float(config.get("random_start_settle_time", 1.0)),
            "retry_on_episode_failure": bool(config.get("retry_on_episode_failure", True)),
            "max_episode_attempts": int(config.get("max_episode_attempts", 10)),
            "abort_on_random_start_fail": bool(config.get("abort_on_random_start_fail", True)),
            "abort_on_joint_limit_clip": bool(config.get("abort_on_joint_limit_clip", True)),
            "max_joint_limit_clip_samples": int(config.get("max_joint_limit_clip_samples", 0)),
            "max_invalid_sample_ratio": float(config.get("max_invalid_sample_ratio", 0.05)),
            "max_consecutive_invalid_samples": int(config.get("max_consecutive_invalid_samples", 20)),
            "failure_return_home": bool(config.get("failure_return_home", False)),
            "failure_cooldown_s": float(config.get("failure_cooldown_s", 1.0)),
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
        episode_paths = self._make_episode_paths(config)
        jsonl_path = episode_paths["jsonl_path"]
        metadata_path = episode_paths["metadata_path"]
        npz_path = episode_paths["npz_path"]

        try:
            if robot is None:
                raise RuntimeError(f"{robot_arm} is not connected")

            self._prepare_episode_output(config, episode_paths)

            if config["move_to_home_first"]:
                self._go_home(robot_arm, robot)

            os.makedirs(episode_paths["root_dir"], exist_ok=True)
            with self._collection_lock:
                self._status.update(
                    {
                        "state": "running",
                        "message": "collection running",
                        "output_path": npz_path,
                        "metadata_path": metadata_path,
                        "jsonl_path": jsonl_path,
                        "dataset_name": episode_paths.get("dataset_name", ""),
                        "episode_index": config["episode_index"],
                        "episode_name": episode_paths["episode_name"],
                        "attempt_samples": 0,
                        "attempt_index": 0,
                        "accepted_episodes": 0,
                        "failed_attempts": 0,
                    }
                )

            max_attempts = max(1, int(config["max_episode_attempts"]))
            accepted = False
            last_failure = None
            for attempt_index in range(max_attempts):
                if self._stop_event.is_set():
                    break
                attempt_config = dict(config)
                attempt_config["attempt_index"] = attempt_index
                global_sample_start = self._count_jsonl_lines(jsonl_path) if episode_paths["append_jsonl"] else 0
                with self._collection_lock:
                    self._status.update(
                        {
                            "state": "running",
                            "message": f"attempt {attempt_index + 1}/{max_attempts}",
                            "attempt_index": attempt_index,
                            "attempt_samples": 0,
                        }
                    )

                result = self._run_episode_attempt(
                    config=attempt_config,
                    robot_arm=robot_arm,
                    robot=robot,
                    episode_paths=episode_paths,
                    global_sample_start=global_sample_start,
                )
                if result["ok"]:
                    samples = result["samples"]
                    metadata = result["metadata"]
                    self._commit_episode_samples(
                        config=attempt_config,
                        paths=episode_paths,
                        metadata=metadata,
                        samples=samples,
                        global_sample_start=global_sample_start,
                    )
                    with self._collection_lock:
                        self._status["samples"] += len(samples)
                        self._status["invalid_samples"] += int(
                            sum(not bool(s.get("valid", False)) for s in samples)
                        )
                        self._status["accepted_episodes"] = 1
                    accepted = True
                    break

                last_failure = result
                self._record_failed_attempt(episode_paths, result["failure"])
                with self._collection_lock:
                    self._status["failed_attempts"] += 1
                    self._status["message"] = f"attempt failed: {result['failure']['reason']}"
                log.warning(
                    "[DYNAMICS] episode %s attempt %s failed: %s",
                    config["episode_index"],
                    attempt_index,
                    result["failure"]["reason"],
                )
                if config["failure_return_home"]:
                    self._go_home(robot_arm, robot)
                if not config["retry_on_episode_failure"] or result.get("fatal", False):
                    break
                if config["failure_cooldown_s"] > 0.0:
                    time.sleep(config["failure_cooldown_s"])

            if not accepted:
                reason = "stopped" if self._stop_event.is_set() else "max_episode_attempts_exhausted"
                if last_failure is not None:
                    reason = f"{reason}:{last_failure['failure']['reason']}"
                raise RuntimeError(reason)

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

    def _run_episode_attempt(
        self,
        *,
        config: Dict[str, Any],
        robot_arm: str,
        robot,
        episode_paths: Dict[str, Any],
        global_sample_start: int,
    ) -> Dict[str, Any]:
        attempt_index = int(config.get("attempt_index", 0))
        samples = []
        q_start = None

        if config["random_start_enabled"]:
            q_start = self._sample_random_start(config, robot_arm)
            reached = self._move_to_joint_target(
                robot_arm=robot_arm,
                robot=robot,
                target=q_start,
                speed_percent=config["random_start_move_speed_percent"],
                timeout=config["random_start_timeout"],
                tolerance=config["random_start_tolerance"],
                startup_grace_s=config["random_start_startup_grace_s"],
                stationary_checks=config["random_start_stationary_checks"],
            )
            if not reached:
                current_q = self._read_current_joint_angles(robot)
                current_text = current_q.tolist() if current_q is not None else None
                log.error(
                    "[%s] failed to reach random_start_q target=%s current=%s",
                    robot_arm,
                    q_start.tolist(),
                    current_text,
                )
                return {
                    "ok": False,
                    "fatal": not config["abort_on_random_start_fail"],
                    "failure": self._make_attempt_failure(
                        config=config,
                        reason="random_start_unreachable",
                        target_q_start=q_start.tolist(),
                        current_q=current_text,
                        samples=samples,
                    ),
                }
            if config["random_start_settle_time"] > 0.0:
                time.sleep(config["random_start_settle_time"])

        q0 = self._read_current_joint_angles(robot)
        if q0 is None:
            return {
                "ok": False,
                "fatal": True,
                "failure": self._make_attempt_failure(
                    config=config,
                    reason="failed_to_read_initial_joint_angles",
                    target_q_start=q_start.tolist() if q_start is not None else None,
                    current_q=None,
                    samples=samples,
                ),
            }

        q_prev_cmd = np.asarray(q0, dtype=float)
        runtime = self._prepare_trajectory_runtime(config, robot_arm, robot, q0)
        episode_metadata = {
            "schema_version": 1,
            "server": "nero_dynamics_server",
            "config": config,
            "dataset_name": episode_paths.get("dataset_name", ""),
            "episode_index": config["episode_index"],
            "attempt_index": attempt_index,
            "episode_name": episode_paths["episode_name"],
            "initial_q": q0.tolist(),
            "random_start_q": q_start.tolist() if q_start is not None else None,
            "trajectory_runtime": _plain(runtime.get("metadata", {})),
            "created_at": time.time(),
        }

        start_perf = time.perf_counter()
        next_tick = start_perf
        period = 1.0 / config["sample_hz"]
        episode_sample_index = 0
        invalid_count = 0
        consecutive_invalid = 0
        joint_limit_clip_count = 0

        while not self._stop_event.is_set():
            now = time.perf_counter()
            elapsed = now - start_perf
            if elapsed >= config["duration"]:
                break

            command = self._command_for_elapsed(config, elapsed, q0, runtime)
            if command is not None:
                q_target = np.asarray(command["q_cmd"], dtype=float)
                q_target_clipped, joint_limit_clipped = self._clip_to_joint_limits_with_report(
                    q_target,
                    robot_arm,
                    margin=config["joint_limit_margin"],
                )
                q_cmd = self._limit_command_step(
                    q_target_clipped,
                    q_prev_cmd,
                    config["max_joint_step_rad"],
                )
                q_cmd, step_joint_limit_clipped = self._clip_to_joint_limits_with_report(
                    q_cmd,
                    robot_arm,
                    margin=config["joint_limit_margin"],
                )
                q_prev_cmd = q_cmd
                command["q_cmd"] = q_cmd.tolist()
                command["joint_limit_margin"] = float(config["joint_limit_margin"])
                command["joint_limit_clipped"] = bool(
                    command.get("joint_limit_clipped", False)
                    or joint_limit_clipped
                    or step_joint_limit_clipped
                )
                if command["joint_limit_clipped"]:
                    command["q_cmd_unclipped"] = q_target.tolist()
                    joint_limit_clip_count += 1
                if command.get("control_mode") == "cartesian_servo_OL":
                    runtime["ik_solver"].init_state(q_cmd)
                if command.get("ik_success", True):
                    move_js = getattr(robot, "move_js", None)
                    if not callable(move_js):
                        raise RuntimeError("move_js API is not available on this Nero driver")
                    move_js(command["q_cmd"])

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
            if sample_valid:
                consecutive_invalid = 0
            else:
                invalid_count += 1
                consecutive_invalid += 1

            global_sample_index = global_sample_start + episode_sample_index
            sample.update(
                {
                    "dataset_name": episode_paths.get("dataset_name", ""),
                    "episode_index": int(config["episode_index"]),
                    "attempt_index": int(attempt_index),
                    "episode_name": episode_paths["episode_name"],
                    "episode_sample_index": int(episode_sample_index),
                    "global_sample_index": int(global_sample_index),
                    "episode_elapsed": float(elapsed),
                    "dataset_elapsed": float(global_sample_index / config["sample_hz"]),
                    "q_start": q_start.tolist() if q_start is not None else q0.tolist(),
                }
            )
            samples.append(sample)

            with self._collection_lock:
                self._latest_sample = sample
                self._status["attempt_samples"] = len(samples)

            abort_reason = self._episode_abort_reason(
                sample=sample,
                config=config,
                invalid_count=invalid_count,
                consecutive_invalid=consecutive_invalid,
                joint_limit_clip_count=joint_limit_clip_count,
            )
            if abort_reason:
                return {
                    "ok": False,
                    "fatal": False,
                    "failure": self._make_attempt_failure(
                        config=config,
                        reason=abort_reason,
                        target_q_start=q_start.tolist() if q_start is not None else None,
                        current_q=sample.get("joint_angles", {}).get("q"),
                        samples=samples,
                    ),
                }

            episode_sample_index += 1
            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        if self._stop_event.is_set():
            return {
                "ok": False,
                "fatal": True,
                "failure": self._make_attempt_failure(
                    config=config,
                    reason="stop_requested",
                    target_q_start=q_start.tolist() if q_start is not None else None,
                    current_q=samples[-1].get("joint_angles", {}).get("q") if samples else None,
                    samples=samples,
                ),
            }

        final_invalid_ratio = (invalid_count / len(samples)) if samples else 1.0
        if final_invalid_ratio > config["max_invalid_sample_ratio"]:
            return {
                "ok": False,
                "fatal": False,
                "failure": self._make_attempt_failure(
                    config=config,
                    reason=f"invalid_sample_ratio:{final_invalid_ratio:.4f}",
                    target_q_start=q_start.tolist() if q_start is not None else None,
                    current_q=samples[-1].get("joint_angles", {}).get("q") if samples else None,
                    samples=samples,
                ),
            }

        episode_metadata["finished_at"] = time.time()
        episode_metadata["num_samples"] = len(samples)
        episode_metadata["valid_samples"] = int(sum(bool(s.get("valid", False)) for s in samples))
        episode_metadata["invalid_samples"] = int(len(samples) - episode_metadata["valid_samples"])
        episode_metadata["joint_limit_clip_samples"] = int(joint_limit_clip_count)
        return {"ok": True, "samples": samples, "metadata": episode_metadata}

    def _episode_abort_reason(
        self,
        *,
        sample: Dict[str, Any],
        config: Dict[str, Any],
        invalid_count: int,
        consecutive_invalid: int,
        joint_limit_clip_count: int,
    ) -> str:
        if (
            config["abort_on_joint_limit_clip"]
            and joint_limit_clip_count > config["max_joint_limit_clip_samples"]
        ):
            return "joint_limit_clipped"
        if self._sample_has_joint_angle_limit(sample):
            return "joint_angle_limit_status"
        if consecutive_invalid > config["max_consecutive_invalid_samples"]:
            return "too_many_consecutive_invalid_samples"
        sample_count = max(1, int(sample.get("episode_sample_index", 0)) + 1)
        if sample_count >= 10 and invalid_count / sample_count > config["max_invalid_sample_ratio"]:
            return "invalid_sample_ratio"
        return ""

    def _sample_has_joint_angle_limit(self, sample: Dict[str, Any]) -> bool:
        err_status = sample.get("arm_status", {}).get("err_status", {})
        if not isinstance(err_status, dict):
            return False
        return any("angle_limit" in str(key) and bool(value) for key, value in err_status.items())

    def _make_attempt_failure(
        self,
        *,
        config: Dict[str, Any],
        reason: str,
        target_q_start,
        current_q,
        samples: list,
    ) -> Dict[str, Any]:
        valid_samples = int(sum(bool(s.get("valid", False)) for s in samples))
        return {
            "episode_index": int(config.get("episode_index", 0)),
            "attempt_index": int(config.get("attempt_index", 0)),
            "episode_name": str(config.get("episode_name", "")),
            "reason": reason,
            "target_q_start": _plain(target_q_start),
            "current_q": _plain(current_q),
            "num_samples": int(len(samples)),
            "valid_samples": valid_samples,
            "invalid_samples": int(len(samples) - valid_samples),
            "created_at": time.time(),
        }

    def _commit_episode_samples(
        self,
        *,
        config: Dict[str, Any],
        paths: Dict[str, Any],
        metadata: Dict[str, Any],
        samples: list,
        global_sample_start: int,
    ) -> None:
        with open(paths["jsonl_path"], "a" if paths["append_jsonl"] else "w", encoding="utf-8") as f_jsonl:
            for sample in samples:
                f_jsonl.write(json.dumps(_plain(sample), ensure_ascii=True) + "\n")

        if paths["append_jsonl"]:
            self._update_dataset_metadata(
                config=config,
                paths=paths,
                episode_metadata=metadata,
                samples=samples,
                global_sample_start=global_sample_start,
                finished_at=time.time(),
            )
        else:
            existing = self._load_dataset_metadata(paths["metadata_path"])
            if existing.get("failed_attempts"):
                metadata["failed_attempts"] = existing["failed_attempts"]
            self._write_json(paths["metadata_path"], metadata)
            self._save_npz(paths["npz_path"], samples)

    def _record_failed_attempt(self, paths: Dict[str, Any], failure: Dict[str, Any]) -> None:
        metadata = self._load_dataset_metadata(paths["metadata_path"])
        if not metadata:
            metadata = {
                "schema_version": 1,
                "server": "nero_dynamics_server",
                "dataset_name": paths.get("dataset_name", ""),
                "save_layout": "single_dataset" if paths.get("append_jsonl") else "episode_files",
                "created_at": time.time(),
                "jsonl_path": paths["jsonl_path"],
                "npz_path": paths["npz_path"],
                "episodes": [],
                "failed_attempts": [],
                "finalized": False,
            }
        failed_attempts = list(metadata.get("failed_attempts", []))
        failed_attempts.append(_plain(failure))
        metadata["failed_attempts"] = failed_attempts
        metadata["updated_at"] = time.time()
        self._write_json(paths["metadata_path"], metadata)

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
        get_enable_status = getattr(robot, "get_joints_enable_status_list", None)
        enable_status = (
            _plain(self._safe_call(get_enable_status))
            if callable(get_enable_status)
            else None
        )
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
            "enable_status": enable_status,
        }

    def _read_motor_states(self, robot) -> Dict[str, Any]:
        position, velocity, torque, current = [], [], [], []
        timestamps, hz, valid = [], [], []
        get_motor_states = getattr(robot, "get_motor_states", None)
        if not callable(get_motor_states):
            return {
                "position": [np.nan] * 7,
                "velocity": [np.nan] * 7,
                "torque": [np.nan] * 7,
                "current": [np.nan] * 7,
                "timestamp": [np.nan] * 7,
                "hz": [np.nan] * 7,
                "valid": [False] * 7,
                "time_skew": np.nan,
                "api_available": False,
            }
        for joint_index in range(1, 8):
            result = self._safe_call(get_motor_states, joint_index)
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
            "api_available": True,
        }

    def _read_driver_states(self, robot) -> Dict[str, Any]:
        vol, foc_temp, motor_temp, bus_current = [], [], [], []
        timestamps, hz, valid, foc_status = [], [], [], []
        get_driver_states = getattr(robot, "get_driver_states", None)
        if not callable(get_driver_states):
            return {
                "vol": [np.nan] * 7,
                "foc_temp": [np.nan] * 7,
                "motor_temp": [np.nan] * 7,
                "bus_current": [np.nan] * 7,
                "timestamp": [np.nan] * 7,
                "hz": [np.nan] * 7,
                "valid": [False] * 7,
                "foc_status": [{} for _ in range(7)],
                "api_available": False,
            }
        for joint_index in range(1, 8):
            result = self._safe_call(get_driver_states, joint_index)
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
            "api_available": True,
        }

    def _read_joint_angles_message(self, robot) -> Dict[str, Any]:
        get_joint_angles = getattr(robot, "get_joint_angles", None)
        result = self._safe_call(get_joint_angles) if callable(get_joint_angles) else None
        if result is None:
            return {
                "q": [np.nan] * 7,
                "timestamp": np.nan,
                "hz": np.nan,
                "valid": False,
                "api_available": callable(get_joint_angles),
            }
        return {
            "q": _plain(result.msg),
            "timestamp": _float_or_none(getattr(result, "timestamp", np.nan)),
            "hz": _float_or_none(getattr(result, "hz", np.nan)),
            "valid": True,
            "api_available": True,
        }

    def _read_current_joint_angles(self, robot, timeout: float = 2.0) -> Optional[np.ndarray]:
        get_joint_angles = getattr(robot, "get_joint_angles", None)
        if not callable(get_joint_angles):
            return None
        start_t = time.monotonic()
        while time.monotonic() - start_t < timeout:
            result = self._safe_call(get_joint_angles)
            if result is not None:
                return np.asarray(result.msg, dtype=float)
            time.sleep(0.005)
        return None

    def _read_arm_status(self, robot) -> Dict[str, Any]:
        get_arm_status = getattr(robot, "get_arm_status", None)
        result = self._safe_call(get_arm_status) if callable(get_arm_status) else None
        if result is None:
            return {"valid": False, "api_available": callable(get_arm_status)}
        msg = result.msg
        return {
            "valid": True,
            "api_available": True,
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
            episode_index=np.asarray([int(s.get("episode_index", 0)) for s in samples], dtype=np.int64),
            episode_sample_index=np.asarray([int(s.get("episode_sample_index", i)) for i, s in enumerate(samples)], dtype=np.int64),
            global_sample_index=np.asarray([int(s.get("global_sample_index", i)) for i, s in enumerate(samples)], dtype=np.int64),
            episode_elapsed=np.asarray([s.get("episode_elapsed", s["elapsed"]) for s in samples], dtype=float),
            dataset_elapsed=np.asarray([s.get("dataset_elapsed", s["elapsed"]) for s in samples], dtype=float),
            q_start=np.asarray([s.get("q_start", [np.nan] * 7) for s in samples], dtype=float),
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
        save_layout = str(config.get("save_layout", "episode_files"))
        if save_layout == "single_dataset":
            dataset_name = config["dataset_name"] or f"{config['arm']}_{config['trajectory']}_{stamp}"
            dataset_root = os.path.join(config["output_dir"], dataset_name)
            episode_name = config["episode_name"] or f"episode_{int(config.get('episode_index', 0)):03d}_qstart"
            base = os.path.join(dataset_root, dataset_name)
            return {
                "root_dir": dataset_root,
                "dataset_name": dataset_name,
                "episode_name": episode_name,
                "jsonl_path": base + ".jsonl",
                "metadata_path": base + ".metadata.json",
                "npz_path": base + ".npz",
                "append_jsonl": True,
            }

        episode_name = config["episode_name"] or f"{config['arm']}_{config['trajectory']}_{stamp}"
        base = os.path.join(config["output_dir"], episode_name)
        return {
            "root_dir": config["output_dir"],
            "dataset_name": "",
            "episode_name": episode_name,
            "jsonl_path": base + ".jsonl",
            "metadata_path": base + ".metadata.json",
            "npz_path": base + ".npz",
            "append_jsonl": False,
        }

    def _prepare_episode_output(self, config: Dict[str, Any], paths: Dict[str, str]) -> None:
        os.makedirs(paths["root_dir"], exist_ok=True)
        if not paths["append_jsonl"] or not config.get("reset_dataset", False):
            return
        for key in ("jsonl_path", "metadata_path", "npz_path"):
            path = paths[key]
            if os.path.exists(path):
                os.remove(path)

    def _count_jsonl_lines(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for count, _ in enumerate(f, start=1):
                pass
        return count

    def _load_jsonl_samples(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _write_json(self, path: str, value: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_plain(value), f, indent=2, ensure_ascii=True)

    def _load_dataset_metadata(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _update_dataset_metadata(
        self,
        *,
        config: Dict[str, Any],
        paths: Dict[str, str],
        episode_metadata: Dict[str, Any],
        samples: list,
        global_sample_start: int,
        finished_at: float,
    ) -> None:
        metadata = self._load_dataset_metadata(paths["metadata_path"])
        if not metadata:
            metadata = {
                "schema_version": 1,
                "server": "nero_dynamics_server",
                "dataset_name": paths["dataset_name"],
                "save_layout": "single_dataset",
                "created_at": episode_metadata["created_at"],
                "config": config,
                "jsonl_path": paths["jsonl_path"],
                "npz_path": paths["npz_path"],
                "episodes": [],
                "failed_attempts": [],
                "finalized": False,
            }
        else:
            metadata.setdefault("schema_version", 1)
            metadata.setdefault("server", "nero_dynamics_server")
            metadata.setdefault("dataset_name", paths.get("dataset_name", ""))
            metadata.setdefault("save_layout", "single_dataset")
            metadata.setdefault("created_at", episode_metadata["created_at"])
            metadata.setdefault("config", config)
            metadata.setdefault("jsonl_path", paths["jsonl_path"])
            metadata.setdefault("npz_path", paths["npz_path"])
            metadata.setdefault("episodes", [])
            metadata.setdefault("failed_attempts", [])
            metadata.setdefault("finalized", False)

        episode_index = int(config["episode_index"])
        valid_samples = int(sum(bool(s.get("valid", False)) for s in samples))
        episode_record = {
            "episode_index": episode_index,
            "attempt_index": int(config.get("attempt_index", 0)),
            "episode_name": paths["episode_name"],
            "trajectory": config["trajectory"],
            "arm": config["arm"],
            "initial_q": episode_metadata.get("initial_q"),
            "random_start_q": episode_metadata.get("random_start_q"),
            "trajectory_runtime": episode_metadata.get("trajectory_runtime", {}),
            "start_global_sample": int(global_sample_start),
            "num_samples": int(len(samples)),
            "valid_samples": valid_samples,
            "invalid_samples": int(len(samples) - valid_samples),
            "joint_limit_clip_samples": int(
                sum(bool(s.get("command", {}).get("joint_limit_clipped", False)) for s in samples)
            ),
            "started_at": episode_metadata["created_at"],
            "finished_at": finished_at,
        }
        episodes = [e for e in metadata.get("episodes", []) if int(e.get("episode_index", -1)) != episode_index]
        episodes.append(episode_record)
        episodes.sort(key=lambda e: int(e.get("episode_index", 0)))
        metadata["episodes"] = episodes
        metadata["total_samples"] = int(sum(int(e.get("num_samples", 0)) for e in episodes))
        metadata["valid_samples"] = int(sum(int(e.get("valid_samples", 0)) for e in episodes))
        metadata["updated_at"] = time.time()
        metadata["finalized"] = False
        self._write_json(paths["metadata_path"], metadata)

    def _sample_random_start(self, config: Dict[str, Any], robot_arm: str) -> np.ndarray:
        seed = int(config["random_start_seed"]) + int(config["episode_index"]) * 1000 + int(config.get("attempt_index", 0))
        rng = np.random.default_rng(seed)

        if config.get("random_start_min") and config.get("random_start_max"):
            q_min = np.asarray(config["random_start_min"], dtype=float).reshape(7)
            q_max = np.asarray(config["random_start_max"], dtype=float).reshape(7)
            low = np.minimum(q_min, q_max)
            high = np.maximum(q_min, q_max)
            q = rng.uniform(low, high, size=7)
        else:
            center = config.get("random_start_center") or self._home_for_arm(robot_arm)
            center = np.asarray(center, dtype=float).reshape(7)
            span = np.asarray(config["random_start_range"], dtype=float).reshape(7)
            q = center + rng.uniform(-span, span, size=7)
        return self._clip_to_joint_limits(q, robot_arm, margin=config["joint_limit_margin"])

    def _home_for_arm(self, robot_arm: str) -> list:
        return DEFAULT_LEFT_HOME if robot_arm == "left_robot" else DEFAULT_RIGHT_HOME

    def _clip_to_joint_limits(self, q: np.ndarray, robot_arm: str, margin: float = 0.0) -> np.ndarray:
        clipped, _ = self._clip_to_joint_limits_with_report(q, robot_arm, margin=margin)
        return clipped

    def _clip_to_joint_limits_with_report(
        self, q: np.ndarray, robot_arm: str, margin: float = 0.0
    ) -> Tuple[np.ndarray, bool]:
        cfg = self.left_cfg if robot_arm == "left_robot" else self.right_cfg
        q_arr = np.asarray(q, dtype=float).reshape(7)
        if cfg is None or "joint_limits" not in cfg:
            return q_arr.copy(), False
        out = q_arr.copy()
        margin = max(0.0, float(margin))
        for i in range(1, 8):
            lo, hi = cfg["joint_limits"][f"joint{i}"]
            safe_lo = float(lo) + margin
            safe_hi = float(hi) - margin
            if safe_lo > safe_hi:
                mid = 0.5 * (float(lo) + float(hi))
                safe_lo = mid
                safe_hi = mid
            out[i - 1] = np.clip(out[i - 1], safe_lo, safe_hi)
        clipped = bool(np.any(np.abs(out - q_arr) > 1e-12))
        return out, clipped

    def _move_to_joint_target(
        self,
        *,
        robot_arm: str,
        robot,
        target: np.ndarray,
        speed_percent: float,
        timeout: float,
        tolerance: float,
        startup_grace_s: float,
        stationary_checks: int,
    ) -> bool:
        set_speed_percent = getattr(robot, "set_speed_percent", None)
        move_j = getattr(robot, "move_j", None)
        if not callable(move_j):
            log.error("[%s] move_j API is not available; cannot move to random start", robot_arm)
            return False
        if callable(set_speed_percent):
            self._safe_call(set_speed_percent, self._normalize_speed_percent(speed_percent))
        target_list = np.asarray(target, dtype=float).reshape(7).tolist()
        move_j(target_list)
        return self._wait_for_motion_complete(
            robot,
            target_list,
            timeout=timeout,
            tolerance=tolerance,
            startup_grace_s=startup_grace_s,
            stationary_checks=stationary_checks,
        )

    def _go_home(self, robot_arm: str, robot) -> bool:
        home = DEFAULT_LEFT_HOME if robot_arm == "left_robot" else DEFAULT_RIGHT_HOME
        set_speed_percent = getattr(robot, "set_speed_percent", None)
        move_j = getattr(robot, "move_j", None)
        if not callable(move_j):
            log.error("[%s] move_j API is not available; cannot go home", robot_arm)
            return False
        if callable(set_speed_percent):
            self._safe_call(set_speed_percent, self._normalize_speed_percent(30))
        move_j(home)
        return self._wait_for_motion_complete(robot, home, timeout=20.0)

    def _normalize_speed_percent(self, speed_percent: float) -> int:
        return int(np.clip(round(float(speed_percent)), 1, 100))

    def _wait_for_motion_complete(
        self,
        robot,
        target_joints: list,
        timeout: float = 10.0,
        tolerance: float = 0.01,
        startup_grace_s: float = 0.0,
        stationary_checks: int = 1,
    ) -> bool:
        start_t = time.monotonic()
        target = np.asarray(target_joints, dtype=float)
        get_joint_angles = getattr(robot, "get_joint_angles", None)
        get_arm_status = getattr(robot, "get_arm_status", None)
        stopped_count = 0
        stationary_checks = max(1, int(stationary_checks))
        startup_grace_s = max(0.0, float(startup_grace_s))
        while time.monotonic() - start_t < timeout:
            elapsed = time.monotonic() - start_t
            result = self._safe_call(get_joint_angles) if callable(get_joint_angles) else None
            if result is None:
                time.sleep(0.05)
                continue
            current = np.asarray(result.msg, dtype=float)
            if np.allclose(current, target, atol=tolerance):
                return True
            status = self._safe_call(get_arm_status) if callable(get_arm_status) else None
            motion_status = getattr(status.msg, "motion_status", None) if status is not None else None
            if elapsed >= startup_grace_s and motion_status == 0:
                stopped_count += 1
                if stopped_count >= stationary_checks:
                    max_error = float(np.max(np.abs(current - target)))
                    log.warning(
                        "[wait_for_motion] stopped before target. max_error=%.4f tolerance=%.4f current=%s target=%s",
                        max_error,
                        tolerance,
                        current.round(4).tolist(),
                        target.round(4).tolist(),
                    )
                    return False
            elif motion_status != 0:
                stopped_count = 0
            time.sleep(0.1)
        result = self._safe_call(get_joint_angles) if callable(get_joint_angles) else None
        if result is not None:
            current = np.asarray(result.msg, dtype=float)
            max_error = float(np.max(np.abs(current - target)))
            log.warning(
                "[wait_for_motion] timeout after %.2fs. max_error=%.4f tolerance=%.4f current=%s target=%s",
                timeout,
                max_error,
                tolerance,
                current.round(4).tolist(),
                target.round(4).tolist(),
            )
        return False

    def robot_stop(self, robot_arm: str) -> bool:
        try:
            robot = self._select_robot(robot_arm)
            if robot is not None:
                stop = getattr(robot, "electronic_emergency_stop", None)
                if callable(stop):
                    stop()
                else:
                    log.warning("[%s] electronic_emergency_stop API is not available", robot_arm)
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
