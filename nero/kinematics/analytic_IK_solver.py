import time
import math
import os
import sys
import numpy as np
from dataclasses import replace


def rpy_to_rot(roll: float, pitch: float, yaw: float):
    """Rotation matrix for roll-pitch-yaw using ZYX convention."""
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def rot_to_rpy(R):
    """Inverse of rpy_to_rot for ZYX convention. Returns [roll, pitch, yaw]."""
    R = np.asarray(R, dtype=float)
    r20 = float(R[2, 0])
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(float(-R[0, 1]), float(R[1, 1]))
    else:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return [roll, pitch, yaw]

# 添加 ik_solver 路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kinematics'))

from nero.kinematics.nero_kinematics.nero_ik.ik_solver import (
    fk,
    NeroParams,
    ContinuityParams,
    ContinuityRuntimeState,
    solve_pose_continuous_with_state,
)

class Solver:
    """
    基于 ik_solver.py 的解析 IK 求解器
    使用 ik_arm_angle_with_report 进行单帧求解
    
    性能优化：
    - n_psi: 全局扫描点数，默认61（原181）
    - local_theta0_count: 局部窗口点数，默认21（原41）
    - 禁用1D QP优化（额外开销）
    """
    def __init__(self, joint_limits, dt, n_psi=61):
        self.joint_limits = joint_limits
        self.dt = dt
        self.n_psi = n_psi  # 减少扫描点数：181→61
        
        # 使用默认的 NERO DH 参数
        self.nero_params = NeroParams.default()
        
        # 连续性参数 - 优化性能
        self.continuity = ContinuityParams(
            # 适度缩小局部臂角窗口，降低跨分支切换概率
            local_theta0_window=0.25,
            local_theta0_count=21,  # 减少局部扫描点：41→21
            # 连续性项权重上调，优先轨迹平滑
            w_vel=1.6,
            w_acc=0.55,
            w_pose=0.1,
            w_theta0=0.35,
            hysteresis_margin=0.08,
            # Default-off to avoid expensive fallback in the fast servo loop.
            enable_global_fallback=False,
            w_qp_joint_inc=0.0,  # 禁用QP优化
            w_qp_pose_err=0.0,   # 禁用QP优化
        )
        self._base_local_theta0_window = float(self.continuity.local_theta0_window)
        self._base_local_theta0_count = int(self.continuity.local_theta0_count)
        self._max_local_theta0_window = 0.45
        self._max_local_theta0_count = 33
        # Adaptive policy: only allow global fallback after multiple consecutive failures.
        self._consecutive_failures = 0
        self._fallback_after_failures = 4

        # 输出侧防跳变参数（rad/s 与 rad）
        # 若知道各轴最大安全速度，建议按实机参数配置。
        self.max_joint_vel = np.array([2.2, 2.0, 2.2, 2.2, 2.6, 2.6, 3.0], dtype=float)
        self.min_step_limit = 0.03
        self.jump_detect_scale = 3.0
        self.hard_jump_limit = 0.90

        # 最近一次求解诊断
        self.last_report = None
        self.last_jump_report = None
        
        # 运行时状态
        self.state = None
    
    def _pose_to_matrix(self, pose):
        """将 6D pose [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵"""
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.array(rpy_to_rot(pose[3], pose[4], pose[5]), dtype=float)
        T[:3, 3] = np.array(pose[:3], dtype=float)
        return T
    
    def _clamp_joints(self, q):
        """关节限位裁剪"""
        q_out = np.array(q, dtype=float)
        for i, (lo, hi) in enumerate(self.joint_limits):
            q_out[i] = min(max(q_out[i], lo), hi)
        return q_out

    def _compute_step_limit(self):
        """根据速度上限与控制周期，得到每个关节单步最大改变量。"""
        return np.maximum(self.max_joint_vel * float(self.dt), self.min_step_limit)

    def _detect_and_guard_output(self, q_cmd):
        """检测并抑制关节角跳变，返回 (q_safe, jump_report)。"""
        q_cmd = np.array(q_cmd, dtype=float)
        if self.state is None or self.state.q_prev is None:
            return self._clamp_joints(q_cmd), {
                "jump_detected": False,
                "joint_indices": [],
                "dq_raw": [0.0] * 7,
                "dq_limited": [0.0] * 7,
                "mode": "no_prev_state",
            }

        q_prev = np.array(self.state.q_prev, dtype=float)
        dq_raw = np.array((q_cmd - q_prev), dtype=float)
        dq_raw = (dq_raw + np.pi) % (2.0 * np.pi) - np.pi

        step_limit = self._compute_step_limit()
        detect_limit = np.maximum(step_limit * self.jump_detect_scale, self.min_step_limit)

        jump_mask = np.abs(dq_raw) > detect_limit
        very_large_jump = np.any(np.abs(dq_raw) > self.hard_jump_limit)

        dq_limited = np.clip(dq_raw, -step_limit, step_limit)
        q_safe = self._clamp_joints(q_prev + dq_limited)

        if very_large_jump:
            # 极端跳变时冻结到上一次状态，避免打杆。
            q_safe = q_prev.copy()

        jump_report = {
            "jump_detected": bool(np.any(jump_mask)),
            "joint_indices": np.where(jump_mask)[0].astype(int).tolist(),
            "dq_raw": dq_raw.astype(float).tolist(),
            "dq_limited": dq_limited.astype(float).tolist(),
            "step_limit": step_limit.astype(float).tolist(),
            "detect_limit": detect_limit.astype(float).tolist(),
            "very_large_jump": bool(very_large_jump),
            "mode": "freeze" if very_large_jump else "rate_limit",
        }
        return q_safe, jump_report
    
    def init_state(self, current_q):
        """初始化求解器状态（仅调用一次）"""
        current_q = self._clamp_joints(np.array(current_q, dtype=float))
        self.state = ContinuityRuntimeState(q_prev=current_q)
    
    def solve(self, target_pose, limit_output_step: bool = True):
        """
        求解目标 TCP 位姿对应的关节角
        :param target_pose: TCP 6D pose [x, y, z, roll, pitch, yaw]
        :return: 7维关节角，失败返回 None
        """
        T_target = self._pose_to_matrix(target_pose)

        # Adapt local window/count when failures accumulate to improve local solve hit-rate.
        fail_k = self._consecutive_failures
        run_continuity = replace(self.continuity)
        if fail_k > 0:
            run_continuity.local_theta0_window = min(
                self._max_local_theta0_window,
                self._base_local_theta0_window + 0.04 * fail_k,
            )
            run_continuity.local_theta0_count = min(
                self._max_local_theta0_count,
                self._base_local_theta0_count + 2 * fail_k,
            )
        run_continuity.enable_global_fallback = fail_k >= self._fallback_after_failures
        
        # 使用 solve_pose_continuous_with_state 求解
        q_cmd, report, new_state = solve_pose_continuous_with_state(
            T_target,
            state=self.state,
            p=self.nero_params,
            n_psi=self.n_psi,
            continuity=run_continuity,
        )
        self.last_report = report
        
        if q_cmd is None:
            self._consecutive_failures += 1
            # IK 求解失败，不更新状态，返回 None
            print(f"[IK] solve failed: {report.get('method')}")
            print(f"   目标位姿: x={target_pose[0]:.3f}, y={target_pose[1]:.3f}, z={target_pose[2]:.3f}")
            print(f"   候选解数量: {report.get('candidate_count', 0)}")
            return None
        self._consecutive_failures = 0
        
        # 关节限位裁剪
        q_cmd_clamped = self._clamp_joints(q_cmd)

        # 输出侧跳变检测与抑制（可按调用方需求关闭）
        if limit_output_step:
            q_out, jump_report = self._detect_and_guard_output(q_cmd_clamped)
        else:
            q_out = q_cmd_clamped
            jump_report = {
                "jump_detected": False,
                "joint_indices": [],
                "dq_raw": [0.0] * 7,
                "dq_limited": [0.0] * 7,
                "mode": "bypass_rate_limit",
            }
        self.last_jump_report = jump_report
        if jump_report["jump_detected"]:
            idx_str = ",".join(str(i + 1) for i in jump_report["joint_indices"])
            print(f"[IK] jump detected on joints [{idx_str}], guard mode={jump_report['mode']}")
        
        # 成功时更新状态（使用裁剪后的关节角）
        self.state = ContinuityRuntimeState(
            q_prev=q_out,
            q_prev2=self.state.q_prev.copy() if self.state.q_prev is not None else q_out.copy(),
            theta0_prev=new_state.theta0_prev,
            q_lock=q_out,
        )
        
        return q_out
    
from dataclasses import dataclass

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

@dataclass
class ContinuityRuntimeState:
    """Runtime state kept for API compatibility with existing server code."""

    q_prev: np.ndarray
    q_prev2: np.ndarray = None
    theta0_prev: float = None
    q_lock: np.ndarray = None

class Pinocchio_Solver:
    """
    基于 Pinocchio 的迭代 IK 求解器（DLS）。
    兼容原 Solver 的外部接口：init_state/solve/state/last_report。
    """
    def __init__(
        self,
        joint_limits,
        dt,
        n_psi=61,
        urdf_path=None,
        ee_frame_name="link7",
        tcp_offset=None,
        max_iterations=60,
        damping=1e-4,
        tol_pos=1e-4,
        tol_rot=5e-3,
    ):
        if pin is None:
            raise RuntimeError(
                "Pinocchio is required for Solver. Install with: "
                "conda install -c conda-forge pinocchio eigenpy -y"
            )

        self.joint_limits = joint_limits
        self.dt = dt
        self.n_psi = n_psi  # Kept for backward compatibility.
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.tol_pos = float(tol_pos)
        self.tol_rot = float(tol_rot)

        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self._resolve_ee_frame_id(ee_frame_name)
        self._active_q_idx, self._active_v_idx = self._resolve_active_joint_indices()
        self.joint_names = self._resolve_active_joint_names()
        self.frame_names = [frame.name for frame in self.model.frames]
        if len(self._active_q_idx) != len(self.joint_limits):
            raise RuntimeError(
                f"URDF active joints({len(self._active_q_idx)}) and joint_limits"
                f"({len(self.joint_limits)}) mismatch"
            )

        self._q_lo = np.array([lo for lo, _ in self.joint_limits], dtype=float)
        self._q_hi = np.array([hi for _, hi in self.joint_limits], dtype=float)
        self.set_tool_offset([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if tcp_offset is None else tcp_offset)

        # 输出侧防跳变参数（rad/s 与 rad）
        # 若知道各轴最大安全速度，建议按实机参数配置。
        self.max_joint_vel = np.array([2.2, 2.0, 2.2, 2.2, 2.6, 2.6, 3.0], dtype=float)
        self.min_step_limit = 0.03
        self.jump_detect_scale = 3.0
        self.hard_jump_limit = 0.90

        # 最近一次求解诊断
        self.last_report = None
        self.last_jump_report = None
        
        # 运行时状态
        self.state = None

    def _resolve_urdf_path(self, urdf_path):
        if urdf_path is not None:
            candidate = os.path.abspath(os.path.expanduser(str(urdf_path)))
            if os.path.isfile(candidate):
                return candidate

        env_path = os.getenv("NERO_URDF_PATH")
        if env_path:
            candidate = os.path.abspath(os.path.expanduser(env_path))
            if os.path.isfile(candidate):
                return candidate

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.abspath(
                os.path.join(
                    base_dir,
                    "..",
                    "..",
                    "asserts",
                    "agx_arm_urdf-main",
                    "nero",
                    "urdf",
                    "nero_description.urdf",
                )
            ),
            # os.path.expanduser("~/pyAgxArm/asserts/agx_arm_urdf-main/nero/urdf/nero_description.urdf"),
            os.path.join(
                base_dir,
                "..",
                "..",
                "pyAgxArm",
                "asserts",
                "agx_arm_urdf",
                "nero",
                "urdf",
                "nero_description.urdf",
            ),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            "NERO URDF not found. Set NERO_URDF_PATH or place URDF at "
            "asserts/agx_arm_urdf-main/nero/urdf/nero_description.urdf"
        )

    def _resolve_ee_frame_id(self, ee_frame_name):
        try:
            frame_id = self.model.getFrameId(ee_frame_name)
            if frame_id < self.model.nframes and self.model.frames[frame_id].name == ee_frame_name:
                return frame_id
        except Exception:
            pass
        raise RuntimeError(f"End-effector frame '{ee_frame_name}' not found in URDF")

    def _resolve_active_joint_indices(self):
        preferred = [f"joint{i}" for i in range(1, len(self.joint_limits) + 1)]
        all_names = list(self.model.names)
        if all(name in all_names for name in preferred):
            q_idx = []
            v_idx = []
            for name in preferred:
                joint_id = self.model.getJointId(name)
                jmodel = self.model.joints[joint_id]
                q_idx.append(jmodel.idx_q)
                v_idx.append(jmodel.idx_v)
            return q_idx, v_idx

        q_idx = []
        v_idx = []
        for joint_id in range(1, self.model.njoints):
            jmodel = self.model.joints[joint_id]
            if jmodel.nq == 1 and jmodel.nv == 1:
                q_idx.append(jmodel.idx_q)
                v_idx.append(jmodel.idx_v)
        q_idx = q_idx[: len(self.joint_limits)]
        v_idx = v_idx[: len(self.joint_limits)]
        return q_idx, v_idx

    def _resolve_active_joint_names(self):
        by_q_idx = {}
        for joint_id in range(1, self.model.njoints):
            jmodel = self.model.joints[joint_id]
            if jmodel.nq == 1 and jmodel.nv == 1:
                by_q_idx[jmodel.idx_q] = self.model.names[joint_id]
        return [by_q_idx.get(q_idx, f"joint{i + 1}") for i, q_idx in enumerate(self._active_q_idx)]

    def _to_full_q(self, q):
        q_full = pin.neutral(self.model)
        q = np.asarray(q, dtype=float).reshape(-1)
        for i, q_idx in enumerate(self._active_q_idx):
            q_full[q_idx] = q[i]
        return q_full

    def _pose6_to_matrix(self, pose6):
        pose6 = np.asarray(pose6, dtype=float).reshape(-1)
        if pose6.size != 6:
            raise ValueError(f"Expected 6 pose values, got {pose6.size}")
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.array(rpy_to_rot(pose6[3], pose6[4], pose6[5]), dtype=float)
        T[:3, 3] = pose6[:3]
        return T

    def set_tool_offset(self, tcp_offset):
        """Set TCP offset relative to ee_frame (ee -> tcp), [x,y,z,roll,pitch,yaw]."""
        self.tcp_offset = np.asarray(tcp_offset, dtype=float).reshape(-1)
        if self.tcp_offset.size != 6:
            raise ValueError(f"Expected 6 tcp_offset values, got {self.tcp_offset.size}")
        self._T_ee_tcp = self._pose6_to_matrix(self.tcp_offset)
        self._T_tcp_ee = np.linalg.inv(self._T_ee_tcp)

    def fk_matrix(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        q_full = self._to_full_q(self._clamp_joints(q))
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        T = np.eye(4, dtype=float)
        placement = self.data.oMf[self.ee_frame_id]
        T[:3, :3] = np.asarray(placement.rotation, dtype=float)
        T[:3, 3] = np.asarray(placement.translation, dtype=float)
        # Return world->tcp instead of world->ee(link7)
        return T @ self._T_ee_tcp

    def fk_pose(self, q):
        T = self.fk_matrix(q)
        rpy = np.asarray(pin.rpy.matrixToRpy(T[:3, :3]), dtype=float)
        return np.concatenate([T[:3, 3], rpy])

    def jacobian_matrix(self, q, reference_frame=None):
        """
        Return the 6xN geometric Jacobian for the configured TCP.

        Rows are ordered as [linear_velocity; angular_velocity]. By default
        velocities are expressed in a world-aligned frame, which matches the
        finite-difference debug tools in this repository.
        """
        if reference_frame is None:
            reference_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        q = np.asarray(q, dtype=float).reshape(-1)
        q_full = self._to_full_q(self._clamp_joints(q))
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_full,
            self.ee_frame_id,
            reference_frame,
        )
        J = np.asarray(J_full[:, self._active_v_idx], dtype=float).copy()

        tcp_translation = self._T_ee_tcp[:3, 3]
        if (
            reference_frame == pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            and np.linalg.norm(tcp_translation) > 1e-12
        ):
            placement = self.data.oMf[self.ee_frame_id]
            offset_world = np.asarray(placement.rotation, dtype=float) @ tcp_translation
            for col in range(J.shape[1]):
                J[:3, col] += np.cross(J[3:, col], offset_world)
        return J

    jacobian = jacobian_matrix
    
    def _pose_to_matrix(self, pose):
        """将 6D pose [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵"""
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.array(rpy_to_rot(pose[3], pose[4], pose[5]), dtype=float)
        T[:3, 3] = np.array(pose[:3], dtype=float)
        return T
    
    def _clamp_joints(self, q):
        """关节限位裁剪"""
        q_out = np.array(q, dtype=float)
        for i, (lo, hi) in enumerate(self.joint_limits):
            q_out[i] = min(max(q_out[i], lo), hi)
        return q_out

    def _compute_step_limit(self):
        """根据速度上限与控制周期，得到每个关节单步最大改变量。"""
        return np.maximum(self.max_joint_vel * float(self.dt), self.min_step_limit)

    def _detect_and_guard_output(self, q_cmd):
        """检测并抑制关节角跳变，返回 (q_safe, jump_report)。"""
        q_cmd = np.array(q_cmd, dtype=float)
        if self.state is None or self.state.q_prev is None:
            return self._clamp_joints(q_cmd), {
                "jump_detected": False,
                "joint_indices": [],
                "dq_raw": [0.0] * 7,
                "dq_limited": [0.0] * 7,
                "mode": "no_prev_state",
            }

        q_prev = np.array(self.state.q_prev, dtype=float)
        dq_raw = np.array((q_cmd - q_prev), dtype=float)
        dq_raw = (dq_raw + np.pi) % (2.0 * np.pi) - np.pi

        step_limit = self._compute_step_limit()
        detect_limit = np.maximum(step_limit * self.jump_detect_scale, self.min_step_limit)

        jump_mask = np.abs(dq_raw) > detect_limit
        very_large_jump = np.any(np.abs(dq_raw) > self.hard_jump_limit)

        dq_limited = np.clip(dq_raw, -step_limit, step_limit)
        q_safe = self._clamp_joints(q_prev + dq_limited)

        if very_large_jump:
            # 极端跳变时冻结到上一次状态，避免打杆。
            q_safe = q_prev.copy()

        jump_report = {
            "jump_detected": bool(np.any(jump_mask)),
            "joint_indices": np.where(jump_mask)[0].astype(int).tolist(),
            "dq_raw": dq_raw.astype(float).tolist(),
            "dq_limited": dq_limited.astype(float).tolist(),
            "step_limit": step_limit.astype(float).tolist(),
            "detect_limit": detect_limit.astype(float).tolist(),
            "very_large_jump": bool(very_large_jump),
            "mode": "freeze" if very_large_jump else "rate_limit",
        }
        return q_safe, jump_report
    
    def init_state(self, current_q):
        """初始化求解器状态（仅调用一次）"""
        current_q = self._clamp_joints(np.array(current_q, dtype=float))
        self.state = ContinuityRuntimeState(q_prev=current_q)
    
    def solve(self, target_pose, limit_output_step: bool = True):
        """
        求解目标 TCP 位姿对应的关节角
        :param target_pose: TCP 6D pose [x, y, z, roll, pitch, yaw]
        :return: 7维关节角，失败返回 None
        """
        target_pose = np.asarray(target_pose, dtype=float).reshape(-1)
        if target_pose.size != 6:
            raise ValueError(f"Expected 6 pose values, got {target_pose.size}")

        if self.state is None or self.state.q_prev is None:
            q_seed = 0.5 * (self._q_lo + self._q_hi)
            self.state = ContinuityRuntimeState(q_prev=q_seed.copy())
        else:
            q_seed = np.asarray(self.state.q_prev, dtype=float).reshape(-1)

        T_target = self._pose_to_matrix(target_pose)
        # Solve IK in ee_frame (link7): T_world_ee = T_world_tcp * inv(T_ee_tcp)
        T_target_ee = T_target @ self._T_tcp_ee
        target_se3 = pin.SE3(T_target_ee[:3, :3], T_target_ee[:3, 3])

        q = self._clamp_joints(q_seed)
        q_full = self._to_full_q(q)
        converged = False
        pos_err = float("inf")
        rot_err = float("inf")
        iter_count = 0

        for i in range(self.max_iterations):
            iter_count = i + 1
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            current_se3 = self.data.oMf[self.ee_frame_id]

            err6 = pin.log6(current_se3.inverse() * target_se3).vector
            pos_err = float(np.linalg.norm(err6[:3]))
            rot_err = float(np.linalg.norm(err6[3:]))
            if pos_err < self.tol_pos and rot_err < self.tol_rot:
                converged = True
                break

            J_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            J = J_full[:, self._active_v_idx]
            H = J @ J.T + self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(H, err6)

            step_limit = self._compute_step_limit()
            dq = np.clip(np.asarray(dq, dtype=float), -step_limit, step_limit)
            q = self._clamp_joints(q + dq)
            q_full = self._to_full_q(q)

        report = {
            "method": "pinocchio_dls",
            "converged": bool(converged),
            "iterations": int(iter_count),
            "pos_error_m": float(pos_err),
            "rot_error_rad": float(rot_err),
            "urdf_path": self.urdf_path,
            "ee_frame": self.ee_frame_name,
            "reason": "converged" if converged else "max_iterations",
            "timed_out": bool((not converged) and iter_count >= self.max_iterations),
            "last_q": q.astype(float).tolist(),
            "best_q": q.astype(float).tolist(),
        }
        self.last_report = report

        if not converged:
            print(f"[IK] solve failed: {report['method']}")
            print(f"   目标位姿: x={target_pose[0]:.3f}, y={target_pose[1]:.3f}, z={target_pose[2]:.3f}")
            print(f"   误差: pos={pos_err:.4f}m, rot={rot_err:.4f}rad, iters={iter_count}")
            return None

        # 关节限位裁剪
        q_cmd_clamped = self._clamp_joints(q)

        # 输出侧跳变检测与抑制（可按调用方需求关闭）
        if limit_output_step:
            q_out, jump_report = self._detect_and_guard_output(q_cmd_clamped)
        else:
            q_out = q_cmd_clamped
            jump_report = {
                "jump_detected": False,
                "joint_indices": [],
                "dq_raw": [0.0] * 7,
                "dq_limited": [0.0] * 7,
                "mode": "bypass_rate_limit",
            }
        self.last_jump_report = jump_report
        if jump_report["jump_detected"]:
            idx_str = ",".join(str(i + 1) for i in jump_report["joint_indices"])
            print(f"[IK] jump detected on joints [{idx_str}], guard mode={jump_report['mode']}")
        
        # 成功时更新状态（使用裁剪后的关节角）
        self.state = ContinuityRuntimeState(
            q_prev=q_out,
            q_prev2=self.state.q_prev.copy() if self.state.q_prev is not None else q_out.copy(),
            theta0_prev=None,
            q_lock=q_out,
        )
        self.last_report["solution_q"] = q_out.astype(float).tolist()
        
        return q_out
