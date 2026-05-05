'''
Nero dual-arm robot interface server.
Provides zerorpc interface for dual-arm control.
'''

import zerorpc
import numpy as np
import logging
import time
import math
from typing import Optional, List
import sys, os
import pdb

from nero.kinematics.analytic_IK_solver import Pinocchio_Solver
# from nero.kinematics.nero_kinematics.nero_ik.ik_solver import fk

log = logging.getLogger(__name__)

# 手动实现四元数乘法 (输入输出均为 [x, y, z, w] 格式)
def quat_multiply(q1, q2):
    """四元数乘法，输入输出格式均为 [x, y, z, w]"""
    x1, y1, z1, w1 = q1  # [x, y, z, w]
    x2, y2, z2, w2 = q2  # [x, y, z, w]
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
    )
class NeroDualArmServer:
    """Dual-arm Nero server interface."""
    
    def __init__(self, gripper_enabled: bool = True):
        self.gripper_enabled = gripper_enabled
        # Initialize IK handles early: go_home may run before IK setup completes.
        self.left_ik_solver = None
        self.right_ik_solver = None
        # Must exist before _setup_gripper triggers *_gripper_goto calls.
        self._last_left_gripper_cmd = None
        self._last_right_gripper_cmd = None

        # Initialize left arm
        self.left_robot = None
        self.left_gripper = None
        
        try:
            from pyAgxArm import create_agx_arm_config, AgxArmFactory
            self.left_cfg = create_agx_arm_config(robot="nero", comm="can", channel="can_left")
            self.left_robot = AgxArmFactory.create_arm(self.left_cfg)
            
            # ⚠️ 关键步骤：在 connect 前初始化末端执行器
            if gripper_enabled:
                self.left_gripper = self.left_robot.init_effector(self.left_robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            
            self.left_robot.connect()
            time.sleep(0.3)

            # 清除底层错误状态（防止之前卡在透传或急停模式）
            self.left_robot.set_normal_mode()
            time.sleep(0.3)

            # Enable all joints
            start_t = time.monotonic()
            while not self.left_robot.enable(255):
                if time.monotonic() - start_t > 5.0:
                    log.warning("[SERVER] Left arm enable timeout")
                    break
                time.sleep(0.01)
            self.left_robot_go_home()
            log.info("[SERVER] Left arm connected and enabled")
            
        except Exception as e:
            log.error(f"[SERVER] Failed to connect to left arm: {e}")

        # Initialize right arm
        self.right_robot = None
        self.right_gripper = None
        
        try:
            from pyAgxArm import create_agx_arm_config, AgxArmFactory
            self.right_cfg = create_agx_arm_config(robot="nero", comm="can", channel="can_right")
            self.right_robot = AgxArmFactory.create_arm(self.right_cfg)
            
            # ⚠️ 关键步骤：在 connect 前初始化末端执行器
            if gripper_enabled:
                self.right_gripper = self.right_robot.init_effector(self.right_robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            
            self.right_robot.connect()
            time.sleep(0.3)

            # 清除底层错误状态（防止之前卡在透传或急停模式）
            self.right_robot.set_normal_mode()
            time.sleep(0.3)

            # Enable all joints
            start_t = time.monotonic()
            while not self.right_robot.enable(255):
                if time.monotonic() - start_t > 5.0:
                    log.warning("[SERVER] Right arm enable timeout")
                    break
                time.sleep(0.01)
            self.right_robot_go_home()
            log.info("[SERVER] Right arm connected and enabled")
            
        except Exception as e:
            log.error(f"[SERVER] Failed to connect to right arm: {e}")
        
        log.info("=" * 50)
        log.info("Nero Dual-Arm Server Ready")
        log.info("=" * 50)

        # Setup grippers (初始化已在 connect 前完成，这里只做开合测试)
        if gripper_enabled:
            try:
                if self.left_gripper is not None:
                    self._setup_gripper(self.left_gripper, "left")
                    log.info("[SERVER] Left gripper initialized")
            except Exception as e:
                log.error(f"[SERVER] Failed to setup left gripper: {e}")
        
            try:
                if self.right_gripper is not None:
                    self._setup_gripper(self.right_gripper, "right")
                    log.info("[SERVER] Right gripper initialized")
            except Exception as e:
                log.error(f"[SERVER] Failed to setup right gripper: {e}")

            log.info("=" * 50)
            log.info("Nero Dual-Gripper Server Ready")
            log.info("=" * 50)

        self.tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Initialize IK solver
        # Keep server-side IK timestep aligned with teleop action send rate.
        self.track_freq = 50.0
        self.dt = 1.0 / self.track_freq
        # Per-cycle safety limits for servo delta commands.
        self.max_cart_step_m = 0.03
        self.max_rot_step_rad = 0.35
        self.max_joint_step_rad = 0.1
        self.ik_resync_thresh_rad = 0.1
        self.max_ik_solve_ms = 30.0
        # If IK falls into expensive global fallback, skip a short window to avoid repeated stalls.
        self.ik_fail_cooldown_s = 0.25
        self._ik_skip_until = {"left_robot": 0.0, "right_robot": 0.0}
        # Adaptive attenuation after IK failures: shrink delta commands and briefly freeze orientation.
        self.ik_delta_decay = 0.6
        self.ik_delta_recover = 0.08
        self.ik_delta_min_scale = 0.2
        self.ik_orientation_freeze_s = 0.20
        self._ik_delta_scale = {"left_robot": 1.0, "right_robot": 1.0}
        self._ik_freeze_rot_until = {"left_robot": 0.0, "right_robot": 0.0}

        # High-frequency stdout printing can introduce control-loop jitter.
        self.enable_servo_timing_print = False
        self.servo_timing_print_every_n = 50
        self.gripper_print = False

        # servo_p 开环控制记录的当前位姿
        self.left_cur_pose = None
        self.right_cur_pose = None
        self.ik_urdf_path = os.getenv("NERO_URDF_PATH")

        try:
            if self.left_robot is not None and hasattr(self, "left_cfg"):
                self.left_ik_solver = self._setup_ik_solver(self.left_robot, self.left_cfg, "Left Arm")
            if self.right_robot is not None and hasattr(self, "right_cfg"):
                self.right_ik_solver = self._setup_ik_solver(self.right_robot, self.right_cfg, "Right Arm")
        except Exception as e:
            log.error(f"[SERVER] IK solvers init failed: {e}")

    def _get_current_joints(self, robot, timeout: float = 2.0):
        """Read current joint angles with timeout protection."""
        current_joints = None
        start_t = time.monotonic()
        while current_joints is None:
            ja = robot.get_joint_angles()
            if ja is not None:
                current_joints = np.asarray(ja.msg, dtype=float)
                break
            if time.monotonic() - start_t > timeout:
                return None
            time.sleep(0.005)
        return current_joints

    def _limit_pose_delta(self, pose_delta: np.ndarray) -> np.ndarray:
        """Clamp per-cycle Cartesian and angular increments for stable servo."""
        pose_delta = np.asarray(pose_delta, dtype=float).reshape(6)
        out = pose_delta.copy()
        out[:3] = np.clip(out[:3], -self.max_cart_step_m, self.max_cart_step_m)
        out[3:] = np.clip(out[3:], -self.max_rot_step_rad, self.max_rot_step_rad)
        return out

    def _sync_ik_state_if_needed(self, ik_solver, q_current: np.ndarray):
        """Re-sync IK continuity state when robot state drifts from solver state."""
        if ik_solver.state is None:
            ik_solver.init_state(q_current)
            return
        q_prev = np.asarray(ik_solver.state.q_prev, dtype=float).reshape(-1)
        if q_prev.shape[0] != 7 or np.linalg.norm(q_current - q_prev) > self.ik_resync_thresh_rad:
            ik_solver.init_state(q_current)

    def _limit_joint_step(self, q_current: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
        """Clamp joint increment per cycle to reduce JS mode shock."""
        q_current = np.asarray(q_current, dtype=float).reshape(7)
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(7)
        dq = q_cmd - q_current
        dq = np.clip(dq, -self.max_joint_step_rad, self.max_joint_step_rad)
        return q_current + dq

    # ==================== Left Arm State Query ====================

    def left_robot_get_joint_positions(self) -> list:
        """Get left arm joint positions (radians)."""
        if self.left_robot is None:
            return [0.0] * 7
        result = self.left_robot.get_joint_angles()
        return result.msg if result is not None else [0.0] * 7
    
    def left_robot_get_joint_velocities(self) -> list:
        """Get left arm joint velocities (rad/s)."""
        if self.left_robot is None:
            return [0.0] * 7
        velocities = []
        for i in range(1, 8):
            result = self.left_robot.get_motor_states(i)
            if result is not None:
                velocities.append(result.msg.velocity)
            else:
                velocities.append(0.0)
        return velocities
    
    def left_robot_get_ee_pose(self) -> list:
        """Get left arm end-effector pose [x, y, z, roll, pitch, yaw] (m, rad)."""
        if self.left_robot is None:
            return [0.0] * 6
        self.left_robot.set_tcp_offset(self.tcp_offset)
        result = self.left_robot.get_tcp_pose()
        return result.msg if result is not None else [0.0] * 6
    

    def left_robot_get_arm_status(self) -> dict:
        """Get left arm overall status."""
        if self.left_robot is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        result = self.left_robot.get_arm_status()
        if result is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return {
            "ctrl_mode": result.msg.ctrl_mode,
            "arm_status": result.msg.arm_status,
            "motion_status": result.msg.motion_status,
            "trajectory_num": result.msg.trajectory_num
        }
    
    # ==================== Right Arm State Query ====================

    def right_robot_get_joint_positions(self) -> list:
        """Get right arm joint positions (radians)."""
        if self.right_robot is None:
            return [0.0] * 7
        result = self.right_robot.get_joint_angles()
        return result.msg if result is not None else [0.0] * 7
    
    def right_robot_get_joint_velocities(self) -> list:
        """Get right arm joint velocities (rad/s)."""
        if self.right_robot is None:
            return [0.0] * 7
        velocities = []
        for i in range(1, 8):
            result = self.right_robot.get_motor_states(i)
            if result is not None:
                velocities.append(result.msg.velocity)
            else:
                velocities.append(0.0)
        return velocities
    
    def right_robot_get_ee_pose(self) -> list:
        """Get right arm end-effector pose [x, y, z, roll, pitch, yaw] (m, rad)."""
        if self.right_robot is None:
            return [0.0] * 6
        self.right_robot.set_tcp_offset(self.tcp_offset)
        result = self.right_robot.get_tcp_pose()
        return result.msg if result is not None else [0.0] * 6
    
    def right_robot_get_arm_status(self) -> dict:
        """Get right arm overall status."""
        if self.right_robot is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        result = self.right_robot.get_arm_status()
        if result is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return {
            "ctrl_mode": result.msg.ctrl_mode,
            "arm_status": result.msg.arm_status,
            "motion_status": result.msg.motion_status,
            "trajectory_num": result.msg.trajectory_num
        }
    
    # ==================== Left Arm Motion ====================
    
    def left_robot_move_to_joint_positions(self, positions: list, delta: bool = False):
        """Move left arm to joint positions (radians)."""
        if self.left_robot is None:
            return
        
        # @Key-Zzs: fix TypeError
        positions = np.asarray(positions, dtype=float)
        if positions.shape[0] != 7:
            raise ValueError(f"Expected 7 joints, got {positions.shape[0]}")
        
        if delta:
            current = np.asarray(self.left_robot_get_joint_positions(), dtype=float)
            target = current + positions
        else:
            target = positions

        target_list = target.tolist()
        log.info("[DEBUG] move_j target: %s", target_list)

        self.left_robot.move_j(target_list)
        # time.sleep(3.0)
        log.info("move to joint positions completed")
        
    def left_robot_move_to_ee_pose(self, pose: list, delta: bool = False):
        """Move left arm to end-effector pose [x, y, z, roll, pitch, yaw] (m, rad)."""
        """use move_p for direct pose control, but it may cause discontinuity and vibration."""
        if self.left_robot is None:
            return
        
        pose = np.asarray(pose, dtype=float)
        if pose.shape[0] != 6:
            raise ValueError(f"Expected 6 joints, got {pose.shape[0]}")

        if delta:
            current = np.asarray(self.left_robot_get_ee_pose(), dtype=float)
            target = current + pose
        else:
            target = pose
        
        target_list = target.tolist()
        log.info("[DEBUG] move_p target: %s", target_list)
    
        self.left_robot.set_speed_percent(30)

        self.left_robot.move_p(target_list)
        # time.sleep(3.0)
        log.info("move to end-effector pose completed")

    # ==================== Right Arm Motion ====================
    
    # same as left_robot_move_to_joint_positions() and left_robot_move_to_ee_pose() 
    def right_robot_move_to_joint_positions(self, positions: list, delta: bool = False):
        """Move right arm to joint positions (radians)."""
        if self.right_robot is None:
            return
        
        positions = np.asarray(positions, dtype=float)
        if positions.shape[0] != 7:
            raise ValueError(f"Expected 7 joints, got {positions.shape[0]}")
        
        if delta:
            current = np.asarray(self.right_robot_get_joint_positions(), dtype=float)
            target = current + positions
        else:
            target = positions

        target_list = target.tolist()
        log.info("[DEBUG] move_j target: %s", target_list)

        self.right_robot.move_j(target_list)
        # time.sleep(3.0)
        log.info("move to joint positions completed")
    
    def right_robot_move_to_ee_pose(self, pose: list, delta: bool = False):
        """Move right arm to end-effector pose [x, y, z, roll, pitch, yaw] (m, rad)."""
        if self.right_robot is None:
            return
        
        if delta:
            current = self.right_robot_get_ee_pose()
            target = np.array(current) + np.array(pose)
        else:
            target = pose

        target_list = target.tolist()
        log.info("[DEBUG] move_p target: %s", target_list)
    
        self.right_robot.move_p(target_list)
        # time.sleep(3.0)
        log.info("move to end-effector pose completed")

    def dual_robot_move_to_ee_pose(self, left_pose: list, right_pose: list, delta: bool = False):
        if self.left_robot is None or self.right_robot is None:
            return
        self.left_robot_move_to_ee_pose(left_pose, delta=delta)
        self.right_robot_move_to_ee_pose(right_pose, delta=delta)

    def left_robot_go_home(self):
        if self.left_robot is None:
            log.error("Left robot not initialized")
            return

        self.left_robot.set_speed_percent(30)

        home = [0.0, -0.2, 0.0, 1.87, -0.7, 0.0, 1.1]
        # home = [0.0, -0.2, 0.0, 1.87, 0.0, 0.0, 1.1]
        # home = [1.22, 1.57, -1.57, 1.90, -1.57, 0.0, 0.0]

        log.info("[DEBUG] Moving to home: %s", home)
        self.left_robot.move_j(home)

        # 等待运动完成
        motion_complete = self._wait_for_motion_complete(self.left_robot, home, timeout=20.0)
        if not motion_complete:
            log.warning("[left_robot_go_home] Motion did not complete within timeout")
        else:
            log.info("已回到初始位置")

        # 更新 left_cur_pose
        if self.left_robot is not None and self.left_ik_solver is not None:
            from pyAgxArm.utiles.tf import rot_to_rpy
            try:
                current_joints = None
                timeout = 2.0
                start_t = time.monotonic()
                while current_joints is None:
                    ja = self.left_robot.get_joint_angles()
                    if ja is not None:
                        current_joints = ja.msg
                        break
                    if time.monotonic() - start_t > timeout:
                        log.warning("[left_robot_go_home] get_joint_angles timeout")
                        break
                    time.sleep(0.01)
                
                if current_joints is not None:
                    q_current = np.array(current_joints, dtype=float)
                    # T_fk = fk(q_current, self.left_ik_solver.nero_params)
                    # fk_xyz = np.asarray(T_fk[:3, 3], dtype=float)
                    # fk_rpy = np.asarray(rot_to_rpy(T_fk[:3, :3].tolist()), dtype=float)
                    # self.left_cur_pose = np.concatenate([fk_xyz, fk_rpy])
                    self.left_cur_pose = self.left_ik_solver.fk_pose(q_current)
                    log.info(f"[left_robot_go_home] Updated left_cur_pose: {self.left_cur_pose}")
                    
                    # ⚠️ 关键：同步 IK solver 状态
                    self.left_ik_solver.init_state(q_current)
                    log.info(f"[left_robot_go_home] IK solver state synced: {q_current.round(3)}")
            except Exception as e:
                log.error(f"[left_robot_go_home] Failed to update pose: {e}")
    
    def right_robot_go_home(self):
        if self.right_robot is None:
            log.error("Right robot not initialized")
            return

        self.right_robot.set_speed_percent(30)
        
        home = [0.0, -0.2, 0.0, 1.87, 0.7, 0.0, 1.1]
        # home = [0.0, -0.2, 0.0, 1.87, 0.0, 0.0, 1.1]
        # home = [-1.22, 1.57, 1.57, 1.90, 1.57, 0.0, 0.0]

        log.info("[DEBUG] Moving to home: %s", home)
        self.right_robot.move_j(home)
        
        # 等待运动完成
        motion_complete = self._wait_for_motion_complete(self.right_robot, home, timeout=20.0)

        if not motion_complete:
            log.warning("[right_robot_go_home] Motion did not complete within timeout")
        else:
            log.info("已回到初始位置")

        # 更新 right_cur_pose
        if self.right_robot is not None and self.right_ik_solver is not None:
            from pyAgxArm.utiles.tf import rot_to_rpy
            try:
                current_joints = None
                timeout = 2.0
                start_t = time.monotonic()
                while current_joints is None:
                    ja = self.right_robot.get_joint_angles()
                    if ja is not None:
                        current_joints = ja.msg
                        break
                    if time.monotonic() - start_t > timeout:
                        log.warning("[right_robot_go_home] get_joint_angles timeout")
                        break
                    time.sleep(0.01)
                
                if current_joints is not None:
                    q_current = np.array(current_joints, dtype=float)
                    # T_fk = fk(q_current, self.right_ik_solver.nero_params)
                    # fk_xyz = np.asarray(T_fk[:3, 3], dtype=float)
                    # fk_rpy = np.asarray(rot_to_rpy(T_fk[:3, :3].tolist()), dtype=float)
                    # self.right_cur_pose = np.concatenate([fk_xyz, fk_rpy])
                    self.right_cur_pose = self.right_ik_solver.fk_pose(q_current)
                    log.info(f"[right_robot_go_home] Updated right_cur_pose: {self.right_cur_pose}")
                    
                    # ⚠️ 关键：同步 IK solver 状态
                    self.right_ik_solver.init_state(q_current)
                    log.info(f"[right_robot_go_home] IK solver state synced: {q_current.round(3)}")
            except Exception as e:
                log.error(f"[right_robot_go_home] Failed to update pose: {e}")

    def robot_go_home(self):
        """双臂同时 go home（并行执行）"""
        if self.left_robot is None or self.right_robot is None:
            return
        
        import threading
        
        # 创建两个线程并行执行
        left_thread = threading.Thread(target=self.left_robot_go_home, name="left_go_home")
        right_thread = threading.Thread(target=self.right_robot_go_home, name="right_go_home")
        
        # 同时启动
        left_thread.start()
        right_thread.start()
        
        # 等待两个线程都完成
        left_thread.join()
        right_thread.join()
        
        log.info("[SERVER] Robot home setup completed for both arms")

    # ==================== ServoJ Control (Joint Servo) ====================

    
    def servo_j(self, robot_arm: str, joints: list, delta: bool) -> bool:
        """
        直接输入某个机械臂名称与目标关节角度（度），控制机械臂运动。

        Args:
            robot_arm: "left_robot" or "right_robot"
            joints: 7维绝对关节角度（度）
            delta: False=绝对控制, True=增量控制

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        try:
            joints = np.asarray(joints, dtype=float)
            if joints.shape[0] != 7:
                raise ValueError(f"Expected 7 joints, got {joints.shape[0]}")
            
            # 根据 robot_arm 选择对应的机械臂和 get_joint_positions 方法
            if robot_arm == "left_robot":
                robot = self.left_robot
                get_joint_positions = self.left_robot_get_joint_positions()
            elif robot_arm == "right_robot":
                robot = self.right_robot
                get_joint_positions = self.right_robot_get_joint_positions()
            else:
                raise ValueError("robot_arm must be 'left_robot' or 'right_robot'")
            
            if robot is None:
                log.error(f"[ERROR] {robot_arm} not initialized")
                return False
            
            # 计算目标关节角度
            if delta:
                current = np.asarray(get_joint_positions, dtype=float)
                for i in range(7):
                    current[i] = np.rad2deg(current[i])
                target = current + joints
            else:
                target = joints

            log.info(f"[DEBUG] servo_j target (degree): {target}")
            
            # 转换为弧度
            for i in range(7):
                target[i] = np.deg2rad(target[i])
            
            target = target.tolist()

            # 下发关节控制
            robot.move_js(target)

            return True

        except Exception as e:
            log.error(f"[ERROR] servo_j failed: {e}")
            return False
        
    
    # ==================== ServoP Control (Pose Servo) ====================

    def servo_p_OL(self, robot_arm: str, pose: list, delta: bool) -> bool:
        """
        Send ServoP open loop with target pose [x, y, z, rx, ry, rz] (m, radians).
        Args:
            robot_arm: "left_robot" or "right_robot"
            pose: 末端位置(m, radians)
            delta: 绝对控制(False)，增量控制(True)
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        try:
            import time as _time
            from pyAgxArm.utiles.tf import rot_to_rpy, euler_convert_quat, quat_convert_euler
            
            limit_z = 0.25

            # ========== 调用间隔追踪 ==========
            _t_call_start = _time.perf_counter()
            if not hasattr(self, '_last_call_time'):
                self._last_call_time = {}
                self._call_count = 0
                self._freq_start_time = _t_call_start
            
            _last_time = self._last_call_time.get(robot_arm, _t_call_start)
            _interval = (_t_call_start - _last_time) * 1000
            self._last_call_time[robot_arm] = _t_call_start
            self._call_count += 1

            # Short-circuit repeated IK fallback stalls for the same arm.
            if _t_call_start < self._ik_skip_until.get(robot_arm, 0.0):
                return False
            
            # 每50次计算实际频率
            if self._call_count % 50 == 0:
                _elapsed = _t_call_start - self._freq_start_time
                _actual_freq = 50 / _elapsed if _elapsed > 0 else 0
                self._freq_start_time = _t_call_start
                if self.enable_servo_timing_print:
                    print(f"[FREQ] 实际调用频率: {_actual_freq:.1f} Hz (目标50Hz)")
            
            # ========== 计时诊断 ==========
            _t_start = _time.perf_counter()
            _timings = {}
            
            # 1. 选择 robot & IK
            _t0 = _time.perf_counter()
            if robot_arm == "left_robot":
                robot = self.left_robot
                ik_solver = self.left_ik_solver
                cur_pose_attr = "left_cur_pose"
            elif robot_arm == "right_robot":
                robot = self.right_robot
                ik_solver = self.right_ik_solver
                cur_pose_attr = "right_cur_pose"
            else:
                log.error(f"[ERROR] invalid robot_arm: {robot_arm}")
                return False

            if robot is None or ik_solver is None:
                log.error("[ERROR] robot or IK solver not ready")
                return False
            _timings['select_robot'] = (_time.perf_counter() - _t0) * 1000

            # 2. 开环控制：仅在首次调用时读取关节角初始化 IK 状态
            # 后续调用跳过 CAN 读取，大幅减少延迟
            _t0 = _time.perf_counter()
            q_current = None  # 初始化变量，避免 UnboundLocalError
            if ik_solver.state is None:
                q_current = self._get_current_joints(robot, timeout=2.0)
                if q_current is None:
                    log.error("[ERROR] get_joint_angles timeout")
                    return False
                ik_solver.init_state(q_current)
                log.info("[servo_p_OL] IK solver state initialized")
            else:
                # 已初始化时，从 IK solver 状态获取当前关节角
                q_current = ik_solver.state.q_prev
            _timings['get_joints'] = (_time.perf_counter() - _t0) * 1000
            
            pose = np.asarray(pose, dtype=float).reshape(-1)
            if pose.size != 6:
                raise ValueError(f"Expected 6 pose values, got {pose.size}")

            # 3. 计算 target_pose
            _t0 = _time.perf_counter()
            if delta:
                pose = self._limit_pose_delta(pose)
                # Failure-aware attenuation: reduce command amplitude for recovery frames.
                delta_scale = float(self._ik_delta_scale.get(robot_arm, 1.0))
                if delta_scale < 0.999:
                    pose = pose * delta_scale
                # Temporarily freeze orientation increments after repeated IK failures.
                if _t_call_start < self._ik_freeze_rot_until.get(robot_arm, 0.0):
                    pose[3:] = 0.0
                
                # ⚠️ 关键修复：增量模式下使用 IK solver 内部状态计算当前位姿
                # 而不是用实际关节角 q_current，保证 IK 求解的连续性
                # 参考 test_pos_flw_ik.py 的做法
                q_for_fk = ik_solver.state.q_prev
                # fk_xyz = np.asarray(T_fk[:3, 3], dtype=float)
                # fk_rpy = np.asarray(rot_to_rpy(T_fk[:3, :3].tolist()), dtype=float)
                # cur_pose = np.concatenate([fk_xyz, fk_rpy])
                cur_pose = np.asarray(ik_solver.fk_pose(q_for_fk), dtype=float)
                
                cur_xyz = np.asarray(cur_pose[:3], dtype=float)
                cur_rpy = np.asarray(cur_pose[3:], dtype=float)
                
                # 调试日志改为 debug 级别，避免高频输出影响性能
                log.debug(f"当前位姿 (from IK state): {cur_pose}")

                # --- 计算目标位姿 ---
                # 1. 位置直接相加
                target_xyz = cur_xyz + np.array(pose[:3], dtype=float)

                # 2. rpy相加转为四元数相乘
                ## 当前姿态 RPY → 四元数
                current_quat = euler_convert_quat(cur_rpy[0], cur_rpy[1], cur_rpy[2])
                ## 增量 RPY → 四元数
                delta_quat = euler_convert_quat(pose[3], pose[4], pose[5])

                ## 末端姿态增量四元数相乘得到目标姿态四元数
                target_quat = quat_multiply(delta_quat, current_quat)
                ## 目标姿态四元数归一化
                # target_quat = quat_normalize(target_quat) # wait for function
                q_norm = np.sqrt(target_quat[0]**2 + target_quat[1]**2 + target_quat[2]**2 + target_quat[3]**2)
                target_quat = tuple(v / q_norm for v in target_quat)
                ## 目标姿态四元数 → RPY
                target_rpy = quat_convert_euler(*target_quat)
                log.debug(f"目标姿态 XYZ: {target_xyz}")
                log.debug(f"目标姿态 RPY: {target_rpy}")

                # 3. 合并位置和姿态
                target_pose = np.concatenate([target_xyz, target_rpy])
            else:
                # 绝对模式：直接使用目标位姿
                target_pose = pose
                target_xyz = np.asarray(target_pose[:3], dtype=float)

            if target_xyz[2] < limit_z:
                log.warning(
                    f"[servo_p_OL] Skip command for {robot_arm}: "
                    f"target z={target_xyz[2]:.4f}m < limit_z={limit_z:.4f}m"
                )
                return False
            _timings['target_compute'] = (_time.perf_counter() - _t0) * 1000

            # 4. IK 求解
            _t0 = _time.perf_counter()
            q_cmd = ik_solver.solve(target_pose, limit_output_step=bool(delta))
            _timings['ik_solve'] = (_time.perf_counter() - _t0) * 1000

            # IK 超时保护：若单次解算超过阈值，则丢弃本次动作，避免控制环路滞后。
            if _timings['ik_solve'] > self.max_ik_solve_ms:
                log.warning(
                    f"[servo_p_OL] Drop action for {robot_arm}: "
                    f"IK solve {_timings['ik_solve']:.1f}ms > {self.max_ik_solve_ms:.1f}ms"
                )
                self._ik_delta_scale[robot_arm] = max(
                    self.ik_delta_min_scale,
                    self._ik_delta_scale.get(robot_arm, 1.0) * self.ik_delta_decay,
                )
                self._ik_freeze_rot_until[robot_arm] = _t_call_start + self.ik_orientation_freeze_s
                self._ik_skip_until[robot_arm] = _t_call_start + self.ik_fail_cooldown_s
                return False
            
            # 调试日志改为 debug 级别
            log.debug(f"计算出的关节角度: {q_cmd}")

            # 增加对求解失败的安全校验
            if q_cmd is None or len(q_cmd) == 0:
                log.error("[ERROR] IK solve failed: returned None/Empty")
                # IK 失败时重新同步状态，避免状态不一致
                ik_solver.init_state(q_current)
                self._ik_delta_scale[robot_arm] = max(
                    self.ik_delta_min_scale,
                    self._ik_delta_scale.get(robot_arm, 1.0) * self.ik_delta_decay,
                )
                self._ik_freeze_rot_until[robot_arm] = _t_call_start + self.ik_orientation_freeze_s
                self._ik_skip_until[robot_arm] = _t_call_start + self.ik_fail_cooldown_s
                return False

            if isinstance(q_cmd, np.ndarray):
                q_cmd = q_cmd.tolist()

            # 开环控制：不限制关节增量，让 IK solver 完全控制轨迹
            # 参考 test_pos_flw_ik.py 的做法
            # q_cmd = self._limit_joint_step(q_current, np.asarray(q_cmd, dtype=float)).tolist()

            # 5. 下发关节控制
            _t0 = _time.perf_counter()
            robot.move_js(q_cmd)
            _timings['send_command'] = (_time.perf_counter() - _t0) * 1000

            # ========== 计时诊断（每次输出，用于调试50Hz性能） ==========
            _timings['total'] = (_time.perf_counter() - _t_start) * 1000
            if self.enable_servo_timing_print and self._call_count % self.servo_timing_print_every_n == 0:
                print(f"[TIMING-{robot_arm}] "
                    f"interval={_interval:.1f}ms, "
                    f"select={_timings['select_robot']:.1f}ms, "
                    f"get_joints={_timings['get_joints']:.1f}ms, "
                    f"target={_timings['target_compute']:.1f}ms, "
                    f"ik={_timings['ik_solve']:.1f}ms, "
                    f"send={_timings['send_command']:.1f}ms, "
                    f"total={_timings['total']:.1f}ms")

            # Successful frame: gradually restore delta scale to normal.
            self._ik_delta_scale[robot_arm] = min(
                1.0,
                self._ik_delta_scale.get(robot_arm, 1.0) + self.ik_delta_recover,
            )

            return True

        except Exception as e:
            log.error(f"[ERROR] servo_p_OL failed", e)
            return False

    def servo_p(self, robot_arm: str, pose: list, delta: bool) -> bool:
        """
        Send ServoP with target pose [x, y, z, rx, ry, rz] (m, radians).
        Args:
            robot_arm: "left_robot" or "right_robot"
            pose: 末端位置(m, radians)
            delta: 绝对控制(False)，增量控制(True)
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        try:
            from pyAgxArm.utiles.tf import euler_convert_quat, quat_convert_euler
            # 1. 选择 robot & IK
            if robot_arm == "left_robot":
                robot = self.left_robot
                ik_solver = self.left_ik_solver
            elif robot_arm == "right_robot":
                robot = self.right_robot
                ik_solver = self.right_ik_solver
            else:
                log.error(f"[ERROR] invalid robot_arm: {robot_arm}")
                return False

            if robot is None or ik_solver is None:
                log.error("[ERROR] robot or IK solver not ready")
                return False

            q_current = self._get_current_joints(robot, timeout=2.0)
            if q_current is None:
                log.error("[ERROR] get_joint_angles timeout")
                return False

            # ⚠️ 开环控制：仅在首次调用时初始化 IK 状态，之后不干预求解器内部状态
            if ik_solver.state is None:
                ik_solver.init_state(q_current)
                log.info("[servo_p] IK solver state initialized")
            
            pose = np.asarray(pose, dtype=float).reshape(-1)
            if pose.size != 6:
                raise ValueError(f"Expected 6 pose values, got {pose.size}")

            # 2. 计算 target_pose
            if delta:
                pose = self._limit_pose_delta(pose)
                
                # ⚠️ 开环控制：用 FK 计算当前位姿，避免累积误差
                current_pose = np.asarray(ik_solver.fk_pose(q_current), dtype=float)
                fk_xyz = np.asarray(current_pose[:3], dtype=float)
                fk_rpy = np.asarray(current_pose[3:], dtype=float)

                log.debug("-------------------------------")
                log.debug(f"FK当前位姿: {current_pose}")

                # --- 计算目标位姿 ---
                # 1. 位置直接相加
                target_fk_xyz = fk_xyz + np.array(pose[:3], dtype=float)

                # # 2. 姿态用小增量近似相加后归一化到 [-pi, pi]
                # target_fk_rpy = (fk_rpy + np.pi) % (2.0 * np.pi) - np.pi

                # 2. rpy相加转为四元数相乘
                ## 当前姿态 RPY → 四元数
                current_quat = euler_convert_quat(fk_rpy[0], fk_rpy[1], fk_rpy[2])
                ## 增量 RPY → 四元数
                delta_quat = euler_convert_quat(pose[3], pose[4], pose[5])

                ## 末端姿态增量四元数相乘得到目标姿态四元数
                target_quat = quat_multiply(current_quat, delta_quat)
                ## 目标姿态四元数归一化
                # target_quat = quat_normalize(target_quat) # wait for function
                q_norm = np.sqrt(target_quat[0]**2 + target_quat[1]**2 + target_quat[2]**2 + target_quat[3]**2)
                target_quat = tuple(v / q_norm for v in target_quat)
                ## 目标姿态四元数 → RPY
                target_fk_rpy = quat_convert_euler(*target_quat)
                log.debug(f"目标姿态 XYZ: {target_fk_xyz}")
                log.debug(f"目标姿态 RPY: {target_fk_rpy}")

                # 3. 合并位置和姿态
                target_pose = np.concatenate([target_fk_xyz, target_fk_rpy])
            else:
                # target_pose = np.array(pose, dtype=float)
                target_pose = pose

            # 3. IK 求解
            q_cmd = ik_solver.solve(target_pose)
            log.debug(f"计算出的关节角度: {q_cmd}")

            # 增加对求解失败的安全校验 (判断是否返回了 None 或者空数组)
            if q_cmd is None or len(q_cmd) == 0:
                log.error("[ERROR] IK solve failed: returned None/Empty")
                # IK 失败时重新同步状态，避免状态不一致
                ik_solver.init_state(q_current)
                return False

            if isinstance(q_cmd, np.ndarray):
                q_cmd = q_cmd.tolist()

            q_cmd = self._limit_joint_step(q_current, np.asarray(q_cmd, dtype=float)).tolist()

            # 4. 下发关节控制
            robot.move_js(q_cmd)

            return True

        except Exception as e:
            log.error(f"[ERROR] servo_p failed: %s", e)
            return False
    
    # ==================== Inverse Kinematics ====================

    def _setup_ik_solver(self, robot, cfg, robot_name: str, timeout_sec: float = 2.0):
        """辅助方法：获取初始关节角，提取限位，并初始化 IK Solver"""
        log.info(f"[{robot_name}] 正在获取当前关节角作为 IK 初始基准...")
        current_pose = None
        current_joints = None
        start_t = time.monotonic()
        while current_pose is None or current_joints is None:
            if time.monotonic() - start_t > timeout_sec:
                log.warning(f"[{robot_name}] get tcp/joint pose timeout after {timeout_sec}s, using default pose")
                current_pose = [0.0] * 6
                current_joints = [0.0] * 7
                break
            robot.set_tcp_offset(self.tcp_offset)
            fp = robot.get_tcp_pose()
            ja = robot.get_joint_angles()
            if fp is not None: current_pose = fp.msg
            if ja is not None: current_joints = ja.msg
            time.sleep(0.1)

        # 获取关节限位
        joint_limits = []
        for i in range(1, 8):
            lo, hi = cfg["joint_limits"][f"joint{i}"]
            joint_limits.append((lo, hi))

        # 实例化解析 IK 求解器
        ik_solver = Pinocchio_Solver(
            joint_limits=joint_limits,
            dt=self.dt,
            n_psi=181,
            tcp_offset=self.tcp_offset,
        )

        # 机器人的真实状态给 IK 求解器初始化
        ik_solver.init_state(current_joints)
        log.info(f"[{robot_name}] IK Solver 初始化完成！初始关节角: {np.array(current_joints).round(3)}")
        
        return ik_solver
    
    # ==================== Gripper (Placeholder) ====================
    
    def _setup_gripper(self, gripper, gripper_name: str):
        """夹爪初始化：先合后张
        
        对每个夹爪执行：
        1. 合夹爪（width=0.0）
        2. 等待 0.5 秒
        3. 张夹爪（width=0.1）
        4. 等待 0.5 秒
        """
        # 夹爪控制
        if gripper is not None:
            try:
                log.info(f"[setup_gripper] Setting up gripper...")

                # 检查夹爪通信状态
                if not gripper.is_ok():
                    log.warning(f"[setup_gripper] [{gripper_name}] communication not ready, waiting...")
                    for i in range(20):
                        time.sleep(0.1)
                        if gripper.is_ok():
                            log.info(f"[setup_gripper] [{gripper_name}] communication ready after {i*0.1:.1f}s")
                            break
                    else:
                        log.error(f"[setup_gripper] [{gripper_name}] communication timeout!")
                        return
                
                # 等待夹爪状态反馈
                status = None
                for i in range(20):
                    status = gripper.get_gripper_status()
                    if status is not None:
                        log.info(f"[setup_gripper] [{gripper_name}] status received after {i*0.1:.1f}s")
                        break
                    time.sleep(0.1)
                
                if status is None:
                    log.warning(f"[setup_gripper] [{gripper_name}] status not available, continuing anyway...")
                else:
                    log.info(f"[setup_gripper] [{gripper_name}] initial state: width={status.msg.width:.4f}m, force={status.msg.force:.2f}N, enabled={status.msg.foc_status.driver_enable_status}")
                

                log.info(f"[setup_gripper] Setting up {gripper_name} gripper...")
                goto = self.left_gripper_goto if gripper_name.lower().startswith("left") else self.right_gripper_goto
                goto(0.0)
                log.info(f"[setup_gripper] [{gripper_name}] closed (width=0.0)")
                time.sleep(0.5)
                goto(0.1)
                log.info(f"[setup_gripper] [{gripper_name}] opened (width=0.1)")
                time.sleep(0.5)
                log.info(f"[setup_gripper] [{gripper_name}] setup completed")

                # if(gripper == self.left_gripper):

                #     # 检查夹爪通信状态
                #     if not self.left_gripper.is_ok():
                #         log.warning("[setup_gripper] Left gripper communication not ready, waiting...")
                #         for i in range(20):
                #             time.sleep(0.1)
                #             if self.left_gripper.is_ok():
                #                 log.info(f"[setup_gripper] Left gripper communication ready after {i*0.1:.1f}s")
                #                 break
                #         else:
                #             log.error("[setup_gripper] Left gripper communication timeout!")
                #             return
                    
                #     # 等待夹爪状态反馈
                #     status = None
                #     for i in range(20):
                #         status = self.left_gripper.get_gripper_status()
                #         if status is not None:
                #             log.info(f"[setup_gripper] Left gripper status received after {i*0.1:.1f}s")
                #             break
                #         time.sleep(0.1)
                    
                #     if status is None:
                #         log.warning("[setup_gripper] Left gripper status not available, continuing anyway...")
                #     else:
                #         log.info(f"[setup_gripper] Left gripper initial state: width={status.msg.width:.4f}m, force={status.msg.force:.2f}N, enabled={status.msg.foc_status.driver_enable_status}")
                    

                #     log.info(f"[setup_gripper] Setting up left gripper...")
                #     self.left_gripper_goto(0.0)
                #     log.info("[setup_gripper] Left gripper closed (width=0.0)")
                #     time.sleep(0.5)
                #     self.left_gripper_goto(0.1)
                #     log.info("[setup_gripper] Left gripper opened (width=0.1)")
                #     time.sleep(0.5)
                #     log.info("[setup_gripper] Left gripper setup completed")
                # else:
                #     log.info(f"[setup_gripper] Setting up right gripper...")

                #     # 检查夹爪通信状态
                #     if not self.right_gripper.is_ok():
                #         log.warning("[setup_gripper] Right gripper communication not ready, waiting...")
                #         for i in range(20):
                #             time.sleep(0.1)
                #             if self.left_gripper.is_ok():
                #                 log.info(f"[setup_gripper] Right gripper communication ready after {i*0.1:.1f}s")
                #                 break
                #         else:
                #             log.error("[setup_gripper] Right gripper communication timeout!")
                #             return
                    
                #     # 等待夹爪状态反馈
                #     status = None
                #     for i in range(20):
                #         status = self.left_gripper.get_gripper_status()
                #         if status is not None:
                #             log.info(f"[setup_gripper] Right gripper status received after {i*0.1:.1f}s")
                #             break
                #         time.sleep(0.1)
                    
                #     if status is None:
                #         log.warning("[setup_gripper] Right gripper status not available, continuing anyway...")
                #     else:
                #         log.info(f"[setup_gripper] Right gripper initial state: width={status.msg.width:.4f}m, force={status.msg.force:.2f}N, enabled={status.msg.foc_status.driver_enable_status}")
                    

                #     self.right_gripper_goto(0.0)
                #     log.info("[setup_gripper] Right gripper closed (width=0.0)")
                #     time.sleep(0.5)
                #     self.right_gripper_goto(0.1)
                #     log.info("[setup_gripper] Right gripper opened (width=0.1)")
                #     time.sleep(0.5)
                #     log.info("[setup_gripper] Right gripper setup completed")
            except Exception as e:
                log.warning(f"[setup_gripper] Failed to setup gripper: {e}")
        
        log.info("[setup_gripper] Gripper setup completed for both arms")

    # TODO: 重构为非阻塞控制
    ## def task():
    ## threading.Thread(target=task, daemon=True).start()
    def left_gripper_goto(self, width: float, force: float = 1.0):
        import time as _time
        _t_start = _time.perf_counter()
        
        if not self.gripper_enabled or self.left_gripper is None:
            log.warning("[SERVER] Left gripper not available")
            return False

        width = float(max(0.0, min(width, 0.1)))
        cmd = (round(width, 4), round(float(force), 3))
        if self._last_left_gripper_cmd == cmd:
            return True

        try:
            _t0 = _time.perf_counter()
            self.left_gripper.move_gripper(width=width, force=force)
            _t_move = (_time.perf_counter() - _t0) * 1000
            _t_total = (_time.perf_counter() - _t_start) * 1000
            self._last_left_gripper_cmd = cmd
            if self.gripper_print:
                print(f"[GRIPPER-L] move={_t_move:.1f}ms, total={_t_total:.1f}ms, width={width:.3f}")
            return True
        except Exception as e:
            log.error(f"[SERVER] Left gripper goto failed: {e}")
            return False
     
    # TODO: 实现逻辑未确定
    def left_gripper_grasp(self, force: float = 1.0, width: float = 0.05):
        if not self.gripper_enabled or self.left_gripper is None:
            log.warning("[SERVER] Left gripper not available")
            return False

        width = float(max(0.0, min(width, 0.1)))

        log.info(f"[SERVER] Left gripper grasp: width={width}, force={force}")

        try:
            self.left_gripper.move_gripper(width=width, force=force)
            time.sleep(1.5)

            status = self.left_gripper.get_gripper_status()
            if status is None:
                return False

            current_width = status.msg.width

            # 抓取判断（核心 heuristic）
            is_grasped = (width < 0.01) and (current_width > 0.005)

            log.info(f"[SERVER] Left grasp result: width={current_width:.4f}, grasped={is_grasped}")

            return is_grasped

        except Exception as e:
            log.error(f"[SERVER] Left gripper grasp failed: {e}")
            return False
        
    def left_gripper_get_state(self):
        if not self.gripper_enabled or self.left_gripper is None:
            return {"is_moving": False, "is_grasped": False}

        try:
            status = self.left_gripper.get_gripper_status()
            if status is None:
                return {"is_moving": False, "is_grasped": False}

            width = status.msg.width
            force = status.msg.force

            is_moving = abs(force) > 0.1
            is_grasped = (width > 0.005) and (force > 0.5)

            return {
                "width": width,
                "force": force,
                "is_moving": is_moving,
                "is_grasped": is_grasped
            }

        except Exception as e:
            log.error(f"[SERVER] Left gripper state failed: {e}")
            return {"is_moving": False, "is_grasped": False}

    def right_gripper_goto(self, width: float, force: float = 1.0):
        import time as _time
        _t_start = _time.perf_counter()
        
        if not self.gripper_enabled or self.right_gripper is None:
            log.warning("[SERVER] Right gripper not available")
            return False

        width = float(max(0.0, min(width, 0.1)))
        cmd = (round(width, 4), round(float(force), 3))
        if self._last_right_gripper_cmd == cmd:
            return True

        try:
            _t0 = _time.perf_counter()
            self.right_gripper.move_gripper(width=width, force=force)
            _t_move = (_time.perf_counter() - _t0) * 1000
            _t_total = (_time.perf_counter() - _t_start) * 1000
            self._last_right_gripper_cmd = cmd
            if self.gripper_print:
                print(f"[GRIPPER-R] move={_t_move:.1f}ms, total={_t_total:.1f}ms, width={width:.3f}")
            return True
        except Exception as e:
            log.error(f"[SERVER] Right gripper goto failed: {e}")
            return False
     
    # TODO: 实现逻辑未确定
    def right_gripper_grasp(self, force: float = 1.0, width: float = 0.05): pass

    def right_gripper_get_state(self):
        if not self.gripper_enabled or self.right_gripper is None:
            return {"is_moving": False, "is_grasped": False}

        try:
            status = self.right_gripper.get_gripper_status()
            if status is None:
                return {"is_moving": False, "is_grasped": False}

            width = status.msg.width
            force = status.msg.force

            is_moving = abs(force) > 0.1
            is_grasped = (width > 0.005) and (force > 0.5)

            return {
                "width": width,
                "force": force,
                "is_moving": is_moving,
                "is_grasped": is_grasped
            }

        except Exception as e:
            log.error(f"[SERVER] Right gripper state failed: {e}")
            return {"is_moving": False, "is_grasped": False}
    
    def _wait_for_motion_complete(self, robot, target_joints: list, timeout: float = 10.0, tolerance: float = 0.01):
        """
        等待机械臂运动完成
        
        Args:
            robot: 机械臂对象
            target_joints: 目标关节角度（弧度）
            timeout: 超时时间（秒）
            tolerance: 位置容差（弧度）
        
        Returns:
            bool: 成功到达目标返回 True，超时返回 False
        """
        start_t = time.monotonic()
        target = np.asarray(target_joints, dtype=float)
        
        while time.monotonic() - start_t < timeout:
            try:
                # 获取当前关节角度
                result = robot.get_joint_angles()
                if result is None:
                    time.sleep(0.05)
                    continue
                
                current = np.asarray(result.msg, dtype=float)
                
                # 检查是否到达目标（所有关节都在容差范围内）
                if np.allclose(current, target, atol=tolerance):
                    log.info(f"[wait_for_motion] Motion completed! Current: {current.round(3)}, Target: {target.round(3)}")
                    return True
                
                # 检查是否还在运动（通过motion_status）
                status = robot.get_arm_status()
                if status is not None and status.msg.motion_status == 0:
                    # motion_status == 0 表示静止
                    if np.allclose(current, target, atol=tolerance):
                        log.info(f"[wait_for_motion] Motion stopped at target. Current: {current.round(3)}, Target: {target.round(3)}")
                        return True
                    else:
                        log.warning(f"[wait_for_motion] Motion stopped but not at target! Current: {current.round(3)}, Target: {target.round(3)}")
                        return False
                
                time.sleep(0.1)
                
            except Exception as e:
                log.error(f"[wait_for_motion] Error checking motion status: {e}")
                time.sleep(0.1)
        
        # 超时：检查当前位置
        result = robot.get_joint_angles()
        if result is not None:
            current = np.asarray(result.msg, dtype=float)
            error = np.linalg.norm(current - target)
            log.warning(f"[wait_for_motion] Timeout after {timeout}s. Error: {error:.4f} rad")
        
        return False

    # ==================== Utility ====================
    
    # TODO: wait for testing
    def robot_stop(self, robot_arm: str):
        """
        Stops the specified robot arm by triggering an emergency stop.

        Args:
            robot_arm (str): The name of the robot arm to stop, either "left_robot" or "right_robot".

        Returns:
            bool: True if the stop command was executed successfully, False otherwise.
        """
        try:
            if robot_arm == "left_robot":
                if self.left_robot is not None:
                    self.left_robot.electronic_emergency_stop()
                    log.info("[SERVER] Left robot emergency stopped")
                else:
                    log.warning("[SERVER] Left robot is not initialized")
            elif robot_arm == "right_robot":
                if self.right_robot is not None:
                    self.right_robot.electronic_emergency_stop()
                    log.info("[SERVER] Right robot emergency stopped")
                else:
                    log.warning("[SERVER] Right robot is not initialized")
            else:
                raise ValueError("robot_arm must be 'left_robot' or 'right_robot'")

            return True

        except Exception as e:
            log.error(f"[SERVER] Robot stop failed: {e}")
            return False

def start_server(ip: str, port: int = 4242, gripper_enabled: bool = True):
    server = zerorpc.Server(NeroDualArmServer(gripper_enabled))
    server.bind(f"tcp://{ip}:{port}")
    log.info(f"[SERVER] Listening on tcp://{ip}:{port}")
    server.run()

# python nero_interface/nero_interface_server.py --ip 0.0.0.0 --port 4242
# sudo iptables -I INPUT -p tcp --dport 4242 -j ACCEPT
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=4242)
    parser.add_argument('--no-gripper', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
    start_server(ip=args.ip, port=args.port, gripper_enabled=not args.no_gripper)
