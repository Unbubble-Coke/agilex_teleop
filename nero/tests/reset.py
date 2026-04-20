import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

def reset_arm(channel: str):
    """重置单个机械臂"""
    print(f"\n{'='*50}")
    print(f"正在重置 {channel}...")
    print('='*50)
    
    try:
        # 1. 初始化并连接
        print(f"[{channel}] 正在连接...")
        cfg = create_agx_arm_config(robot="nero", comm="can", channel=channel)
        robot = AgxArmFactory.create_arm(cfg)
        robot.connect()
        time.sleep(0.3)

        # 2. 电子急停
        print(f"[{channel}] 触发电子急停...")
        robot.electronic_emergency_stop()

        # 清除急停状态
        print(f"[{channel}] 清除急停状态...")
        robot.reset()
        time.sleep(0.2)

        # 3. 切换回正常控制模式
        print(f"[{channel}] 切换至正常控制模式...")
        robot.set_normal_mode()
        time.sleep(0.3)

        # 4. 失能
        print(f"[{channel}] 失能...")
        # robot.disable()
        # time.sleep(0.1)
        while not robot.disable():
            time.sleep(0.1)

        # 5. 重新使能验证
        print(f"[{channel}] 重新使能...")
        start_t = time.monotonic()
        is_enabled = False
        
        while time.monotonic() - start_t < 5.0:
            if robot.enable(255):
                is_enabled = True
                break
            time.sleep(0.1)

        if is_enabled:
            print(f"✅ [{channel}] 重置成功！")
        else:
            print(f"❌ [{channel}] 重置失败！未能重新使能。")
        
        return is_enabled
        
    except Exception as e:
        print(f"❌ [{channel}] 重置失败: {e}")
        return False

def main():
    # print("==================================================")
    # print("⚠️ 安全警告：执行重置会导致机械臂瞬间失去力矩！")
    # print("如果机械臂当前在半空中，它会【立刻掉落】。")
    # print("==================================================")
    
    # 强制要求用户确认，防止误触导致砸机
    # input("请【用手扶稳机械臂】，确认安全后按 Enter 键继续...")

    # 重置左臂和右臂
    left_ok = reset_arm("can_left")
    right_ok = reset_arm("can_right")
    
    print("\n" + "="*50)
    print("重置结果:")
    print(f"  左臂 (can_left): {'✅ 成功' if left_ok else '❌ 失败'}")
    print(f"  右臂 (can_right): {'✅ 成功' if right_ok else '❌ 失败'}")
    print("="*50)

if __name__ == "__main__":
    main()
