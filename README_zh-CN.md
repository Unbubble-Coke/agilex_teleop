# AgxArm_teleop

[English](./README.md) | [中文](./README_zh-CN.md)

---

AgileX nero 机械臂双臂遥操作系统，基于 AgileX 机械臂 SDK 开发。

本项目所需设备为：
- **AgileX nero 机械臂** x 2
- **AgileX gripper（集成 Inter RealSense D405 相机传感器）** x 2
- **Inter RealSense D435i 相机传感器** x 1
- **Server 端 PC**（要求运行内存较高） x 1：用于运行 Server 端项目控制机械臂（连接到 Agilex 机械臂的 CAN-USB 转换器）
- **Client 端 PC**（要求磁盘空间较大） x 1：用于运行 Client 端项目遥操作机械臂及采集数据（连接到三个相机传感器）
- AgileX 官方 **CAN-USB 转换器** x2

---

## 待办事项

- [x] 臂体控制问题
  - [ ] 运动滤波（[1€ filter](https://github.com/casiez/OneEuroFilter)）
  - [x] ik问题（拷打松灵技术）
    - [x] 控制频率？
    - [x] 解算方法（可考虑采用其他 ik 库配合 [nero urdf 模型](https://github.com/agilexrobotics/agx_arm_urdf) 使用，推荐 [ikpy](https://github.com/ikpy/ikpy) 和 [frax](https://github.com/danielpmorton/frax)）
- [x] 夹爪问题
  - [x] [Pika 夹爪 sdk](https://github.com/agilexrobotics/pika_sdk) 适配
  - [x] 响应速度（先移植 Pika 夹爪 sdk 到项目中，测试后问题依旧存在再尝试研究 sdk 和 [Server 端项目](nero/teleop/interface/nero_interface_server.py) 中 `left_gripper_goto()` 和 `right_gripper_goto()` 实现方式）

---

## 0 环境配置

### 0.1 创建 Conda 虚拟环境

```bash
# 创建名为 agilex_teleop 的 Python 3.10 环境
conda create -n agilex_teleop python=3.10 -y

# 激活环境
conda activate agilex_teleop
```

### 0.2 克隆 lerobot 项目并安装 lerobot 框架
```bash
# 安装指定版本 0.3.4
# git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

### 0.3 克隆并初始化项目依赖

#### 0.3.1 Server 端项目

1. 克隆 Server 端项目到工作目录

   ```bash
   cd lerobot
   mkdir agilex_ws && cd agilex_ws
   git clone --recursive https://github.com/Key-Zzs/agilex_teleop.git
   cd agilex_teleop
   ```

   如果忘记添加 `--recursive` 选项，需要手动克隆子模块：

   ```bash
   cd agilex_ws/agilex_teleop

   # 1. 初始化 submodule 配置
   git submodule init

   # 2. 拉取所有 submodule 的实际代码（递归，如果子模块还有子模块）
   git submodule update --recursive
   ```

2. 安装项目依赖

   - 方式一：使用 requirements.txt + pyproject.toml 安装所有依赖（测试推荐）

      ```bash
      # 安装所有依赖
      pip install -r requirements.txt

      # 安装项目（开发模式）
      pip install -e .
      ```

      PS: 首次安装后运行 `pip install -r requirements.txt`，如果遇到冲突，可尝试升级 sympy 或降级 torch

   - 方式二：使用 pyproject.toml 安装所需依赖

      ```bash
      # 基础安装
      pip install -e .

      # 包含仿真功能
      pip install -e ".[sim]"

      # 包含动力学功能
      pip install -e ".[dynamics]"
      ```

   - 安装 Pinocchio（可不选）

      Pinocchio 依赖较多，如果 `requirements.txt` 安装失败，可使用 conda-forge：

      ```bash
      conda install -c conda-forge pinocchio eigenpy -y
      ```

#### 0.3.2 Client 端项目

1. 克隆 Server 端项目到工作目录并安装依赖

   ```bash
   cd lerobot
   mkdir agilex_ws && cd agilex_ws
   git clone https://github.com/Shenzhaolong1330/dual_arm_teleop.git
   # 备选：使用 Key-Zzs fork 的 dual_arm_teleop 项目
   # git clone https://github.com/Key-Zzs/dual_arm_teleop.git
   cd dual_arm_teleop
   pip install -e .
   ```

2. 安装 Oculus Reader APK（具体安装方式请自行搜索或咨询 @Shenzhaolong1330 ）

   ```bash
   cd dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
   git clone https://github.com/rail-berkeley/oculus_reader.git
   cd oculus_reader
   pip install -e .
   ```

3. 确定 Server 端 PC 的 IP 地址并在 [record_cfg.yaml](../dual_arm_teleop/scripts/config/record_cfg.yaml) 中配置

   ```bash
   ifconfig
   ```

   ```yaml
   robot:
      robot_ip: &ip "192.168.110.41" # Server 端 PC 的 IP 地址
      robot_port: 4242 # Server 端 PC 的端口号
   ```

---

## 1 遥操作前置工作

### 1.1 Server 端电脑配置

#### 1.1.1 激活双臂 can 设备

> 使用 `can_muti_activate.sh` 脚本

首先将两个官方 can 模块插入到 Server 端电脑，建议先插入左臂 can 模块再插入右臂 can 模块。

1. 记录每个 CAN 模块对应的 USB 端口硬件地址

   ```bash
   bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh
   ```

   记录下两个 CAN 模块的 `USB port` 的数值，例如 `3-1.4:1.0` 和 `3-1.1:1.0`。

   > **提示：** 如果未曾激活过，则第一个插入的 CAN 模块默认为 `can0`，第二个为 `can1` ，若按前述建议顺序插入，即左臂 can 模块为 `can0`，右臂为 `can1` ；若激活过，名字为之前激活过的名称。

2. 预定义 USB 端口、目标接口名称及波特率

   假设上面记录的 `USB port` 数值分别为 `3-1.4:1.0` 和 `3-1.1:1.0`，则将 [agilex_ws/agilex_teleop/pyAgxArm/scripts/ubuntu/can_muti_activate.sh](./pyAgxArm/scripts/ubuntu/can_muti_activate.sh) 中的参数修改为：

   ```bash
   USB_PORTS["3-1.4:1.0"]="can_left:1000000"
   USB_PORTS["3-1.1:1.0"]="can_right:1000000"
   ```

   含义：`3-1.4:1.0` 端口的 CAN 设备重命名为 `can_left`，波特率 `1000000`，并激活。

3. 激活多个 CAN 模块

   执行：

   ```bash
   bash pyAgxArm/scripts/ubuntu/can_muti_activate.sh
   ```

4. 验证是否设置成功

   ```bash
   bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh
   ```
   查看是否有 `can_left` 和 `can_right`。


can 模块使用手册详见官方文档：[docs/can_user.md](./docs/can_user.md#can-模块使用手册)

#### 1.1.2 运行 nero 测试脚本（最好运行，否则后续机械臂可能会处于未使能状态）

**注意**：[reset.py](./nero/tests/reset.py) 和 [test_pos_flw_ik.py](./nero/tests/test_pos_flw_ik.py) 均为单臂测试脚本，请运行单臂后修改文件中的 can 设备名，如 `can_left` 或 `can_right`，再运行下一个。

```bash
# nero 关节重置脚本
python nero/tests/reset.py
# nero 位置跟随 IK 测试脚本
python nero/tests/test_pos_flw_ik.py
```

### 1.2 Client 端电脑配置

#### 1.2.1 相机环境配置

1. RealSense 相机环境配置

   运行 `realsense-viewer` 查看 RealSense 相机是否正常工作：

      ```bash
      realsense-viewer
      ```

   > 若无 `realsense-viewer` 命令，说明未安装 `realsense-viewer` 工具，请自行网上搜索安装

2. 序列号获取与配置（若使用当前设备，无需此步）

   运行 `realsense-viewer` 查看 RealSense 相机 info，记录下相机序列号 `Serial Number`，例如 `412622270929`。然后依次将 `agilex_ws/dual_arm_teleop/scripts/config/record_cfg.yaml` 中的 `left_wrist_cam_serial` 和 `right_wrist_cam_serial` 和 `head_cam_serial` 后的序列号替换为对应相机序列号。

   ```yaml
   teleop:
      cameras:
         left_wrist_cam_serial: "412622270929"
         right_wrist_cam_serial: "412622270929"
         head_cam_serial: "412622270929"
   ```

#### 1.2.2 Oculus Quest 设置

1. 安装 ADB（Android 调试桥）：Oculus Quest 与计算机之间通信必需的工具

   ```bash
   # 在 Ubuntu 上
   sudo apt install android-tools-adb

   # 验证安装
   adb version
   ```

2. 在 Oculus Quest 上启用开发者模式

   1. 在 [Meta for Developers](https://developer.oculus.com/manage/organizations/create/) 创建或加入开发者组织
   2. 在手机上打开 Meta Quest 应用
   3. 进入 **设置** → 选择您的设备 → **更多设置** → **开发者模式**
   4. 启用 **开发者模式** 开关

3. 连接 Oculus Quest 到计算机

   方式 1：USB 连接（推荐用于初始设置，或对实时性要求高的场景）

   1. 使用 USB-C 线缆将 Oculus Quest 连接到计算机
   2. 佩戴头显并在提示时允许 USB 调试
   3. 勾选 `始终允许来自此计算机`
   4. 验证连接：
   
      ```bash
      adb devices
      # 预期输出：
      # List of devices attached
      # <device_id>    device
      adb shell ip route
      # 查找 "src" 后面的 IP 地址，例如 192.168.110.62
      ```

   方式 2：无线连接（操作更便捷）

   1. 首先通过 USB 线缆连接 Oculus Quest 到计算机执行方案 1
   2. 确保 Oculus Quest 和计算机连接到同一网络
   3. 验证连接：
   
      ```bash
      adb connect <获取到的IP地址>:5555
      adb shell ip route
      # 查找 "src" 后面的 IP 地址，例如 192.168.110.62
      ```

   4. 在 `record_cfg.yaml` 中配置 IP：
   
      ```yaml
      teleop:
         oculus_config:
            ip: "192.168.110.62"  # 您的 Oculus Quest IP 地址
      ```

---

## 2 启动遥操作 Server 端服务

**注意**：
请保证 `bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh` 输出有 `can_left` 和 `can_right` 两个 can 设备！！

```bash
# 启动 Server 服务
python nero/teleop/interface/nero_interface_server.py --ip 0.0.0.0 --port 4242

# 开放端口 4242（若 Server 端 PC 默认开放端口，无需此步）
udo iptables -I INPUT -p tcp --dport 4242 -j ACCEPT # iptables 方式
```

---

## 3 启动遥操作 Client 端服务

**注意**：
1. 启动前请 `adb devices` 检查 Oculus Quest 是否连接成功
2. 每次修改项目中的 python 文件后，需在项目根目录 `agilex_ws/dual_arm_teleop`  下执行 `pip install -e .` 更新依赖

```bash
# 重置机械臂
robot-reset
# 开始遥操作及数据采集
robot-record
# 右箭头：停止采集数据
# enter：继续遥操作
```

> 其余操作请执行 `robot-help` 查看

- 补充：Oculus 控制器操作说明

   | 控制键 | 功能 |
   |--------|------|
   | **RG（右手握持键）** | 按住以启动机器人运动 |
   | **RTr（右手扳机）** | 按下关闭夹爪，松开打开夹爪 |
   | **A 按钮** | 请求机器人复位 |
   | **右手控制器位姿** | 控制末端执行器增量位姿 |

---

## 项目核心结构

```
AgxArm_teleop/
├── pyAgxArm/              # Agilex 机械臂 SDK
│   ├── api/               # API 接口
│   ├── protocols/         # 通信协议
│   └── utiles/            # 工具函数
├── nero/                  # Nero 双臂系统
│   ├── kinematics/        # 运动学
│   ├── teleop/            # 遥操作
│   └── tests/             # 测试脚本
├── requirements.txt       # 依赖列表
└── pyproject.toml         # 项目配置
```

---

## 官方开发资料

| 说明 | 文档 |
| --- | --- |
| ROS | [agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) |
| CAN 模块手册 | [docs/can_user.md](./docs/can_user.md#can-模块使用手册) |
| Nero 首次使用 CAN 指南 | [docs/nero/first_time_user_guide_can.md](./docs/nero/first_time_user_guide_can.md#nero-首次使用指南can) |
| Nero API | [docs/nero/nero_api.md](./docs/nero/nero_api.md#nero-机械臂-api-使用文档) |
| AgxGripper API | [docs/effector/agx_gripper/agx_gripper_api.md](./docs/effector/agx_gripper/agx_gripper_api.md#agxgripper-夹爪-api-使用文档) |


---

## 鸣谢

该代码基于以下开源代码库构建，在此表示感谢。

- [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)
- [dual_arm_teleop](https://github.com/Shenzhaolong1330/dual_arm_teleop)