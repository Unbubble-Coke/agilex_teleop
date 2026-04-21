# AgxArm_teleop

[English](./README.md) | [中文](./README_zh-CN.md)

---

AgileX nero dual-arm teleoperation system, developed based on the AgileX robotic arm SDK.

Required equipment for this project:
- **AgileX nero robotic arm** x 2
- **AgileX gripper (integrated with Intel RealSense D405 camera sensor)** x 2
- **Intel RealSense D435i camera sensor** x 1
- **Server PC** (requires higher running memory) x 1: Used to run the Server project to control the robotic arms (connected to CAN-USB adapters linked to AgileX robotic arms)
- **Client PC** (requires larger disk space) x 1: Used to run the Client project for teleoperating the robotic arms and collecting data (connected to three camera sensors)
- AgileX official **CAN-USB adapter** x 2

---

## 0 Environment Setup

### 0.1 Create Conda Virtual Environment

```bash
# Create a Python 3.10 environment named agilex_teleop
conda create -n agilex_teleop python=3.10 -y

# Activate the environment
conda activate agilex_teleop
```

### 0.2 Clone lerobot Project and Install lerobot Framework
```bash
# Install specific version 0.3.4
# git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

### 0.3 Clone and Initialize Project Dependencies

#### 0.3.1 Server Project

1. Clone the Server project to your workspace

   ```bash
   cd lerobot
   mkdir agilex_ws && cd agilex_ws
   git clone --recursive https://github.com/Key-Zzs/agilex_teleop.git
   cd agilex_teleop
   ```

   If you forgot to add the `--recursive` option, you need to manually clone submodules:

   ```bash
   cd agilex_ws/agilex_teleop

   # 1. Initialize submodule configuration
   git submodule init

   # 2. Fetch all submodule code (recursive, if submodules have submodules)
   git submodule update --recursive
   ```

2. Install project dependencies

   - Method 1: Use requirements.txt + pyproject.toml to install all dependencies (recommended for testing)

      ```bash
      # Install all dependencies
      pip install -r requirements.txt

      # Install project (development mode)
      pip install -e .
      ```

      PS: After initial installation, run `pip install -r requirements.txt`. If you encounter conflicts, try upgrading sympy or downgrading torch.

   - Method 2: Use pyproject.toml to install required dependencies

      ```bash
      # Basic installation
      pip install -e .

      # Include simulation features
      pip install -e ".[sim]"

      # Include dynamics features
      pip install -e ".[dynamics]"
   ```

   - Install Pinocchio (optional)

      Pinocchio has many dependencies. If `requirements.txt` installation fails, you can use conda-forge:

      ```bash
      conda install -c conda-forge pinocchio eigenpy -y
      ```

#### 0.3.2 Client Project

1. Clone the Server project to your workspace and install dependencies

   ```bash
   cd lerobot
   mkdir agilex_ws && cd agilex_ws
   git clone https://github.com/Shenzhaolong1330/dual_arm_teleop.git
   # Alternative: Use Key-Zzs fork of dual_arm_teleop project
   # git clone https://github.com/Key-Zzs/dual_arm_teleop.git
   cd dual_arm_teleop
   pip install -e .
   ```

2. Install Oculus Reader APK (please search for specific installation method or consult @Shenzhaolong1330)

   ```bash
   cd dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
   git clone https://github.com/rail-berkeley/oculus_reader.git
   cd oculus_reader
   pip install -e .
   ```

3. Determine the Server PC's IP address and configure in [record_cfg.yaml](../dual_arm_teleop/scripts/config/record_cfg.yaml)

   ```bash
   ifconfig
   ```

   ```yaml
   robot:
      robot_ip: &ip "192.168.110.41" # Server PC's IP address
      robot_port: 4242 # Server PC's port number
    ```

---

## 1 Teleoperation Prerequisites

### 1.1 Server PC Configuration

#### 1.1.1 Activate Dual-Arm CAN Devices

> Use the `can_muti_activate.sh` script

First, insert the two official CAN modules into the Server PC. It is recommended to insert the left arm CAN module first, then the right arm CAN module.

1. Record the USB port hardware addresses for each CAN module

   ```bash
   bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh
   ```

   Record the `USB port` values for the two CAN modules, for example `3-1.4:1.0` and `3-1.1:1.0`.

   > **Tip:** If never activated before, the first inserted CAN module defaults to `can0`, the second to `can1`. If inserting in the recommended order (left arm first), the left arm CAN module is `can0` and the right arm is `can1`. If previously activated, the names will be whatever was used before.

2. Pre-define USB ports, target interface names, and baud rates

   Assuming the recorded `USB port` values are `3-1.4:1.0` and `3-1.1:1.0`, modify the parameters in `agilex_ws/agilex_teleop/pyAgxArm/scripts/ubuntu/can_muti_activate.sh`:

   ```bash
   USB_PORTS["3-1.4:1.0"]="can_left:1000000"
   USB_PORTS["3-1.1:1.0"]="can_right:1000000"
   ```

   Meaning: Rename the CAN device on port `3-1.4:1.0` to `can_left`, with baud rate `1000000`, and activate it.

3. Activate Multiple CAN Modules

   Execute:

   ```bash
   bash pyAgxArm/scripts/ubuntu/can_muti_activate.sh
   ```

4. Verify Settings

   ```bash
   bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh
   ```
   Check if `can_left` and `can_right` appear.


For CAN module user manual, see official documentation: [docs/can_user.md](./docs/can_user.md#can-module-user-manual)

#### 1.1.2 Run nero Test Scripts (recommended to run, otherwise the robotic arm may remain in a disabled state)

**Note**: [reset.py](./nero/tests/reset.py) and [test_pos_flw_ik.py](./nero/tests/test_pos_flw_ik.py) are single-arm test scripts. After running one arm, modify the CAN device name in the file (e.g., `can_left` or `can_right`), then run the other.

```bash
# nero joint reset script
python nero/tests/reset.py
# nero position following IK test script
python nero/tests/test_pos_flw_ik.py
```

### 1.2 Client PC Configuration

#### 1.2.1 Camera Environment Setup

1. RealSense Camera Environment Setup

   Run `realsense-viewer` to check if the RealSense camera is working properly:

      ```bash
      realsense-viewer
      ```

   > If `realsense-viewer` command is not found, it means `realsense-viewer` is not installed. Please search for installation instructions online.

2. Obtain and Configure Serial Numbers (skip if using current device)

   Run `realsense-viewer` to view RealSense camera info, record the camera serial number `Serial Number`, for example `412622270929`. Then replace the serial numbers after `left_wrist_cam_serial`, `right_wrist_cam_serial`, and `head_cam_serial` in `agilex_ws/dual_arm_teleop/scripts/config/record_cfg.yaml` with the corresponding camera serial numbers.

   ```yaml
   teleop:
      cameras:
         left_wrist_cam_serial: "412622270929"
         right_wrist_cam_serial: "412622270929"
         head_cam_serial: "412622270929"
   ```

#### 1.2.2 Oculus Quest Setup

1. Install ADB (Android Debug Bridge): Required tool for communication between Oculus Quest and computer

   ```bash
   # On Ubuntu
   sudo apt install android-tools-adb

   # Verify installation
   adb version
   ```

2. Enable Developer Mode on Oculus Quest

   1. Create or join a developer organization at [Meta for Developers](https://developer.oculus.com/manage/organizations/create/)
   2. Open the Meta Quest app on your phone
   3. Go to **Settings** → Select your device → **More Settings** → **Developer Mode**
   4. Enable the **Developer Mode** switch

3. Connect Oculus Quest to Computer

   Method 1: USB Connection (recommended for initial setup or high real-time requirements)

   1. Connect Oculus Quest to computer using a USB-C cable
   2. Put on the headset and allow USB debugging when prompted
   3. Check "Always allow from this computer"
   4. Verify connection:
   
      ```bash
      adb devices
      # Expected output:
      # List of devices attached
      # <device_id>    device
      adb shell ip route
      # Find the IP address after "src", for example 192.168.110.62
      ```

   Method 2: Wireless Connection (more convenient operation)

   1. First connect Oculus Quest to computer via USB cable to execute Method 1
   2. Ensure Oculus Quest and computer are on the same network
   3. Verify connection:
   
      ```bash
      adb connect <obtained_IP_address>:5555
      adb shell ip route
      # Find the IP address after "src", for example 192.168.110.62
      ```

   4. Configure IP in `record_cfg.yaml`:
   
      ```yaml
      teleop:
         oculus_config:
            ip: "192.168.110.62"  # Your Oculus Quest IP address
      ```

---

## 2 Start Teleoperation Server Service

**Note**:
1. Please ensure `bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh` outputs `can_left` and `can_right` two CAN devices!!
2. **When starting the Server service** [nero_interface_server.py](./nero/teleop/interface/nero_interface_server.py) **be sure to hold the robotic arm steady, the program is not optimized yet, there is a risk of falling!!!!!!**

```bash
# Start Server service
python nero/teleop/interface/nero_interface_server.py --ip 0.0.0.0 --port 4242

# Open port 4242 (if Server PC has ports open by default, skip this step)
sudo iptables -I INPUT -p tcp --dport 4242 -j ACCEPT # iptables method
```

---

## 3 Start Teleoperation Client Service

**Note**:
1. Before starting, please check if Oculus Quest is connected successfully with `adb devices`
2. After modifying any Python file in the project, run `pip install -e .` in the project root directory `agilex_ws/dual_arm_teleop` to update dependencies

```bash
# Reset robotic arm
robot-record
# Right arrow: Stop data collection
# Enter: Continue teleoperation
```

> For other operations, run `robot-help` to view

- Supplement: Oculus Controller Operation Instructions

   | Control Key | Function |
   |-------------|----------|
   | **RG (Right Grip)** | Hold to start robot movement |
   | **RTr (Right Trigger)** | Press to close gripper, release to open gripper |
   | **A Button** | Request robot reset |
   | **Right controller pose** | Control end effector incremental pose |

---

## Project Core Structure

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

## Official Development Resources

| Description | Documentation |
| --- | --- |
| ROS | [agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) |
| CAN Module Manual | [docs/can_user.md](./docs/can_user.md#can-module-user-manual) |
| Nero First-Time User CAN Guide | [docs/nero/first_time_user_guide_can.md](./docs/nero/first_time_user_guide_can.md#nero-first-time-user-guidecan) |
| Nero API | [docs/nero/nero_api.md](./docs/nero/nero_api.md#nero-robotic-arm-api-usage-documentation) |
| AgxGripper API | [docs/effector/agx_gripper/agx_gripper_api.md](./docs/effector/agx_gripper/agx_gripper_api.md#agxgripper-gripper-api-usage-documentation) |

---

## Acknowledgments

This code is built upon the following open-source repositories, which we would like to thank.

- [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)
- [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)
- [dual_arm_teleop](https://github.com/Shenzhaolong1330/dual_arm_teleop)