#!/usr/bin/env python3
"""采集前的只读硬件检查：机械臂连通性与 Orbbec 相机序列号。"""

from __future__ import annotations

import argparse
import sys


DEFAULT_SERIALS = {
    "cam_top": "CP0P76300026",
    "cam_bottom": "CP0N7630000A",
    "cam_wrist": "CP0N763000HE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--skip-arm", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    return parser.parse_args()


def check_arm(ip: str, port: int) -> bool:
    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    except ImportError as exc:
        print(f"[FAIL] Robotic_Arm SDK 不可用: {exc}")
        return False

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    try:
        handle = arm.rm_create_robot_arm(ip, port)
        state = arm.rm_get_current_arm_state()
        if not isinstance(state, tuple) or state[0] != 0:
            print(f"[FAIL] 机械臂状态读取失败: {state}")
            return False
        joints = [round(float(value), 2) for value in state[1]["joint"][:6]]
        pose = [round(float(value), 4) for value in state[1]["pose"][:6]]
        print(f"[OK] arm {ip}:{port}; handle={handle}")
        print(f"     joints(deg): {joints}")
        print(f"     pose(m,rad): {pose}")
        try:
            ret_info, robot_info = arm.rm_get_robot_info()
            if ret_info == 0:
                print(
                    "     robot: "
                    f"model={robot_info.get('arm_model')}, "
                    f"force_type={robot_info.get('force_type')}, "
                    f"controller={robot_info.get('robot_controller_version')}"
                )
        except Exception as exc:
            print(f"[WARN] 机械臂型号/力传感器信息读取失败: {exc}")
        try:
            ret_sens, sensitivity = arm.rm_get_drag_teach_sensitivity()
            if ret_sens == 0:
                print(f"     drag sensitivity: {sensitivity}/100")
            else:
                print(f"[WARN] 拖动灵敏度读取失败: code {ret_sens}")
        except Exception as exc:
            print(f"[WARN] 拖动灵敏度读取异常: {exc}")
        try:
            ret_tool, tool = arm.rm_get_current_tool_frame()
            if ret_tool == 0:
                print(
                    "     tool: "
                    f"name={str(tool.get('name', '')).strip(chr(0))}, "
                    f"payload={float(tool.get('payload', 0.0)):.4f} kg, "
                    f"com_raw={[tool.get(k) for k in ('x', 'y', 'z')]}"
                )
        except Exception as exc:
            print(f"[WARN] 工具坐标系读取失败: {exc}")
        return True
    except Exception as exc:
        print(f"[FAIL] 机械臂检查异常: {exc}")
        return False
    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass


def check_cameras() -> bool:
    try:
        from lerobot.cameras.orbbec import OrbbecCamera
    except ImportError as exc:
        print(f"[FAIL] LeRobot Orbbec 驱动不可用: {exc}")
        return False

    try:
        devices = OrbbecCamera.find_cameras()
    except Exception as exc:
        print(f"[FAIL] 相机枚举失败: {exc}")
        return False

    found = {str(device.get("serial_number")) for device in devices}
    success = True
    for name, serial in DEFAULT_SERIALS.items():
        if serial in found:
            print(f"[OK] {name}: {serial}")
        else:
            print(f"[FAIL] {name}: 未找到 {serial}")
            success = False
    unknown = found.difference(DEFAULT_SERIALS.values())
    if unknown:
        print(f"[INFO] 额外相机: {sorted(unknown)}")
    return success


def main() -> None:
    args = parse_args()
    results = []
    if not args.skip_arm:
        results.append(check_arm(args.ip, args.port))
    if not args.skip_cameras:
        results.append(check_cameras())
    if not results:
        print("未执行任何检查。")
        return
    if all(results):
        print("\n硬件检查通过。")
        return
    print("\n硬件检查未通过。")
    sys.exit(1)


if __name__ == "__main__":
    main()
