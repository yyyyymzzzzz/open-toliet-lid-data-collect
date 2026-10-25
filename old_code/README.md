# Open Toilet Lid 数据采集

基于 Realman ECO63F、三台 Orbbec Gemini 335 和 LeRobot 0.5.2 的键盘拖动示教采集工具。代码源自 `rm-collect-wwx` 的采集流程，默认任务和数据集已改为 `open_toilet_lid`。

## 1. 登录与环境

采集程序需要交互式终端，SSH 时使用 `-t`：

```bash
ssh -t user@192.168.110.101
source /home/user/miniconda3/bin/activate lerobot_env
cd /home/user/workspaces/ymz/open_toilet_lid
```

当前 `lerobot_env` 中的 LeRobot 是可编辑安装，源码位于 `/home/user/workspaces/jiangjiaying/rm-collect`。

## 2. 采集前检查

```bash
python check_hardware.py
```

脚本只读取机械臂状态并枚举相机，不会运动机械臂。已核对的默认相机为：

| 视角 | 序列号 |
|---|---|
| `cam_top` | `CP0P76300026` |
| `cam_bottom` | `CP0N7630000A` |
| `cam_wrist` | `CP0N763000HE` |

## 3. 启动采集

```bash
python collect_open_toilet_lid.py
```

默认参数：

- 机械臂：`192.168.1.18:8080`
- 采集频率：30 Hz
- 目标条数：100
- task：`Open the toilet lid`
- repo id：`local/open_toilet_lid_v1`
- 数据目录：`episodes/datasets`
- init home：`3.2 45.72 -132.48 88.6 85.96 0.73` deg

完整示例：

```bash
python collect_open_toilet_lid.py \
  --ip 192.168.1.18 \
  --freq 30 \
  --drag-mode 0 \
  --drag-sensitivity 100 \
  --home-speed 20 \
  --repo-id local/open_toilet_lid_v1 \
  --task "Open the toilet lid" \
  --num-episodes 100 \
  --init-home 3.2 45.72 -132.48 88.6 85.96 0.73
```

只调试机械臂和 CSV，不连相机：

```bash
python collect_open_toilet_lid.py --no-cameras
```

默认要求三台相机全部连接，防止误创建缺少视角的数据集。仅在调试时可使用 `--allow-partial-cameras`。

## 4. 每条 episode 的操作

| 状态 | 按键 | 操作 |
|---|---|---|
| IDLE | `h` | 移动到 init home |
| IDLE | `Enter` | 记录当前位姿为本条起点，开始拖动示教 |
| RECORDING | `Enter` | 结束示教，保存 CSV，自动回到本条起点 |
| RECORDING | `b` / `d` | 放弃本次录制 |
| 待回放 | `r` | 回放轨迹，同步采集相机、关节、位姿和力 |
| REPLAYING | `b` | 缓停并中止回放 |
| REVIEW | `s` | 保存到 LeRobot dataset |
| REVIEW | `d` | 删除本条数据 |
| REVIEW | `r` | 回到起点后重新回放采集 |
| 任意 | `q` / `Esc` | 安全退出 |

数据集视觉与状态数据在回放阶段同步采集，人工拖动阶段仅保存轨迹 CSV。

## 5. 安全要求

- 按 `h`、结束示教或开始回放前，确认机械臂到目标位姿的路径无人、无障碍物。
- 首次运行先使用低 `--home-speed`，并随时准备急停。
- 每条示教要从统一的马桶盖初始状态和机械臂位姿开始。
- 当前工具坐标系沿用 `rm-collect-wwx`：TCP Z 偏移 204.4 mm，负载 0.5892 kg，质心 `[-12.9, 0.2, 92.6]` mm。如果更换末端工具，必须先修改并标定 `drag_teach_utils.py` 中的参数，不要直接采集。
- 当前为无电机夹具，gripper 数值固定为 0。

## 6. 输出格式

```text
episodes/
├── <timestamp>/
│   ├── recording.csv
│   ├── arm_trajectory.txt
│   ├── meta.json
│   └── cam_frames/
└── datasets/
    ├── meta/
    ├── data/
    └── videos/
```

LeRobot features：

- `observation.state` (17, float32): `j0..j5(deg), xyz(m), rx/ry/rz(rad), qw/qx/qy/qz, gripper`
- `observation.force` (6, float32): `Fx/Fy/Fz(N), Mx/My/Mz(N·m)`
- `action` (7, float32): `dx/dy/dz(m), drx/dry/drz(rad), gripper`
- `observation.images.cam_top/cam_bottom/cam_wrist`: 640×480 RGB video

## 7. 检查与恢复

检查 CSV 轨迹：

```bash
python inspect_episode.py
python inspect_episode.py --episode <timestamp> --export-plot --export-csv
```

视频编码失败时，主程序会尝试把临时 PNG 保全到 `frames_dump/`。离线重建：

```bash
python build_dataset.py --base-dir episodes --dry-run
python build_dataset.py --base-dir episodes
```

仅在重建成功且确认数据可用后，才考虑删除备份帧：

```bash
python build_dataset.py --base-dir episodes --remove-frames-dump
```
