#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f /mnt/Data/xzrobot/env/storage.sh ]]; then
  source /mnt/Data/xzrobot/env/storage.sh
fi
set +u
source /opt/ros/humble/setup.bash
source /mnt/Data/robot-stack/current/setup.bash
set -u
export ROS_DOMAIN_ID=211 ROS_LOCALHOST_ONLY=1
export PYTHONNOUSERSITE=1
export ROS_LOG_DIR="$PWD/logs/ros"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$ROS_LOG_DIR"
exec "$PWD/.venv/bin/python" "$@"
