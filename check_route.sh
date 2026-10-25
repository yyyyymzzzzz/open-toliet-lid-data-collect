#!/usr/bin/env bash
set -euo pipefail
# This is inspection only. A missing host route causes the Wi-Fi subnet to win.
ip route get 192.168.1.18
ip route show 192.168.1.18/32
if ! ip route get 192.168.1.18 | grep -q 'dev eth2'; then
  echo 'Wrong arm route. After the operator has stopped the robot, run:' >&2
  echo 'sudo ip route add 192.168.1.18/32 dev eth2 src 192.168.1.100' >&2
  exit 2
fi
