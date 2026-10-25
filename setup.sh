#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
storage=/mnt/Data/pi05_open_toilet_r5
mkdir -p "$storage/cache"
export UV_CACHE_DIR="$storage/cache"
uv_bin=/home/xzrobot/.local/bin/uv
if [[ ! -e "$storage/.venv/bin/python" ]]; then
  "$uv_bin" venv --python /usr/bin/python3 --system-site-packages "$storage/.venv"
fi
if [[ ! -e .venv ]]; then ln -s "$storage/.venv" .venv; fi
for name in episodes evidence logs cache; do
  mkdir -p "$storage/$name"
  if [[ ! -e "$name" ]]; then ln -s "$storage/$name" "$name"; fi
done
"$uv_bin" pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements.txt
bash run.sh -m unittest -v test_pipeline
