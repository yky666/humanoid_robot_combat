#!/bin/bash

set -eo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run this launcher as root (for example, with sudo)." >&2
  exit 1
fi

if pgrep -f '[s]rc_executor' >/dev/null; then
  echo "Another src_executor process is already running." >&2
  pgrep -af src_executor >&2
  exit 1
fi

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
executor="$root_dir/_install/bin/src_executor"

if [[ ! -x "$executor" ]]; then
  echo "Missing executable: $executor" >&2
  exit 1
fi

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is not installed." >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
set -u

export ROS_DOMAIN_ID=69
export ENGINEAI_ROBOTICS_RUNTIME_ENV=robot
export ENGINEAI_ROBOTICS_DIR="$root_dir"
export ENGINEAI_ROBOTICS_ASSETS="$root_dir/assets"
export ENGINEAI_ROBOTICS_CONFIG="$root_dir/assets/config"
export ENGINEAI_ROBOTICS_THIRD_PARTY="$root_dir/_install/engineai_robotics_third_party"
export ENGINEAI_ROBOTICS_HARDWARE=/opt/engineai_robotics_hardware
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ENGINEAI_ROBOTICS_THIRD_PARTY/lib:$ENGINEAI_ROBOTICS_THIRD_PARTY/lib/runtime:$ENGINEAI_ROBOTICS_HARDWARE/lib:$root_dir/_install/lib"

cd "$root_dir/_install/bin"
exec "$executor" t800
