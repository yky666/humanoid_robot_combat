#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sdk_root="$(cd "$script_dir/../.." && pwd)"
third_party="${ENGINEAI_ROBOTICS_THIRD_PARTY:-/opt/engineai_robotics_third_party}"
output="${1:-$script_dir/t800_keyboard_control}"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -I"$sdk_root/core/include/data" \
  -I"$third_party/include" \
  "$script_dir/t800_keyboard_control.cc" \
  -L"$third_party/lib" -Wl,-rpath,'$ORIGIN/../../_install/engineai_robotics_third_party/lib' \
  -llcm -pthread -o "$output"

echo "built $output"
