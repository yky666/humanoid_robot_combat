#!/usr/bin/env bash
set -euo pipefail

ACTION=""
ORIENTATION="unknown"
OUTPUT_ROOT="/home/ubuntu/source/engineai_workspace/src/interface_example/scripts/logs"
DURATION=""
NOTES=""
DOMAIN_ID="${ROS_DOMAIN_ID:-69}"
ALLOW_MISSING_REQUIRED=false
EXTRA_TOPICS=()

usage() {
  cat <<'EOF'
Record synchronized T800 telemetry into a ROS 2 bag.

Usage:
  record_t800_motion.sh --action NAME [options]

Required:
  --action NAME                  Motion label, for example prone_to_stance.

Options:
  --orientation VALUE           upright, supine, prone, or unknown.
  --output-root PATH            Recording root directory.
  --duration SECONDS            Stop automatically after this duration.
  --notes TEXT                  Free-form capture notes.
  --domain-id ID                ROS_DOMAIN_ID, default 69.
  --extra-topic TOPIC           Record an additional topic; repeatable.
  --allow-missing-required      Record even if a required topic is absent.
  -h, --help                    Show this help.

The script only subscribes to topics. It never publishes a robot command.
EOF
}

while (($#)); do
  case "$1" in
    --action)
      ACTION="${2:?missing value for --action}"
      shift 2
      ;;
    --orientation)
      ORIENTATION="${2:?missing value for --orientation}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value for --output-root}"
      shift 2
      ;;
    --duration)
      DURATION="${2:?missing value for --duration}"
      shift 2
      ;;
    --notes)
      NOTES="${2:?missing value for --notes}"
      shift 2
      ;;
    --domain-id)
      DOMAIN_ID="${2:?missing value for --domain-id}"
      shift 2
      ;;
    --extra-topic)
      EXTRA_TOPICS+=("${2:?missing value for --extra-topic}")
      shift 2
      ;;
    --allow-missing-required)
      ALLOW_MISSING_REQUIRED=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  echo "[ERROR] --action is required" >&2
  usage >&2
  exit 2
fi

if [[ ! "$ACTION" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[ERROR] --action may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi

case "$ORIENTATION" in
  upright|supine|prone|unknown) ;;
  *)
    echo "[ERROR] --orientation must be upright, supine, prone, or unknown" >&2
    exit 2
    ;;
esac

if [[ -n "$DURATION" ]] && ! [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] --duration must be a positive number" >&2
  exit 2
fi

if ! command -v ros2 >/dev/null 2>&1; then
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
  fi
fi

workspace_setups=(
  "/home/ubuntu/source/engineai_workspace/install/setup.bash"
  "/home/ubuntu/source/engineai_workspace/_install/setup.bash"
  "/home/ubuntu/source/engineai_workspace/build/ros2_env/install/setup.bash"
)
for setup_file in "${workspace_setups[@]}"; do
  if [[ -f "$setup_file" ]]; then
    set +u
    source "$setup_file"
    set -u
    break
  fi
done

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[ERROR] ros2 is unavailable; source the EngineAI ROS 2 environment first" >&2
  exit 1
fi

export ROS_DOMAIN_ID="$DOMAIN_ID"

mapfile -t DISCOVERED_TOPICS < <(ros2 topic list | sed '/^[[:space:]]*$/d' | sort -u)
if ((${#DISCOVERED_TOPICS[@]} == 0)); then
  echo "[ERROR] no ROS 2 topics discovered in domain $ROS_DOMAIN_ID" >&2
  exit 1
fi

topic_exists() {
  local wanted="$1"
  local topic
  for topic in "${DISCOVERED_TOPICS[@]}"; do
    [[ "$topic" == "$wanted" ]] && return 0
  done
  return 1
}

REQUIRED_TOPICS=(
  "/hardware/joint_state"
  "/hardware/joint_command_feedback"
  "/hardware/imu_info"
)

missing_required=()
for topic in "${REQUIRED_TOPICS[@]}"; do
  if ! topic_exists "$topic"; then
    missing_required+=("$topic")
  fi
done

if ((${#missing_required[@]} > 0)); then
  echo "[ERROR] missing required telemetry topics:" >&2
  printf '  %s\n' "${missing_required[@]}" >&2
  if [[ "$ALLOW_MISSING_REQUIRED" != true ]]; then
    echo "[ERROR] start the ROS2 bridge, or pass --allow-missing-required for diagnostics only" >&2
    exit 1
  fi
fi

SELECTED_TOPICS=()
add_topic() {
  local candidate="$1"
  local existing
  topic_exists "$candidate" || return 0
  for existing in "${SELECTED_TOPICS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  SELECTED_TOPICS+=("$candidate")
}

DEFAULT_TOPICS=(
  "/hardware/joint_state"
  "/hardware/joint_command_feedback"
  "/hardware/motor_state"
  "/hardware/motor_command"
  "/hardware/imu_info"
  "/hardware/motor_debug"
  "/hardware/power_info"
  "/hardware/gamepad_keys"
  "/motion/motion_state"
  "/tf"
  "/tf_static"
)
for topic in "${DEFAULT_TOPICS[@]}"; do
  add_topic "$topic"
done

for topic in "${DISCOVERED_TOPICS[@]}"; do
  if [[ "$topic" =~ (base|odom|link_info|contact|foot|wrench|force) ]]; then
    add_topic "$topic"
  fi
done

for topic in "${EXTRA_TOPICS[@]}"; do
  if ! topic_exists "$topic"; then
    echo "[ERROR] requested extra topic does not exist: $topic" >&2
    exit 1
  fi
  add_topic "$topic"
done

has_base_pose=false
has_contact_data=false
for topic in "${SELECTED_TOPICS[@]}"; do
  topic_type="$(ros2 topic type "$topic" 2>/dev/null || true)"
  case "$topic_type" in
    tf2_msgs/msg/TFMessage)
      if [[ "$topic" == "/tf" ]]; then
        has_base_pose=true
      fi
      ;;
    interface_protocol/msg/LinkInfo|nav_msgs/msg/Odometry|geometry_msgs/msg/PoseStamped)
      if [[ "$topic" =~ (base|root|odom) ]]; then
        has_base_pose=true
      fi
      ;;
  esac
  if [[ "$topic" =~ (contact|foot|wrench|force) ]]; then
    has_contact_data=true
  fi
done

if [[ "$has_contact_data" != true ]]; then
  echo "[WARNING] no foot/contact/force/wrench topic was discovered" >&2
  echo "[WARNING] the bag will preserve joint and IMU data, but contact labels must be reconstructed later" >&2
fi

if ((${#SELECTED_TOPICS[@]} == 0)); then
  echo "[ERROR] none of the requested telemetry topics exist" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
recording_name="${timestamp}_${ACTION}_${ORIENTATION}"
mkdir -p "$OUTPUT_ROOT"
bag_dir="$OUTPUT_ROOT/$recording_name"
manifest_tmp="$OUTPUT_ROOT/.${recording_name}_capture_manifest.yaml"

if [[ -e "$bag_dir" || -e "$manifest_tmp" ]]; then
  echo "[ERROR] recording target already exists: $bag_dir" >&2
  exit 1
fi

{
  printf 'format_version: 1\n'
  printf 'action: "%s"\n' "$ACTION"
  printf 'orientation: "%s"\n' "$ORIENTATION"
  printf 'notes: "%s"\n' "${NOTES//\"/\\\"}"
  printf 'hostname: "%s"\n' "$(hostname)"
  printf 'operator: "%s"\n' "${USER:-unknown}"
  printf 'start_time_local: "%s"\n' "$(date --iso-8601=seconds)"
  printf 'start_time_utc: "%s"\n' "$(date --utc --iso-8601=seconds)"
  printf 'ros_domain_id: %s\n' "$ROS_DOMAIN_ID"
  printf 'has_dynamic_base_pose_topic: %s\n' "$has_base_pose"
  printf 'has_contact_topic: %s\n' "$has_contact_data"
  printf 'topics:\n'
  for topic in "${SELECTED_TOPICS[@]}"; do
    topic_type="$(ros2 topic type "$topic" 2>/dev/null || true)"
    printf '  - name: "%s"\n' "$topic"
    printf '    type: "%s"\n' "$topic_type"
  done
} > "$manifest_tmp"

cleanup_manifest() {
  if [[ -d "$bag_dir" && -f "$manifest_tmp" ]]; then
    printf 'end_time_local: "%s"\n' "$(date --iso-8601=seconds)" >> "$manifest_tmp"
    printf 'end_time_utc: "%s"\n' "$(date --utc --iso-8601=seconds)" >> "$manifest_tmp"
    mv "$manifest_tmp" "$bag_dir/capture_manifest.yaml"
  fi
}
trap cleanup_manifest EXIT

echo "[INFO] action: $ACTION"
echo "[INFO] orientation: $ORIENTATION"
echo "[INFO] ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "[INFO] output: $bag_dir"
echo "[INFO] recording topics:"
printf '  %s\n' "${SELECTED_TOPICS[@]}"
echo "[INFO] start the official action after rosbag reports 'Recording...'"
echo "[INFO] press Ctrl-C only after the robot has reached and held the terminal pose"

record_command=(ros2 bag record --storage sqlite3 --max-cache-size 1073741824 -o "$bag_dir")
record_command+=("${SELECTED_TOPICS[@]}")

if [[ -n "$DURATION" ]]; then
  set +e
  timeout --signal=INT --kill-after=10 "$DURATION" "${record_command[@]}"
  status=$?
  set -e
  if [[ $status -ne 0 && $status -ne 124 && $status -ne 130 ]]; then
    exit "$status"
  fi
else
  set +e
  "${record_command[@]}"
  status=$?
  set -e
  if [[ $status -ne 0 && $status -ne 130 ]]; then
    exit "$status"
  fi
fi

if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
  echo "[WARNING] metadata.yaml is missing; attempting rosbag reindex" >&2
  ros2 bag reindex "$bag_dir"
fi

if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
  echo "[ERROR] recording contains SQLite data but no metadata.yaml" >&2
  echo "[ERROR] run: ros2 bag reindex '$bag_dir'" >&2
  exit 1
fi

ros2 bag info "$bag_dir"
echo "[OK] recording completed: $bag_dir"
