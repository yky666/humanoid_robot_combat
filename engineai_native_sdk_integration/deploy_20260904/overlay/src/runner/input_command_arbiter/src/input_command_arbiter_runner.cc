#include "input_command_arbiter/input_command_arbiter_runner.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <glog/logging.h>

#include "input_command_arbiter/gamepad_input_adapter.h"
#include "input_command_arbiter/rc02_input_adapter.h"
#include "input_command_arbiter/virtual_gamepad_input_adapter.h"

namespace runner {
namespace {

constexpr uint16_t kDefaultAudioFeedbackPort = 45800;
constexpr std::array<int, 11> kAudioFeedbackKeys = {3, 5, 6, 9, 10, 17, 18, 33, 34, 129, 384};

bool IsFeedbackKey(int value) {
  for (const int key : kAudioFeedbackKeys) {
    if (value == key) return true;
  }
  return false;
}

}  // namespace

InputCommandArbiterRunner::InputCommandArbiterRunner(
    std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
    : BasicRunner(name, data_store) {
  RegisterHardwareSource("rc02", std::make_shared<Rc02InputAdapter>("rc02", data_store));
  RegisterHardwareSource("gamepad", std::make_shared<GamepadInputAdapter>("gamepad", data_store));
  RegisterOverrideSource("virtual_gamepad",
                         std::make_shared<VirtualGamepadInputAdapter>("virtual_gamepad", data_store));

  for (int i = 0; i < static_cast<int>(hardware_sources_.size()); ++i) {
    if (!hardware_sources_[i]->Init()) {
      LOG(WARNING) << "InputCommandArbiterRunner: Init failed for hardware source '"
                   << hardware_sources_[i]->GetName() << "'.";
      continue;
    }
    LOG(INFO) << "InputCommandArbiterRunner: hardware source '" << hardware_sources_[i]->GetName()
              << "' Init succeeded at index " << i << ".";
    selected_hardware_idx_ = i;
    break;
  }

  for (auto& source : override_sources_) {
    if (!source->Init()) {
      LOG(WARNING) << "InputCommandArbiterRunner: Init failed for override source '" << source->GetName() << "'.";
    }
  }

  InitializeAudioFeedback();
}

InputCommandArbiterRunner::~InputCommandArbiterRunner() {
  if (audio_feedback_socket_ >= 0) close(audio_feedback_socket_);
}

void InputCommandArbiterRunner::Run() {
  data::GamepadInfo result;
  result.Reset();

  if (selected_hardware_idx_ < 0) {
    for (int i = 0; i < static_cast<int>(hardware_sources_.size()); ++i) {
      if (hardware_sources_[i]->Run() == InputAdapterStatus::NORMAL && selected_hardware_idx_ < 0) {
        selected_hardware_idx_ = i;
      }
      if (hardware_sources_[i]->IsActive()) {
        hardware_sources_[i]->Process(result);
        LOG(INFO) << "Hardware source locked: " << hardware_sources_[i]->GetName();
        break;
      }
    }
  } else {
    auto& selected = hardware_sources_[selected_hardware_idx_];
    static_cast<void>(selected->Run());
    if (selected->IsActive()) selected->Process(result);
  }

  for (auto& source : override_sources_) {
    static_cast<void>(source->Run());
    if (source->IsActive()) source->Process(result);
  }

  NotifyAudioFeedback(result);
  data_store_->gamepad_info.Set(result);
}

void InputCommandArbiterRunner::RegisterHardwareSource(const std::string& name,
                                                       std::shared_ptr<BaseInputAdapter> adapter) {
  hardware_sources_.emplace_back(std::move(adapter));
  LOG(INFO) << "Registered hardware source: " << name;
}

void InputCommandArbiterRunner::RegisterOverrideSource(const std::string& name,
                                                       std::shared_ptr<BaseInputAdapter> adapter) {
  override_sources_.emplace_back(std::move(adapter));
  LOG(INFO) << "Registered override source: " << name;
}

void InputCommandArbiterRunner::InitializeAudioFeedback() {
  const char* host = std::getenv("ENGINEAI_AUDIO_FEEDBACK_HOST");
  if (host == nullptr || host[0] == '\0') return;

  long port = kDefaultAudioFeedbackPort;
  if (const char* value = std::getenv("ENGINEAI_AUDIO_FEEDBACK_PORT")) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0 && parsed <= 65535) port = parsed;
  }

  if (inet_pton(AF_INET, host, &audio_feedback_address_) != 1) {
    LOG(WARNING) << "Audio feedback disabled: invalid IPv4 address " << host;
    return;
  }

  audio_feedback_socket_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
  if (audio_feedback_socket_ < 0) {
    LOG(WARNING) << "Audio feedback disabled: socket creation failed: " << std::strerror(errno);
    return;
  }

  audio_feedback_port_ = static_cast<uint16_t>(port);
  LOG(INFO) << "Audio feedback UDP target: " << host << ':' << audio_feedback_port_;
}

void InputCommandArbiterRunner::NotifyAudioFeedback(const data::GamepadInfo& input) {
  if (audio_feedback_socket_ < 0) return;

  if (input.combined_key_value != last_audio_key_value_ && IsFeedbackKey(input.combined_key_value)) {
    SendAudioFeedback("KEY " + std::to_string(input.combined_key_value));
  }
  last_audio_key_value_ = input.combined_key_value;

  const auto motion_state = data_store_->current_motion_task_name.Get();
  if (motion_state && *motion_state != last_audio_motion_state_) {
    last_audio_motion_state_ = *motion_state;
    SendAudioFeedback("STATE " + *motion_state);
  }
}

void InputCommandArbiterRunner::SendAudioFeedback(const std::string& event) const {
  sockaddr_in target{};
  target.sin_family = AF_INET;
  target.sin_port = htons(audio_feedback_port_);
  target.sin_addr.s_addr = audio_feedback_address_;
  const ssize_t sent = sendto(audio_feedback_socket_, event.data(), event.size(), MSG_DONTWAIT,
                              reinterpret_cast<const sockaddr*>(&target), sizeof(target));
  if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    VLOG(1) << "Audio feedback UDP send failed: " << std::strerror(errno);
  }
}

}  // namespace runner
