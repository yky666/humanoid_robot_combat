#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "basic/basic_runner.h"
#include "basic/runner_registry.h"
#include "input_command_arbiter/base_input_adapter.h"

namespace runner {

// Collects inputs from multiple sources and resolves the final command
// by applying adapters in ascending priority order.
class InputCommandArbiterRunner : public BasicRunner {
 public:
  InputCommandArbiterRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store);
  ~InputCommandArbiterRunner();

  void Run() override;

 private:
  void RegisterHardwareSource(const std::string& name, std::shared_ptr<BaseInputAdapter> adapter);
  void RegisterOverrideSource(const std::string& name, std::shared_ptr<BaseInputAdapter> adapter);
  void InitializeAudioFeedback();
  void NotifyAudioFeedback(const data::GamepadInfo& input);
  void SendAudioFeedback(const std::string& event) const;

  std::vector<std::shared_ptr<BaseInputAdapter>> hardware_sources_;
  std::vector<std::shared_ptr<BaseInputAdapter>> override_sources_;
  int selected_hardware_idx_{-1};

  int audio_feedback_socket_{-1};
  uint32_t audio_feedback_address_{0};
  uint16_t audio_feedback_port_{0};
  int last_audio_key_value_{0};
  std::string last_audio_motion_state_;
};

}  // namespace runner

REGISTER_RUNNER(InputCommandArbiterRunner, "input_command_arbiter_runner", kResident)
