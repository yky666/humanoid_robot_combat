#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <lcm/lcm-cpp.hpp>

#include "lcm_data/GamepadKeys.hpp"
#include "lcm_data/TaskState.hpp"

namespace {

constexpr char kGamepadChannel[] = "virtual_gamepad/gamepad_keys";
constexpr char kTaskStateChannel[] = "task_state";
constexpr char kDefaultLcmUrl[] = "udpm://239.255.76.67:7667?ttl=0";
constexpr auto kFramePeriod = std::chrono::milliseconds(20);
constexpr int kHeldFrameCount = 18;
constexpr int kReleaseFrameCount = 5;

enum KeyIndex {
  kLb = 0,
  kRb = 1,
  kA = 2,
  kB = 3,
  kX = 4,
  kY = 5,
  kBack = 6,
  kStart = 7,
};

struct Action {
  std::string name;
  std::vector<int> keys;
  std::set<std::string> allowed_states;
};

const std::map<char, Action> kActions = {
    {'i', {"idle", {kLb, kStart}, {"passive"}}},
    {'p', {"passive", {kLb, kRb}, {"idle", "pd_stand", "pd_stand_x", "pd_stand_y",
                                      "qualifier_front_kick", "qualifier_straight_punch",
                                      "qualifier_hook_punch", "qualifier_jab_left",
                                      "qualifier_recovery_supine"}}},
    {'t', {"pd_stand", {kLb, kA}, {"passive", "pd_stand_x", "pd_stand_y",
                                      "qualifier_front_kick", "qualifier_straight_punch",
                                      "qualifier_hook_punch", "qualifier_jab_left"}}},
    {'x', {"pd_stand_x", {kLb, kX}, {"passive", "pd_stand", "pd_stand_y"}}},
    {'y', {"pd_stand_y", {kLb, kY}, {"passive", "pd_stand", "pd_stand_x"}}},
    {'f', {"qualifier_front_kick", {kRb, kA}, {"pd_stand"}}},
    {'c', {"qualifier_straight_punch", {kRb, kY}, {"pd_stand"}}},
    {'h', {"qualifier_hook_punch", {kLb, kB}, {"pd_stand"}}},
    {'j', {"qualifier_jab_left", {kRb, kB}, {"pd_stand"}}},
    {'r', {"qualifier_recovery_supine", {kBack, kA}, {"passive"}}},
};

class TerminalMode {
 public:
  TerminalMode() {
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &original_) != 0) return;
    termios raw = original_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    active_ = tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0;
  }

  ~TerminalMode() {
    if (active_) tcsetattr(STDIN_FILENO, TCSANOW, &original_);
  }

  TerminalMode(const TerminalMode&) = delete;
  TerminalMode& operator=(const TerminalMode&) = delete;

 private:
  termios original_{};
  bool active_ = false;
};

class TaskStateListener {
 public:
  void Handle(const lcm::ReceiveBuffer*, const std::string&, const data::TaskState* message) {
    last_message_time_ = std::chrono::steady_clock::now();
    if (message->current_motion_task_name != state_) {
      state_ = message->current_motion_task_name;
      std::cout << "\n[state] " << state_ << std::endl;
    }
  }

  bool IsReady() const {
    return !state_.empty() &&
           std::chrono::steady_clock::now() - last_message_time_ < std::chrono::seconds(1);
  }

  const std::string& state() const { return state_; }

 private:
  std::string state_;
  std::chrono::steady_clock::time_point last_message_time_{};
};

int64_t TimestampMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

data::GamepadKeys MakeMessage(const std::vector<int>& pressed_keys) {
  data::GamepadKeys message{};
  message.timestamp = TimestampMicros();
  for (const int key : pressed_keys) message.digital_states[key] = 1;
  return message;
}

void PublishAction(lcm::LCM& lcm, const Action& action) {
  for (int frame = 0; frame < kHeldFrameCount; ++frame) {
    auto message = MakeMessage(action.keys);
    lcm.publish(kGamepadChannel, &message);
    lcm.handleTimeout(0);
    std::this_thread::sleep_for(kFramePeriod);
  }
  for (int frame = 0; frame < kReleaseFrameCount; ++frame) {
    auto message = MakeMessage({});
    lcm.publish(kGamepadChannel, &message);
    lcm.handleTimeout(0);
    std::this_thread::sleep_for(kFramePeriod);
  }
}

bool ReadCharacter(char* value, int timeout_ms) {
  fd_set input;
  FD_ZERO(&input);
  FD_SET(STDIN_FILENO, &input);
  timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
  const int result = select(STDIN_FILENO + 1, &input, nullptr, nullptr, &timeout);
  return result > 0 && read(STDIN_FILENO, value, 1) == 1;
}

void PrintHelp() {
  std::cout << "Keys: p=passive t=pd_stand x=prone-PD y=supine-PD "
               "j=jab h=hook c=straight-punch f=front-kick r=recovery "
               "i=idle ?=help q=quit\n"
               "Spinning kick is intentionally unavailable (qualification gate failed).\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool armed = false;
  std::string lcm_url = kDefaultLcmUrl;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--arm") {
      armed = true;
    } else if (argument == "--url" && index + 1 < argc) {
      lcm_url = argv[++index];
    } else if (argument == "--help") {
      std::cout << "Usage: " << argv[0] << " [--arm] [--url LCM_URL]\n";
      PrintHelp();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argument << std::endl;
      return 2;
    }
  }

  lcm::LCM lcm(lcm_url);
  if (!lcm.good()) {
    std::cerr << "Failed to open LCM URL: " << lcm_url << std::endl;
    return 1;
  }

  TaskStateListener listener;
  lcm.subscribe(kTaskStateChannel, &TaskStateListener::Handle, &listener);
  const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!listener.IsReady() && std::chrono::steady_clock::now() < ready_deadline) {
    lcm.handleTimeout(100);
  }
  if (!listener.IsReady()) {
    std::cerr << "No task_state received. The custom executor is not ready; refusing input.\n";
    return 3;
  }

  if (!armed) {
    std::cout << "Connected to state '" << listener.state()
              << "'. Dry run only; restart with --arm to publish controls.\n";
    return 0;
  }

  TerminalMode terminal_mode;
  PrintHelp();
  std::cout << "[armed] current state: " << listener.state() << std::endl;
  while (true) {
    lcm.handleTimeout(0);
    char input = 0;
    if (!ReadCharacter(&input, 100)) continue;
    if (input == 'q') break;
    if (input == '?') {
      PrintHelp();
      continue;
    }

    const auto action_it = kActions.find(input);
    if (action_it == kActions.end()) continue;
    if (!listener.IsReady()) {
      std::cerr << "\nInput link stale; refusing action.\n";
      continue;
    }

    const Action& action = action_it->second;
    if (!action.allowed_states.count(listener.state())) {
      std::cerr << "\nRefused " << action.name << " from state " << listener.state() << std::endl;
      continue;
    }

    std::cout << "\n[send] " << action.name << std::endl;
    PublishAction(lcm, action);
  }

  auto release = MakeMessage({});
  lcm.publish(kGamepadChannel, &release);
  std::cout << "\nKeyboard control stopped.\n";
  return 0;
}
