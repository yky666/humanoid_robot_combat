#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "app/app.h"
#include "glog/logging.h"

#include "client/crashpad_client.h"

namespace {

bool InitializeCrashpad() {
  std::filesystem::path coredump_dir = std::filesystem::temp_directory_path() / "crashpad" / "coredump";
  try {
    std::filesystem::create_directories(coredump_dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(ERROR) << "Failed to create coredump directory: " << e.what();
    return false;
  }

  base::FilePath db = base::FilePath(coredump_dir.c_str());
  base::FilePath metrics = db;
  const char* third_party_env = std::getenv("ENGINEAI_ROBOTICS_THIRD_PARTY");
  std::filesystem::path third_party =
      third_party_env != nullptr && third_party_env[0] != '\0' ? third_party_env : "/opt/engineai_robotics_third_party";
  std::filesystem::path handler_path = third_party / "bin" / "handler";
  if (!std::filesystem::is_regular_file(handler_path)) {
    LOG(ERROR) << "Crashpad handler not found: " << handler_path;
    return false;
  }
  base::FilePath handler = base::FilePath(handler_path.c_str());
  std::string url = "";

  std::map<std::string, std::string> annotations;
  annotations["prod"] = "engineai_robotics";
  annotations["ver"] = "1.0.0";

  std::vector<std::string> arguments;
  // upload crash report immediately
  arguments.push_back("--no-rate-limit");

  crashpad::CrashpadClient client;
  return client.StartHandler(handler, db, metrics, url, annotations, arguments, true, false);
}

}  // namespace

int main(int argc, char* argv[]) {
  InitializeCrashpad();
  engineai_robotics::App app;

  try {
    app.RegisterApps();
    app.RunMain(argc, argv);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Catch Exception: " << e.what();
    app.GracefulExit(SIGINT);
  }

  return 0;
}
