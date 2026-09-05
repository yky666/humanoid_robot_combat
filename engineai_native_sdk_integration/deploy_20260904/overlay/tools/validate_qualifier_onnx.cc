#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> Run(const std::string& model_path, const std::vector<float>& observations, bool has_time_input) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "t800_policy_validation");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  Ort::Session session(env, model_path.c_str(), options);
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 2> obs_shape{1, static_cast<int64_t>(observations.size())};
  std::vector<float> mutable_obs = observations;
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, mutable_obs.data(), mutable_obs.size(), obs_shape.data(), 2));
  std::vector<const char*> input_names{"obs"};

  std::array<float, 1> time_step{17.0f};
  std::array<int64_t, 2> time_shape{1, 1};
  if (has_time_input) {
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, time_step.data(), 1, time_shape.data(), 2));
    input_names.push_back("time_step");
  }

  const std::array<const char*, 1> output_names{"actions"};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                             output_names.data(), output_names.size());
  const auto count = outputs.at(0).GetTensorTypeAndShapeInfo().GetElementCount();
  const float* values = outputs.at(0).GetTensorData<float>();
  return {values, values + count};
}

void WriteValues(const std::string& path, const std::vector<float>& values) {
  std::ofstream output(path);
  output << std::setprecision(10);
  for (float value : values) output << value << '\n';
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: validate_qualifier_onnx ORIGINAL ACTOR INPUT_TXT EXPECTED_TXT\n";
    return 2;
  }
  try {
    std::vector<float> observations(140);
    for (size_t i = 0; i < observations.size(); ++i) observations[i] = 0.5f * std::sin(0.17f * i);
    const auto original = Run(argv[1], observations, true);
    const auto actor = Run(argv[2], observations, false);
    if (original.size() != 25 || actor.size() != 25) throw std::runtime_error("action output must contain 25 values");
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
      if (!std::isfinite(original[i]) || !std::isfinite(actor[i])) throw std::runtime_error("non-finite action");
      max_abs_error = std::max(max_abs_error, std::abs(original[i] - actor[i]));
    }
    if (max_abs_error > 1e-7f) throw std::runtime_error("actor extraction changed policy actions");
    WriteValues(argv[3], observations);
    WriteValues(argv[4], actor);
    std::cout << "ONNX original/actor max_abs_error=" << max_abs_error << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
