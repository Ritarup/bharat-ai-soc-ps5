/*
 * Persistent Industrial Daemon for Llama 3.2 ExecuTorch Runner
 * Keeps model resident in RAM to eliminate per-inference loading lag.
 */

#include <executorch/examples/models/llama/runner/runner.h>
#include <gflags/gflags.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "llama2.pte", "Model serialized in flatbuffer format.");
DEFINE_string(data_paths, "", "Data files for the model.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");
DEFINE_double(temperature, 0.0f, "Temperature; 0.0 = greedy deterministic sampling.");
DEFINE_int32(seq_len, 256, "Total sequence length.");
DEFINE_int32(max_new_tokens, -1, "Total tokens to generate.");
DEFINE_int32(cpu_threads, -1, "Number of CPU threads.");
DEFINE_int32(num_bos, 0, "Number of BOS tokens.");
DEFINE_int32(num_eos, 0, "Number of EOS tokens.");
DEFINE_bool(ignore_eos, false, "Ignore EOS token.");
DEFINE_string(method_name, "forward", "Method name.");

std::vector<std::string> parseStringList(const std::string& input) {
  std::vector<std::string> result;
  if (input.empty()) return result;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t first = item.find_first_not_of(" \t");
    if (first == std::string::npos) continue;
    size_t last = item.find_last_not_of(" \t");
    item = item.substr(first, last - first + 1);
    if (!item.empty()) result.push_back(item);
  }
  return result;
}

bool readFileToString(const std::string& path, std::string& out) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) return false;
  std::ostringstream ss;
  ss << file.rdbuf();
  out = ss.str();
  return true;
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();
  std::vector<std::string> data_paths = parseStringList(FLAGS_data_paths);
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  float temperature = FLAGS_temperature;
  int32_t seq_len = FLAGS_seq_len;
  int32_t cpu_threads = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::cerr << "[DAEMON_INIT] Loading model: " << model_path << "..." << std::endl;
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
      example::create_llama_runner(
          model_path,
          tokenizer_path,
          data_paths,
          temperature,
          nullptr,
          FLAGS_method_name);

  if (runner == nullptr) {
    std::cerr << "[DAEMON_ERROR] Failed to create llama runner" << std::endl;
    return 1;
  }

  executorch::extension::llm::GenerationConfig config{};
  config.temperature = temperature;
  config.ignore_eos = FLAGS_ignore_eos;
  config.num_bos = FLAGS_num_bos;
  config.num_eos = FLAGS_num_eos;
  if (FLAGS_max_new_tokens != -1) {
    config.max_new_tokens = FLAGS_max_new_tokens;
  } else {
    config.seq_len = seq_len;
  }

  std::cout << "[DAEMON_READY]" << std::endl;
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    size_t first = line.find_first_not_of(" \r\n\t");
    if (first == std::string::npos) continue;
    size_t last = line.find_last_not_of(" \r\n\t");
    line = line.substr(first, last - first + 1);
    if (line.empty()) continue;

    if (line == "PING") {
      std::cout << "PONG" << std::endl;
      std::cout.flush();
      continue;
    }
    if (line == "QUIT" || line == "EXIT") {
      break;
    }

    std::string prompt;
    if (line.rfind("FILE:", 0) == 0) {
      std::string p_path = line.substr(5);
      if (!readFileToString(p_path, prompt)) {
        std::cerr << "[DAEMON_ERROR] Failed to read prompt file: " << p_path << std::endl;
        std::cout << "[END_GEN]" << std::endl;
        std::cout.flush();
        continue;
      }
    } else {
      prompt = line;
    }

    std::cout << "[START_GEN]" << std::endl;
    std::cout.flush();

    runner->reset();
    auto token_cb = [&](const std::string& piece) {
      if (piece.find("<|eot_id|>") != std::string::npos ||
          piece.find("<|end_of_text|>") != std::string::npos) {
        runner->stop();
      }
    };
    auto error = runner->generate(prompt.c_str(), config, token_cb);
    if (error != executorch::runtime::Error::Ok) {
      std::cerr << "[DAEMON_ERROR] runner->generate failed" << std::endl;
    }

    std::cout << std::endl << "[END_GEN]" << std::endl;
    std::cout.flush();
  }

  std::cerr << "[DAEMON_EXIT] Shutting down." << std::endl;
  return 0;
}
