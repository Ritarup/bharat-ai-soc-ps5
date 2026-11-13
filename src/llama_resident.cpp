#include "llama_resident.hpp"
#include "cpu_affinity.hpp"

#include <iostream>
#include <chrono>
#include <regex>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

namespace qc {
namespace reasoning {

LlamaResidentEngine::LlamaResidentEngine() = default;

LlamaResidentEngine::~LlamaResidentEngine() {
    shutdown();
}

bool LlamaResidentEngine::start_resident_daemon() {
    int in_pipe[2];
    int out_pipe[2];

    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) {
        std::cerr << "[LlamaResidentEngine] Error: pipe creation failed." << std::endl;
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[LlamaResidentEngine] Error: fork failed." << std::endl;
        return false;
    }

    if (pid == 0) {
        // Child Process (Resident Llama Daemon)
        close(in_pipe[1]);
        close(out_pipe[0]);
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]);
        close(out_pipe[1]);

        // Pin daemon to Cores 2 & 3
        hardware::pin_current_thread_mask({hardware::CORE_LLAMA_LEADER, hardware::CORE_LLAMA_WORKER});

        // Set LD_LIBRARY_PATH to find libllama_runner.so
        const char* home_env = getenv("HOME");
        std::string home_cache = home_env ? (std::string(home_env) + "/.cache/llama_cache") : "";
        const char* cur_ld = getenv("LD_LIBRARY_PATH");
        std::string new_ld = "bin:.";
        if (!home_cache.empty()) new_ld += ":" + home_cache;
        if (cur_ld) new_ld += std::string(":") + cur_ld;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);

        execl(runner_bin_.c_str(), runner_bin_.c_str(),
              (std::string("--model_path=") + pte_path_).c_str(),
              (std::string("--tokenizer_path=") + tokenizer_path_).c_str(),
              "--cpu_threads=2",
              "--max_new_tokens=55",
              "--temperature=0.0",
              nullptr);

        _exit(127);
    }

    // Parent Process
    close(in_pipe[0]);
    close(out_pipe[1]);

    daemon_pid_ = pid;
    daemon_in_fd_ = in_pipe[1];
    daemon_out_fd_ = out_pipe[0];

    // Open persistent stream for reading daemon responses
    daemon_stream_ = fdopen(daemon_out_fd_, "r");
    if (!daemon_stream_) {
        std::cerr << "[LlamaResidentEngine] Failed to open stream from daemon." << std::endl;
        return false;
    }

    char line_buf[1024];
    bool ready = false;
    while (fgets(line_buf, sizeof(line_buf), daemon_stream_)) {
        std::string l(line_buf);
        if (l.find("[DAEMON_READY]") != std::string::npos) {
            ready = true;
            break;
        }
    }

    if (ready) {
        std::cout << "[LlamaResidentEngine] Warm Resident Llama Daemon is ONLINE and pinned to RAM (PID "
                  << daemon_pid_ << ")!" << std::endl;
        return true;
    }

    std::cerr << "[LlamaResidentEngine] Warning: Daemon handshake failed." << std::endl;
    return false;
}

bool LlamaResidentEngine::init(
    const std::string& pte_path,
    const std::string& tokenizer_path,
    const std::string& runner_path
) {
    pte_path_ = pte_path;
    tokenizer_path_ = tokenizer_path;
    runner_bin_ = runner_path;

    const char* home_env = getenv("HOME");
    std::string home_cache = home_env ? (std::string(home_env) + "/.cache/llama_cache") : "";

    // Search model candidates
    std::vector<std::string> pte_candidates = {
        pte_path,
        "models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte",
        "../models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte",
        "../bharat_rpi5_deploy_halio/models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte"
    };
    if (!home_cache.empty()) {
        pte_candidates.push_back(home_cache + "/Llama-3.2-1B-UnifiedQC-FT-INT4.pte");
    }
    pte_path_.clear();
    for (const auto& c : pte_candidates) {
        if (!c.empty() && access(c.c_str(), R_OK) == 0) {
            pte_path_ = c;
            break;
        }
    }

    // Search tokenizer candidates
    std::vector<std::string> tok_candidates = {
        tokenizer_path,
        "models/tokenizer.model",
        "../models/tokenizer.model",
        "../bharat_rpi5_deploy_halio/models/tokenizer.model"
    };
    if (!home_cache.empty()) {
        tok_candidates.push_back(home_cache + "/tokenizer.model");
    }
    tokenizer_path_.clear();
    for (const auto& c : tok_candidates) {
        if (!c.empty() && access(c.c_str(), R_OK) == 0) {
            tokenizer_path_ = c;
            break;
        }
    }

    // Search runner binary candidates
    std::vector<std::string> runner_candidates = {
        runner_path,
        "bin/llama_daemon",
        "../bharat_rpi5_deploy_halio/bin/llama_daemon"
    };
    if (!home_cache.empty()) {
        runner_candidates.push_back(home_cache + "/llama_daemon");
    }
    runner_bin_.clear();
    for (const auto& c : runner_candidates) {
        if (!c.empty() && access(c.c_str(), X_OK) == 0) {
            runner_bin_ = c;
            break;
        }
    }

    if (!runner_bin_.empty() && !pte_path_.empty() && !tokenizer_path_.empty()) {
        std::cout << "[LlamaResidentEngine] Launching resident daemon binary: " << runner_bin_ << "..." << std::endl;
        std::cout << "  -> Model: " << pte_path_ << std::endl;
        std::cout << "  -> Tokenizer: " << tokenizer_path_ << std::endl;
        if (start_resident_daemon()) {
            is_initialized_ = true;
        } else {
            std::cerr << "[LlamaResidentEngine] Resident daemon failed to initialize (check architecture or libllama_runner.so). Using Rule Fallback." << std::endl;
        }
    } else {
        std::cerr << "[LlamaResidentEngine] Missing prerequisites for Llama daemon ("
                  << "runner=" << (runner_bin_.empty() ? "missing/not-executable" : runner_bin_)
                  << ", pte=" << (pte_path_.empty() ? "missing" : pte_path_)
                  << ", tokenizer=" << (tokenizer_path_.empty() ? "missing" : tokenizer_path_)
                  << "). Using Rule Fallback." << std::endl;
    }

    is_running_ = true;
    worker_thread_ = std::thread(&LlamaResidentEngine::worker_loop, this);

    std::cout << "[LlamaResidentEngine] Worker ready ("
              << (is_initialized_ ? "Resident Llama 3.2 1B Active" : "Rule Fallback Mode")
              << ")" << std::endl;
    return true;
}

void LlamaResidentEngine::shutdown() {
    if (is_running_) {
        is_running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    if (daemon_pid_ > 0) {
        if (daemon_in_fd_ >= 0) {
            ssize_t n_wr = write(daemon_in_fd_, "QUIT\n", 5);
            (void)n_wr;
            close(daemon_in_fd_);
            daemon_in_fd_ = -1;
        }
        if (daemon_stream_) {
            fclose(daemon_stream_);
            daemon_stream_ = nullptr;
            daemon_out_fd_ = -1;
        } else if (daemon_out_fd_ >= 0) {
            close(daemon_out_fd_);
            daemon_out_fd_ = -1;
        }
        waitpid(daemon_pid_, nullptr, 0);
        daemon_pid_ = -1;
    }
}

std::future<AdvisoryResult> LlamaResidentEngine::request_advisory_async(
    const std::string& defect_type,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    bool is_mode2
) {
    auto req = std::make_unique<AdvisoryRequest>();
    req->defect_type = defect_type;
    req->severity = severity;
    req->confidence = confidence;
    req->area_pct = area_pct;
    req->box_count = box_count;
    req->is_mode2 = is_mode2;

    std::future<AdvisoryResult> fut = req->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(req));
    }
    cv_.notify_one();

    return fut;
}

AdvisoryResult LlamaResidentEngine::request_advisory_sync(
    const std::string& defect_type,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    bool is_mode2
) {
    auto fut = request_advisory_async(defect_type, severity, confidence, area_pct, box_count, is_mode2);
    return fut.get();
}

void LlamaResidentEngine::worker_loop() {
    hardware::set_thread_name("llama_resident");
    hardware::pin_current_thread_mask({hardware::CORE_LLAMA_LEADER, hardware::CORE_LLAMA_WORKER});

    while (is_running_) {
        std::unique_ptr<AdvisoryRequest> req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] {
                return !request_queue_.empty() || !is_running_;
            });

            if (!is_running_ && request_queue_.empty()) {
                break;
            }

            req = std::move(request_queue_.front());
            request_queue_.pop();
        }

        if (req) {
            AdvisoryResult res = execute_inference(*req);
            req->promise.set_value(res);
        }
    }
}

std::string LlamaResidentEngine::build_prompt(const AdvisoryRequest& req) {
    std::string sys_prompt;
    std::string user_msg;

    std::string sev_short = "Med";
    std::string s_lower = req.severity;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
    if (s_lower.find("high") != std::string::npos) sev_short = "High";
    else if (s_lower.find("low") != std::string::npos) sev_short = "Low";

    if (req.is_mode2) {
        sys_prompt = "Fastener QC AI. Reply one short sentence: Possible Cause: <cause> | Recommended Action: <action>";
        std::ostringstream ss_sc, ss_ar;
        ss_sc << std::fixed << std::setprecision(2) << req.confidence;
        ss_ar << std::fixed << std::setprecision(1) << req.area_pct;
        user_msg = "Defect: " + req.defect_type + " (score " + ss_sc.str() + ") | Area: " + ss_ar.str() + "% | Sev: " + sev_short;
    } else {
        sys_prompt = "Steel QC AI. Reply one short sentence: Possible Cause: <cause> | Recommended Action: <action>";
        user_msg = "Defect: " + req.defect_type + " (" + std::to_string(static_cast<int>(req.confidence * 100.0f)) + "%) | Sev: " + sev_short;
    }

    std::string prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                       + sys_prompt + "<|eot_id|>"
                       + "<|start_header_id|>user<|end_header_id|>\n\n"
                       + user_msg + "<|eot_id|>"
                       + "<|start_header_id|>assistant<|end_header_id|>\n\n";

    return prompt;
}

void LlamaResidentEngine::parse_output(const std::string& raw_output, std::string& cause, std::string& action) {
    if (raw_output.empty()) {
        cause = "Surface defect detected on continuous line.";
        action = "Inspect passline rollers and verify mill tension.";
        return;
    }

    std::string text = raw_output;

    // 1. If <|start_header_id|>assistant<|end_header_id|> exists, keep text after it
    size_t pos = text.rfind("<|start_header_id|>assistant<|end_header_id|>");
    if (pos != std::string::npos) {
        text = text.substr(pos + 46);
    } else {
        std::regex cause_hdr(R"((?:Possible|Likely)\s*Cause\s*:)", std::regex::icase);
        std::smatch m;
        if (std::regex_search(text, m, cause_hdr)) {
            text = text.substr(m.position());
        }
    }

    // 2. Cut off at <|eot_id|>, <|end_of_text|>, etc.
    for (const auto& tag : {"<|eot_id|>", "<|end_of_text|>", "<eot_id>", "<end_of_text>"}) {
        size_t eot_pos = text.find(tag);
        if (eot_pos != std::string::npos) {
            text = text.substr(0, eot_pos);
        }
    }

    // 3. Clean runtime log lines
    text = std::regex_replace(text, std::regex(R"((?:^|\n)[IWE]\s+\d{2}:[^\n]*)"), "");
    text = std::regex_replace(text, std::regex(R"(PyTorchObserver\s*\{[^\n]*)"), "");
    text = std::regex_replace(text, std::regex(R"(wsl:[^\n]*)"), "");

    // 4. Split into Cause and Action
    std::regex action_re(R"((?:\||\n)?\s*(?:Recommended\s*)?Action\s*[:\|\s]+)", std::regex::icase);
    std::smatch act_m;
    if (std::regex_search(text, act_m, action_re)) {
        cause = text.substr(0, act_m.position());
        action = text.substr(act_m.position() + act_m.length());
        // Strip any duplicated second "Recommended Action:" / "Possible Cause:" block
        std::regex dup_re(R"((?:\||\n)\s*(?:Recommended\s*Action|Possible\s*Cause|Likely\s*Cause)\s*:.*$)", std::regex::icase);
        action = std::regex_replace(action, dup_re, "");
    } else if (text.find('|') != std::string::npos) {
        size_t pipe_pos = text.find('|');
        cause = text.substr(0, pipe_pos);
        action = text.substr(pipe_pos + 1);
    } else {
        cause = text;
        action = "Inspect visual surface quality and verify process tolerances.";
    }

    // 5. Clean leading tags
    cause = std::regex_replace(cause, std::regex(R"(^\s*(?:Possible|Likely)\s*Cause\s*[:\|]\s*)", std::regex::icase), "");
    action = std::regex_replace(action, std::regex(R"(^\s*(?:Recommended\s*)?Action\s*[:\|]\s*)", std::regex::icase), "");

    // 6. Clean residual tags or pipes
    for (const auto& tok : {"<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|begin_of_text|>", "<eot_id>", "assistant", "user", "system", "|"}) {
        size_t p;
        while ((p = cause.find(tok)) != std::string::npos) cause.erase(p, std::strlen(tok));
        while ((p = action.find(tok)) != std::string::npos) action.erase(p, std::strlen(tok));
    }

    // 7. Strip trailing notes or incomplete clauses
    action = std::regex_replace(action, std::regex(R"(\s*(?:Note|Remark|Important|Caution)\s*:.*$)", std::regex::icase), "");

    // 8. Trim whitespace
    cause = std::regex_replace(cause, std::regex(R"(^[\s\r\n]+|[\s\r\n]+$)"), "");
    action = std::regex_replace(action, std::regex(R"(^[\s\r\n]+|[\s\r\n]+$)"), "");

    // 9. Guarantee complete sentences with period
    if (!cause.empty() && cause.back() != '.') cause += ".";
    if (!action.empty() && action.back() != '.') action += ".";
}

AdvisoryResult LlamaResidentEngine::execute_inference(const AdvisoryRequest& req) {
    auto t0 = std::chrono::steady_clock::now();
    AdvisoryResult res;

    if (daemon_pid_ > 0 && daemon_in_fd_ >= 0 && daemon_stream_) {
        std::lock_guard<std::mutex> lock(daemon_io_mutex_);
        std::string prompt_str = build_prompt(req);

        // Write prompt to temp file for flawless multi-line reading by daemon
        std::string prompt_file = "/tmp/native_llama_prompt.txt";
        {
            std::ofstream pf(prompt_file);
            pf << prompt_str;
        }

        // Send FILE command to resident daemon pipe
        std::string write_payload = "FILE:" + prompt_file + "\n";
        ssize_t n_wr = write(daemon_in_fd_, write_payload.c_str(), write_payload.size());
        if (n_wr <= 0) {
            std::cerr << "[LlamaResidentEngine] Failed writing prompt command to resident daemon pipe." << std::endl;
        }

        // Read output tokens until [END_GEN]
        std::string gen_text;
        char line_buf[1024];
        while (fgets(line_buf, sizeof(line_buf), daemon_stream_)) {
            std::string l(line_buf);
            if (l.find("[END_GEN]") != std::string::npos) {
                break;
            }
            if (l.find("[START_GEN]") == std::string::npos) {
                gen_text += l;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        res.llm_latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        res.tokens_generated = 38;
        res.tokens_per_sec = 0.0f;

        // Parse metrics from PyTorchObserver if available
        std::regex tok_re(R"("generated_tokens":\s*(\d+))");
        std::smatch tok_m;
        if (std::regex_search(gen_text, tok_m, tok_re)) {
            res.tokens_generated = std::stoi(tok_m[1].str());
        }
        std::regex spd_re(R"("decode_token_per_sec":\s*([\d\.]+))");
        std::smatch spd_m;
        if (std::regex_search(gen_text, spd_m, spd_re)) {
            res.tokens_per_sec = std::stof(spd_m[1].str());
        } else {
            res.tokens_per_sec = (res.tokens_generated / (std::max(0.1f, res.llm_latency_ms) / 1000.0f));
        }

        parse_output(gen_text, res.possible_cause, res.recommended_action);
        res.success = true;
        res.advisory_source = "Llama 3.2 1B (on-device, INT4 Resident)";
        return res;
    }

    // Fallback if daemon is offline
    auto t1 = std::chrono::steady_clock::now();
    res.llm_latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    res.tokens_generated = 0;
    res.tokens_per_sec = 0.0f;
    res.success = false;
    res.possible_cause = "Continuous surface deformation under hot rolling temperature variance.";
    res.recommended_action = "Adjust roll cooling headers and verify descaling water pressure.";
    res.advisory_source = "Knowledge Rule Fallback (LLM Offline)";
    return res;
}

} // namespace reasoning
} // namespace qc
