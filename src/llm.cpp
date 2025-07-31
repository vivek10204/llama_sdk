#include "api.h"
#include "llama.h"
#include "llama_cpp/vendor/nlohmann/json.hpp"
#include "params.hpp"
#include <cassert>
#include <vector>
#include <atomic>
#include <mutex>


#include <iostream> // For std::cerr
#include <sys/stat.h> // For stat()
#include <errno.h>    // For errno
#include <filesystem> // For std::filesystem (C++17)


static std::atomic_bool stop_generation(false);
static std::mutex continue_mutex;

static llama_model * model = nullptr;
static llama_context * ctx = nullptr;
static llama_sampler * smpl = nullptr;
static int prev_len = 0;

std::vector<llama_chat_message> llama_parse_messages(char * messages) {
    auto json_messages = json::parse(messages);
    std::vector<llama_chat_message> result;

    for (auto & message : json_messages) {
        auto role = message["role"].get<std::string>();
        auto content = message["content"].get<std::string>();

        llama_chat_message msg = {
            strdup(role.c_str()),
            strdup(content.c_str())
        };

        result.push_back(msg);
    }

    return result;
}

char * llama_default_params(void) {
    json params = json::object();

    /// Model parameters
    auto default_model_params = llama_model_default_params();

    params["vocab_only"] = default_model_params.vocab_only;
    params["use_mmap"] = default_model_params.use_mmap;
    params["use_mlock"] = default_model_params.use_mlock;
    params["check_tensors"] = default_model_params.check_tensors;

    /// Context parameters
    auto default_context_params = llama_context_default_params();

    params["n_ctx"] = default_context_params.n_ctx;
    params["n_batch"] = default_context_params.n_batch;
    params["n_ubatch"] = default_context_params.n_ubatch;
    params["n_seq_max"] = default_context_params.n_seq_max;
    params["n_threads"] = default_context_params.n_threads;
    params["n_threads_batch"] = default_context_params.n_threads_batch;
    params["rope_scaling_type"] = default_context_params.rope_scaling_type;
    params["pooling_type"] = default_context_params.pooling_type;
    params["attention_type"] = default_context_params.attention_type;
    params["rope_freq_base"] = default_context_params.rope_freq_base;
    params["rope_freq_scale"] = default_context_params.rope_freq_scale;
    params["yarn_ext_factor"] = default_context_params.yarn_ext_factor;
    params["yarn_attn_factor"] = default_context_params.yarn_attn_factor;
    params["yarn_beta_fast"] = default_context_params.yarn_beta_fast;
    params["yarn_beta_slow"] = default_context_params.yarn_beta_slow;
    params["yarn_orig_ctx"] = default_context_params.yarn_orig_ctx;
    params["defrag_thold"] = default_context_params.defrag_thold;
    params["embeddings"] = default_context_params.embeddings;
    params["offload_kqv"] = default_context_params.offload_kqv;
    params["flash_attn"] = default_context_params.flash_attn;
    params["no_perf"] = default_context_params.no_perf;

    /// Sampler parameters
    params["greedy"] = true;

    return strdup(params.dump().c_str());
}

/*int llama_llm_init(char * params) {
    auto json_params = json::parse(params);

    if (!json_params.contains("model_path") || !json_params["model_path"].is_string()) {
        fprintf(stderr, "Missing 'model_path' in parameters\n");
        return 1;
    }

    auto model_path = json_params["model_path"].get<std::string>();

    auto model_params = llama_model_params_from_json(json_params);
    auto context_params = llama_context_params_from_json(json_params);

    ggml_backend_load_all();

    model = llama_model_load_from_file(model_path.c_str(), model_params);
    ctx = llama_init_from_model(model, context_params);
    smpl = llama_sampler_from_json(model, json_params);

    prev_len = 0;

    return 0;
}*/

int llama_llm_init(char * params_json_str) { // Use a different parameter name to avoid confusion
    auto json_params = json::parse(params_json_str);

    if (!json_params.contains("model_path") || !json_params["model_path"].is_string()) {
        std::cerr << "ERROR (C++): Missing 'model_path' in parameters\n";
        return 1;
    }

    std::string s_model_path = json_params["model_path"].get<std::string>();

    std::cerr << "DEBUG (C++): Received model_path: " << s_model_path << std::endl;

    // Use std::filesystem for robust path handling and check
    std::filesystem::path fs_model_path(s_model_path);

    std::error_code ec; // For error handling without throwing

    // Attempt to canonicalize the path first. This resolves symlinks, ".." etc.
    // It requires the path to exist.
    std::filesystem::path canonical_path = std::filesystem::canonical(fs_model_path, ec);

    if (ec) {
        std::cerr << "ERROR (C++): Failed to canonicalize path '" << s_model_path << "': " << ec.message() << std::endl;
        // If canonicalization fails (e.g., path doesn't exist yet, or permission issue),
        // fall back to getting an absolute path.
        canonical_path = std::filesystem::absolute(fs_model_path, ec);
        if (ec) {
            std::cerr << "ERROR (C++): Also failed to get absolute path: " << ec.message() << std::endl;
            return 1; // Return error if path cannot be resolved.
        }
    }
    
    std::cerr << "DEBUG (C++): Final model path being used (absolute/canonical): " << canonical_path.string() << std::endl;

    // Verify the file exists and is a regular file using std::filesystem
    if (!std::filesystem::exists(canonical_path, ec)) {
        std::cerr << "ERROR (C++): Model file does NOT exist at path: " << canonical_path.string() << " (Error: " << ec.message() << ")" << std::endl;
        return 1;
    }
    if (!std::filesystem::is_regular_file(canonical_path, ec)) {
        std::cerr << "ERROR (C++): Path is not a regular file: " << canonical_path.string() << " (Error: " << ec.message() << ")" << std::endl;
        return 1;
    }

    // Use the canonicalized or absolute path for llama_load_model_from_file
    const char* final_model_path_c_str = canonical_path.string().c_str();

    auto model_params = llama_model_params_from_json(json_params);
    auto context_params = llama_context_params_from_json(json_params);

    ggml_backend_load_all();

    std::cerr << "DEBUG (C++): Calling llama_model_load_from_file with path: " << final_model_path_c_str << std::endl;
    model = llama_model_load_from_file(final_model_path_c_str, model_params);

    if (model == nullptr) {
        std::cerr << "ERROR (C++): llama_model_load_from_file returned nullptr for: " << final_model_path_c_str << std::endl;
        return 1;
    }

    std::cerr << "DEBUG (C++): Model loaded successfully. Initializing context." << std::endl;

    ctx = llama_init_from_model(model, context_params);
    
    if (ctx == nullptr) {
        std::cerr << "ERROR (C++): llama_init_from_model returned nullptr." << std::endl;
        llama_free_model(model);
        model = nullptr;
        return 1;
    }

    smpl = llama_sampler_from_json(model, json_params);

    prev_len = 0;

    std::cerr << "DEBUG (C++): LLM initialization successful." << std::endl;

    return 0;
}


int llama_prompt(char * msgs, dart_output * output) {
    auto messages = llama_parse_messages(msgs);

    std::lock_guard<std::mutex> lock(continue_mutex);
    stop_generation.store(false);

    assert(model != nullptr);
    assert(ctx != nullptr);
    assert(smpl != nullptr);

    auto vocab = llama_model_get_vocab(model);

    std::vector<char> formatted(llama_n_ctx(ctx));

    const char * tmpl = llama_model_chat_template(model, nullptr);
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int) formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }

    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return 1;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

    std::string response;

    const bool is_first = llama_kv_self_seq_pos_max(ctx, 0) == 0;

    // tokenize the prompt
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (!stop_generation.load()) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            break;
        }
        
        if (llama_decode(ctx, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }

        std::string piece(buf, n);
        output(piece.c_str());
        response += piece;

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    // add the response to the messages
    messages.push_back({"assistant", strdup(response.c_str())});
    prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return 1;
    }
    
    output(nullptr);
    return 0;
}

void llama_llm_stop(void) {
    stop_generation.store(true);
}

void llama_llm_free(void) {
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
}
