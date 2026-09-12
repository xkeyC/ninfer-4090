#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures +=
        check(defaults.slot_save_path.empty(), "slot persistence is not disabled by default");
    failures += check(defaults.turn_checkpoint_ring == 0,
                      "turn checkpoint ring is not disabled by default");
    failures += check(defaults.host_prefix_cache_bytes == 0,
                      "host prefix cache is not disabled by default");
    failures += check(defaults.kv_affinity_burst == 5 && defaults.kv_affinity_grace_ms == 1500,
                      "KV affinity defaults mismatch");

    const ServeOptions media =
        parse({"ninfer-serve", "model.ninfer", "--vision-max-attention-pairs", "4294967296",
               "--vision-max-media-items", "128"});
    failures += check(media.vision_max_attention_pairs == (1ULL << 32) &&
                          media.vision_max_media_items == 128,
                      "custom aggregate vision budgets were not applied");
    for (const std::string flag : {"--vision-max-attention-pairs", "--vision-max-media-items"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", flag, "0"});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "zero aggregate vision budget was accepted");
    }

    const ServeOptions ring = parse({"ninfer-serve", "model.ninfer", "--turn-checkpoints", "8"});
    failures += check(ring.turn_checkpoint_ring == 8, "--turn-checkpoints was not applied");
    failures += check(!ring.auto_save_evicted, "auto-save-evicted is not disabled by default");

    const ServeOptions host_cache =
        parse({"ninfer-serve", "model.ninfer", "--host-prefix-cache-mib", "20480"});
    failures += check(host_cache.host_prefix_cache_bytes == (20480ULL << 20),
                      "--host-prefix-cache-mib was not applied");
    const ServeOptions affinity = parse({"ninfer-serve", "model.ninfer", "--kv-affinity-burst", "9",
                                         "--kv-affinity-grace-ms", "750"});
    failures += check(affinity.kv_affinity_burst == 9 && affinity.kv_affinity_grace_ms == 750,
                      "KV affinity options were not applied");

    const ServeOptions auto_save = parse(
        {"ninfer-serve", "model.ninfer", "--slot-save-path", "/tmp/slots", "--auto-save-evicted"});
    failures += check(auto_save.auto_save_evicted, "--auto-save-evicted was not applied");
    bool auto_save_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--auto-save-evicted"});
    } catch (const std::invalid_argument&) { auto_save_rejected = true; }
    failures +=
        check(auto_save_rejected, "--auto-save-evicted without --slot-save-path was not rejected");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions rotor = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk8v4"});
    failures += check(rotor.kv_cache == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                      "--kv-dtype rk8v4 did not select rotated K8/V4 storage");
    failures += check(defaults.kv_cache == ninfer::KvCacheStorage::BFloat16,
                      "rk8v4 unexpectedly changed the default KV storage");

    const ServeOptions k4e8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk4v4-e8"});
    failures += check(k4e8.kv_cache == ninfer::KvCacheStorage::RK4V4E8,
                      "--kv-dtype rk4v4-e8 did not select RK4V4E8 storage");

    const ServeOptions k2e8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk2v4-e8"});
    failures += check(k2e8.kv_cache == ninfer::KvCacheStorage::RK2V4E8,
                      "--kv-dtype rk2v4-e8 did not select RK2V4E8 storage");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15",
                     "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash and Vision were accepted together");
    bool dflash_host_cache_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15",
                     "--host-prefix-cache-mib", "1"});
    } catch (const std::invalid_argument&) { dflash_host_cache_rejected = true; }
    failures += check(dflash_host_cache_rejected,
                      "DFlash and the host prefix cache were accepted together");
    bool disabled_reuse_host_cache_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--host-prefix-cache-mib", "1", "--no-prefix-reuse"});
    } catch (const std::invalid_argument&) { disabled_reuse_host_cache_rejected = true; }
    failures += check(disabled_reuse_host_cache_rejected,
                      "host prefix cache was accepted with prefix reuse disabled");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse(
        {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--vision", "--max-concurrency", "4",
         "--max-pending-requests", "12", "--pending-timeout-ms", "2500", "--max-context", "4096",
         "--kv-capacity", "8192", "--log-stats-interval-ms", "0", "--preserve-thinking"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "40", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    failures += check(to_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!to_request_options(request, configured).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");
    const ninfer::RequestOptions inherited_sampling = to_request_options(request, sampling);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling).execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--host-prefix-cache-mib") != std::string::npos,
              "serve help omits --host-prefix-cache-mib");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");

    const ServeOptions slots =
        parse({"ninfer-serve", "model.ninfer", "--slot-save-path", "/var/lib/ninfer/slots"});
    failures += check(slots.slot_save_path == "/var/lib/ninfer/slots",
                      "--slot-save-path did not preserve its directory");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--slot-save-path") != std::string::npos,
              "serve help omits --slot-save-path");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
