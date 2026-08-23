#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/host_prefix_cache.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;
    using HostCache = HostPrefixCache<targets::qwen3_6::RetainedSessionSnapshot>;
    using HostEntryId = typename HostCache::EntryId;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          auto_save_evicted_(options.auto_save_evicted),
          host_prefix_cache_(options.host_prefix_cache_bytes),
          kv_affinity_burst_(options.kv_affinity_burst),
          kv_affinity_grace_(std::chrono::milliseconds(options.kv_affinity_grace_ms)),
          admission_capacity_(instance.program->admission_capacity()) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        if (host_prefix_cache_.enabled() &&
            options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument("host prefix cache does not support the DFlash backend");
        }
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}, HostInputLease host_input = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                pending_deadline, submitted, std::move(host_input));
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

    // Session persistence entry points. Each claims the execution mutex, so GPU copies land at
    // a request boundary; the worker resumes as soon as the device round trip completes. A lane
    // with an active request is refused rather than drained. A non-empty expected_digest is a
    // precondition on the lane's resident session, checked atomically with the operation.
    // session_path is the file the operation targets; a successful save or restore binds the
    // lane to it so an involuntary eviction can spill the session back (see
    // spill_retained_lane).
    [[nodiscard]] targets::qwen3_6::RetainedSessionSnapshot
    save_retained_lane(std::uint32_t lane, std::string_view model_binding,
                       std::string_view expected_digest, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        auto snapshot = instance_.program->save_retained_lane(lane, model_binding);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        return snapshot;
    }

    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_retained_lane(std::uint32_t lane, std::span<const std::uint8_t> snapshot,
                          std::string_view model_binding, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        if (instance_.program->has_retained_lane(lane)) {
            // Involuntary for whatever session held the lane: the client asked for a restore,
            // not for that session's destruction.
            spill_retained_lane(lane);
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
        }
        release_lane_host_entry(lane);
        lane_session_path_[lane].clear();
        const std::uint32_t tokens =
            instance_.program->restore_retained_lane(lane, snapshot, model_binding);
        invalidate_lane_plans(lane);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        retained_digest_cache_[lane]      = instance_.program->retained_lane_digest(lane);
        retained_checkpoints_cache_[lane] = instance_.program->retained_lane_checkpoints(lane);
        publish_runtime_stats();
        return {tokens, retained_digest_cache_[lane]};
    }

    std::uint32_t erase_retained_lane(std::uint32_t lane, std::string_view expected_digest) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        const std::uint32_t tokens = instance_.program->retained_lane_depth(lane);
        // Explicit erase is a deletion request: never auto-save, and drop the binding.
        lane_session_path_[lane].clear();
        release_lane_host_entry(lane);
        if (instance_.program->has_retained_lane(lane)) {
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
            publish_runtime_stats();
        }
        return tokens;
    }

    // Installs the auto-save sink: the model binding save_retained_lane needs, and a consumer
    // that receives (path, snapshot) for each spilled session and writes the file off-thread.
    void set_eviction_sink(
        std::string model_binding,
        std::function<void(std::string, targets::qwen3_6::RetainedSessionSnapshot&&)> sink) {
        std::scoped_lock lock(execution_mutex_);
        eviction_model_binding_ = std::move(model_binding);
        eviction_sink_          = std::move(sink);
    }

    void set_host_prefix_cache_model_binding(std::string model_binding) {
        std::scoped_lock lock(execution_mutex_);
        host_prefix_cache_model_binding_ = std::move(model_binding);
    }

    // Truthful per-lane occupancy: an active request's prompt size, or the retained session's
    // depth and identifying digest. Served from the snapshot the worker publishes at every unit
    // boundary - the execution mutex is held nearly continuously while a request runs, so a
    // scraper that waited on it would starve for the length of a deep prefill.
    [[nodiscard]] std::vector<SlotState> slot_states() const {
        std::lock_guard lock(stats_mutex_);
        std::vector<SlotState> states = published_slots_;
        states.resize(max_concurrency_);
        return states;
    }

private:
    void require_idle_lane(std::uint32_t lane) const {
        if (lane >= max_concurrency_) {
            throw std::invalid_argument("slot id is outside the Engine lane count");
        }
        if (slots_[lane] != nullptr) {
            throw RequestError(RequestErrorKind::Overloaded, "slot is processing a request");
        }
    }

    void require_session_digest(std::uint32_t lane, std::string_view expected_digest) const {
        if (expected_digest.empty()) { return; }
        if (instance_.program->retained_lane_digest(lane) != expected_digest) {
            throw SlotSessionMismatch("slot session does not match if_digest");
        }
    }

    void release_lane_host_entry(std::uint32_t lane) noexcept {
        if (lane >= kMaximumConcurrency || !lane_host_cache_entry_[lane]) { return; }
        (void)host_prefix_cache_.unpin(*lane_host_cache_entry_[lane]);
        lane_host_cache_entry_[lane].reset();
    }

    // Best-effort preservation of a retained session about to be destroyed involuntarily. One
    // device snapshot feeds the automatic host victim cache and, when configured and file-bound,
    // the background disk writer. A preservation failure never blocks the eviction itself.
    void spill_retained_lane(std::uint32_t lane) noexcept {
        if (lane >= kMaximumConcurrency || !instance_.program->has_retained_lane(lane)) { return; }
        const bool save_host =
            host_prefix_cache_.enabled() && !host_prefix_cache_model_binding_.empty();
        const bool save_disk =
            auto_save_evicted_ && eviction_sink_ && !lane_session_path_[lane].empty();
        if (!save_host && !save_disk) { return; }
        const auto capture_started = Clock::now();
        bool capture_time_recorded = false;
        try {
            const std::string_view binding =
                save_host ? std::string_view(host_prefix_cache_model_binding_)
                          : std::string_view(eviction_model_binding_);
            const std::optional<HostEntryId> base_entry_id = lane_host_cache_entry_[lane];
            std::optional<typename HostCache::View> base_view;
            if (save_host && base_entry_id) {
                base_view = host_prefix_cache_.view(*base_entry_id, false);
            }
            targets::qwen3_6::RetainedSessionCacheView target_base;
            if (base_view) {
                target_base.manifest = base_view->image;
                target_base.blocks   = base_view->blocks;
            }
            auto snapshot = save_host
                                ? instance_.program->capture_retained_lane_cache(lane, binding,
                                                                                 target_base)
                                : instance_.program->save_retained_lane(lane, binding);
            release_lane_host_entry(lane);
            if (save_host) {
                cumulative_stats_.host_prefix_cache_capture_bytes +=
                    snapshot.device_transfer_bytes;
                cumulative_stats_.host_prefix_cache_capture_seconds +=
                    std::chrono::duration<double>(Clock::now() - capture_started).count();
                capture_time_recorded = true;
            }
            if (save_disk) {
                // The ordinary deployment uses either sink. Keeping both configured is valid;
                // the writer receives its own immutable image while the cache takes the original.
                auto disk_snapshot =
                    save_host ? snapshot : targets::qwen3_6::RetainedSessionSnapshot{};
                if (save_host) {
                    std::size_t total = 0;
                    for (const auto& block : disk_snapshot.cache_blocks) {
                        total += block.bytes;
                    }
                    disk_snapshot.bytes.reserve(total);
                    disk_snapshot.cache_block_sizes.reserve(disk_snapshot.cache_blocks.size());
                    for (const auto& block : disk_snapshot.cache_blocks) {
                        if (block.base_block_index !=
                            targets::qwen3_6::RetainedSessionSnapshot::CacheBlock::kNew) {
                            if (!base_view || block.base_block_index >= base_view->blocks.size()) {
                                throw std::logic_error("cache delta references an invalid base block");
                            }
                            const auto bytes = base_view->blocks[block.base_block_index];
                            disk_snapshot.bytes.insert(disk_snapshot.bytes.end(), bytes.begin(),
                                                       bytes.end());
                        } else {
                            const auto begin = disk_snapshot.cache_delta_bytes.begin() +
                                               static_cast<std::ptrdiff_t>(block.delta_offset);
                            disk_snapshot.bytes.insert(
                                disk_snapshot.bytes.end(), begin,
                                begin + static_cast<std::ptrdiff_t>(block.bytes));
                        }
                        disk_snapshot.cache_block_sizes.push_back(block.bytes);
                    }
                    disk_snapshot.cache_delta_bytes.clear();
                    disk_snapshot.cache_blocks.clear();
                    eviction_sink_(lane_session_path_[lane], std::move(disk_snapshot));
                } else {
                    eviction_sink_(lane_session_path_[lane], std::move(snapshot));
                }
            }
            if (save_host) {
                std::vector<typename HostCache::BlockSource> blocks;
                blocks.reserve(snapshot.cache_blocks.size());
                for (const auto& block : snapshot.cache_blocks) {
                    blocks.push_back(typename HostCache::BlockSource{
                        .base_block_index = block.base_block_index,
                        .delta_offset     = block.delta_offset,
                        .bytes            = block.bytes,
                    });
                }
                auto delta = std::move(snapshot.cache_delta_bytes);
                snapshot.cache_blocks.clear();
                snapshot.cache_blocks.shrink_to_fit();
                const auto inserted = host_prefix_cache_.insert_delta(
                    std::move(snapshot), base_entry_id, std::move(delta), blocks);
                cumulative_stats_.host_prefix_cache_evictions += inserted.evicted;
                if (inserted.inserted) {
                    ++cumulative_stats_.host_prefix_cache_captures;
                } else {
                    ++cumulative_stats_.host_prefix_cache_drops;
                }
            }
        } catch (...) {
            // The session was going to be destroyed either way; losing the spill costs the
            // client one cold prefill, exactly the pre-feature behavior.
            if (save_host) {
                ++cumulative_stats_.host_prefix_cache_capture_failures;
                if (!capture_time_recorded) {
                    cumulative_stats_.host_prefix_cache_capture_seconds +=
                        std::chrono::duration<double>(Clock::now() - capture_started).count();
                }
            }
        }
        // spill_retained_lane is only called immediately before eviction or destructive branch
        // reuse. The old manifest remains in the immutable cache but is no longer lane-pinned.
        release_lane_host_entry(lane);
    }

    void publish_runtime_stats() {
        RuntimeStats snapshot              = cumulative_stats_;
        snapshot.host_prefix_cache_entries = static_cast<std::uint32_t>(host_prefix_cache_.size());
        snapshot.host_prefix_cache_blocks =
            static_cast<std::uint32_t>(host_prefix_cache_.block_count());
        snapshot.host_prefix_cache_bytes   = host_prefix_cache_.used_bytes();
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }

        // Per-lane occupancy for /slots-style readers. Digests come from the cache the
        // completion and restore paths maintain, so publishing costs no ledger hashing.
        std::vector<SlotState> slot_snapshot(max_concurrency_);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            SlotState& state    = slot_snapshot[lane];
            const auto& request = slots_[lane];
            if (request != nullptr) {
                state.processing    = true;
                state.prompt_tokens = request->prompt_summary.prompt_tokens;
                if (request->begin) { state.cached_tokens = request->begin->reused_prompt_tokens; }
            } else if (instance_.program->has_retained_lane(lane)) {
                state.retained       = true;
                state.prompt_tokens  = instance_.program->retained_lane_depth(lane);
                state.cached_tokens  = state.prompt_tokens;
                state.session_digest = retained_digest_cache_[lane];
                state.checkpoints    = retained_checkpoints_cache_[lane];
            }
        }

        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
        published_slots_ = std::move(slot_snapshot);
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time, HostInputLease input_lease)
            : id(request_identity), host_input(std::move(input_lease)), prompt(std::move(input)),
              output(std::move(output_session)), prompt_summary(summary),
              prepare_seconds(frontend_seconds), options(std::move(request_options)),
              deadline(limit), submitted(submit_time) {}

        const std::uint64_t id;
        HostInputLease host_input;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> admission_started;
        double queue_seconds        = 0.0;
        double host_restore_seconds = 0.0;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;
        bool affinity_contended = false;
        bool affinity_continuation = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
        AffinityWait,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
        std::optional<HostEntryId> host_cache_entry;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
            result.slot        = static_cast<std::int32_t>(*request->lane);
            // Empty unless the lane retained the finished session (aborts and cancels clear it).
            result.session_digest = instance_.program->retained_lane_digest(*request->lane);
            // Completion and restore are the only paths that make a lane retained, so keeping
            // the cache here means publish_runtime_stats never has to hash a ledger.
            retained_digest_cache_[*request->lane] = result.session_digest;
            retained_checkpoints_cache_[*request->lane] =
                instance_.program->retained_lane_checkpoints(*request->lane);
            if (!result.session_digest.empty() && kv_affinity_burst_ != 0 &&
                host_prefix_cache_.enabled()) {
                if (!affinity_lane_ || *affinity_lane_ != *request->lane) {
                    affinity_lane_ = *request->lane;
                    affinity_burst_used_ = 0;
                }
                affinity_grace_deadline_ = Clock::now() + kv_affinity_grace_;
            }
        }
        result.timings.queue_seconds        = request->queue_seconds;
        result.timings.host_restore_seconds = request->host_restore_seconds;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        if (!instance_.program->has_retained_lane(lane)) { release_lane_host_entry(lane); }
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        consume_service_work(request, 1);
        if (step.host_input_consumed || step.complete) { request->host_input.reset(); }
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            instance_.program->abort_lane(lane);
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const auto unit_started      = Clock::now();
        const PrefillStepResult step = instance_.program->advance_prefill_lane(lane);
        cumulative_stats_.prefill_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    [[nodiscard]] bool matches_resident_affinity(const std::shared_ptr<Request>& request) {
        if (!affinity_lane_ || *affinity_lane_ >= max_concurrency_ ||
            slots_[*affinity_lane_] != nullptr ||
            !instance_.program->has_retained_lane(*affinity_lane_)) {
            return false;
        }
        ensure_base_plan(request);
        ensure_lane_plan(request, *affinity_lane_);
        return request->lane_plans[*affinity_lane_]->summary().reusable_prompt_tokens != 0;
    }

    // Reorder only the local admission snapshot; the bounded FIFO remains the ownership queue.
    // During contention, a resident continuation may bypass older cold work up to the configured
    // burst. Once the burst is spent, the oldest competitor is promoted and starts a new epoch.
    // If the continuation has not arrived yet, a short grace window avoids paying a multi-second
    // owner switch for the sub-second frontend gap between sequential tool turns.
    [[nodiscard]] bool apply_affinity_order(std::vector<std::shared_ptr<Request>>& queued) {
        for (const auto& request : queued) {
            request->affinity_contended = false;
            request->affinity_continuation = false;
        }
        if (kv_affinity_burst_ == 0 || !host_prefix_cache_.enabled() || queued.empty() ||
            !affinity_lane_) {
            return false;
        }
        if (*affinity_lane_ >= max_concurrency_ || slots_[*affinity_lane_] != nullptr ||
            !instance_.program->has_retained_lane(*affinity_lane_)) {
            affinity_lane_.reset();
            affinity_burst_used_ = 0;
            affinity_grace_deadline_.reset();
            return false;
        }

        std::vector<bool> matching(queued.size(), false);
        bool have_match = false;
        bool have_competitor = false;
        for (std::size_t index = 0; index < queued.size(); ++index) {
            try {
                matching[index] = matches_resident_affinity(queued[index]);
            } catch (...) {
                matching[index] = false;
            }
            have_match = have_match || matching[index];
            have_competitor = have_competitor || !matching[index];
        }
        if (!have_competitor) { return false; }

        std::size_t selected = queued.size();
        bool continuation = false;
        if (affinity_burst_used_ < kv_affinity_burst_ && have_match) {
            selected = static_cast<std::size_t>(
                std::find(matching.begin(), matching.end(), true) - matching.begin());
            continuation = true;
        } else if (affinity_burst_used_ >= kv_affinity_burst_) {
            selected = static_cast<std::size_t>(
                std::find(matching.begin(), matching.end(), false) - matching.begin());
        } else if (affinity_grace_deadline_ && Clock::now() < *affinity_grace_deadline_) {
            return true;
        }

        if (selected == queued.size()) { selected = 0; }
        if (selected != 0) {
            std::rotate(queued.begin(), queued.begin() + static_cast<std::ptrdiff_t>(selected),
                        queued.begin() + static_cast<std::ptrdiff_t>(selected + 1));
        }
        queued.front()->affinity_contended = true;
        queued.front()->affinity_continuation = continuation;
        return false;
    }

    // Lane choice maximizes reusable prefix; ties break toward the lane whose occupation costs
    // least to replace - an empty lane before any retained session, then the shallowest
    // retained session - so a fresh request never clobbers a deep resident session while a
    // cheaper lane is available.
    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        std::uint32_t selected_cost  = 0;
        const auto prefer            = [&](std::uint32_t reuse, std::uint32_t cost) {
            return !selected || reuse > selected_reuse ||
                   (reuse == selected_reuse && cost < selected_cost);
        };
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const std::uint32_t cost  = instance_.program->retained_lane_depth(lane);
            if (instance_.program->can_admit_lane(lane, plan) && prefer(reuse, cost)) {
                selected       = LaneChoice{.lane = lane};
                selected_reuse = reuse;
                selected_cost  = cost;
            }
        }
        if (!selected) {
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                ensure_lane_plan(request, lane);
                const Plan& plan          = *request->lane_plans[lane];
                const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
                const std::uint32_t cost  = instance_.program->retained_lane_depth(lane);
                if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                    prefer(reuse, cost)) {
                    selected = LaneChoice{
                        .lane           = lane,
                        .evict_retained = true,
                    };
                    selected_reuse = reuse;
                    selected_cost  = cost;
                }
            }
        }

        const auto host_match = host_prefix_cache_.best_match(
            [&](const targets::qwen3_6::RetainedSessionSnapshot& snapshot) {
                return instance_.program->reusable_snapshot_prefix(
                    snapshot, request->prompt, request->options.execution.allow_prefix_reuse);
            });
        if (host_match && host_match->reused_tokens > selected_reuse) {
            std::optional<LaneChoice> host_choice;
            std::uint32_t host_cost = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                ensure_lane_plan(request, lane);
                const std::uint32_t cost = instance_.program->retained_lane_depth(lane);
                if (instance_.program->can_admit_lane_after_retained_eviction(
                        lane, *request->lane_plans[lane]) &&
                    (!host_choice || cost < host_cost)) {
                    host_choice = LaneChoice{
                        .lane             = lane,
                        .evict_retained   = true,
                        .host_cache_entry = host_match->id,
                    };
                    host_cost = cost;
                }
            }
            if (host_choice) { return host_choice; }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        if (!request->admission_started) {
            request->admission_started = Clock::now();
            request->queue_seconds =
                std::chrono::duration<double>(*request->admission_started - request->submitted)
                    .count();
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        // Pin the immutable block view before preserving other victims. Restore consumes the
        // spans directly; the manifest stays matchable and a later capture reuses unchanged
        // full pages without assembling or re-hashing a contiguous snapshot.
        std::optional<typename HostCache::View> host_snapshot;
        if (choice.host_cache_entry) {
            if (!host_prefix_cache_.pin(*choice.host_cache_entry)) {
                throw std::logic_error("selected host prefix cache entry disappeared");
            }
            host_snapshot = host_prefix_cache_.view(*choice.host_cache_entry);
            if (!host_snapshot) {
                (void)host_prefix_cache_.unpin(*choice.host_cache_entry);
                throw std::logic_error("selected host prefix cache entry disappeared");
            }
        }
        if (choice.evict_retained) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    spill_retained_lane(retained_lane);
                    lane_session_path_[retained_lane].clear();
                    instance_.program->evict_retained_lane(retained_lane);
                    invalidate_lane_plans(retained_lane);
                }
            }
            if (!choice.host_cache_entry &&
                !instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        if (choice.host_cache_entry) {
            if (instance_.program->has_retained_lane(lane)) {
                spill_retained_lane(lane);
                lane_session_path_[lane].clear();
                instance_.program->evict_retained_lane(lane);
                invalidate_lane_plans(lane);
            }
            std::size_t restore_bytes = 0;
            for (const auto block : host_snapshot->blocks) { restore_bytes += block.size(); }
            const auto restore_started      = Clock::now();
            try {
                (void)instance_.program->restore_retained_lane_cache(
                    lane,
                    targets::qwen3_6::RetainedSessionCacheView{
                        .manifest = host_snapshot->image,
                        .blocks   = host_snapshot->blocks,
                    },
                    host_prefix_cache_model_binding_);
                invalidate_lane_plans(lane);
                request->lane_plans[lane].reset();
                ensure_lane_plan(request, lane);
                if (request->lane_plans[lane]->summary().reusable_prompt_tokens == 0 ||
                    !instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                    throw std::logic_error(
                        "restored host prefix did not produce an admissible hit");
                }
                ++cumulative_stats_.host_prefix_cache_hits;
                lane_host_cache_entry_[lane] = *choice.host_cache_entry;
                cumulative_stats_.host_prefix_cache_restore_bytes += restore_bytes;
                const double restore_seconds =
                    std::chrono::duration<double>(Clock::now() - restore_started).count();
                request->host_restore_seconds += restore_seconds;
                cumulative_stats_.host_prefix_cache_restore_seconds += restore_seconds;
            } catch (...) {
                ++cumulative_stats_.host_prefix_cache_restore_failures;
                cumulative_stats_.host_prefix_cache_restore_seconds +=
                    std::chrono::duration<double>(Clock::now() - restore_started).count();
                if (instance_.program->has_retained_lane(lane)) {
                    instance_.program->evict_retained_lane(lane);
                    invalidate_lane_plans(lane);
                }
                (void)host_prefix_cache_.unpin(*choice.host_cache_entry);
                lane_host_cache_entry_[lane].reset();
                throw;
            }
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (request->affinity_contended) {
            if (request->affinity_continuation) {
                if (affinity_burst_used_ != std::numeric_limits<std::uint32_t>::max()) {
                    ++affinity_burst_used_;
                }
            } else {
                affinity_lane_.reset();
                affinity_burst_used_ = 0;
            }
            affinity_grace_deadline_.reset();
        }
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        clear_protection_if_head(request);

        if (!choice.host_cache_entry && lane_host_cache_entry_[lane] &&
            summary.reusable_prompt_tokens != 0) {
            (void)host_prefix_cache_.recall(*lane_host_cache_entry_[lane]);
        }

        // Full reset and checkpoint restore both destroy the lane's current continuation. Preserve
        // that branch before the target truncates in place; otherwise two sessions which share an
        // older turn checkpoint repeatedly re-prefill the entire divergent suffix without ever
        // crossing the ordinary eviction sink.
        const bool destructive_reuse =
            summary.prefix_reuse_path == PrefixReusePath::FullReset ||
            summary.prefix_reuse_path == PrefixReusePath::RestoreTurnCheckpoint;
        if (destructive_reuse) { spill_retained_lane(lane); }
        // A full reset starts an unrelated session, so it must not inherit a slot-file binding.
        if (summary.prefix_reuse_path == PrefixReusePath::FullReset) {
            lane_session_path_[lane].clear();
        }

        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            target_started                = true;
            const auto unit_started       = Clock::now();
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient);
            cumulative_stats_.prefill_seconds_total +=
                std::chrono::duration<double>(Clock::now() - unit_started).count();
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            if (!instance_.program->has_retained_lane(lane)) { release_lane_host_entry(lane); }
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            if (apply_affinity_order(queued)) { return AdmissionProgress::AffinityWait; }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        const auto unit_started                    = Clock::now();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());
        cumulative_stats_.decode_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count)) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published));
            if (terminal[row]) {
                complete_success(request, finish_reasons[row]);
                remove_completed_slot(lane);
            }
        }
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        publish_runtime_stats();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
            release_lane_host_entry(lane);
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::unique_lock execution_lock(execution_mutex_);
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                if (prefill_lane_) {
                    if (!membership.empty() && !previous_unit_was_decode) {
                        run_decode_round(membership);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                    if (progress == AdmissionProgress::AffinityWait && membership.empty()) {
                        const auto deadline = affinity_grace_deadline_.value_or(Clock::now());
                        execution_lock.unlock();
                        std::unique_lock queue_lock(queue_mutex_);
                        queue_cv_.wait_until(queue_lock, deadline);
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
            } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const bool auto_save_evicted_;
    HostCache host_prefix_cache_;
    const std::uint32_t kv_affinity_burst_;
    const std::chrono::milliseconds kv_affinity_grace_;
    const AdmissionResources admission_capacity_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    std::optional<std::uint32_t> affinity_lane_;
    std::uint32_t affinity_burst_used_ = 0;
    std::optional<Clock::time_point> affinity_grace_deadline_;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    std::vector<SlotState> published_slots_;
    // Digest of each lane's retained session, maintained by the completion and restore paths
    // (the only ones that set `retained`) so publishing needs no ledger hashing.
    std::array<std::string, kMaximumConcurrency> retained_digest_cache_{};
    std::array<std::vector<SlotCheckpoint>, kMaximumConcurrency> retained_checkpoints_cache_{};
    // Slot file each lane's resident session was last saved to or restored from; empty means
    // unbound. Guarded by execution_mutex_ like the lane state it describes.
    std::array<std::string, kMaximumConcurrency> lane_session_path_{};
    // Immutable host manifest supplying the resident lane's unchanged prefix blocks. It remains
    // pinned until the lane is destroyed or destructively rewound.
    std::array<std::optional<HostEntryId>, kMaximumConcurrency> lane_host_cache_entry_{};
    std::string eviction_model_binding_;
    std::string host_prefix_cache_model_binding_;
    std::function<void(std::string, targets::qwen3_6::RetainedSessionSnapshot&&)> eviction_sink_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
