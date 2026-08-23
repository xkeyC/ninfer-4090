#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// One retained lane's complete session image (host bytes) for save/restore persistence. The
// byte layout is a target-private format; callers treat it as opaque and durable only across
// processes serving the identical model and KV configuration.
struct RetainedSessionSnapshot {
    struct CacheBlock {
        static constexpr std::size_t kNew = std::numeric_limits<std::size_t>::max();

        // kNew means [delta_offset, delta_offset + bytes) is a newly captured block.
        // Otherwise the block is an immutable reference to that index in the supplied base view.
        std::size_t base_block_index = kNew;
        std::size_t delta_offset     = 0;
        std::size_t bytes            = 0;
    };

    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;

    // Host-cache-only framing. `cache_block_sizes` partitions `bytes` into immutable deduplication
    // units; the target emits one unit for metadata/state and one per 64-token KV page group.
    // `device_transfer_bytes` is the actual D2H payload for this capture, which can be smaller
    // than the durable image when unchanged full KV pages came from a previous manifest.
    std::vector<std::size_t> cache_block_sizes;
    // Host-cache capture form. `cache_blocks` describes the complete logical image while
    // `cache_delta_bytes` owns only blocks that changed since the base manifest. Durable slot
    // serialization continues to use `bytes`; the online cache never materializes it.
    std::vector<std::uint8_t> cache_delta_bytes;
    std::vector<CacheBlock> cache_blocks;
    std::size_t cache_metadata_bytes = 0;
    std::size_t device_transfer_bytes = 0;

    // Host-only reuse metadata. It is deliberately outside the durable byte format: disk restore
    // validates and rebuilds the same state, while the in-memory victim cache uses this image to
    // reject non-matching prompts before paying an H2D restore.
    std::uint32_t execution_frontier       = 0;
    std::uint32_t mtp_kv_valid             = 0;
    std::uint32_t turn_checkpoint_frontier = 0;
    bool tail_hidden_valid                 = false;
    std::vector<std::uint32_t> checkpoint_frontiers;
    std::vector<TokenId> ledger;
    std::vector<std::uint8_t> token_types;
    std::array<std::vector<std::int32_t>, 3> positions;
    std::vector<VisionItem> vision_items;
};

struct RetainedSessionCacheView {
    const RetainedSessionSnapshot* manifest = nullptr;
    std::span<const std::span<const std::uint8_t>> blocks;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] std::uint32_t retained_lane_depth(std::uint32_t lane) const noexcept;
    // Stable identifier (FNV-1a 64 hex) of the lane's resident token ledger; empty unless the
    // lane holds a retained session.
    [[nodiscard]] std::string retained_lane_digest(std::uint32_t lane) const;
    // Retained turn checkpoints of the lane's resident session, oldest first: the frontiers a
    // diverging prompt can restore from, each with the digest of the ledger prefix it covers.
    [[nodiscard]] std::vector<SlotCheckpoint> retained_lane_checkpoints(std::uint32_t lane) const;
    // Session persistence for one idle retained lane. `model_binding` pins the snapshot to the
    // serving weights identity; restore rejects a mismatched binding or configuration. Both
    // synchronize the device before returning and require the lane to hold no active request.
    [[nodiscard]] RetainedSessionSnapshot
    save_retained_lane(std::uint32_t lane, std::string_view model_binding);
    [[nodiscard]] RetainedSessionSnapshot
    capture_retained_lane_cache(std::uint32_t lane, std::string_view model_binding,
                                RetainedSessionCacheView base = {});
    [[nodiscard]] std::uint32_t reusable_snapshot_prefix(const RetainedSessionSnapshot& snapshot,
                                                         const PreparedPrompt& prompt,
                                                         bool allow_prefix_reuse) const;
    [[nodiscard]] std::uint32_t restore_retained_lane(std::uint32_t lane,
                                                      std::span<const std::uint8_t> snapshot,
                                                      std::string_view model_binding);
    [[nodiscard]] std::uint32_t restore_retained_lane_cache(std::uint32_t lane,
                                                            RetainedSessionCacheView snapshot,
                                                            std::string_view model_binding);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
