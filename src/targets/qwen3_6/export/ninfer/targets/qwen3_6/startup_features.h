#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision                     = false;
    std::uint32_t vision_max_tokens = 8192;
    // Aggregate preprocessing work across all images/videos in one request.
    std::uint64_t vision_max_attention_pairs = 128ULL << 20;
    std::uint32_t vision_max_media_items     = 16;
    SpeculativeBackend speculative           = SpeculativeBackend::None;
    ProposalHead proposal_head               = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision            = options.enable_vision,
        .vision_max_tokens = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .vision_max_attention_pairs = options.vision_max_attention_pairs,
        .vision_max_media_items     = options.vision_max_media_items,
        .speculative                = options.speculative.backend,
        .proposal_head              = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6
