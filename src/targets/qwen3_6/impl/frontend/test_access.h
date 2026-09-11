#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

namespace ninfer::targets::qwen3_6 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled               = true,
                                                   std::uint32_t vision_max_tokens   = 8192,
                                                   std::uint64_t max_attention_pairs = 128ULL << 20,
                                                   std::uint32_t max_media_items     = 16);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::targets::qwen3_6
