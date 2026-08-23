#include "runtime/engine/host_prefix_cache.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Image {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
    std::vector<std::size_t> cache_block_sizes;
    std::size_t cache_metadata_bytes = 16;
    std::size_t device_transfer_bytes = 0;
};

struct FakeClock {
    using rep = std::int64_t;
    using period = std::ratio<1>;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<FakeClock>;
    static constexpr bool is_steady = true;

    static time_point now() noexcept { return current; }
    static inline time_point current{};
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

Image image(std::vector<std::uint8_t> bytes, std::uint32_t tokens, std::string digest,
            std::vector<std::size_t> blocks = {}) {
    Image out;
    out.bytes = std::move(bytes);
    out.tokens = tokens;
    out.session_digest = std::move(digest);
    out.cache_block_sizes = std::move(blocks);
    return out;
}

Image filled(std::size_t bytes, std::uint8_t value, std::uint32_t tokens, std::string digest) {
    return image(std::vector<std::uint8_t>(bytes, value), tokens, std::move(digest), {bytes});
}

} // namespace

int main() {
    using Cache = ninfer::runtime::HostPrefixCache<Image, FakeClock>;
    int failures = 0;

    Cache disabled;
    failures += check(!disabled.insert(filled(1, 1, 1, "off")).inserted,
                      "disabled cache accepted an entry");

    Cache shared(4096, FakeClock::duration(10));
    const auto first = shared.insert(image({1, 2, 3, 4}, 40, "a", {2, 2}));
    const auto second = shared.insert(image({1, 2, 5, 6}, 80, "b", {2, 2}));
    failures += check(first.inserted && second.inserted && shared.block_count() == 3,
                      "identical prefix blocks were not deduplicated");
    const auto deepest = shared.best_match([](const Image& held) { return held.tokens; });
    failures += check(deepest && deepest->reused_tokens == 80,
                      "cache did not select the deepest reusable prefix");
    if (deepest) {
        const auto restored = shared.view(deepest->id);
        failures += check(restored && restored->blocks.size() == 2 &&
                              restored->blocks[0][0] == 1 && restored->blocks[1][1] == 6,
                          "manifest view did not preserve immutable block order");
        failures += check(shared.pin(deepest->id) && shared.unpin(deepest->id),
                          "manifest pin lifecycle failed");
    }

    Cache delta_cache(4096, FakeClock::duration(10));
    const auto delta_base = delta_cache.insert(image({10, 11, 20, 21}, 40, "delta-base", {2, 2}));
    Image delta_image;
    delta_image.tokens = 80;
    delta_image.session_digest = "delta-child";
    const std::vector<Cache::BlockSource> delta_sources{
        {.base_block_index = 0, .bytes = 2},
        {.delta_offset = 0, .bytes = 2},
    };
    const auto delta_child = delta_cache.insert_delta(
        std::move(delta_image), delta_base.id, std::vector<std::uint8_t>{30, 31}, delta_sources);
    failures += check(delta_base.inserted && delta_child.inserted &&
                          delta_cache.block_count() == 3,
                      "delta insertion did not reuse the immutable base block");
    if (delta_child.inserted) {
        const auto view = delta_cache.view(delta_child.id, false);
        failures += check(view && view->blocks.size() == 2 && view->blocks[0][0] == 10 &&
                              view->blocks[1][1] == 31,
                          "block view did not expose the delta manifest without materialization");
    }

    Cache probe(4096, FakeClock::duration(10));
    (void)probe.insert(filled(100, 1, 10, "probe-a"));
    const std::size_t one_entry_capacity = probe.used_bytes();
    (void)probe.insert(filled(100, 2, 20, "probe-b"));
    const std::size_t two_entry_capacity = probe.used_bytes();

    Cache replacement(one_entry_capacity, FakeClock::duration(10));
    const auto replace_old = replacement.insert(filled(100, 7, 10, "replace-old"));
    const auto replace_new = replacement.insert(filled(100, 7, 20, "replace-new"));
    failures += check(replace_old.inserted && replace_new.inserted && replace_new.evicted == 1 &&
                          replacement.size() == 1,
                      "shared protected blocks prevented manifest replacement");

    FakeClock::current = FakeClock::time_point{};
    Cache ranked(two_entry_capacity, FakeClock::duration(10));
    const auto hot = ranked.insert(filled(100, 1, 10, "hot"));
    failures += check(hot.inserted && ranked.recall(hot.id) && ranked.recall(hot.id) &&
                          ranked.recall(hot.id),
                      "recall accounting failed");
    FakeClock::current += FakeClock::duration(5);
    const auto recent = ranked.insert(filled(100, 2, 20, "recent"));
    FakeClock::current += FakeClock::duration(1);
    const auto newcomer = ranked.insert(filled(100, 3, 30, "new"));
    failures += check(recent.inserted && newcomer.inserted && newcomer.evicted == 1,
                      "bounded cache did not evict under pressure");
    const auto kept_hot = ranked.best_match(
        [](const Image& held) { return held.session_digest == "hot" ? held.tokens : 0U; });
    const auto evicted_recent = ranked.best_match(
        [](const Image& held) { return held.session_digest == "recent" ? held.tokens : 0U; });
    failures += check(kept_hot && !evicted_recent,
                      "frequency bonus did not protect a repeatedly recalled older block");

    FakeClock::current += FakeClock::duration(100);
    const auto aged = ranked.insert(filled(100, 4, 40, "aged-replacement"));
    FakeClock::current += FakeClock::duration(1);
    const auto aged_second = ranked.insert(filled(100, 5, 50, "aged-replacement-2"));
    const auto old_hot = ranked.best_match(
        [](const Image& held) { return held.session_digest == "hot" ? held.tokens : 0U; });
    failures += check(aged.inserted && aged_second.inserted && !old_hot,
                      "time aging did not eventually expire a historical hot block");

    Cache too_small(32, FakeClock::duration(10));
    failures += check(!too_small.insert(filled(100, 9, 90, "too-large")).inserted &&
                          too_small.used_bytes() == 0,
                      "oversized manifest changed cache occupancy");

    return failures == 0 ? 0 : 1;
}
