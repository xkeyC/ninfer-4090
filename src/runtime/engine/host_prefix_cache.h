#pragma once

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Byte-bounded immutable block store for target-owned retained-sequence images. Image supplies
// cache_block_sizes which partition its durable bytes. Manifests keep the host-only prefix
// identity used by the matcher, while identical blocks are stored once across session branches.
//
// Recall heat belongs to blocks. Eviction selects the coldest unpinned block by recency plus a
// logarithmic frequency bonus, then removes every dependent unpinned manifest transactionally.
// A manifest therefore never survives with a missing middle block.
template <class Image, class Clock = std::chrono::steady_clock>
class HostPrefixCache {
public:
    using EntryId = std::uint64_t;

    struct Match {
        EntryId id                  = 0;
        std::uint32_t reused_tokens = 0;
    };

    struct InsertResult {
        bool inserted              = false;
        EntryId id                 = 0;
        std::size_t evicted        = 0;
        std::size_t evicted_bytes  = 0;
        std::size_t inserted_bytes = 0;
    };

    static constexpr std::size_t kNewBlock = std::numeric_limits<std::size_t>::max();

    // One logical block in a delta image. A reused block names its index in `base`; a new block
    // names a byte range in the supplied delta payload. The cache resolves reuse directly to its
    // immutable BlockId, so unchanged pages are neither copied nor hashed again.
    struct BlockSource {
        std::size_t base_block_index = kNewBlock;
        std::size_t delta_offset     = 0;
        std::size_t bytes            = 0;
    };

    struct View {
        const Image* image = nullptr;
        std::vector<std::span<const std::uint8_t>> blocks;
    };

    explicit HostPrefixCache(
        std::size_t capacity_bytes = 0,
        typename Clock::duration frequency_window = std::chrono::minutes(5)) noexcept
        : capacity_bytes_(capacity_bytes), frequency_window_(frequency_window) {}

    [[nodiscard]] bool enabled() const noexcept { return capacity_bytes_ != 0; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }

    InsertResult insert(Image image) {
        if (!enabled() || image.bytes.empty()) { return {}; }

        std::vector<std::uint8_t> source = std::move(image.bytes);
        std::vector<std::size_t> block_sizes = std::move(image.cache_block_sizes);
        if (block_sizes.empty()) { block_sizes.push_back(source.size()); }
        std::vector<BlockSource> blocks;
        blocks.reserve(block_sizes.size());
        std::size_t represented = 0;
        for (const std::size_t bytes : block_sizes) {
            if (bytes == 0 || represented > source.size() ||
                bytes > source.size() - represented) {
                return {};
            }
            blocks.push_back(BlockSource{.delta_offset = represented, .bytes = bytes});
            represented += bytes;
        }
        if (represented != source.size()) { return {}; }
        return insert_delta(std::move(image), std::nullopt, std::move(source), blocks);
    }

    InsertResult insert_delta(Image image, std::optional<EntryId> base,
                              std::vector<std::uint8_t> delta,
                              std::span<const BlockSource> sources) {
        InsertResult result;
        if (!enabled() || sources.empty()) { return result; }

        const Entry* base_entry = base ? find_entry(*base) : nullptr;
        if (base && base_entry == nullptr) { return result; }
        for (const BlockSource& source : sources) {
            if (source.bytes == 0) { return result; }
            if (source.base_block_index != kNewBlock) {
                if (base_entry == nullptr || source.base_block_index >= base_entry->blocks.size()) {
                    return result;
                }
            } else if (source.delta_offset > delta.size() ||
                       source.bytes > delta.size() - source.delta_offset) {
                return result;
            }
        }

        Entry entry;
        entry.id             = next_entry_id_++;
        entry.metadata_bytes = image.cache_metadata_bytes +
                               sources.size() * sizeof(BlockId) + sizeof(Entry);
        entry.image          = std::move(image);
        entry.blocks.reserve(sources.size());

        const auto now = Clock::now();
        for (const BlockSource& source : sources) {
            if (source.base_block_index != kNewBlock) {
                const BlockId id = base_entry->blocks[source.base_block_index];
                ++blocks_.at(id).references;
                entry.blocks.push_back(id);
            } else {
                const std::span<const std::uint8_t> payload(delta.data() + source.delta_offset,
                                                            source.bytes);
                entry.blocks.push_back(intern(payload, now, result.inserted_bytes));
            }
        }
        entry.unique_blocks = entry.blocks;
        std::sort(entry.unique_blocks.begin(), entry.unique_blocks.end());
        entry.unique_blocks.erase(
            std::unique(entry.unique_blocks.begin(), entry.unique_blocks.end()),
            entry.unique_blocks.end());
        used_bytes_ += entry.metadata_bytes;
        result.inserted_bytes += entry.metadata_bytes;
        const EntryId protected_entry = entry.id;
        entries_.push_front(std::move(entry));
        for (const BlockId id : entries_.front().unique_blocks) {
            blocks_.at(id).dependents.push_back(protected_entry);
        }

        std::size_t required = entries_.front().metadata_bytes;
        for (const BlockId id : entries_.front().unique_blocks) {
            required += blocks_.at(id).bytes.size();
        }
        if (required > capacity_bytes_) {
            (void)erase_entry(protected_entry);
            return result;
        }

        while (used_bytes_ > capacity_bytes_) {
            const auto coldest = coldest_evictable_block(protected_entry);
            if (!coldest) {
                (void)erase_entry(protected_entry);
                return result;
            }
            result.evicted_bytes +=
                evict_block_dependents(*coldest, protected_entry, result.evicted);
        }
        result.inserted = true;
        result.id       = protected_entry;
        return result;
    }

    [[nodiscard]] std::optional<View> view(EntryId id, bool count_recall = true) {
        Entry* entry = find_entry(id);
        if (entry == nullptr) { return std::nullopt; }
        if (count_recall) { recall_entry(*entry); }
        View out;
        out.image = &entry->image;
        out.blocks.reserve(entry->blocks.size());
        for (const BlockId id : entry->blocks) {
            const Block& block = blocks_.at(id);
            out.blocks.emplace_back(block.bytes.data(), block.bytes.size());
        }
        return out;
    }

    template <class Matcher>
    [[nodiscard]] std::optional<Match> best_match(Matcher&& matcher) const {
        std::optional<Match> best;
        for (const Entry& entry : entries_) {
            const std::uint32_t reused = matcher(entry.image);
            if (reused != 0 && (!best || reused > best->reused_tokens)) {
                best = Match{entry.id, reused};
            }
        }
        return best;
    }

    [[nodiscard]] bool recall(EntryId id) {
        Entry* entry = find_entry(id);
        if (entry == nullptr) { return false; }
        recall_entry(*entry);
        return true;
    }

    [[nodiscard]] bool pin(EntryId id) {
        Entry* entry = find_entry(id);
        if (entry == nullptr) { return false; }
        if (entry->pin_count++ != 0) { return true; }
        for_each_unique_block(*entry, [&](Block& block) { ++block.pin_count; });
        return true;
    }

    [[nodiscard]] bool unpin(EntryId id) noexcept {
        Entry* entry = find_entry(id);
        if (entry == nullptr || entry->pin_count == 0) { return false; }
        if (--entry->pin_count != 0) { return true; }
        try {
            for_each_unique_block(*entry, [](Block& block) {
                if (block.pin_count != 0) { --block.pin_count; }
            });
        } catch (...) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool contains(EntryId id) const noexcept { return find_entry(id) != nullptr; }

private:
    using BlockId = std::uint64_t;

    struct Block {
        BlockId id              = 0;
        std::uint64_t hash      = 0;
        std::vector<std::uint8_t> bytes;
        std::uint32_t references = 0;
        std::uint32_t pin_count  = 0;
        std::uint64_t recall_count = 0;
        typename Clock::time_point last_recall{};
        std::vector<EntryId> dependents;
    };

    struct Entry {
        EntryId id                 = 0;
        std::size_t metadata_bytes = 0;
        std::uint32_t pin_count    = 0;
        Image image;
        std::vector<BlockId> blocks;
        std::vector<BlockId> unique_blocks;
    };

    static std::uint64_t hash_bytes(std::span<const std::uint8_t> bytes) noexcept {
        std::uint64_t hash = 0x9e3779b97f4a7c15ULL ^ bytes.size();
        std::size_t offset = 0;
        while (bytes.size() - offset >= sizeof(std::uint64_t)) {
            std::uint64_t word = 0;
            std::memcpy(&word, bytes.data() + offset, sizeof(word));
            word ^= word >> 30;
            word *= 0xbf58476d1ce4e5b9ULL;
            word ^= word >> 27;
            word *= 0x94d049bb133111ebULL;
            word ^= word >> 31;
            hash ^= word + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
            offset += sizeof(word);
        }
        std::uint64_t tail = 0;
        if (offset != bytes.size()) {
            std::memcpy(&tail, bytes.data() + offset, bytes.size() - offset);
            hash ^= tail + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        return hash;
    }

    [[nodiscard]] BlockId find_identical(std::uint64_t hash,
                                         std::span<const std::uint8_t> bytes) const {
        const auto [first, last] = hash_index_.equal_range(hash);
        for (auto it = first; it != last; ++it) {
            const Block& held = blocks_.at(it->second);
            if (held.bytes.size() == bytes.size() &&
                std::memcmp(held.bytes.data(), bytes.data(), bytes.size()) == 0) {
                return held.id;
            }
        }
        return 0;
    }

    BlockId intern(std::span<const std::uint8_t> bytes, typename Clock::time_point now,
                   std::size_t& inserted_bytes) {
        const std::uint64_t hash = hash_bytes(bytes);
        if (const BlockId existing = find_identical(hash, bytes); existing != 0) {
            ++blocks_.at(existing).references;
            return existing;
        }
        Block block;
        block.id          = next_block_id_++;
        block.hash        = hash;
        block.bytes.assign(bytes.begin(), bytes.end());
        block.references  = 1;
        block.last_recall = now;
        const BlockId id  = block.id;
        inserted_bytes += block.bytes.size();
        used_bytes_ += block.bytes.size();
        hash_index_.emplace(hash, id);
        blocks_.emplace(id, std::move(block));
        return id;
    }

    void erase_hash_index(const Block& block) noexcept {
        const auto [first, last] = hash_index_.equal_range(block.hash);
        for (auto it = first; it != last; ++it) {
            if (it->second == block.id) {
                hash_index_.erase(it);
                return;
            }
        }
    }

    void release_block_reference(BlockId id) noexcept {
        auto it = blocks_.find(id);
        if (it == blocks_.end() || it->second.references == 0) { return; }
        Block& block = it->second;
        if (--block.references != 0) { return; }
        used_bytes_ -= block.bytes.size();
        erase_hash_index(block);
        blocks_.erase(it);
    }

    std::size_t erase_entry(EntryId id) noexcept {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id != id || it->pin_count != 0) { continue; }
            const std::size_t before = used_bytes_;
            used_bytes_ -= it->metadata_bytes;
            for (const BlockId block_id : it->unique_blocks) {
                auto block = blocks_.find(block_id);
                if (block == blocks_.end()) { continue; }
                auto& dependents = block->second.dependents;
                dependents.erase(std::remove(dependents.begin(), dependents.end(), id),
                                 dependents.end());
            }
            for (const BlockId block : it->blocks) { release_block_reference(block); }
            entries_.erase(it);
            return before - used_bytes_;
        }
        return 0;
    }

    [[nodiscard]] typename Clock::time_point effective_recall(const Block& block) const noexcept {
        using TimePoint = typename Clock::time_point;
        const std::uint64_t frequency =
            block.recall_count == std::numeric_limits<std::uint64_t>::max()
                ? block.recall_count
                : block.recall_count + 1U;
        const unsigned levels = std::bit_width(frequency) - 1U;
        if (levels == 0) { return block.last_recall; }
        const auto maximum = TimePoint::max() - block.last_recall;
        if (frequency_window_ > maximum / levels) { return TimePoint::max(); }
        return block.last_recall + frequency_window_ * levels;
    }

    [[nodiscard]] std::optional<BlockId> coldest_evictable_block(EntryId protected_entry) const {
        std::optional<BlockId> selected;
        typename Clock::time_point selected_heat{};
        for (const auto& [id, block] : blocks_) {
            bool has_evictable_dependent = false;
            for (const EntryId dependent : block.dependents) {
                const Entry* entry = find_entry(dependent);
                if (entry != nullptr && dependent != protected_entry && entry->pin_count == 0) {
                    has_evictable_dependent = true;
                    break;
                }
            }
            if (!has_evictable_dependent) { continue; }
            const auto heat = effective_recall(block);
            if (!selected || heat < selected_heat ||
                (heat == selected_heat && block.recall_count < blocks_.at(*selected).recall_count)) {
                selected = id;
                selected_heat = heat;
            }
        }
        return selected;
    }

    std::size_t evict_block_dependents(BlockId block, EntryId protected_entry,
                                       std::size_t& evicted_entries) noexcept {
        std::vector<EntryId> victims;
        const auto held = blocks_.find(block);
        if (held == blocks_.end()) { return 0; }
        for (const EntryId dependent : held->second.dependents) {
            const Entry* entry = find_entry(dependent);
            if (entry != nullptr && entry->id != protected_entry && entry->pin_count == 0) {
                victims.push_back(entry->id);
            }
        }
        std::size_t bytes = 0;
        for (const EntryId victim : victims) {
            const std::size_t released = erase_entry(victim);
            if (released != 0) {
                bytes += released;
                ++evicted_entries;
            }
        }
        return bytes;
    }

    void recall_entry(Entry& entry) {
        const auto now = Clock::now();
        for_each_unique_block(entry, [&](Block& block) {
            block.last_recall = now;
            if (block.recall_count != std::numeric_limits<std::uint64_t>::max()) {
                ++block.recall_count;
            }
        });
    }

    template <class Fn>
    void for_each_unique_block(Entry& entry, Fn&& fn) {
        for (const BlockId id : entry.unique_blocks) { fn(blocks_.at(id)); }
    }

    Entry* find_entry(EntryId id) noexcept {
        for (Entry& entry : entries_) {
            if (entry.id == id) { return &entry; }
        }
        return nullptr;
    }

    const Entry* find_entry(EntryId id) const noexcept {
        for (const Entry& entry : entries_) {
            if (entry.id == id) { return &entry; }
        }
        return nullptr;
    }

    std::size_t capacity_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    typename Clock::duration frequency_window_;
    EntryId next_entry_id_ = 1;
    BlockId next_block_id_ = 1;
    std::list<Entry> entries_;
    std::unordered_map<BlockId, Block> blocks_;
    std::unordered_multimap<std::uint64_t, BlockId> hash_index_;
};

} // namespace ninfer::runtime
