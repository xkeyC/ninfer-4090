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
        InsertResult result;
        if (!enabled() || image.bytes.empty()) { return result; }

        std::vector<std::uint8_t> source = std::move(image.bytes);
        std::vector<std::size_t> block_sizes = std::move(image.cache_block_sizes);
        if (block_sizes.empty()) { block_sizes.push_back(source.size()); }
        std::size_t represented = 0;
        for (const std::size_t bytes : block_sizes) {
            if (bytes == 0 || represented > source.size() ||
                bytes > source.size() - represented) {
                return result;
            }
            represented += bytes;
        }
        if (represented != source.size()) { return result; }

        Entry entry;
        entry.id             = next_entry_id_++;
        entry.metadata_bytes = image.cache_metadata_bytes +
                               block_sizes.size() * sizeof(BlockId) + sizeof(Entry);
        entry.image          = std::move(image);
        entry.blocks.reserve(block_sizes.size());

        const auto now = Clock::now();
        std::size_t offset = 0;
        for (const std::size_t bytes : block_sizes) {
            const std::span<const std::uint8_t> payload(source.data() + offset, bytes);
            entry.blocks.push_back(intern(payload, now, result.inserted_bytes));
            offset += bytes;
        }
        used_bytes_ += entry.metadata_bytes;
        result.inserted_bytes += entry.metadata_bytes;
        const EntryId protected_entry = entry.id;
        entries_.push_front(std::move(entry));

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

    [[nodiscard]] std::optional<Image> materialize(EntryId id, bool count_recall = true) {
        Entry* entry = find_entry(id);
        if (entry == nullptr) { return std::nullopt; }
        if (count_recall) { recall_entry(*entry); }

        Image image = entry->image;
        std::size_t total = 0;
        for (const BlockId block_id : entry->blocks) { total += blocks_.at(block_id).bytes.size(); }
        image.bytes.reserve(total);
        image.cache_block_sizes.reserve(entry->blocks.size());
        for (const BlockId block_id : entry->blocks) {
            const Block& block = blocks_.at(block_id);
            image.bytes.insert(image.bytes.end(), block.bytes.begin(), block.bytes.end());
            image.cache_block_sizes.push_back(block.bytes.size());
        }
        return image;
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
    };

    struct Entry {
        EntryId id                 = 0;
        std::size_t metadata_bytes = 0;
        std::uint32_t pin_count    = 0;
        Image image;
        std::vector<BlockId> blocks;
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
            if (block.pin_count != 0) { continue; }
            bool referenced = false;
            bool protected_reference = false;
            for (const Entry& entry : entries_) {
                if (std::find(entry.blocks.begin(), entry.blocks.end(), id) == entry.blocks.end()) {
                    continue;
                }
                referenced = true;
                if (entry.pin_count != 0 || entry.id == protected_entry) {
                    protected_reference = true;
                    break;
                }
            }
            if (!referenced || protected_reference) { continue; }
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
        for (const Entry& entry : entries_) {
            if (entry.id == protected_entry || entry.pin_count != 0) { continue; }
            if (std::find(entry.blocks.begin(), entry.blocks.end(), block) != entry.blocks.end()) {
                victims.push_back(entry.id);
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
        std::vector<BlockId> unique = entry.blocks;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (const BlockId id : unique) { fn(blocks_.at(id)); }
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
