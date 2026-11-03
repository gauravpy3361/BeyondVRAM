#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>

// BeyondVRAM: Native C++ SLRU Hot Expert Cache Controller
// Integrates with llama.cpp backend scheduler and CUDA streams.

namespace beyond_vram {

struct ExpertKey {
    int32_t layer;
    int32_t expert;

    bool operator==(const ExpertKey & other) const {
        return layer == other.layer && expert == other.expert;
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey & k) const {
        return std::hash<int32_t>()(k.layer) ^ (std::hash<int32_t>()(k.expert) << 16);
    }
};

struct CacheStats {
    uint64_t total_requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_uploaded = 0;

    double hit_rate_pct() const {
        return total_requests > 0 ? (100.0 * hits / total_requests) : 0.0;
    }
};

class HotExpertCache {
public:
    HotExpertCache(int32_t n_layers, int32_t n_experts_per_layer, int32_t n_slots, size_t bytes_per_expert);
    ~HotExpertCache() = default;

    // Checks slot mapping: returns slot index in VRAM (>= 0) on HIT, -1 on MISS
    int32_t lookup(int32_t layer, int32_t expert) const;

    // Plans slot allocations for Top-K experts in the given layer
    // Fills contains list of (expert_id, assigned_slot) that need async DMA transfer
    void plan_layer_access(int32_t layer, const int32_t * expert_ids, int32_t n_experts,
                           std::vector<std::pair<int32_t, int32_t>> & out_dma_fills,
                           int32_t & out_hits);

    // Pre-warms the cache with calibrated hot experts before inference begins
    void prewarm(const std::map<int32_t, std::vector<int32_t>> & calibrated_hot_experts);

    const CacheStats & get_stats() const { return stats; }

private:
    int32_t evict_victim();

    int32_t n_layers;
    int32_t n_experts;
    int32_t n_slots;
    size_t  bytes_per_expert;
    int32_t protected_capacity;

    // slot_map[layer][expert] -> slot_id
    std::vector<std::vector<int32_t>> slot_map;

    // Inverted index: slot_id -> ExpertKey
    std::unordered_map<int32_t, ExpertKey> slot_to_key;

    // Free slots available
    std::vector<int32_t> free_slots;

    // Doubly linked lists for SLRU probation (0) and protected (1) segments
    std::vector<int32_t> lru_prev;
    std::vector<int32_t> lru_next;
    std::vector<bool>    is_protected;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    int32_t n_protected = 0;

    CacheStats stats;
};

} // namespace beyond_vram
