#include "beyond_vram_cache.h"
#include <algorithm>
#include <iostream>

namespace beyond_vram {

HotExpertCache::HotExpertCache(int32_t n_layers, int32_t n_experts_per_layer, int32_t n_slots, size_t bytes_per_expert)
    : n_layers(n_layers),
      n_experts(n_experts_per_layer),
      n_slots(n_slots),
      bytes_per_expert(bytes_per_expert),
      protected_capacity(static_cast<int32_t>(n_slots * 0.70)) { // 70% protected segment
    
    // Initialize slot mapping table with -1 (not cached)
    slot_map.resize(n_layers, std::vector<int32_t>(n_experts, -1));

    // Linked list arrays for LRU tracking
    lru_prev.resize(n_slots, -1);
    lru_next.resize(n_slots, -1);
    is_protected.resize(n_slots, false);

    // Populate free slot list
    free_slots.reserve(n_slots);
    for (int32_t i = n_slots - 1; i >= 0; --i) {
        free_slots.push_back(i);
    }
}

int32_t HotExpertCache::lookup(int32_t layer, int32_t expert) const {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
        return -1;
    }
    return slot_map[layer][expert];
}

void HotExpertCache::plan_layer_access(int32_t layer, const int32_t * expert_ids, int32_t n_req_experts,
                                      std::vector<std::pair<int32_t, int32_t>> & out_dma_fills,
                                      int32_t & out_hits) {
    out_dma_fills.clear();
    out_hits = 0;

    for (int32_t i = 0; i < n_req_experts; ++i) {
        int32_t expert = expert_ids[i];
        stats.total_requests++;

        int32_t existing_slot = lookup(layer, expert);
        if (existing_slot >= 0) {
            // HIT
            out_hits++;
            stats.hits++;

            // SLRU Promotion: if in probation segment (0), promote to protected segment (1)
            if (!is_protected[existing_slot]) {
                // Detach from probation list
                int32_t p = lru_prev[existing_slot];
                int32_t n = lru_next[existing_slot];
                if (p != -1) lru_next[p] = n; else head[0] = n;
                if (n != -1) lru_prev[n] = p; else tail[0] = p;

                // Check protected capacity; if full, demote LRU of protected to probation
                if (n_protected >= protected_capacity && tail[1] != -1) {
                    int32_t demoted = tail[1];
                    int32_t dp = lru_prev[demoted];
                    if (dp != -1) lru_next[dp] = -1; else head[1] = -1;
                    tail[1] = dp;

                    is_protected[demoted] = false;
                    n_protected--;

                    // Insert demoted at head of probation
                    lru_prev[demoted] = -1;
                    lru_next[demoted] = head[0];
                    if (head[0] != -1) lru_prev[head[0]] = demoted;
                    head[0] = demoted;
                    if (tail[0] == -1) tail[0] = demoted;
                }

                // Insert into head of protected
                is_protected[existing_slot] = true;
                n_protected++;
                lru_prev[existing_slot] = -1;
                lru_next[existing_slot] = head[1];
                if (head[1] != -1) lru_prev[head[1]] = existing_slot;
                head[1] = existing_slot;
                if (tail[1] == -1) tail[1] = existing_slot;
            }
        } else {
            // MISS
            stats.misses++;
            stats.bytes_uploaded += bytes_per_expert;

            int32_t target_slot = -1;
            if (!free_slots.empty()) {
                target_slot = free_slots.back();
                free_slots.pop_back();
            } else {
                target_slot = evict_victim();
            }

            // Assign slot
            slot_map[layer][expert] = target_slot;
            slot_to_key[target_slot] = {layer, expert};

            // Insert into head of probation segment (0)
            is_protected[target_slot] = false;
            lru_prev[target_slot] = -1;
            lru_next[target_slot] = head[0];
            if (head[0] != -1) lru_prev[head[0]] = target_slot;
            head[0] = target_slot;
            if (tail[0] == -1) tail[0] = target_slot;

            out_dma_fills.push_back({expert, target_slot});
        }
    }
}

int32_t HotExpertCache::evict_victim() {
    // Evict from tail of probation segment first
    int32_t victim = tail[0];
    int32_t seg = 0;

    if (victim == -1) {
        // If probation is empty, evict from tail of protected
        victim = tail[1];
        seg = 1;
        if (victim != -1) n_protected--;
    }

    if (victim == -1) {
        return 0; // fallback if somehow both empty
    }

    // Detach victim
    int32_t p = lru_prev[victim];
    if (p != -1) lru_next[p] = -1; else head[seg] = -1;
    tail[seg] = p;

    // Invalidate slot in slot_map
    auto it = slot_to_key.find(victim);
    if (it != slot_to_key.end()) {
        slot_map[it->second.layer][it->second.expert] = -1;
        slot_to_key.erase(it);
    }

    return victim;
}

void HotExpertCache::prewarm(const std::map<int32_t, std::vector<int32_t>> & calibrated_hot_experts) {
    for (const auto & kv : calibrated_hot_experts) {
        int32_t layer = kv.first;
        if (layer < 0 || layer >= n_layers) continue;

        for (int32_t expert : kv.second) {
            if (expert < 0 || expert >= n_experts) continue;
            if (free_slots.empty()) return;

            int32_t slot = free_slots.back();
            free_slots.pop_back();

            slot_map[layer][expert] = slot;
            slot_to_key[slot] = {layer, expert};

            // Pre-warmed items start in protected segment
            is_protected[slot] = true;
            n_protected++;

            lru_prev[slot] = -1;
            lru_next[slot] = head[1];
            if (head[1] != -1) lru_prev[head[1]] = slot;
            head[1] = slot;
            if (tail[1] == -1) tail[1] = slot;
        }
    }
}

} // namespace beyond_vram
