#include "../src/beyond_vram_cache.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[*] Running BeyondVRAM HotExpertCache Unit Tests..." << std::endl;

    const int n_layers = 4;
    const int n_experts = 8;
    const int n_slots = 4;
    const size_t bytes_per_expert = 1024 * 1024; // 1 MB

    beyond_vram::HotExpertCache cache(n_layers, n_experts, n_slots, bytes_per_expert);

    // Initial lookup should all be miss
    assert(cache.lookup(0, 0) == -1);

    // Access 4 experts: should fill free slots
    int32_t req1[] = {0, 1, 2, 3};
    std::vector<std::pair<int32_t, int32_t>> fills;
    int32_t hits = 0;
    cache.plan_layer_access(0, req1, 4, fills, hits);

    assert(hits == 0);
    assert(fills.size() == 4);
    assert(cache.lookup(0, 0) >= 0);
    assert(cache.lookup(0, 1) >= 0);
    assert(cache.lookup(0, 2) >= 0);
    assert(cache.lookup(0, 3) >= 0);
    std::cout << "  [+] Free slot allocation test passed." << std::endl;

    // Access expert 0 again: should HIT and promote to protected segment
    int32_t req2[] = {0};
    cache.plan_layer_access(0, req2, 1, fills, hits);
    assert(hits == 1);
    assert(fills.empty());
    std::cout << "  [+] Cache hit and SLRU promotion test passed." << std::endl;

    // Access expert 4 (MISS, needs eviction): should evict an unpromoted probationary victim
    int32_t req3[] = {4};
    cache.plan_layer_access(0, req3, 1, fills, hits);
    assert(hits == 0);
    assert(fills.size() == 1);
    // Expert 0 was promoted to protected, so it should still be in cache!
    assert(cache.lookup(0, 0) >= 0);
    std::cout << "  [+] Protected segment immunity from eviction test passed." << std::endl;

    const auto & stats = cache.get_stats();
    std::cout << "  [+] Cache Statistics: " << stats.hits << " hits, " 
              << stats.misses << " misses (" << stats.hit_rate_pct() << "% hit rate)" << std::endl;

    std::cout << "[SUCCESS] All HotExpertCache tests passed cleanly!" << std::endl;
    return 0;
}
