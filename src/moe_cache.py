"""
BeyondVRAM: Step 1 & 2 - SLRU Hot Expert Cache Implementation & Simulator
Models the Segmented Least-Recently-Used (SLRU) cache allocation for
MoE architectures, tracking hit/miss ratios, DMA uploads, and VRAM residency.
"""

from typing import List, Dict, Tuple, Optional, Any
import collections


class SLRUEntry:
    def __init__(self, layer: int, expert: int, slot: int):
        self.layer = layer
        self.expert = expert
        self.slot = slot
        self.is_protected = False


class BeyondVRAMHotCache:
    """
    Segmented LRU (SLRU) Hot Expert Cache for MoE models.
    Partitions VRAM into fixed expert slots and divides cache management
    into Probation (newly fetched) and Protected (frequently accessed) segments.
    """
    def __init__(self, num_layers: int, experts_per_layer: int,
                 cache_capacity_slots: int, bytes_per_expert: int):
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.capacity = cache_capacity_slots
        self.bytes_per_expert = bytes_per_expert
        
        # Max protected size: 50% of cache capacity
        self.protected_capacity = self.capacity // 2

        # Slot map: slot_map[layer][expert] -> slot_id (or -1 if Cold/RAM)
        self.slot_map: List[List[int]] = [
            [-1 for _ in range(experts_per_layer)]
            for _ in range(num_layers)
        ]

        # Inverted index: slot_to_key[slot_id] -> (layer, expert)
        self.slot_to_key: Dict[int, Tuple[int, int]] = {}

        # Free slot pool
        self.free_slots: List[int] = list(range(self.capacity - 1, -1, -1))

        # SLRU segments using OrderedDict for O(1) MRU/LRU operations
        # key: (layer, expert), value: slot_id
        self.probation: collections.OrderedDict[Tuple[int, int], int] = collections.OrderedDict()
        self.protected: collections.OrderedDict[Tuple[int, int], int] = collections.OrderedDict()

        # Cumulative statistics
        self.total_requests = 0
        self.hits = 0
        self.misses = 0
        self.bytes_uploaded = 0

    def access(self, layer: int, expert_ids: List[int]) -> Dict[str, Any]:
        """
        Process expert routing requests for a given layer.
        Promotes hits to protected tier, selects victims for misses from probation tier,
        and returns which slots were assigned.
        """
        batch_hits = 0
        batch_misses = 0
        fills: List[Tuple[int, int]] = []  # List of (expert_id, assigned_slot)

        for expert in expert_ids:
            self.total_requests += 1
            key = (layer, expert)
            slot = self.slot_map[layer][expert]

            if slot >= 0:
                # Cache HIT!
                batch_hits += 1
                self.hits += 1

                # If in probation, promote to protected
                if key in self.probation:
                    del self.probation[key]
                    self.protected[key] = slot
                    # If protected tier exceeds 50%, demote its LRU to probation
                    if len(self.protected) > self.protected_capacity:
                        demoted_key, demoted_slot = self.protected.popitem(last=False)
                        self.probation[demoted_key] = demoted_slot
                elif key in self.protected:
                    # Move to MRU position of protected
                    self.protected.move_to_end(key)
            else:
                # Cache MISS!
                batch_misses += 1
                self.misses += 1

                # Allocate slot: prefer free slot, else evict victim
                if self.free_slots:
                    new_slot = self.free_slots.pop()
                else:
                    new_slot = self._evict_victim()

                # Assign slot to new expert
                self.slot_map[layer][expert] = new_slot
                self.slot_to_key[new_slot] = key
                self.probation[key] = new_slot

                self.bytes_uploaded += self.bytes_per_expert
                fills.append((expert, new_slot))

        hit_rate = (self.hits / self.total_requests) * 100.0 if self.total_requests > 0 else 0.0

        return {
            "layer": layer,
            "requested_experts": expert_ids,
            "hits": batch_hits,
            "misses": batch_misses,
            "fills": fills,
            "cumulative_hit_rate": round(hit_rate, 2),
            "vram_uploaded_mib": round(self.bytes_uploaded / (1024 * 1024), 2)
        }

    def _evict_victim(self) -> int:
        """
        Picks victim for eviction:
        1. First from LRU of probation tier (unproven, single-hit experts).
        2. If probation is empty, from LRU of protected tier.
        """
        if self.probation:
            (v_layer, v_expert), victim_slot = self.probation.popitem(last=False)
        elif self.protected:
            (v_layer, v_expert), victim_slot = self.protected.popitem(last=False)
        else:
            raise RuntimeError("Out of cache slots with no eviction candidates")

        # Invalidate old slot map entry
        self.slot_map[v_layer][v_expert] = -1
        del self.slot_to_key[victim_slot]
        return victim_slot

    def prewarm(self, hot_expert_map: Dict[int, List[int]]):
        """
        Pre-populates the cache with known high-frequency experts
        (e.g., from offline calibration or imatrix data) before inference begins.
        """
        for layer, experts in hot_expert_map.items():
            for e in experts:
                if len(self.free_slots) > 0 and self.slot_map[layer][e] == -1:
                    slot = self.free_slots.pop()
                    self.slot_map[layer][e] = slot
                    key = (layer, e)
                    self.slot_to_key[slot] = key
                    self.protected[key] = slot
                    self.bytes_uploaded += self.bytes_per_expert

    def get_status(self) -> Dict[str, Any]:
        used_slots = self.capacity - len(self.free_slots)
        return {
            "capacity_slots": self.capacity,
            "used_slots": used_slots,
            "probation_count": len(self.probation),
            "protected_count": len(self.protected),
            "free_slots": len(self.free_slots),
            "occupancy_pct": round((used_slots / self.capacity) * 100.0, 2),
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round((self.hits / max(1, self.total_requests)) * 100.0, 2),
            "vram_bandwidth_saved_gib": round((self.hits * self.bytes_per_expert) / (1024**3), 2)
        }


if __name__ == "__main__":
    import random

    print("=" * 65)
    print(" BeyondVRAM: Hot Cache (SLRU) Simulation on Groot-19B")
    print("=" * 65)

    # Groot-19B parameters
    NUM_LAYERS = 41
    EXPERTS_PER_LAYER = 128
    TOP_K = 8
    EXPERT_SIZE_BYTES = int(1.29 * 1024 * 1024)  # ~1.29 MB per expert
    CACHE_SLOTS = 3100  # 4000 MiB VRAM budget

    cache = BeyondVRAMHotCache(
        num_layers=NUM_LAYERS,
        experts_per_layer=EXPERTS_PER_LAYER,
        cache_capacity_slots=CACHE_SLOTS,
        bytes_per_expert=EXPERT_SIZE_BYTES
    )

    print(f" Initialized SLRU Cache: {CACHE_SLOTS} slots ({CACHE_SLOTS * EXPERT_SIZE_BYTES / (1024**2):.1f} MiB)")
    print(f" Simulating 100 tokens generation across {NUM_LAYERS} layers (Top-{TOP_K} routing)...")

    # Simulate realistic topic-locality: 30% of experts are cluster favorites
    # 70% of selections come from a subset of 35 experts per layer
    random.seed(42)
    for token_idx in range(100):
        for layer_idx in range(NUM_LAYERS):
            # 80% probability to pick from hot core (experts 0-35), 20% random
            if random.random() < 0.80:
                selected = random.sample(range(35), TOP_K)
            else:
                selected = random.sample(range(EXPERTS_PER_LAYER), TOP_K)
            cache.access(layer_idx, selected)

    status = cache.get_status()
    print("-" * 65)
    print(f" Total Expert Requests : {status['total_requests']:,}")
    print(f" Cache Hits (VRAM)     : {status['hits']:,} ({status['hit_rate_pct']}%)")
    print(f" Cache Misses (RAM)    : {status['misses']:,}")
    print(f" VRAM Occupancy        : {status['used_slots']} / {status['capacity_slots']} ({status['occupancy_pct']}%)")
    print(f" Protected Hot Experts : {status['protected_count']}")
    print(f" Probationary Experts  : {status['probation_count']}")
    print(f" PCIe Traffic Saved    : {status['vram_bandwidth_saved_gib']} GiB of PCIe bandwidth avoided!")
    print("=" * 65)
