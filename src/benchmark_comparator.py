"""
BeyondVRAM: Comparative Benchmark & PCIe Transit Savings Calculator
Reconciles both architectural execution modes:
1. Dynamic GPU VRAM Cache Mode (Current Production Engine):
   - Cache hits run in GDDR6 (0 PCIe traffic)
   - Cache misses promote/stream weights to GPU (expert_size PCIe traffic)
2. Hybrid Dynamic Cache + Zero-Transfer CPU FFN Mode (Asymmetric Architecture):
   - Cache hits run in GDDR6 (0 PCIe traffic)
   - Single-shot cold misses execute in-place on CPU (activation_size PCIe traffic)
"""

import sys
from typing import Dict, Any


class PCIeTransitComparator:
    """
    Evaluates PCIe bus bottlenecks:
    Calculates total data shuttled over PCIe bus and time penalty incurred
    under different bus configurations (PCIe 3.0/4.0 on consumer laptops).
    """
    def __init__(self,
                 num_layers: int = 41,
                 expert_count: int = 128,
                 top_k: int = 8,
                 expert_size_mb: float = 1.29,
                 activation_size_kb: float = 10.0,
                 pcie_bandwidth_gbps: float = 4.8):
        """
        :param pcie_bandwidth_gbps: Real-world bidirectional bandwidth
               (e.g., PCIe 4.0 x4 on laptop is ~4.5 - 5.0 GB/s)
        """
        self.num_layers = num_layers
        self.expert_count = expert_count
        self.top_k = top_k
        self.expert_size_mb = expert_size_mb
        self.activation_size_kb = activation_size_kb
        self.pcie_bandwidth_gbps = pcie_bandwidth_gbps

    def compare_run(self, num_tokens: int, cache_hit_rate_pct: float = 89.0) -> Dict[str, Any]:
        total_expert_calls = num_tokens * self.num_layers * self.top_k
        single_expert_gb = self.expert_size_mb / 1024.0
        activation_gb = self.activation_size_kb / (1024.0 * 1024.0)

        # 1. Baseline: Naive offload transfers every expert weight across PCIe per token
        baseline_traffic_gb = total_expert_calls * single_expert_gb
        baseline_transit_sec = baseline_traffic_gb / self.pcie_bandwidth_gbps

        # Cache accounting
        hits = int(total_expert_calls * (cache_hit_rate_pct / 100.0))
        misses = total_expert_calls - hits

        # 2. Mode 1: Dynamic GPU VRAM Cache (Production Engine)
        # Misses upload expert weights into GPU SLRU cache; hits transfer 0 bytes
        dyn_cache_traffic_gb = misses * single_expert_gb
        dyn_cache_transit_sec = dyn_cache_traffic_gb / self.pcie_bandwidth_gbps
        dyn_cache_saved_gb = baseline_traffic_gb - dyn_cache_traffic_gb
        dyn_cache_reduction_pct = (dyn_cache_saved_gb / max(0.001, baseline_traffic_gb)) * 100.0

        # 3. Mode 2: Hybrid Cache + Zero-Transfer CPU FFN (Asymmetric Architecture)
        # Misses execute in-place on CPU (only 10 KB activations move over PCIe)
        zero_transfer_traffic_gb = misses * activation_gb
        zero_transfer_transit_sec = zero_transfer_traffic_gb / self.pcie_bandwidth_gbps
        zero_transfer_saved_gb = baseline_traffic_gb - zero_transfer_traffic_gb
        zero_transfer_reduction_pct = (zero_transfer_saved_gb / max(0.001, baseline_traffic_gb)) * 100.0

        return {
            "num_tokens": num_tokens,
            "total_expert_calls": total_expert_calls,
            "cache_hit_rate_pct": cache_hit_rate_pct,
            "baseline_traffic_gb": round(baseline_traffic_gb, 2),
            "baseline_transit_sec": round(baseline_transit_sec, 2),
            # Dynamic Cache
            "dyn_cache_traffic_gb": round(dyn_cache_traffic_gb, 2),
            "dyn_cache_transit_sec": round(dyn_cache_transit_sec, 2),
            "dyn_cache_saved_gb": round(dyn_cache_saved_gb, 2),
            "dyn_cache_reduction_pct": round(dyn_cache_reduction_pct, 2),
            # Zero-Transfer CPU FFN
            "zero_transfer_traffic_gb": round(zero_transfer_traffic_gb, 4),
            "zero_transfer_transit_sec": round(zero_transfer_transit_sec, 4),
            "zero_transfer_saved_gb": round(zero_transfer_saved_gb, 2),
            "zero_transfer_reduction_pct": round(zero_transfer_reduction_pct, 2),
        }

    def print_comparison_card(self, results: Dict[str, Any]):
        tokens = results['num_tokens']
        print("=" * 80)
        print(f" BeyondVRAM: PCIe Transit Savings Benchmark ({tokens} Tokens Generated)")
        print("=" * 80)
        print(f" Hardware Bus Model : PCIe Interface @ {self.pcie_bandwidth_gbps} GB/s real-world")
        print(f" Total Expert Calls : {results['total_expert_calls']:,} across {self.num_layers} layers (Top-{self.top_k})")
        print(f" Cache Hit Rate     : {results['cache_hit_rate_pct']}% resident in GDDR6")
        print("-" * 80)
        print(f" {'METRIC':<32} | {'NAIVE OFFLOAD':<14} | {'DYNAMIC CACHE':<14} | {'+ ZERO-TRANSFER':<14}")
        print("-" * 80)
        print(f" {'PCIe Weight Shuttled':<32} | {results['baseline_traffic_gb']:>8.2f} GB     | {results['dyn_cache_traffic_gb']:>8.2f} GB     | {results['zero_transfer_traffic_gb']:>8.4f} GB")
        print(f" {'Bus Transit Penalty':<32} | {results['baseline_transit_sec']:>8.2f} s      | {results['dyn_cache_transit_sec']:>8.2f} s      | {results['zero_transfer_transit_sec']:>8.4f} s")
        print(f" {'Bus Traffic Eliminated':<32} | {'0.00 GB':>8}       | {results['dyn_cache_saved_gb']:>8.2f} GB     | {results['zero_transfer_saved_gb']:>8.2f} GB")
        print(f" {'Data Shuttling Reduction':<32} | {'0.0%':>8}       | {results['dyn_cache_reduction_pct']:>8.1f} %      | {results['zero_transfer_reduction_pct']:>8.1f} %")
        print("=" * 80)


if __name__ == "__main__":
    comparator = PCIeTransitComparator()
    
    # Run comparison for 100, 500, and 1,000 tokens
    for token_count in [100, 500, 1000]:
        res = comparator.compare_run(num_tokens=token_count, cache_hit_rate_pct=89.0)
        comparator.print_comparison_card(res)
        print()
