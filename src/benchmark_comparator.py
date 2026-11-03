"""
BeyondVRAM: Analytical PCIe Bus Traffic & Latency Calculator
Models theoretical PCIe weight-streaming upper bounds vs. actual `llama_moe_cache`
SLRU hit rates (~89%) and cold-expert CPU execution (`--n-cpu-moe`).
"""

from typing import Dict, Any


class PCIeTransitComparator:
    """
    Analytical calculator comparing:
      1. Theoretical Uncached Streaming (if every Top-K expert call were fetched over PCIe)
      2. GPU SLRU Dynamic Cache (`--moe-cache-mib`): Hits execute in GDDR6 (0 B PCIe);
         misses upload 1.29 MiB expert weights over PCIe into the SLRU cache.
      3. CPU-Resident Cold Execution (`--n-cpu-moe` without GPU upload): Cold experts
         execute on the CPU; only 16 KiB activations (8 KiB in + 8 KiB out) cross PCIe.
    """

    def __init__(
        self,
        num_layers: int = 41,
        expert_count: int = 128,
        top_k: int = 8,
        expert_size_mib: float = 1.289,
        activation_roundtrip_kib: float = 16.0,
        pcie_bandwidth_gibps: float = 11.0,
    ):
        self.num_layers = num_layers
        self.expert_count = expert_count
        self.top_k = top_k
        self.expert_size_mib = expert_size_mib
        self.activation_roundtrip_kib = activation_roundtrip_kib
        self.pcie_bandwidth_gibps = pcie_bandwidth_gibps

    def compare_run(self, num_tokens: int, cache_hit_rate_pct: float = 89.0) -> Dict[str, Any]:
        total_expert_calls = num_tokens * self.num_layers * self.top_k
        single_expert_gib = self.expert_size_mib / 1024.0
        activation_gib = self.activation_roundtrip_kib / (1024.0 * 1024.0)

        # 1. Theoretical upper bound if every Top-K expert were streamed over PCIe per token
        uncached_stream_gib = total_expert_calls * single_expert_gib
        uncached_transit_sec = uncached_stream_gib / self.pcie_bandwidth_gibps

        hits = int(total_expert_calls * (cache_hit_rate_pct / 100.0))
        misses = total_expert_calls - hits

        # 2. GPU SLRU Cache (--moe-cache-mib): only misses upload expert weights
        slru_upload_gib = misses * single_expert_gib
        slru_transit_sec = slru_upload_gib / self.pcie_bandwidth_gibps
        slru_saved_gib = uncached_stream_gib - slru_upload_gib
        slru_reduction_pct = (slru_saved_gib / max(0.001, uncached_stream_gib)) * 100.0

        # 3. CPU-Resident Cold Misses: only 16 KiB activations cross PCIe per layer miss
        cpu_act_gib = misses * activation_gib
        cpu_act_transit_sec = cpu_act_gib / self.pcie_bandwidth_gibps

        return {
            "num_tokens": num_tokens,
            "total_expert_calls": total_expert_calls,
            "cache_hit_rate_pct": cache_hit_rate_pct,
            "uncached_stream_gib": round(uncached_stream_gib, 2),
            "uncached_transit_sec": round(uncached_transit_sec, 2),
            "slru_upload_gib": round(slru_upload_gib, 2),
            "slru_transit_sec": round(slru_transit_sec, 2),
            "slru_saved_gib": round(slru_saved_gib, 2),
            "slru_reduction_pct": round(slru_reduction_pct, 2),
            "cpu_act_gib": round(cpu_act_gib, 3),
            "cpu_act_transit_sec": round(cpu_act_transit_sec, 3),
        }

    def print_comparison_card(self, results: Dict[str, Any]):
        tokens = results["num_tokens"]
        print("=" * 82)
        print(f" BeyondVRAM: Analytical PCIe Traffic Model ({tokens:,} Decode Tokens)")
        print(" NOTE: Hypothetical stream is an upper bound, NOT stock llama.cpp behaviour (CPU layers run in place).")
        print(f" Measured PCIe H2D Bandwidth : {self.pcie_bandwidth_gibps} GiB/s (pinned host -> RTX 3050)")
        print(f" Single-Token Expert Lookups : {results['total_expert_calls']:,} ({self.num_layers} layers x Top-{self.top_k})")
        print(f" SLRU Hit Rate (from logs)   : {results['cache_hit_rate_pct']}% resident in GDDR6")
        print("-" * 82)
        print(f" {'METRIC':<30} | {'HYPOTHETICAL STREAM':<19} | {'SLRU CACHE (89%)':<16} | {'CPU COLD PATH'}")
        print("-" * 82)
        print(
            f" {'PCIe Data Transferred':<30} | {results['uncached_stream_gib']:>10.2f} GiB   | "
            f"{results['slru_upload_gib']:>10.2f} GiB   | {results['cpu_act_gib']:>8.3f} GiB"
        )
        print(
            f" {'Pure PCIe Transfer Time':<30} | {results['uncached_transit_sec']:>10.2f} s     | "
            f"{results['slru_transit_sec']:>10.2f} s     | {results['cpu_act_transit_sec']:>8.3f} s"
        )
        print(
            f" {'Traffic Avoided vs Stream':<30} | {'0.00 GiB':>12}     | "
            f"{results['slru_saved_gib']:>10.2f} GiB   | {results['uncached_stream_gib'] - results['cpu_act_gib']:>8.2f} GiB"
        )
        print("=" * 82)


if __name__ == "__main__":
    comparator = PCIeTransitComparator()
    for token_count in [100, 500, 1000]:
        res = comparator.compare_run(num_tokens=token_count, cache_hit_rate_pct=89.0)
        comparator.print_comparison_card(res)
        print()
