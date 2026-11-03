"""
BeyondVRAM: Step 2 - Async Pipeline & N+2 Lookahead Scheduler Simulator
Calculates timeline overlap, hides PCIe DMA transit behind Attention math,
and evaluates prefetch hit masking efficiency.
"""

from typing import List, Dict, Any, Tuple


class AsyncPipelineScheduler:
    """
    Simulates dual-stream asynchronous execution of Attention math
    and PCIe DMA expert prefetching with N+1 / N+2 lookahead.
    """
    def __init__(self,
                 num_layers: int = 41,
                 attn_compute_ms_per_layer: float = 0.35,
                 dma_transfer_ms_per_expert: float = 0.28,
                 pcie_bandwidth_gbps: float = 4.5):
        self.num_layers = num_layers
        self.t_attn = attn_compute_ms_per_layer
        self.t_dma = dma_transfer_ms_per_expert
        self.pcie_bandwidth = pcie_bandwidth_gbps

    def simulate_token_timeline(self, misses_per_layer: List[int], lookahead_layers: int = 2) -> Dict[str, Any]:
        """
        Simulates one token's progression across all layers:
        - Sequential Mode: T_layer = T_attn + (misses * T_dma) + T_ffn
        - Pipelined Mode: T_dma is initiated `lookahead_layers` in advance,
          overlapping with preceding Attention layers.
        """
        sequential_time_ms = 0.0
        pipelined_time_ms = 0.0
        total_dma_time_ms = 0.0
        masked_dma_time_ms = 0.0

        # For lookahead tracking
        pending_dma_queue: List[float] = []

        for layer in range(self.num_layers):
            n_miss = misses_per_layer[layer]
            layer_dma_cost = n_miss * self.t_dma
            total_dma_time_ms += layer_dma_cost

            # Sequential: Attention must wait for all DMA transfers
            sequential_time_ms += (self.t_attn + layer_dma_cost)

            # Pipelined: While Attention computes, background DMA finishes
            # Lookahead window allows Attention from preceding layers to absorb DMA cost
            available_compute_window = self.t_attn * lookahead_layers

            if layer_dma_cost <= available_compute_window:
                # Fully masked!
                visible_dma = 0.0
                masked_dma_time_ms += layer_dma_cost
            else:
                # Partially masked: only the residual stalls the pipeline
                visible_dma = layer_dma_cost - available_compute_window
                masked_dma_time_ms += available_compute_window

            pipelined_time_ms += (self.t_attn + visible_dma)

        masking_efficiency_pct = (masked_dma_time_ms / max(0.001, total_dma_time_ms)) * 100.0
        latency_reduction_pct = ((sequential_time_ms - pipelined_time_ms) / sequential_time_ms) * 100.0
        speedup = sequential_time_ms / max(0.001, pipelined_time_ms)

        return {
            "num_layers": self.num_layers,
            "lookahead_window": lookahead_layers,
            "sequential_time_ms": round(sequential_time_ms, 2),
            "pipelined_time_ms": round(pipelined_time_ms, 2),
            "total_dma_ms": round(total_dma_time_ms, 2),
            "masked_dma_ms": round(masked_dma_time_ms, 2),
            "masking_efficiency_pct": round(masking_efficiency_pct, 2),
            "latency_reduction_pct": round(latency_reduction_pct, 2),
            "speedup_factor": round(speedup, 2)
        }

    def print_timeline_report(self, stats: Dict[str, Any]):
        print("=" * 65)
        print(" BeyondVRAM: Step 2 Async Pipelining Simulation Report")
        print("=" * 65)
        print(f" Layer Count          : {stats['num_layers']} Transformer layers")
        print(f" Lookahead Prefetch   : N+{stats['lookahead_window']} layers lookahead window")
        print(f" Total PCIe DMA Load  : {stats['total_dma_ms']} ms of transfers required")
        print("-" * 65)
        print(f" Sequential Latency   : {stats['sequential_time_ms']} ms / token (Blocking stalls)")
        print(f" Pipelined Latency    : {stats['pipelined_time_ms']} ms / token (Overlapped)")
        print(f" DMA Masked Behind GPU: {stats['masked_dma_ms']} ms hidden ({stats['masking_efficiency_pct']}% masked!)")
        print("-" * 65)
        print(f" [*] LATENCY REDUCTION : {stats['latency_reduction_pct']}% faster per token")
        print(f" [*] PIPELINE SPEEDUP  : {stats['speedup_factor']}x hardware execution boost")
        print("=" * 65)


if __name__ == "__main__":
    scheduler = AsyncPipelineScheduler()

    # In our live benchmark on Groot-19B, ~15-20% of expert calls are misses
    # Across 41 layers with Top-8 routing, that's roughly 1 to 2 misses per layer
    misses_profile = [1 if (i % 2 == 0) else 2 for i in range(41)]

    # Compare N+1 vs N+2 lookahead
    for lookahead in [1, 2]:
        res = scheduler.simulate_token_timeline(misses_profile, lookahead_layers=lookahead)
        scheduler.print_timeline_report(res)
        print()
