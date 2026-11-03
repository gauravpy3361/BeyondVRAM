"""
BeyondVRAM: Step 2 - Analytical Model of Dual-Stream Async DMA & Compute Overlap
Models how per-layer PCIe H2D expert transfers (`ggml_backend_tensor_set_async` /
`cudaMemcpyAsync`) overlap with GPU compute streams using measured RTX 3050 constants.
"""

from typing import List, Dict, Any


class AsyncPipelineScheduler:
    """
    Analytical model of sequential vs. dual-stream overlapped execution across 41 MoE layers,
    calibrated to measured RTX 3050 + PCIe 4.0 microbenchmark latencies (`src/async_pipeline.cu`).
    """

    def __init__(
        self,
        num_layers: int = 41,
        attn_compute_ms_per_layer: float = 0.250,
        dma_transfer_ms_per_expert: float = 0.114,
        cold_ffn_ms_per_expert: float = 0.012,
    ):
        self.num_layers = num_layers
        self.t_attn = attn_compute_ms_per_layer
        self.t_dma = dma_transfer_ms_per_expert
        self.t_ffn = cold_ffn_ms_per_expert

    def simulate_token_timeline(self, misses_per_layer: List[int], lookahead_layers: int = 1) -> Dict[str, Any]:
        sequential_time_ms = 0.0
        pipelined_time_ms = 0.0
        total_dma_time_ms = 0.0
        masked_dma_time_ms = 0.0

        for layer in range(self.num_layers):
            n_miss = misses_per_layer[layer]
            layer_dma_cost = n_miss * self.t_dma
            layer_ffn_cost = n_miss * self.t_ffn
            total_dma_time_ms += layer_dma_cost

            # Sequential: Attention + blocking H2D DMA + FFN
            sequential_time_ms += self.t_attn + layer_dma_cost + layer_ffn_cost

            # Pipelined: DMA overlaps with Attention compute window
            available_compute_window = self.t_attn * lookahead_layers
            if layer_dma_cost <= available_compute_window:
                visible_dma = 0.0
                masked_dma_time_ms += layer_dma_cost
            else:
                visible_dma = layer_dma_cost - available_compute_window
                masked_dma_time_ms += available_compute_window

            pipelined_time_ms += self.t_attn + visible_dma + layer_ffn_cost

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
            "speedup_factor": round(speedup, 2),
        }

    def print_timeline_report(self, stats: Dict[str, Any]):
        print("=" * 68)
        print(" BeyondVRAM: Step 2 Async Pipeline Analytical Model (Calibrated)")
        print("=" * 68)
        print(f" Layer Count          : {stats['num_layers']} Transformer layers")
        print(f" Overlap Window       : {stats['lookahead_window']}x layer Attention compute ({self.t_attn} ms/layer)")
        print(f" Total PCIe DMA Time  : {stats['total_dma_ms']} ms ({self.t_dma} ms per 1.29 MiB expert)")
        print("-" * 68)
        print(f" Sequential Latency   : {stats['sequential_time_ms']} ms / token (same-stream H2D + compute)")
        print(f" Overlapped Latency   : {stats['pipelined_time_ms']} ms / token (dual-stream H2D + compute)")
        print(f" DMA Hidden Behind GPU: {stats['masked_dma_ms']} ms ({stats['masking_efficiency_pct']}% masked)")
        print("-" * 68)
        print(f" [*] LATENCY REDUCTION : {stats['latency_reduction_pct']}% faster per token")
        print(f" [*] PIPELINE SPEEDUP  : {stats['speedup_factor']}x")
        print("=" * 68)


if __name__ == "__main__":
    scheduler = AsyncPipelineScheduler()
    # At ~89% hit rate across Top-8 experts (41 layers = 328 expert calls/token),
    # ~36 misses occur per token (~0.88 misses/layer on average; modeled as 1 miss on 36 layers).
    misses_profile = [1 if i < 36 else 0 for i in range(41)]
    res = scheduler.simulate_token_timeline(misses_profile, lookahead_layers=1)
    scheduler.print_timeline_report(res)
