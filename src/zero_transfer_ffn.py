"""
BeyondVRAM: Step 3 - Zero-Transfer CPU FFN & Asymmetric Execution Engine
Simulates in-place host RAM FFN execution for single-shot cold experts,
eliminating weight transfer across PCIe and exchanging only activation vectors.
"""

import sys
from typing import Dict, Any, List


class ZeroTransferFFNRouter:
    """
    Asymmetric Hybrid Dispatcher:
    Decides whether an expert call warrants VRAM promotion or in-place CPU execution.
    """
    def __init__(self,
                 d_model: int = 2048,
                 expert_ffn_dim: int = 512,
                 expert_weight_bytes: int = 1351680,  # ~1.29 MB
                 pcie_bandwidth_gbps: float = 4.5,
                 cpu_gflops: float = 120.0):
        self.d_model = d_model
        self.expert_ffn_dim = expert_ffn_dim
        self.expert_weight_bytes = expert_weight_bytes
        self.pcie_bandwidth_gbps = pcie_bandwidth_gbps
        self.cpu_gflops = cpu_gflops

        # Activation vector: FP16 (2 bytes per element)
        self.activation_bytes_in = d_model * 2  # 4 KB
        self.activation_bytes_out = d_model * 2  # 4 KB
        self.total_activation_transfer_bytes = self.activation_bytes_in + self.activation_bytes_out

    def evaluate_expert_call(self, access_count_window: int) -> Dict[str, Any]:
        """
        Calculates the breakeven trade-off:
        - Path A (Transfer to GPU): Shuttles full 1.29 MB weight matrix over PCIe.
        - Path B (Zero-Transfer CPU FFN): Shuttles only 8 KB activations, computes on host CPU.
        """
        # Time to transfer full weights over PCIe
        pcie_bytes_per_sec = self.pcie_bandwidth_gbps * (1024**3)
        weight_transfer_time_ms = (self.expert_weight_bytes / pcie_bytes_per_sec) * 1000.0

        # Time to transfer activations over PCIe
        activation_transfer_time_ms = (self.total_activation_transfer_bytes / pcie_bytes_per_sec) * 1000.0

        # Floating point operations for SwiGLU FFN:
        # Gate: 2 * d_model * ffn_dim
        # Up:   2 * d_model * ffn_dim
        # Down: 2 * ffn_dim * d_model
        # Total = 6 * d_model * ffn_dim FLOPs
        flops = 6 * self.d_model * self.expert_ffn_dim
        cpu_compute_time_ms = (flops / (self.cpu_gflops * 1e9)) * 1000.0

        total_cpu_path_ms = activation_transfer_time_ms + cpu_compute_time_ms
        total_gpu_transfer_path_ms = weight_transfer_time_ms  # (excluding GPU compute which is near-zero)

        bytes_saved = self.expert_weight_bytes - self.total_activation_transfer_bytes
        transfer_reduction_pct = (bytes_saved / self.expert_weight_bytes) * 100.0

        # Heuristic decision:
        # If single-shot (access_count <= 2), CPU path is faster and avoids VRAM eviction
        use_cpu_ffn = (access_count_window <= 2) or (total_cpu_path_ms < total_gpu_transfer_path_ms)

        return {
            "d_model": self.d_model,
            "expert_weight_bytes": self.expert_weight_bytes,
            "activation_bytes": self.total_activation_transfer_bytes,
            "weight_transfer_time_ms": round(weight_transfer_time_ms, 3),
            "activation_transfer_time_ms": round(activation_transfer_time_ms, 4),
            "cpu_compute_time_ms": round(cpu_compute_time_ms, 3),
            "total_cpu_path_ms": round(total_cpu_path_ms, 3),
            "bytes_saved_per_call": bytes_saved,
            "transfer_reduction_pct": round(transfer_reduction_pct, 2),
            "recommended_dispatch": "CPU_IN_PLACE" if use_cpu_ffn else "VRAM_HOT_PROMOTION",
            "traffic_compression_ratio": round(self.expert_weight_bytes / self.total_activation_transfer_bytes, 1)
        }

    def print_analysis(self, report: Dict[str, Any]):
        print("=" * 65)
        print(" BeyondVRAM: Step 3 Zero-Transfer CPU FFN Decision Matrix")
        print("=" * 65)
        print(f" Activation Vector Size   : {report['activation_bytes']:,} bytes (2 x {self.d_model} FP16)")
        print(f" Expert Weight Matrix Size : {report['expert_weight_bytes']:,} bytes (~1.29 MB)")
        print("-" * 65)
        print(" Bus Transfer Comparison (PCIe 4.0 x4):")
        print(f"  * Traditional Offload  : {report['expert_weight_bytes']:,} bytes moved per cold call")
        print(f"  * Zero-Transfer CPU FFN: {report['activation_bytes']:,} bytes moved per cold call")
        print(f"  * Data Transit Cut     : {report['transfer_reduction_pct']}% REDUCTION in bus traffic!")
        print(f"  * Compression Ratio    : {report['traffic_compression_ratio']}x less data over PCIe")
        print("-" * 65)
        print(" Latency Breakdown:")
        print(f"  * Shuttling Weight to GPU : {report['weight_transfer_time_ms']} ms (PCIe bus bottleneck)")
        print(f"  * Sending Activation Only : {report['activation_transfer_time_ms']} ms")
        print(f"  * In-Place CPU SIMD Math  : {report['cpu_compute_time_ms']} ms")
        print(f"  * Total CPU In-Place Path : {report['total_cpu_path_ms']} ms")
        print("-" * 65)
        print(f" [*] DISPATCH DECISION    : >>> {report['recommended_dispatch']} <<<")
        print(f" [*] WHY IT WINS          : Computing in host RAM is {report['weight_transfer_time_ms'] / report['total_cpu_path_ms']:.2f}x faster")
        print(f"                            than wasting {report['weight_transfer_time_ms']} ms transferring the weights!")
        print("=" * 65)


if __name__ == "__main__":
    dispatcher = ZeroTransferFFNRouter()
    # Evaluate for a single-shot cold expert
    analysis = dispatcher.evaluate_expert_call(access_count_window=1)
    dispatcher.print_analysis(analysis)
