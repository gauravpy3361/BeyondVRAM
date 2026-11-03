"""
BeyondVRAM: Step 3 - Analytical Model of Cold-Expert Execution Paths
Models the latency & PCIe bandwidth trade-offs across three execution paths for
a Groot-19B-A3B 3-matrix SwiGLU expert (1.29 MiB Q3_K):
  - Path 0: GPU SLRU Cache Hit (0 B PCIe transfer, warp-parallel GPU SwiGLU)
  - Path A: Cold Miss via PCIe Weight Upload (1.29 MiB H2D) + GPU SwiGLU
  - Path B: Cold Miss via In-Place CPU AVX2 SwiGLU + 16 KiB Activation Roundtrip & Sync
"""

from typing import Dict, Any


class ZeroTransferFFNRouter:
    """
    Analytical latency & bus-traffic model parameterized by measured RTX 3050 +
    Core 5 210H microbenchmark constants (see `src/zero_transfer_ffn.cpp`).
    """

    def __init__(
        self,
        d_model: int = 2048,
        expert_ffn_dim: int = 512,
        expert_weight_bytes: int = 1351680,  # 1.289 MiB (Q3_K 3-matrix SwiGLU expert)
        pcie_h2d_ms_per_expert: float = 0.114,
        gpu_swiglu_ms_per_expert: float = 0.042,
        cpu_avx2_ms_per_expert: float = 0.086,
        gpu_cpu_sync_roundtrip_ms: float = 0.100,
    ):
        self.d_model = d_model
        self.expert_ffn_dim = expert_ffn_dim
        self.expert_weight_bytes = expert_weight_bytes
        self.pcie_h2d_ms = pcie_h2d_ms_per_expert
        self.gpu_swiglu_ms = gpu_swiglu_ms_per_expert
        self.cpu_avx2_ms = cpu_avx2_ms_per_expert
        self.sync_roundtrip_ms = gpu_cpu_sync_roundtrip_ms

        self.activation_bytes_in = d_model * 4   # 8 KiB FP32 input
        self.activation_bytes_out = d_model * 4  # 8 KiB FP32 output
        self.total_activation_bytes = self.activation_bytes_in + self.activation_bytes_out

    def evaluate_layer_misses(self, k_misses: int = 1) -> Dict[str, Any]:
        """
        Evaluates total latency for `k_misses` cold experts in a layer:
        - Path 0 (If all were GPU Cache Hits): k * gpu_swiglu_ms
        - Path A (Upload k experts to GPU + GPU SwiGLU): k * (pcie_h2d_ms + gpu_swiglu_ms)
        - Path B (1 activation roundtrip sync + k experts computed in-place on CPU):
          sync_roundtrip_ms + k * cpu_avx2_ms
        """
        gpu_hit_total_ms = k_misses * self.gpu_swiglu_ms
        path_a_total_ms = k_misses * (self.pcie_h2d_ms + self.gpu_swiglu_ms)
        path_b_total_ms = self.sync_roundtrip_ms + (k_misses * self.cpu_avx2_ms)

        weight_bytes_moved = k_misses * self.expert_weight_bytes
        bytes_saved = weight_bytes_moved - self.total_activation_bytes
        transfer_reduction_pct = (bytes_saved / weight_bytes_moved) * 100.0

        return {
            "k_misses": k_misses,
            "weight_bytes_moved": weight_bytes_moved,
            "activation_bytes_moved": self.total_activation_bytes,
            "gpu_hit_total_ms": round(gpu_hit_total_ms, 3),
            "path_a_upload_gpu_ms": round(path_a_total_ms, 3),
            "path_b_cpu_inplace_ms": round(path_b_total_ms, 3),
            "transfer_reduction_pct": round(transfer_reduction_pct, 2),
            "faster_cold_path": "CPU_IN_PLACE" if path_b_total_ms < path_a_total_ms else "GPU_CACHE_UPLOAD",
        }

    def print_analysis(self):
        print("=" * 72)
        print(" BeyondVRAM: Step 3 Cold-Expert Analytical Model (Calibrated to Hardware)")
        print("=" * 72)
        print(f" Expert Footprint (Q3_K 3-Matrix SwiGLU) : {self.expert_weight_bytes:,} bytes (1.29 MiB)")
        print(f" Activation Roundtrip (D2H + H2D)        : {self.total_activation_bytes:,} bytes (16.0 KiB)")
        print("-" * 72)
        print(f" {'Cold Misses/Layer':<18} | {'GPU Cache Hit':<14} | {'Path A (H2D+GPU)':<16} | {'Path B (CPU+Sync)':<17} | {'Winner'}")
        print("-" * 72)
        for k in [1, 2, 4, 8]:
            r = self.evaluate_layer_misses(k)
            print(
                f" k = {k:<14} | {r['gpu_hit_total_ms']:>9.3f} ms   | "
                f"{r['path_a_upload_gpu_ms']:>11.3f} ms   | {r['path_b_cpu_inplace_ms']:>12.3f} ms   | {r['faster_cold_path']}"
            )
        print("=" * 72)


if __name__ == "__main__":
    router = ZeroTransferFFNRouter()
    router.print_analysis()
