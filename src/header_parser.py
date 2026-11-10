"""
BeyondVRAM: Step 1 - GGUF Binary Header Parser & VRAM Budgeter
Extracts model geometry from GGUF metadata, partitions Non-Expert vs. MoE Expert
weights, accounts for hybrid Recurrent (SSM) + Grouped-Query Attention (GQA) KV
cache footprints, and calculates optimal SLRU Hot Cache VRAM allocation.
"""

import os
import argparse
from typing import Dict, Any
import gguf


class MoEModelGeometry:
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"Model file not found: {gguf_path}")

        self.reader = gguf.GGUFReader(gguf_path)
        self.file_size_bytes = os.path.getsize(gguf_path)
        self._parse_metadata()
        self._analyze_tensors()

    def _parse_metadata(self):
        arch_field = self.reader.fields.get("general.architecture")
        if arch_field is None:
            raise ValueError("Invalid GGUF: Missing general.architecture")
        self.arch = str(bytes(arch_field.parts[arch_field.data[0]]), "utf-8")

        def get_val(key_suffix: str, default=0):
            field = self.reader.fields.get(f"{self.arch}.{key_suffix}")
            if field is not None and field.data:
                return int(field.parts[field.data[0]][0])
            return default

        self.block_count = get_val("block_count", 0)
        self.nextn_layers = get_val("nextn_predict_layers", 0)
        self.total_layers = self.block_count + self.nextn_layers
        self.expert_count = get_val("expert_count", 0)
        self.expert_used_count = get_val("expert_used_count", 0)
        self.context_length = get_val("context_length", 4096)
        self.embedding_length = get_val("embedding_length", 2048)
        self.head_count = get_val("attention.head_count", 16)
        self.head_count_kv = get_val("attention.head_count_kv", 2)
        self.key_length = get_val("attention.key_length", 128)
        self.value_length = get_val("attention.value_length", 128)
        self.full_attn_interval = get_val("full_attention_interval", 1)

        if self.full_attn_interval > 1:
            self.n_full_attn_layers = self.block_count // self.full_attn_interval
            self.n_recurrent_layers = self.block_count - self.n_full_attn_layers
        else:
            self.n_full_attn_layers = self.block_count
            self.n_recurrent_layers = 0

    def _analyze_tensors(self):
        self.non_expert_bytes = 0
        self.expert_bytes = 0
        self.router_bytes = 0
        self.attention_bytes = 0
        self.shared_expert_bytes = 0
        self.moe_layer_indices = set()

        self.tensor_categories = {
            "attention": 0,
            "router": 0,
            "shared_expert": 0,
            "moe_experts": 0,
            "embeddings_head": 0,
            "norms": 0,
            "mtp": 0,
            "other": 0,
        }

        for tensor in self.reader.tensors:
            n_bytes = tensor.n_bytes
            name = tensor.name.lower()

            if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name or "ffn_gate_up_exps" in name:
                self.expert_bytes += n_bytes
                self.tensor_categories["moe_experts"] += n_bytes
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    self.moe_layer_indices.add(int(parts[1]))
            elif "shexp" in name:
                self.non_expert_bytes += n_bytes
                self.shared_expert_bytes += n_bytes
                self.tensor_categories["shared_expert"] += n_bytes
            elif "gate_inp" in name or "router" in name:
                self.non_expert_bytes += n_bytes
                self.router_bytes += n_bytes
                self.tensor_categories["router"] += n_bytes
            elif "attn" in name or "ssm" in name:
                self.non_expert_bytes += n_bytes
                self.attention_bytes += n_bytes
                self.tensor_categories["attention"] += n_bytes
            elif "nextn" in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories["mtp"] += n_bytes
            elif "token_embd" in name or "output" in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories["embeddings_head"] += n_bytes
            elif "norm" in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories["norms"] += n_bytes
            else:
                self.non_expert_bytes += n_bytes
                self.tensor_categories["other"] += n_bytes

        self.n_moe_layers = len(self.moe_layer_indices) if self.moe_layer_indices else self.total_layers
        total_experts = self.n_moe_layers * max(1, self.expert_count)
        self.bytes_per_expert = self.expert_bytes / max(1, total_experts)

    def compute_budget(
        self,
        total_vram_mib: int = 6143,
        os_reserve_mib: int = 364,
        ctx_size: int = 131072,
        kv_type: str = "q4_0",
    ) -> Dict[str, Any]:
        """
        Computes exact VRAM partition matching llama.cpp's memory allocator:
        - Non-expert weights on GPU (or CPU-mapped MTP embeddings)
        - Full-attention KV cache buffer (scaled by ctx_size and full_attention_interval)
        - Recurrent / SSM state buffer (fixed ~256 MiB for hybrid Qwen3.5-MoE layers)
        - Compute & graph reservation (~471 MiB)
        """
        non_expert_mib = self.non_expert_bytes / (1024 * 1024)

        if kv_type == "q4_0":
            bpe = 0.5625  # 18 bytes per 32 elements
        elif kv_type == "q8_0":
            bpe = 1.0625  # 34 bytes per 32 elements
        else:
            bpe = 2.0

        # Full-attention KV cache bytes across full-attention layers
        kv_bytes_per_layer = ctx_size * self.head_count_kv * (self.key_length + self.value_length) * bpe
        kv_mib = (self.n_full_attn_layers * kv_bytes_per_layer) / (1024 * 1024)

        # Recurrent / SSM state buffer for hybrid recurrent layers (~256 MiB in llama.cpp)
        ssm_state_mib = 256.0 if self.n_recurrent_layers > 0 else 0.0
        total_context_mib = kv_mib + ssm_state_mib

        compute_scratch_mib = 471.0
        # Note: in llama.cpp with -cmoe, ~208 MiB of token embeddings/MTP stays CPU-mapped,
        # leaving ~1037 MiB of non-expert weights resident in CUDA0.
        cuda_non_expert_mib = non_expert_mib - 208.4 if non_expert_mib > 1100 else non_expert_mib

        used_non_cache_mib = os_reserve_mib + cuda_non_expert_mib + total_context_mib + compute_scratch_mib
        hot_cache_budget_mib = max(0.0, total_vram_mib - used_non_cache_mib)

        mb_per_expert = self.bytes_per_expert / (1024 * 1024)
        slots = int(hot_cache_budget_mib / max(0.001, mb_per_expert))
        total_model_experts = self.n_moe_layers * self.expert_count
        ratio = (slots / max(1, total_model_experts)) * 100.0

        return {
            "total_vram_mib": total_vram_mib,
            "os_reserve_mib": os_reserve_mib,
            "non_expert_total_mib": round(non_expert_mib, 2),
            "cuda_non_expert_mib": round(cuda_non_expert_mib, 2),
            "ctx_size": ctx_size,
            "kv_cache_mib": round(kv_mib, 2),
            "ssm_state_mib": round(ssm_state_mib, 2),
            "total_context_mib": round(total_context_mib, 2),
            "compute_scratch_mib": round(compute_scratch_mib, 2),
            "hot_cache_budget_mib": round(hot_cache_budget_mib, 2),
            "mb_per_expert": round(mb_per_expert, 3),
            "bytes_per_expert": int(round(self.bytes_per_expert)),
            "hot_expert_slots": min(total_model_experts, slots),
            "total_model_experts": total_model_experts,
            "vram_resident_ratio_pct": round(min(100.0, ratio), 2),
        }

    def print_report(self, budget: Dict[str, Any]):
        print("=" * 68)
        print(" BeyondVRAM: Step 1 GGUF Geometry & VRAM Budget Report")
        print("=" * 68)
        print(f" Model File          : {os.path.basename(self.gguf_path)}")
        print(f" Architecture        : {self.arch} ({self.n_full_attn_layers} Full-Attn + {self.n_recurrent_layers} Recurrent/SSM + {self.nextn_layers} MTP)")
        print(f" Total File Size     : {self.file_size_bytes / (1024**3):.2f} GiB")
        print(f" MoE Layer Count     : {self.n_moe_layers} layers x {self.expert_count} experts (Top-{self.expert_used_count} active)")
        print(f" Total Model Experts : {budget['total_model_experts']:,}")
        print("-" * 68)
        print(" Tensor Footprint Classification:")
        print(f"  * Attention + SSM     : {self.attention_bytes / (1024**2):.2f} MiB")
        print(f"  * Router Gate Heads   : {self.router_bytes / (1024**2):.2f} MiB")
        print(f"  * Non-Expert Total    : {budget['non_expert_total_mib']:.2f} MiB ({budget['cuda_non_expert_mib']:.2f} MiB on CUDA0)")
        print(f"  * MoE Experts Total   : {self.expert_bytes / (1024**2):.2f} MiB ({self.expert_bytes / (1024**3):.2f} GiB in Host RAM)")
        print(f"  * Single Expert Size  : {budget['mb_per_expert']:.3f} MiB ({budget['bytes_per_expert']:,} bytes)")
        print("-" * 68)
        print(f" Hardware Budget (RTX 3050 {budget['total_vram_mib']} MiB, Context = {budget['ctx_size']:,} tokens):")
        print(f"  * Total VRAM Capacity :  {budget['total_vram_mib']:>7.1f} MiB")
        print(f"  * Non-Expert Weights  : -{budget['cuda_non_expert_mib']:>7.1f} MiB")
        print(f"  * KV Cache (Q4_0)     : -{budget['kv_cache_mib']:>7.1f} MiB ({self.n_full_attn_layers} full-attn layers)")
        print(f"  * SSM Recurrent State : -{budget['ssm_state_mib']:>7.1f} MiB ({self.n_recurrent_layers} recurrent layers)")
        print(f"  * Compute Buffers     : -{budget['compute_scratch_mib']:>7.1f} MiB")
        print(f"  * OS / WDDM Reserve   : -{budget['os_reserve_mib']:>7.1f} MiB")
        print("  ------------------------------------------------------------------")
        print(f"  [+] MOE CACHE BUDGET  :  {budget['hot_cache_budget_mib']:>7.0f} MiB (--moe-cache-mib)")
        print(f"  [+] CACHE CAPACITY    :  {budget['hot_expert_slots']:,} / {budget['total_model_experts']:,} experts ({budget['vram_resident_ratio_pct']}%)")
        print("=" * 68)


if __name__ == "__main__":
    default_model = os.environ.get(
        "BEYONDVRAM_MODEL",
        r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf",
    )
    parser = argparse.ArgumentParser(description="BeyondVRAM GGUF Header Parser & VRAM Budgeter")
    parser.add_argument("--model", type=str, default=default_model, help="Path to GGUF model")
    parser.add_argument("--vram", type=int, default=6143, help="Total GPU VRAM in MiB")
    parser.add_argument("--ctx", type=int, default=131072, help="Context length in tokens")
    args = parser.parse_args()

    geom = MoEModelGeometry(args.model)
    for ctx in [4096, 131072] if args.ctx == 131072 else [args.ctx]:
        budget = geom.compute_budget(total_vram_mib=args.vram, ctx_size=ctx)
        geom.print_report(budget)
        print()
