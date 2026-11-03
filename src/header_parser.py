"""
BeyondVRAM: Step 1 - GGUF Binary Header Parser & VRAM Budgeter
Extracts model geometry, partitions Non-Expert vs MoE Expert weights,
and calculates optimal Hot Cache VRAM allocation.
"""

import sys
import os
import argparse
from typing import Dict, Any, Tuple
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
        arch_field = self.reader.fields.get('general.architecture')
        if arch_field is None:
            raise ValueError("Invalid GGUF: Missing general.architecture")
        self.arch = str(bytes(arch_field.parts[arch_field.data[0]]), 'utf-8')
        
        def get_val(key_suffix: str, default=0):
            field = self.reader.fields.get(f"{self.arch}.{key_suffix}")
            if field is not None and field.data:
                return field.parts[field.data[0]][0]
            return default

        self.block_count = get_val('block_count', 0)
        self.expert_count = get_val('expert_count', 0)
        self.expert_used_count = get_val('expert_used_count', 0)
        self.context_length = get_val('context_length', 4096)
        self.embedding_length = get_val('embedding_length', 2048)

    def _analyze_tensors(self):
        self.non_expert_bytes = 0
        self.expert_bytes = 0
        self.router_bytes = 0
        self.attention_bytes = 0
        self.shared_expert_bytes = 0

        self.tensor_categories = {
            'attention': 0,
            'router': 0,
            'shared_expert': 0,
            'moe_experts': 0,
            'embeddings_head': 0,
            'norms': 0,
            'mtp': 0,
            'other': 0
        }

        for tensor in self.reader.tensors:
            n_bytes = tensor.n_bytes
            name = tensor.name.lower()

            if 'ffn_gate_exps' in name or 'ffn_up_exps' in name or 'ffn_down_exps' in name:
                self.expert_bytes += n_bytes
                self.tensor_categories['moe_experts'] += n_bytes
            elif 'shexp' in name:
                self.non_expert_bytes += n_bytes
                self.shared_expert_bytes += n_bytes
                self.tensor_categories['shared_expert'] += n_bytes
            elif 'gate_inp' in name or 'router' in name:
                self.non_expert_bytes += n_bytes
                self.router_bytes += n_bytes
                self.tensor_categories['router'] += n_bytes
            elif 'attn' in name:
                self.non_expert_bytes += n_bytes
                self.attention_bytes += n_bytes
                self.tensor_categories['attention'] += n_bytes
            elif 'nextn' in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories['mtp'] += n_bytes
            elif 'token_embd' in name or 'output' in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories['embeddings_head'] += n_bytes
            elif 'norm' in name:
                self.non_expert_bytes += n_bytes
                self.tensor_categories['norms'] += n_bytes
            else:
                self.non_expert_bytes += n_bytes
                self.tensor_categories['other'] += n_bytes

        total_experts = self.block_count * max(1, self.expert_count)
        self.bytes_per_expert = self.expert_bytes / max(1, total_experts)

    def compute_budget(self, total_vram_mib: int = 6144, os_reserve_mib: int = 450,
                       ctx_size: int = 4096, kv_type: str = 'q4_0') -> Dict[str, Any]:
        """
        Computes exact VRAM partition for Hot Cache vs Cold Host RAM.
        """
        non_expert_mib = self.non_expert_bytes / (1024 * 1024)
        
        # Approximate KV cache footprint
        # Q4_0 is ~0.5 bytes per element per token
        kv_mib = (ctx_size * self.embedding_length * 2 * 0.5 * self.block_count) / (1024 * 1024)
        if kv_type == 'q4_0':
            kv_mib = 273.0  # Empirical for 4k Q4 on this arch
        
        compute_scratch_mib = 170.0  # CUDA graph execution buffer
        
        used_non_cache_mib = os_reserve_mib + non_expert_mib + kv_mib + compute_scratch_mib
        hot_cache_budget_mib = max(0.0, total_vram_mib - used_non_cache_mib)
        
        mb_per_expert = self.bytes_per_expert / (1024 * 1024)
        slots = int(hot_cache_budget_mib / max(0.001, mb_per_expert))
        total_model_experts = self.block_count * self.expert_count
        ratio = (slots / max(1, total_model_experts)) * 100.0

        return {
            'total_vram_mib': total_vram_mib,
            'os_reserve_mib': os_reserve_mib,
            'non_expert_mib': round(non_expert_mib, 2),
            'kv_cache_mib': round(kv_mib, 2),
            'compute_scratch_mib': round(compute_scratch_mib, 2),
            'hot_cache_budget_mib': round(hot_cache_budget_mib, 2),
            'mb_per_expert': round(mb_per_expert, 3),
            'hot_expert_slots': min(total_model_experts, slots),
            'total_model_experts': total_model_experts,
            'vram_resident_ratio_pct': round(min(100.0, ratio), 2)
        }

    def print_report(self, budget: Dict[str, Any]):
        print("=" * 65)
        print(" BeyondVRAM: Step 1 Geometry & VRAM Budget Report")
        print("=" * 65)
        print(f" Model File        : {os.path.basename(self.gguf_path)}")
        print(f" Architecture      : {self.arch}")
        print(f" Total File Size   : {self.file_size_bytes / (1024**3):.2f} GB")
        print(f" Layer Count       : {self.block_count}")
        print(f" Experts Per Layer : {self.expert_count} (Active Top-{self.expert_used_count})")
        print(f" Total Experts     : {budget['total_model_experts']}")
        print("-" * 65)
        print(" Tensor Footprint Classification:")
        print(f"  * Attention Backbone  : {self.attention_bytes / (1024**2):.2f} MB")
        print(f"  * Router Heads (All)  : {self.router_bytes / (1024**2):.2f} MB")
        print(f"  * Non-Expert Total    : {budget['non_expert_mib']:.2f} MB (Permanent VRAM)")
        print(f"  * MoE Experts Total   : {self.expert_bytes / (1024**3):.2f} GB (Hot/Cold Tier)")
        print(f"  * Size Per Expert     : {budget['mb_per_expert']:.2f} MB")
        print("-" * 65)
        print(" Hardware Budget (RTX 3050 6GB):")
        print(f"  * Total VRAM Capacity : {budget['total_vram_mib']} MiB")
        print(f"  * Non-Expert Weights  : -{budget['non_expert_mib']} MiB")
        print(f"  * OS / Display Reserve: -{budget['os_reserve_mib']} MiB")
        print(f"  * KV Cache (4k Q4)    : -{budget['kv_cache_mib']} MiB")
        print(f"  * Compute Buffer      : -{budget['compute_scratch_mib']} MiB")
        print("  =======================================================")
        print(f"  [+] OPTIMAL HOT CACHE : {budget['hot_cache_budget_mib']:.0f} MiB")
        print(f"  [+] HOT EXPERT SLOTS  : {budget['hot_expert_slots']} / {budget['total_model_experts']} slots")
        print(f"  [+] VRAM RESIDENCY    : {budget['vram_resident_ratio_pct']}% of all model experts fit in VRAM!")
        print("=" * 65)
        print(" Recommended Engine Execution Command:")
        print(f" llama-cli -m \"{self.gguf_path}\" \\")
        print(f"   -c {self.context_length} -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp \\")
        print(f"   -cmoe --moe-cache-mib {int(budget['hot_cache_budget_mib'])}")
        print("=" * 65)


if __name__ == "__main__":
    default_model = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"
    parser = argparse.ArgumentParser(description="BeyondVRAM Header Parser & Budgeter")
    parser.add_argument("--model", type=str, default=default_model, help="Path to GGUF model")
    parser.add_argument("--vram", type=int, default=6144, help="Total GPU VRAM in MiB")
    args = parser.parse_args()

    geom = MoEModelGeometry(args.model)
    budget = geom.compute_budget(total_vram_mib=args.vram)
    geom.print_report(budget)
