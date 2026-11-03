import subprocess
import os
import re
import sys
import json
import time

CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"
MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"

# Tested context sizes and calculated safe MoE cache size for 6000 MiB available VRAM
# Total Budget = 6000 MiB
# Base (Backbone 1245 MiB + Compute 150 MiB) = ~1400 MiB
# Budget for (KV Cache + MoE Cache) = 4600 MiB
TEST_CONFIGS = [
    {"ctx": 4096,   "cache_mib": 4400, "label": "4K (Standard)"},
    {"ctx": 8192,   "cache_mib": 4300, "label": "8K (Double)"},
    {"ctx": 16384,  "cache_mib": 4100, "label": "16K (Document)"},
    {"ctx": 32768,  "cache_mib": 3700, "label": "32K (Large Doc)"},
    {"ctx": 65536,  "cache_mib": 2900, "label": "64K (Book/Codebase)"},
    {"ctx": 131072, "cache_mib": 1400, "label": "128K (Ultra-Long)"},
]

results = []

print("======================================================================", flush=True)
print(" BeyondVRAM: Full Context Scaling Benchmark (RTX 3050 6GB + iGPU)", flush=True)
print(" Available VRAM: ~6000 MiB (Intel iGPU driving Windows display)", flush=True)
print("======================================================================", flush=True)

for cfg in TEST_CONFIGS:
    ctx = cfg["ctx"]
    cache_mib = cfg["cache_mib"]
    label = cfg["label"]
    
    print(f"\n[*] Evaluating Context: {ctx:,} tokens ({label})...", flush=True)
    print(f"    Configured MoE Cache: {cache_mib} MiB", flush=True)
    
    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", str(ctx),
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "-cmoe",
        "--moe-cache-mib", str(cache_mib),
        "-p", "Context test.",
        "-n", "5",
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]
    
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore", stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    
    output = proc.stderr + "\n" + proc.stdout
    
    kv_match = re.search(r'CUDA0 KV buffer size\s*=\s*([\d\.]+)\s*MiB', output)
    mem_match = re.search(r'CUDA0.*\|.*?6143\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)\s*\+\s*([-\d]+)', output)
    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', output)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', output)
    
    kv_size = float(kv_match.group(1)) if kv_match else 0.0
    allocated = int(mem_match.group(2)) if mem_match else 0
    free_vram = int(mem_match.group(6)) if mem_match else 0
    tg_speed = float(tg_match[-1][2]) if tg_match else 0.0
    
    print(f"    -> KV Cache Footprint: {kv_size:.1f} MiB", flush=True)
    print(f"    -> VRAM Allocated:     {allocated} MiB / 6143 MiB", flush=True)
    print(f"    -> Free Headroom:      {free_vram} MiB", flush=True)
    print(f"    -> Status:             {'SUCCESS (Clean VRAM)' if free_vram >= 0 else 'OVERCOMMITTED'}", flush=True)
    
    results.append({
        "context": ctx,
        "label": label,
        "kv_size_mib": kv_size,
        "moe_cache_mib": cache_mib,
        "allocated_mib": allocated,
        "free_vram_mib": free_vram,
        "success": free_vram >= 0
    })

print("\n" + "="*80, flush=True)
print(" SUMMARY: CONTEXT SCALING ENVELOPE FOR GROOT-19B ON RTX 3050 (6GB)", flush=True)
print("="*80, flush=True)
print(f"{'Context Window':<16} | {'KV Size (Q4)':<14} | {'MoE Cache':<12} | {'VRAM Used':<12} | {'Free VRAM':<10} | {'Status'}", flush=True)
print("-" * 80, flush=True)
for r in results:
    status_str = "OK (Zero Thrashing)" if r["success"] else "Oversubscribed"
    print(f"{r['context']:<16,d} | {r['kv_size_mib']:<10.1f} MiB | {r['moe_cache_mib']:<8d} MiB | {r['allocated_mib']:<8d} MiB | {r['free_vram_mib']:<6d} MiB | {status_str}", flush=True)

with open(r"C:\Users\gaurav\context_scaling_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to C:\\Users\\gaurav\\context_scaling_results.json", flush=True)
