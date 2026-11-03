import subprocess
import os
import re
import json
import time

CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"
MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"

PROMPTS = [
    {
        "id": "Q1",
        "title": "Distributed 3D Parallelism & Infrastructure",
        "domain": "Exascale Distributed AI Systems",
        "file": r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q1_distributed.txt",
        "log": r"C:\Users\gaurav\bench_1k_q1.log",
        "done": True,
        "pp_tokens": 916,
        "pp_speed": 61.9,
        "tg_tokens": 1000,
        "tg_speed": 23.6,
        "cache_rate": 87.8,
        "draft_rate": 46.0,
        "elapsed": 69.3
    },
    {
        "id": "Q2",
        "title": "Autonomous Agent Operating System (AgentOS)",
        "domain": "Agentic Kernel & Concurrency Architecture",
        "file": r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q2_agent_os.txt",
        "log": r"C:\Users\gaurav\bench_1k_q2.log",
        "done": False
    },
    {
        "id": "Q3",
        "title": "Fused FlashAttention & Quantized MoE Kernel",
        "domain": "LLVM IR, Polyhedral Loops & CUDA PTX",
        "file": r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q3_compiler_simd.txt",
        "log": r"C:\Users\gaurav\bench_1k_q3.log",
        "done": False
    }
]

def parse_metrics(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    cache_match = re.search(r'llama_moe_cache:.*hits = (\d+), misses = (\d+), hit rate = ([\d\.]+)%, uploaded = ([\d\.]+) MiB', text)
    draft_match = re.search(r'draft acceptance =\s+([\d\.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\)', text)

    return {
        'pp_tokens': int(pp_match.group(2)) if pp_match else 0,
        'pp_speed': float(pp_match.group(3)) if pp_match else 0.0,
        'tg_tokens': int(tg_match[-1][1]) if tg_match else 0,
        'tg_speed': float(tg_match[-1][2]) if tg_match else 0.0,
        'cache_rate': float(cache_match.group(3)) if cache_match else 0.0,
        'draft_rate': float(draft_match.group(1))*100.0 if draft_match else 0.0,
    }

results = []

for q in PROMPTS:
    if q["done"]:
        print(f"[*] {q['id']}: {q['title']} already completed.")
        results.append(q)
        continue

    print(f"\n==================================================================")
    print(f"[*] Running {q['id']}: {q['title']}")
    print(f"    Domain: {q['domain']}")
    print(f"    Target: 1,000 Tokens Prompt Ingest -> 1,000 Tokens Decode Output")
    print(f"==================================================================")

    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "131072",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--spec-type", "draft-mtp",
        "-cmoe",
        "--moe-cache-mib", "3300",
        "-f", q["file"],
        "-n", "1000",
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]

    t0 = time.time()
    with open(q["log"], "w", encoding="utf-8") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    metrics = parse_metrics(q["log"])
    q["pp_tokens"] = metrics["pp_tokens"]
    q["pp_speed"] = metrics["pp_speed"]
    q["tg_tokens"] = metrics["tg_tokens"]
    q["tg_speed"] = metrics["tg_speed"]
    q["cache_rate"] = metrics["cache_rate"]
    q["draft_rate"] = metrics["draft_rate"]
    q["elapsed"] = elapsed
    results.append(q)

    print(f"    -> Finished in {elapsed:.1f}s")
    print(f"    -> Prompt: {metrics['pp_tokens']} tokens @ {metrics['pp_speed']:.1f} t/s")
    print(f"    -> Decode: {metrics['tg_tokens']} tokens @ {metrics['tg_speed']:.1f} t/s")
    print(f"    -> Cache Hit: {metrics['cache_rate']:.1f}% | MTP Accept: {metrics['draft_rate']:.1f}%")

print("\n" + "="*95)
print(" 3x (1K PROMPT x 1K DECODE) BENCHMARK SCORECARD")
print("="*95)
print(f"{'ID':<4} | {'Domain':<35} | {'Prompt':<12} | {'Decode':<12} | {'Cache Hit':<10} | {'MTP Acc'}")
print("-" * 95)
for r in results:
    print(f"{r['id']:<4} | {r['domain']:<35} | {r['pp_tokens']} tok ({r['pp_speed']:.1f} t/s) | {r['tg_tokens']} tok ({r['tg_speed']:.1f} t/s) | {r['cache_rate']:.1f}%      | {r['draft_rate']:.1f}%")

avg_pp = sum(r["pp_speed"] for r in results) / len(results)
avg_tg = sum(r["tg_speed"] for r in results) / len(results)
avg_cache = sum(r["cache_rate"] for r in results) / len(results)
avg_draft = sum(r["draft_rate"] for r in results) / len(results)
total_tokens = sum(r["tg_tokens"] for r in results)

print("-" * 95)
print(f"AVERAGES: | Decode: {avg_tg:.1f} t/s | PP: {avg_pp:.1f} t/s | Cache Hit: {avg_cache:.1f}% | MTP: {avg_draft:.1f}% | Total: {total_tokens} tokens")
print("="*95)

with open(r"C:\Users\gaurav\bench_3x1k_results.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "summary": {"avg_pp": avg_pp, "avg_tg": avg_tg, "avg_cache": avg_cache, "avg_draft": avg_draft}}, f, indent=2)

print("\nSaved full 3x1k results to C:\\Users\\gaurav\\bench_3x1k_results.json")
