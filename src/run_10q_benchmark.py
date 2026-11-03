import subprocess
import os
import re
import sys
import json
import time

CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"
MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"
PROMPTS_DIR = r"C:\Users\gaurav\bench_prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)

QUESTIONS = [
    {
        "id": "Q01",
        "category": "Coding (Easy)",
        "title": "LRU Cache (Double-Linked List)",
        "prompt": "Implement a high-performance LRU Cache in Python with O(1) get and put operations using a doubly linked list and hash map. Include edge cases for capacity 1, existing key updates, and eviction order. Provide clean, well-commented code."
    },
    {
        "id": "Q02",
        "category": "Coding (Easy-Med)",
        "title": "Async Token Bucket Rate Limiter",
        "prompt": "Write an asynchronous rate limiter middleware in Python using asyncio and FastAPI. Implement a Token Bucket algorithm that supports per-client IP rate limiting with sliding window replenish and clean thread-safe locking."
    },
    {
        "id": "Q03",
        "category": "Systems (Medium)",
        "title": "C++20 Arena Memory Allocator",
        "prompt": "Write a C++20 custom memory pool arena allocator with std::pmr integration. Implement a contiguous memory block that resets in O(1) time, enforces 64-byte alignment for SIMD operations, and handles thread-local arenas for low-latency trading."
    },
    {
        "id": "Q04",
        "category": "Agentic Work (Medium)",
        "title": "ReAct Autonomous Tool-Calling Loop",
        "prompt": "Design and implement in Python a robust ReAct (Reasoning + Acting) autonomous agent core loop. It must dynamically discover available tools, inspect JSON schemas, execute multi-step tool calls with error handling, and implement an exponential backoff retry gate."
    },
    {
        "id": "Q05",
        "category": "Distributed Systems (Medium-Hard)",
        "title": "Raft Leader Election Protocol",
        "prompt": "Implement a core Raft leader election mechanism in Go. Detail the state transitions (Follower, Candidate, Leader), randomized election timeouts, RequestVote RPC dispatch, vote tallying, and term increment logic."
    },
    {
        "id": "Q06",
        "category": "GPU Systems (Hard)",
        "title": "CUDA Fused FlashAttention Kernel",
        "prompt": "Explain and write a CUDA C++ kernel outline for fused FlashAttention forward pass. Detail shared memory tiling of Q, K, V blocks, online Welford/Milakov softmax normalization without materializing the N x N attention matrix, and write back to HBM."
    },
    {
        "id": "Q07",
        "category": "Agentic Work (Hard)",
        "title": "Hierarchical Multi-Agent Orchestration",
        "prompt": "Architect an enterprise-grade hierarchical multi-agent framework in Python. Include a Supervisor Manager Agent that decomposes complex tasks into DAGs, delegates subtasks to specialized worker agents via an async message bus, and validates output against acceptance criteria."
    },
    {
        "id": "Q08",
        "category": "AI Systems (Hard)",
        "title": "Tree Speculative Decoding Verifier",
        "prompt": "Write a speculative decoding verification engine in Python. Formulate how multiple draft tokens generated from a candidate tree are validated in parallel against the target model's logits, implementing greedy or top-p token acceptance and KV cache rollback."
    },
    {
        "id": "Q09",
        "category": "Agentic Work (Hard)",
        "title": "Autonomous Self-Healing CI/CD Agent",
        "prompt": "Design an autonomous self-healing code agent pipeline. The agent ingests CI/CD test failure stack traces and git diffs, queries the codebase AST to locate root cause functions, generates candidate bugfix patches, and validates them in an isolated test harness."
    },
    {
        "id": "Q10",
        "category": "Compiler / SIMD (Very Hard)",
        "title": "AVX2 4-Bit Dequantization Microkernel",
        "prompt": "Implement an optimized AVX2 SIMD matrix-vector multiplication (GEMV) microkernel in C++ for Q4_0 quantized weights. Detail unpacking 4-bit nibbles into 8-bit integers, multiplying with fp32 activations, and accumulating using FMA instructions."
    }
]

N_GEN = 250

def parse_metrics(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    draft_match = re.search(r'draft acceptance =\s+([\d\.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\)', text)
    cache_match = re.search(r'llama_moe_cache:.*hits = (\d+), misses = (\d+), hit rate = ([\d\.]+)%, uploaded = ([\d\.]+) MiB', text)

    tg_data = tg_match[-1] if tg_match else None

    return {
        'pp_tokens': int(pp_match.group(2)) if pp_match else 0,
        'pp_speed': float(pp_match.group(3)) if pp_match else 0.0,
        'tg_tokens': int(tg_data[1]) if tg_data else 0,
        'tg_speed': float(tg_data[2]) if tg_data else 0.0,
        'draft_rate': float(draft_match.group(1)) * 100.0 if draft_match else 0.0,
        'cache_rate': float(cache_match.group(3)) if cache_match else 0.0,
        'cache_uploaded_mib': float(cache_match.group(4)) if cache_match else 0.0
    }

results = []

print("=========================================================================================", flush=True)
print(" BeyondVRAM 10-Question Deep Stress-Test Suite (128K Context | 3300MB Cache | MTP)", flush=True)
print(" Range: Coding & Agentic Work (Easy to Very Hard) | 250 Tokens Output Per Question", flush=True)
print("=========================================================================================", flush=True)

total_t0 = time.time()

for idx, q in enumerate(QUESTIONS):
    q_file = os.path.join(PROMPTS_DIR, f"{q['id']}.txt")
    with open(q_file, "w", encoding="utf-8") as f:
        f.write(q["prompt"])

    log_file = rf"C:\Users\gaurav\bench_{q['id']}.log"
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
        "-f", q_file,
        "-n", str(N_GEN),
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]

    print(f"\n[{idx+1}/10] Running {q['id']}: {q['title']} ({q['category']})...", flush=True)
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    metrics = parse_metrics(log_file)
    q_res = {
        "id": q["id"],
        "category": q["category"],
        "title": q["title"],
        "pp_tokens": metrics["pp_tokens"],
        "pp_speed": metrics["pp_speed"],
        "tg_tokens": metrics["tg_tokens"],
        "tg_speed": metrics["tg_speed"],
        "cache_rate": metrics["cache_rate"],
        "draft_rate": metrics["draft_rate"],
        "elapsed_s": elapsed
    }
    results.append(q_res)
    print(f"      -> Completed in {elapsed:.1f}s | PP: {metrics['pp_speed']:.1f} t/s | Decode: {metrics['tg_speed']:.1f} t/s | Cache Hit: {metrics['cache_rate']:.1f}% | MTP: {metrics['draft_rate']:.1f}%", flush=True)

total_elapsed = time.time() - total_t0

print("\n" + "="*95, flush=True)
print(f" BENCHMARK SUMMARY (Total Suite Time: {total_elapsed:.1f}s)", flush=True)
print("="*95, flush=True)
print(f"{'ID':<4} | {'Title':<35} | {'Category':<15} | {'PP (t/s)':<8} | {'Decode':<8} | {'Cache Hit':<9} | {'MTP Acc'}", flush=True)
print("-" * 95, flush=True)
for r in results:
    print(f"{r['id']:<4} | {r['title']:<35} | {r['category']:<15} | {r['pp_speed']:<8.1f} | {r['tg_speed']:<8.1f} | {r['cache_rate']:<7.1f}%  | {r['draft_rate']:<6.1f}%", flush=True)

avg_pp = sum(r["pp_speed"] for r in results) / len(results)
avg_tg = sum(r["tg_speed"] for r in results) / len(results)
avg_cache = sum(r["cache_rate"] for r in results) / len(results)
avg_mtp = sum(r["draft_rate"] for r in results) / len(results)

print("-" * 95, flush=True)
print(f"AVERAGE OVER 10 QUESTIONS: | PP: {avg_pp:.1f} t/s | Decode: {avg_tg:.1f} t/s | Cache Hit: {avg_cache:.1f}% | MTP Acc: {avg_mtp:.1f}%", flush=True)
print("="*95, flush=True)

with open(r"C:\Users\gaurav\benchmark_10q_results.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "summary": {"avg_pp": avg_pp, "avg_tg": avg_tg, "avg_cache": avg_cache, "avg_mtp": avg_mtp, "total_time": total_elapsed}}, f, indent=2)

print("\nFull benchmark JSON saved to C:\\Users\\gaurav\\benchmark_10q_results.json", flush=True)
