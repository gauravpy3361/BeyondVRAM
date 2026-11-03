import subprocess
import os
import re
import json
import time

MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"
CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"

PROMPTS = [
    ("Q1 (Memory Wall & Hardware)", r"C:\Users\gaurav\prompt_q1.txt"),
    ("Q2 (Distributed 3D Parallelism)", r"C:\Users\gaurav\prompt_q2.txt"),
    ("Q3 (Compiler IRs & Microkernels)", r"C:\Users\gaurav\prompt_q3.txt")
]

N_GEN = 400

def parse_metrics(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    total_match = re.search(r'total time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens', text)
    draft_match = re.search(r'draft acceptance =\s+([\d\.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\)', text)
    cache_match = re.search(r'llama_moe_cache:.*hits = (\d+), misses = (\d+), hit rate = ([\d\.]+)%, uploaded = ([\d\.]+) MiB', text)

    tg_data = tg_match[-1] if tg_match else None

    return {
        'pp_tokens': int(pp_match.group(2)) if pp_match else 0,
        'pp_time_ms': float(pp_match.group(1)) if pp_match else 0.0,
        'pp_speed': float(pp_match.group(3)) if pp_match else 0.0,
        'tg_tokens': int(tg_data[1]) if tg_data else 0,
        'tg_time_ms': float(tg_data[0]) if tg_data else 0.0,
        'tg_speed': float(tg_data[2]) if tg_data else 0.0,
        'total_time_ms': float(total_match.group(1)) if total_match else 0.0,
        'draft_rate': float(draft_match.group(1)) * 100.0 if draft_match else 0.0,
        'draft_acc': int(draft_match.group(2)) if draft_match else 0,
        'draft_gen': int(draft_match.group(3)) if draft_match else 0,
        'cache_hits': int(cache_match.group(1)) if cache_match else 0,
        'cache_misses': int(cache_match.group(2)) if cache_match else 0,
        'cache_rate': float(cache_match.group(3)) if cache_match else 0.0,
        'cache_uploaded_mib': float(cache_match.group(4)) if cache_match else 0.0
    }

hybrid_results = []

for q_idx, (q_name, prompt_file) in enumerate(PROMPTS):
    print(f"\n==================================================================")
    print(f" >>> RUNNING HYBRID STAGING ON QUESTION {q_idx+1}/3: {q_name}")
    print(f"==================================================================")

    log_file = rf"C:\Users\gaurav\hybrid_q{q_idx+1}.log"
    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "4096",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--spec-type", "draft-mtp",
        "-ngl", "99",
        "-ncmoe", "31",
        "--moe-cache-mib", "1900",
        "-f", prompt_file,
        "-n", str(N_GEN),
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    metrics = parse_metrics(log_file)
    metrics["question"] = q_name
    metrics["elapsed"] = elapsed
    hybrid_results.append(metrics)
    print(f"     -> Finished in {elapsed:.1f}s")
    print(f"     -> PP: {metrics['pp_speed']} t/s | Decode: {metrics['tg_speed']} t/s | Cache Hit: {metrics['cache_rate']:.1f}% | MTP Acc: {metrics['draft_rate']:.1f}%")

with open(r"C:\Users\gaurav\hybrid_bench_results.json", "w", encoding="utf-8") as f:
    json.dump(hybrid_results, f, indent=2)

print("\nHybrid Staging Deep Benchmark completed!")
