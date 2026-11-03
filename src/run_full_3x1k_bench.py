import subprocess
import os
import re
import json
import time

CLI_PATH = os.environ.get("LLAMA_CLI_PATH", r"C:\ai\llama.cpp\llama-cli.exe")
MODEL_PATH = os.environ.get(
    "BEYONDVRAM_MODEL",
    r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf",
)
OUT_JSON = os.path.join("benchmarks", "results_3x1k.json")
LOG_DIR = os.path.join("benchmarks", "logs")

PROMPTS = [
    {
        "id": "Q1",
        "title": "Distributed 3D Parallelism & Infrastructure",
        "domain": "Exascale Distributed AI Systems",
        "file": os.path.join("benchmarks", "prompts", "prompt_1k_q1_distributed.txt"),
        "log": os.path.join(LOG_DIR, "bench_1k_q1.log"),
        "done": False,
    },
    {
        "id": "Q2",
        "title": "Autonomous Agent Operating System (AgentOS)",
        "domain": "Agentic Kernel & Concurrency Architecture",
        "file": os.path.join("benchmarks", "prompts", "prompt_1k_q2_agent_os.txt"),
        "log": os.path.join(LOG_DIR, "bench_1k_q2.log"),
        "done": False,
    },
    {
        "id": "Q3",
        "title": "Fused FlashAttention & Quantized MoE Kernel",
        "domain": "LLVM IR, Polyhedral Loops & CUDA PTX",
        "file": os.path.join("benchmarks", "prompts", "prompt_1k_q3_compiler_simd.txt"),
        "log": os.path.join(LOG_DIR, "bench_1k_q3.log"),
        "done": False,
    },
]


def parse_metrics(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    timings = re.findall(r'"timings":(\{[^\}]+\})', text)
    t_obj = json.loads(timings[-1]) if timings else {}

    cache_small = re.search(
        r"llama_moe_cache: ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d\.]+)%, uploaded = ([\d\.]+) MiB",
        text,
    )
    cache_large = re.search(
        r"llama_moe_cache: large batches: (\d+) experts copied from the cache, ([\d\.]+) MiB",
        text,
    )

    draft_n = t_obj.get("draft_n", 0)
    draft_acc = t_obj.get("draft_n_accepted", 0)

    return {
        "pp_tokens": t_obj.get("prompt_n", 0),
        "pp_ms": round(t_obj.get("prompt_ms", 0.0), 2),
        "pp_speed": round(t_obj.get("prompt_per_second", 0.0), 2),
        "tg_tokens": t_obj.get("predicted_n", 0),
        "tg_ms": round(t_obj.get("predicted_ms", 0.0), 2),
        "tg_speed": round(t_obj.get("predicted_per_second", 0.0), 2),
        "cache_hits_decode": int(cache_small.group(1)) if cache_small else 0,
        "cache_misses_decode": int(cache_small.group(2)) if cache_small else 0,
        "cache_rate": float(cache_small.group(3)) if cache_small else 0.0,
        "cache_uploaded_mib": float(cache_small.group(4)) if cache_small else 0.0,
        "prefill_cache_copied_experts": int(cache_large.group(1)) if cache_large else 0,
        "prefill_cache_copied_mib": float(cache_large.group(2)) if cache_large else 0.0,
        "draft_accepted": draft_acc,
        "draft_n": draft_n,
        "draft_rate": round((draft_acc / draft_n) * 100.0, 2) if draft_n > 0 else 0.0,
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    results = []

    for q in PROMPTS:
        print(f"\n==================================================================")
        print(f"[*] Running {q['id']}: {q['title']}")
        print(f"    Domain: {q['domain']}")
        print(f"    Target: ~1,000 Tokens Prompt Ingest -> 1,000 Tokens Decode Output")
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
            "-v",
        ]

        t0 = time.time()
        with open(q["log"], "w", encoding="utf-8") as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        elapsed = round(time.time() - t0, 2)

        metrics = parse_metrics(q["log"])
        q.update(metrics)
        q["done"] = True
        q["elapsed"] = elapsed
        results.append(q)

        print(f"    -> Finished in {elapsed:.1f}s")
        print(f"    -> Prompt: {metrics['pp_tokens']} tokens @ {metrics['pp_speed']:.2f} t/s")
        print(f"    -> Decode: {metrics['tg_tokens']} tokens @ {metrics['tg_speed']:.2f} t/s")
        print(f"    -> Cache Hit: {metrics['cache_rate']:.2f}% | MTP Accept: {metrics['draft_rate']:.1f}%")

    avg_pp = round(sum(r["pp_speed"] for r in results) / len(results), 2)
    avg_tg = round(sum(r["tg_speed"] for r in results) / len(results), 2)
    avg_cache = round(sum(r["cache_rate"] for r in results) / len(results), 2)
    avg_draft = round(sum(r["draft_rate"] for r in results) / len(results), 2)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "summary": {
                    "avg_pp": avg_pp,
                    "avg_tg": avg_tg,
                    "avg_cache": avg_cache,
                    "avg_draft": avg_draft,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved full 3x1k results to {OUT_JSON}")


if __name__ == "__main__":
    main()
