"""
BeyondVRAM: Controlled Deterministic Benchmark Suite
Evaluates Default `llama.cpp` vs. BeyondVRAM Tuned configurations (MTP On & Non-MTP)
under deterministic greedy decoding (`--temp 0`) on identical token sequences:
  1. Default `llama.cpp` (`-fit on`, default 1024 MiB margin -> 7 layers GPU, 35 CPU MoE)
  2. BeyondVRAM Tuned — MTP On (`-cmoe --moe-cache-mib 3300 --spec-type draft-mtp`)
  3. BeyondVRAM Hybrid — MTP On (`-fit off -ngl 99 -ncmoe 31 --moe-cache-mib 1650 --spec-type draft-mtp`)
  4. BeyondVRAM Tuned — Non-MTP (`-cmoe --moe-cache-mib 3300 --spec-type none`, `ubatch=1`)
"""

import os
import re
import json
import time
import statistics
import subprocess

CLI_PATH = os.environ.get("LLAMA_CLI_PATH", r"C:\ai\llama.cpp\llama-cli.exe")
MODEL_PATH = os.environ.get(
    "BEYONDVRAM_MODEL",
    r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf",
)
PROMPT_PATH = os.path.join("benchmarks", "prompts", "prompt_1k_q1_distributed.txt")
OUT_JSON = os.path.join("benchmarks", "ablation_results.json")
N_GEN = 200

CONFIGS = [
    {
        "id": "autofit_default",
        "label": "Default llama.cpp (-fit on, MTP On)",
        "args": ["-fit", "on", "--spec-type", "draft-mtp"],
        "repeats": 3,
    },
    {
        "id": "slru_cache_3300",
        "label": "BeyondVRAM Tuned (3300 MiB Cache, MTP On)",
        "args": ["-cmoe", "--moe-cache-mib", "3300", "--spec-type", "draft-mtp"],
        "repeats": 3,
    },
    {
        "id": "hybrid_ncmoe31_c1650",
        "label": "BeyondVRAM Hybrid (Static + 1650 MiB Cache, MTP On)",
        "args": ["-fit", "off", "-ngl", "99", "-ncmoe", "31", "--moe-cache-mib", "1650", "--spec-type", "draft-mtp"],
        "repeats": 3,
    },
    {
        "id": "slru_cache_3300_no_mtp",
        "label": "BeyondVRAM Tuned (3300 MiB Cache, Non-MTP)",
        "args": ["-cmoe", "--moe-cache-mib", "3300", "--spec-type", "none"],
        "repeats": 2,
    },
]


def run_once(cfg_args, n_gen=N_GEN):
    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "131072",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--temp", "0",
        "--seed", "42",
        "-f", PROMPT_PATH,
        "-n", str(n_gen),
        "-st",
        "--simple-io",
        "--perf",
        "-v",
    ] + cfg_args

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdin=subprocess.DEVNULL,
    )
    wall_s = time.time() - t0
    text = proc.stdout

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
    mem_cuda_all = re.findall(
        r"CUDA0 \(RTX 3050 6GB Laptop GPU\)\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)",
        text,
    )
    mem_cuda = mem_cuda_all[-1] if mem_cuda_all else None

    draft_n = t_obj.get("draft_n", 0)
    draft_acc = t_obj.get("draft_n_accepted", 0)

    return {
        "pp_tokens": t_obj.get("prompt_n", 0),
        "pp_ms": round(t_obj.get("prompt_ms", 0.0), 2),
        "pp_tps": round(t_obj.get("prompt_per_second", 0.0), 2),
        "tg_tokens": t_obj.get("predicted_n", 0),
        "tg_ms": round(t_obj.get("predicted_ms", 0.0), 2),
        "tg_tps": round(t_obj.get("predicted_per_second", 0.0), 2),
        "draft_n": draft_n,
        "draft_accepted": draft_acc,
        "draft_rate_pct": round((draft_acc / draft_n) * 100.0, 2) if draft_n > 0 else 0.0,
        "cache_hits_decode": int(cache_small.group(1)) if cache_small else 0,
        "cache_misses_decode": int(cache_small.group(2)) if cache_small else 0,
        "cache_hit_rate_pct": float(cache_small.group(3)) if cache_small else 0.0,
        "cache_uploaded_mib": float(cache_small.group(4)) if cache_small else 0.0,
        "prefill_cache_copied_experts": int(cache_large.group(1)) if cache_large else 0,
        "prefill_cache_copied_mib": float(cache_large.group(2)) if cache_large else 0.0,
        "vram_used_mib": int(mem_cuda[2]) if mem_cuda else 0,
        "vram_model_mib": int(mem_cuda[3]) if mem_cuda else 0,
        "vram_ctx_mib": int(mem_cuda[4]) if mem_cuda else 0,
        "vram_compute_mib": int(mem_cuda[5]) if mem_cuda else 0,
        "wall_s": round(wall_s, 2),
    }


def main():
    print("=" * 88, flush=True)
    print(" BeyondVRAM: Controlled Deterministic Benchmark Suite (--temp 0, 128K Context)", flush=True)
    print("=" * 88, flush=True)

    summary_rows = []
    for cfg in CONFIGS:
        print(f"\n[*] Configuration: {cfg['label']} ({cfg['repeats']} repeats)", flush=True)
        runs = []
        for r_idx in range(cfg["repeats"]):
            res = run_once(cfg["args"], n_gen=N_GEN)
            runs.append(res)
            print(
                f"    Run {r_idx+1}/{cfg['repeats']}: PP = {res['pp_tps']:6.2f} t/s | "
                f"Decode = {res['tg_tps']:5.2f} t/s | Hit = {res['cache_hit_rate_pct']:5.2f}% | "
                f"VRAM = {res['vram_used_mib']} MiB | MTP = {res['draft_rate_pct']:5.1f}%",
                flush=True,
            )
            time.sleep(2.0)

        pp_list = [x["pp_tps"] for x in runs]
        tg_list = [x["tg_tps"] for x in runs]
        summary_rows.append({
            "id": cfg["id"],
            "label": cfg["label"],
            "repeats": cfg["repeats"],
            "pp_median_tps": round(statistics.median(pp_list), 2),
            "pp_min_tps": round(min(pp_list), 2),
            "pp_max_tps": round(max(pp_list), 2),
            "tg_median_tps": round(statistics.median(tg_list), 2),
            "tg_min_tps": round(min(tg_list), 2),
            "tg_max_tps": round(max(tg_list), 2),
            "cache_hit_rate_pct": runs[0]["cache_hit_rate_pct"],
            "cache_hits_decode": runs[0]["cache_hits_decode"],
            "cache_misses_decode": runs[0]["cache_misses_decode"],
            "cache_uploaded_mib": runs[0]["cache_uploaded_mib"],
            "prefill_cache_copied_mib": runs[0]["prefill_cache_copied_mib"],
            "draft_rate_pct": runs[0]["draft_rate_pct"],
            "vram_used_mib": runs[0]["vram_used_mib"],
            "runs": runs,
        })
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"configs": summary_rows}, f, indent=2)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"configs": summary_rows}, f, indent=2)
    print(f"\nSaved benchmark results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
