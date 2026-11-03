import subprocess
import time
import os
import re
import json

MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"
CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"

PROMPTS = [
    ("Q1", "Exascale Distributed AI Systems", r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q1_distributed.txt", r"C:\Users\gaurav\bench_1k_std_q1.log"),
    ("Q2", "Autonomous Agent Operating System", r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q2_agent_os.txt", r"C:\Users\gaurav\bench_1k_std_q2.log"),
    ("Q3", "High-Performance Compiler & Kernels", r"C:\Users\gaurav\bench_1k_prompts\prompt_1k_q3_compiler_simd.txt", r"C:\Users\gaurav\bench_1k_std_q3.log")
]

results = []

for q_id, q_title, prompt_path, log_path in PROMPTS:
    print(f"============================================================")
    print(f"Running Standard Baseline for {q_id}: {q_title} (N=1000)")
    print(f"============================================================")

    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "131072",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--spec-type", "draft-mtp",
        "-f", prompt_path,
        "-n", "1000",
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as out:
        p = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0

    # Parse log
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    draft_match = re.search(r'draft acceptance =\s+([\d\.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\)', text)

    pp_tok = int(pp_match.group(2)) if pp_match else 0
    pp_spd = float(pp_match.group(3)) if pp_match else 0.0
    tg_tok = int(tg_match[-1][1]) if tg_match else 0
    tg_spd = float(tg_match[-1][2]) if tg_match else 0.0
    draft_pct = float(draft_match.group(1)) * 100.0 if draft_match else 0.0

    print(f"Finished {q_id} in {elapsed:.1f}s | PP: {pp_spd:.2f} t/s | Decode: {tg_spd:.2f} t/s | Draft: {draft_pct:.1f}%")

    results.append({
        "id": q_id,
        "title": q_title,
        "pp_tokens": pp_tok,
        "pp_speed": pp_spd,
        "tg_tokens": tg_tok,
        "tg_speed": tg_spd,
        "draft_rate": draft_pct,
        "elapsed": elapsed
    })

avg_pp = sum(r["pp_speed"] for r in results) / len(results)
avg_tg = sum(r["tg_speed"] for r in results) / len(results)

summary = {
    "results": results,
    "summary": {
        "avg_pp": avg_pp,
        "avg_tg": avg_tg
    }
}

with open(r"C:\Users\gaurav\bench_3x1k_std_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nALL STANDARD RUNS FINISHED SUCCESSFULLY!")
print(f"Average Standard Decode: {avg_tg:.2f} t/s | Average Standard PP: {avg_pp:.2f} t/s")
