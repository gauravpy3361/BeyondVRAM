import subprocess
import time
import os
import re
import json

CLI_PATH = os.environ.get("LLAMA_CLI_PATH", r"C:\ai\llama.cpp\llama-cli.exe")
MODEL_PATH = os.environ.get(
    "BEYONDVRAM_MODEL",
    r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf",
)
OUT_JSON = os.path.join("benchmarks", "results_3x1k_std.json")
LOG_DIR = os.path.join("benchmarks", "logs")

PROMPTS = [
    ("Q1", "Exascale Distributed AI Systems", os.path.join("benchmarks", "prompts", "prompt_1k_q1_distributed.txt"), os.path.join(LOG_DIR, "bench_1k_std_q1.log")),
    ("Q2", "Autonomous Agent Operating System", os.path.join("benchmarks", "prompts", "prompt_1k_q2_agent_os.txt"), os.path.join(LOG_DIR, "bench_1k_std_q2.log")),
    ("Q3", "High-Performance Compiler & Kernels", os.path.join("benchmarks", "prompts", "prompt_1k_q3_compiler_simd.txt"), os.path.join(LOG_DIR, "bench_1k_std_q3.log")),
]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    results = []

    for q_id, q_title, prompt_path, log_path in PROMPTS:
        print("============================================================")
        print(f"Running Stock Auto-Fit Baseline for {q_id}: {q_title} (N=1000)")
        print("============================================================")

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
            "-v",
        ]

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        elapsed = round(time.time() - t0, 2)

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        timings = re.findall(r'"timings":(\{[^\}]+\})', text)
        t_obj = json.loads(timings[-1]) if timings else {}

        draft_n = t_obj.get("draft_n", 0)
        draft_acc = t_obj.get("draft_n_accepted", 0)
        draft_pct = round((draft_acc / draft_n) * 100.0, 2) if draft_n > 0 else 0.0

        pp_tok = t_obj.get("prompt_n", 0)
        pp_ms = round(t_obj.get("prompt_ms", 0.0), 2)
        pp_spd = round(t_obj.get("prompt_per_second", 0.0), 2)
        tg_tok = t_obj.get("predicted_n", 0)
        tg_ms = round(t_obj.get("predicted_ms", 0.0), 2)
        tg_spd = round(t_obj.get("predicted_per_second", 0.0), 2)

        print(f"Finished {q_id} in {elapsed:.1f}s | PP: {pp_spd:.2f} t/s | Decode: {tg_spd:.2f} t/s | Draft: {draft_pct:.1f}%")

        results.append({
            "id": q_id,
            "title": q_title,
            "file": prompt_path.replace("\\", "/"),
            "done": True,
            "pp_tokens": pp_tok,
            "pp_ms": pp_ms,
            "pp_speed": pp_spd,
            "tg_tokens": tg_tok,
            "tg_ms": tg_ms,
            "tg_speed": tg_spd,
            "draft_accepted": draft_acc,
            "draft_n": draft_n,
            "draft_rate": draft_pct,
            "elapsed": elapsed,
        })

    avg_pp = round(sum(r["pp_speed"] for r in results) / len(results), 2)
    avg_tg = round(sum(r["tg_speed"] for r in results) / len(results), 2)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "summary": {
                    "avg_pp": avg_pp,
                    "avg_tg": avg_tg,
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved standard results to {OUT_JSON}")


if __name__ == "__main__":
    main()
