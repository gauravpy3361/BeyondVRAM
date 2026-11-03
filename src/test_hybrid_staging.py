import subprocess
import os
import re
import json
import time

MODEL_PATH = r"C:\Users\gaurav\Downloads\ABDM\Groot-19B-A3B-261023.i1-Q3_K_S.gguf"
CLI_PATH = r"C:\ai\llama.cpp\llama-cli.exe"

def test_config(name, ncmoe, cache_mib, prompt_file, n_gen=100):
    log_path = rf"C:\Users\gaurav\test_{name}.log"
    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-c", "4096",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--spec-type", "draft-mtp",
        "-ngl", "99",
        "-ncmoe", str(ncmoe),
        "--moe-cache-mib", str(cache_mib),
        "-f", prompt_file,
        "-n", str(n_gen),
        "-st",
        "--simple-io",
        "--perf",
        "-v"
    ]
    
    print(f"\n=======================================================")
    print(f"[*] Testing Config: {name}")
    print(f"    ncmoe={ncmoe} (first {ncmoe} layers on CPU/Cache, {41-ncmoe} layers permanently in VRAM)")
    print(f"    moe_cache_mib={cache_mib} MiB")
    print(f"=======================================================")
    
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    pp_match = re.search(r'prompt eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    tg_match = re.findall(r'eval time =\s+([\d\.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+[\d\.]+\s+ms per token,\s+([\d\.]+)\s+tokens per second\)', text)
    mem_match = re.search(r'CUDA0.*\|.*?(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)', text)
    vram_buf_match = re.search(r'CUDA0 model buffer size\s*=\s*([\d\.]+)\s*MiB', text)
    cache_match = re.search(r'llama_moe_cache:.*hits = (\d+), misses = (\d+), hit rate = ([\d\.]+)%', text)
    draft_match = re.search(r'draft acceptance =\s+([\d\.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\)', text)
    
    tg_data = tg_match[-1] if tg_match else None
    pp_speed = float(pp_match.group(3)) if pp_match else 0.0
    tg_speed = float(tg_data[2]) if tg_data else 0.0
    cache_rate = float(cache_match.group(3)) if cache_match else 0.0
    draft_rate = float(draft_match.group(1))*100.0 if draft_match else 0.0
    
    print(f"    -> Finished in {elapsed:.1f}s")
    print(f"    -> PP Speed:     {pp_speed:.1f} t/s")
    print(f"    -> Decode Speed: {tg_speed:.1f} t/s")
    print(f"    -> Cache Hit:    {cache_rate:.1f}%")
    print(f"    -> MTP Accept:   {draft_rate:.1f}%")
    
    # Check VRAM breakdown
    for line in text.splitlines():
        if "common_memory_breakdown_print" in line and "CUDA0" in line:
            print(f"    -> Memory Breakdown: {line.strip()}")
            
    return {
        "name": name,
        "ncmoe": ncmoe,
        "cache_mib": cache_mib,
        "pp_speed": pp_speed,
        "tg_speed": tg_speed,
        "cache_rate": cache_rate,
        "draft_rate": draft_rate,
        "elapsed": elapsed
    }

if __name__ == "__main__":
    prompt_q1 = r"C:\Users\gaurav\prompt_q1.txt"
    # Test 1: ncmoe=31, cache=1900 (10 layers fully on GPU)
    res1 = test_config("ncmoe_31_cache_1900", ncmoe=31, cache_mib=1900, prompt_file=prompt_q1, n_gen=100)
