"""
BeyondVRAM: CLI Auto-Tuner & Launcher for llama.cpp MoE Inference
Usage:
    python beyondvram.py --model "path/to/model.gguf" [--mode non-mtp|mtp|hybrid] [--ctx-size 131072] [--dry-run]

Automatically:
1. Detects NVIDIA GPU VRAM capacity via CUDA Driver API (nvcuda.dll) or nvidia-smi.
2. Parses GGUF header metadata & tensor footprints via src/header_parser.py.
3. Computes the exact SLRU expert cache allocation (--moe-cache-mib) that fits within VRAM without WDDM paging.
4. Launches (or prints with --dry-run) the tuned llama-cli command.
"""

import os
import sys
import ctypes
import argparse
import subprocess

from src.header_parser import MoEModelGeometry


def detect_vram_mib() -> int:
    try:
        cuda = ctypes.windll.LoadLibrary("nvcuda.dll")
        if cuda.cuInit(0) == 0:
            dev = ctypes.c_int()
            cuda.cuDeviceGet(ctypes.byref(dev), 0)
            total = ctypes.c_size_t()
            cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev)
            return int(total.value // (1024 * 1024))
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 6143


def main():
    parser = argparse.ArgumentParser(
        description="BeyondVRAM: Auto-Tuning VRAM Budgeter & Launcher for llama.cpp MoE Models"
    )
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF MoE model file")
    parser.add_argument(
        "--cli",
        default=os.environ.get("LLAMA_CLI_PATH", "llama-cli.exe"),
        help="Path to llama-cli executable (or set LLAMA_CLI_PATH)",
    )
    parser.add_argument(
        "--mode",
        choices=["non-mtp", "mtp", "hybrid"],
        default="non-mtp",
        help=(
            "non-mtp: Peak single-token decode throughput (33.8 tok/s, --spec-type none)\n"
            "mtp: Multi-Token Prediction speculative decoding (23.4 tok/s, --spec-type draft-mtp)\n"
            "hybrid: Static GPU layers + SLRU cache with MTP (24.0 tok/s)"
        ),
    )
    parser.add_argument("-c", "--ctx-size", type=int, default=131072, help="Context window size (default: 131072)")
    parser.add_argument("--vram", type=int, default=None, help="Override detected GPU VRAM in MiB")
    parser.add_argument("-p", "--prompt", default=None, help="Optional single prompt string")
    parser.add_argument("-n", "--predict", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--dry-run", action="store_true", help="Print computed VRAM budget and command without executing")
    args = parser.parse_args()

    vram_mib = args.vram if args.vram is not None else detect_vram_mib()
    geom = MoEModelGeometry(args.model)
    budget = geom.compute_budget(total_vram_mib=vram_mib, ctx_size=args.ctx_size, kv_type="q4_0")
    geom.print_report(budget)

    cache_mib = max(500, (int(budget["hot_cache_budget_mib"]) // 50) * 50)

    cmd = [
        args.cli,
        "-m", args.model,
        "-c", str(args.ctx_size),
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
    ]

    if args.mode == "non-mtp":
        cmd += ["-cmoe", "--moe-cache-mib", str(cache_mib), "--spec-type", "none"]
    elif args.mode == "mtp":
        cmd += ["-cmoe", "--moe-cache-mib", str(cache_mib), "--spec-type", "draft-mtp"]
    elif args.mode == "hybrid":
        cmd += [
            "-fit", "off",
            "-ngl", "99",
            "-ncmoe", "31",
            "--moe-cache-mib", str(cache_mib // 2),
            "--spec-type", "draft-mtp",
        ]

    if args.prompt:
        cmd += ["-p", args.prompt, "-n", str(args.predict)]

    formatted_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n[+] Mode Selected : {args.mode.upper()}")
    print(f"[+] Command       : {formatted_cmd}\n")

    if not args.dry_run:
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
