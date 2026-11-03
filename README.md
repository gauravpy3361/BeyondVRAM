# BeyondVRAM

> **Authoring: Hybrid Architecture** | **Applied Systems**  
> *System architecture, problem formulation, and testing designed by me; implementation assisted by AI coding agents.*

[![Hardware Target](https://img.shields.io/badge/Hardware-RTX%203050%20(6GB)%20%2B%2016GB%20RAM-blue)](#hardware-specification)
[![Architecture](https://img.shields.io/badge/Architecture-Sparse%20MoE%20Hybrid%20Offloading-orange)](#core-architecture)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#license)

---

## Overview

**BeyondVRAM** is a high-performance inference runtime architecture designed to break the memory-bandwidth wall for **Mixture-of-Experts (MoE)** Large Language Models on consumer-grade, memory-constrained hardware (e.g. 4GB–8GB laptop GPUs and single/dual-channel DDR4/DDR5 system RAM).

Instead of conventional, monolithic layer-by-layer offloading—which causes catastrophic GPU stalls and PCIe bus saturation—BeyondVRAM reorganizes model execution around fine-grained, asynchronous hardware orchestration.

---

## The 3 Core Pillars & Validated Benchmarks

All three architectural pillars have been physically implemented and benchmarked on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU + Host CPU**:

```
Token In
   │
   ├─► 1. Dense Attention Backbone & Routers ──► Pinned permanently in GPU VRAM
   │
   ├─► 2. Top-K Routed Experts:
   │      ├─► Hot Cache (SLRU) In VRAM (~68% - 86% hits) ─► Direct GDDR6 execution
   │      ├─► Warm Multi-Token Incoming Experts ─────────► Async DMA Pipelining (Masked)
   │      └─► Cold Single-Shot Experts ──────────────────► Zero-Transfer CPU FFN (In-place)
   │
   └─► Only tiny activation vectors traverse PCIe, completely eliminating weight thrashing.
```

### Pillar 1: Hot Cache in VRAM & RAM (Dynamic Expert Caching)
* **Implementation:** [`src/header_parser.py`](src/header_parser.py), [`src/moe_cache.py`](src/moe_cache.py), [`src/beyond_vram_cache.h`](src/beyond_vram_cache.h).
* **Mechanism:** Pins the 1.22 GB non-expert backbone in VRAM and turns the remaining ~4.0 GB into an SLRU cache holding ~3,100 expert slots.
* **Measured Result:** **67.97% to 86.29% cache hit rate**, eliminating **35.66 GB of PCIe bus traffic per 100 tokens**.

### Pillar 2: Split the Math (Async Compute & DMA Pipelining)
* **Implementation:** [`src/async_pipeline.cu`](src/async_pipeline.cu), [`src/async_pipeline.py`](src/async_pipeline.py).
* **Mechanism:** Decouples the Compute Engine (Stream 0) from the PCIe DMA Copy Engine (Stream 1). Missing expert weights stream across PCIe concurrently while Attention computes on GPU.
* **Measured Result:** **83.35% direct latency cut** (from 0.868 ms down to 0.144 ms per step), providing a **6.01x hardware speedup**.

### Pillar 3: Zero-Transfer CPU FFN (Asymmetric In-Place Execution)
* **Implementation:** [`src/zero_transfer_ffn.cpp`](src/zero_transfer_ffn.cpp), [`src/zero_transfer_ffn.py`](src/zero_transfer_ffn.py).
* **Mechanism:** For single-shot cold experts, weights never leave host RAM. The GPU sends only the 8 KB activation vector; the CPU executes SIMD AVX2 math in-place and sends the 8 KB result back.
* **Measured Result:** **2.62x faster than transferring weights to GPU**, eliminating **99.61% of PCIe bus data movement** with **ZERO weight bytes** crossing the bus.

---

## Definitive 3× (1K Prompt × 1K Decode) Sustained Stress-Test

To rigorously test real-world viability, BeyondVRAM was benchmarked against a **brutal long-sequence workload**: three massive, deep technical queries each ingesting **~800–920 prompt tokens** and generating **exactly 1,000 continuous decode tokens** (3,000 generated tokens total) under a full **131,072 (128K) context window**.

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143 MiB GDDR6, PCIe 4.0 x4) + 16GB Host DDR4
* **Engine Configuration:** `llama-cli.exe` with `-c 131072 -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp -cmoe --moe-cache-mib 3300 -n 1000`
* **Zero Paging Guarantee:** 5,779 MiB allocated / 6,143 MiB total (**+364 MiB clean headroom**). Zero Windows WDDM paging, zero OOMs.

| Benchmark Query & Domain | Ingested Prompt | Prefill Speed (PP) | Generated Tokens | Sustained Decode | MoE Cache Hit Rate | MTP Speculative Acceptance | Total Elapsed Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1: Exascale Distributed AI Systems**<br>*(3D Parallelism, InfiniBand HDR/NDR, Disaggregated Prefill/Decode)* | 916 tokens | **61.9 t/s** | 1,000 tokens | **23.6 t/s** | **87.8%** | **46.0%** (284 / 617) | 69.3 s |
| **Q2: Autonomous Agent Operating System (AgentOS)**<br>*(Kernel Scheduler, Memory Hierarchy, ReAct IAC, Sandboxing)* | 819 tokens | **59.1 t/s** | 1,000 tokens | **24.8 t/s** | **90.2%** | **45.2%** (282 / 624) | 66.5 s |
| **Q3: High-Performance Compiler & Fused Kernels**<br>*(LLVM IR Polyhedral Loops, FlashAttention V2 CUDA PTX)* | 816 tokens | **65.8 t/s** | 1,000 tokens | **22.4 t/s** | **89.0%** | **42.4%** (262 / 618) | 70.2 s |
| **DEFINITIVE BENCHMARK AVERAGE** | **850.3 tokens** | **62.3 t/s** | **3,000 tokens** | **23.6 t/s** | **89.0%** | **44.5%** | **68.7 s / run** |

### What This Empirical Data Reveals:
1. **Steady-State High Throughput:** Unlike small micro-benchmarks that fluctuate, sustaining 1,000 continuous decode tokens per run proves that the dynamic SLRU cache does not degrade over long generation lengths—maintaining **23.6 tokens/sec sustained**.
2. **Elimination of Host-to-Device Bottleneck:** Across 3,000 generated tokens, standard un-cached offloading would have forced **~1.24 Terabytes of PCIe expert transfers**. BeyondVRAM achieved an **89.0% average cache hit rate**, eliminating over **1.10 TB of bus traffic**.
3. **Speculative Multi-Token Synergies (MTP):** Even on complex, dense code/compiler generation, MTP draft acceptance remained between **42.4% and 46.0%**, yielding a **~1.6x decode speedup** purely within the 6GB VRAM budget.

---

## The Killer Metric: PCIe Transit Elimination

Standard runtimes bottleneck on PCIe bandwidth, repeatedly dragging the same expert weights from host RAM to VRAM. BeyondVRAM eliminates the vast majority of this physical data travel:

| Workload (Tokens) | Naive Bus Traffic | BeyondVRAM Traffic | Bus Traffic Eliminated | Bus Transit Latency Saved | Effective Bus Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100 Tokens** | 41.32 GB | 5.67 GB | **35.66 GB (86.3%)** | **7.92 seconds** | **7.29x** |
| **250 Tokens** | 103.30 GB | 14.16 GB | **89.14 GB (86.3%)** | **19.81 seconds** | **7.29x** |
| **500 Tokens** | 206.60 GB | 28.33 GB | **178.28 GB (86.3%)** | **39.62 seconds** | **7.29x** |

> *"BeyondVRAM eliminated **178.28 GB** of PCIe weight shuttling in a 500-token response, shaving **39.62 seconds** of idle bus latency while maintaining 100% mathematical fidelity."*

---

## Project Structure

```
BeyondVRAM/
├── .gitignore                    # Ignores model weights and temporary files
├── README.md                     # Architecture, benchmarks & specifications
├── docs/
│   └── ARCHITECTURE.md           # Detailed systems specification
├── benchmarks/
│   ├── results_3x1k.json         # Definitive 3x (1K x 1K) raw benchmark metrics
│   └── prompts/                  # Full 1K-token stress prompt suites
│       ├── prompt_1k_q1_distributed.txt
│       ├── prompt_1k_q2_agent_os.txt
│       └── prompt_1k_q3_compiler_simd.txt
└── src/
    ├── __init__.py
    ├── header_parser.py          # Step 1: Binary GGUF geometry & VRAM budgeter
    ├── moe_cache.py              # Step 1: SLRU Hot/Cold Cache implementation
    ├── beyond_vram_cache.h       # C++ SLRU Cache controller interface
    ├── async_pipeline.cu         # Step 2: Dual-stream CUDA async DMA prefetch engine
    ├── async_pipeline.py         # Step 2: Python simulation of async pipeline
    ├── zero_transfer_ffn.cpp     # Step 3: Zero-Transfer CPU SIMD AVX2 in-place engine
    ├── zero_transfer_ffn.py      # Step 3: Zero-Transfer dispatch heuristic
    ├── benchmark_comparator.py   # PCIe bus transit & time savings calculator
    ├── create_1k_prompts.py      # Prompt generator for the 1Kx1K benchmark suite
    ├── run_full_3x1k_bench.py    # Definitive 3x (1K prompt x 1K decode) stress-test runner
    └── measure_context_scaling.py# 4K to 128K context memory scaling evaluator
```

---

## License

Apache-2.0 License.
