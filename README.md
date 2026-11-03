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

## Head-to-Head Comparison: Standard `llama.cpp` vs BeyondVRAM

To rigorously evaluate real-world performance, BeyondVRAM was benchmarked side-by-side against **Standard `llama.cpp`** on a brutal long-sequence workload: three massive, production-grade technical queries each ingesting **~800–920 prompt tokens** and generating **exactly 1,000 continuous decode tokens** (3,000 generated tokens total) under a full **131,072 (128K) context window**.

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143 MiB GDDR6, PCIe 4.0 x4) + 16GB Host DDR4
* **Workload Scope:** 3 runs $\times$ 1,000 continuous decode tokens each = **6,000 total benchmark tokens evaluated**
* **VRAM Safety:** Strict 3,300 MiB cache cap with +364 MiB clean headroom. Zero Windows WDDM paging thrashing.

| Benchmark Query & Domain | Metric | Standard `llama.cpp` (Auto-Fit) | BeyondVRAM (Dynamic Cache) | BeyondVRAM Advantage |
| :--- | :--- | :---: | :---: | :---: |
| **Q1: Exascale Distributed Systems**<br>*(916 Prompt / 1,000 Decode)* | **Sustained Decode**<br>Prompt Prefill (PP)<br>MoE Cache Hit Rate<br>MTP Speculative Rate<br>**Total Run Time** | 14.10 t/s<br>**96.97 t/s**<br>0.0% (No Cache)<br>44.6%<br>96.58 s | **23.60 t/s**<br>61.90 t/s<br>**87.8%**<br>**46.0%**<br>**69.30 s** | **+67.4% Decode Speedup**<br>*(Dynamic cache offload)*<br>**+87.8% Cache Hits**<br>+1.4% MTP Draft<br>**27.3s Faster (-28.2%)** |
| **Q2: Autonomous AgentOS Kernel**<br>*(819 Prompt / 1,000 Decode)* | **Sustained Decode**<br>Prompt Prefill (PP)<br>MoE Cache Hit Rate<br>MTP Speculative Rate<br>**Total Run Time** | 14.84 t/s<br>**129.39 t/s**<br>0.0% (No Cache)<br>37.8%<br>95.37 s | **24.75 t/s**<br>59.13 t/s<br>**90.2%**<br>**45.2%**<br>**66.49 s** | **+66.8% Decode Speedup**<br>*(Dynamic cache offload)*<br>**+90.2% Cache Hits**<br>+7.4% MTP Draft<br>**28.9s Faster (-30.3%)** |
| **Q3: High-Performance Compiler & Kernels**<br>*(816 Prompt / 1,000 Decode)* | **Sustained Decode**<br>Prompt Prefill (PP)<br>MoE Cache Hit Rate<br>MTP Speculative Rate<br>**Total Run Time** | 18.45 t/s<br>**165.53 t/s**<br>0.0% (No Cache)<br>**48.6%**<br>75.06 s | **22.42 t/s**<br>65.84 t/s<br>**89.0%**<br>42.4%<br>**70.15 s** | **+21.5% Decode Speedup**<br>*(Dynamic cache offload)*<br>**+89.0% Cache Hits**<br>-6.2% MTP Draft<br>**4.9s Faster (-6.5%)** |
| **SUITE BENCHMARK AVERAGE** | **Sustained Decode**<br>Prompt Prefill (PP)<br>MoE Cache Hit Rate<br>MTP Speculative Rate<br>**Total Wall-Clock Time** | 15.80 t/s<br>**130.63 t/s**<br>0.0% (No Cache)<br>43.7%<br>267.01 s | **23.59 t/s**<br>62.29 t/s<br>**89.0%**<br>**44.5%**<br>**205.94 s** | **+49.3% Sustained Decode**<br>*(Long-generation speedup)*<br>**89.0% Transit Cut**<br>+0.8% MTP Draft<br>**61.1s Total Time Saved** |

### Critical Engineering Insights:
1. **+49.3% Sustained Decode Acceleration:** On heavy, long-sequence generation (1,000 tokens), Standard `llama.cpp` bottlenecks on the PCIe bus as it drags un-cached expert weights back and forth, stalling generation at **14–15 t/s**. BeyondVRAM delivers **23.6–24.8 t/s** on distributed systems and agentic workflows (**+67% faster**).
2. **61.1 Seconds of Waiting Eliminated:** Across all three 1,000-token answers, BeyondVRAM completed the suite in **205.9 seconds** versus **267.0 seconds** for standard `llama.cpp`—saving over **a full minute of idle waiting** on a budget 6GB laptop GPU.
3. **Transparent Architectural Trade-Off:** Standard `llama.cpp` achieves faster prefill (130 t/s vs 62 t/s) because it leaves all GPU VRAM unreserved for static dense layer offloading. However, once decoding begins, BeyondVRAM's dynamic SLRU hot cache keeps **89.0%** of all active expert lookups in GDDR6, eliminating over **1.1 Terabytes of PCIe weight transfers** and overwhelmingly winning total generation time.

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
│   ├── results_3x1k.json         # BeyondVRAM 3x (1K x 1K) raw metrics
│   ├── results_3x1k_std.json     # Standard llama.cpp 3x (1K x 1K) baseline metrics
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
    ├── run_full_3x1k_bench.py    # BeyondVRAM 3x (1K x 1K) stress runner
    ├── run_full_3x1k_std_bench.py# Standard baseline 3x (1K x 1K) runner
    └── measure_context_scaling.py# 4K to 128K context memory scaling evaluator
```

---

## License

Apache-2.0 License.
