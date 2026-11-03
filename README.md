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

## The 3 Core Pillars

```
Token In
   │
   ├─► 1. Dense Attention Backbone & Routers ──► Pinned permanently in GPU VRAM
   │
   ├─► 2. Top-K Routed Experts:
   │      ├─► Hot Cache (SLRU) In VRAM (~68% hits) ─► Execute at full GDDR6 bus speed
   │      ├─► Cache Miss / Incoming Experts ─────────► Async DMA Pipelining (Overlapped)
   │      └─► Cold Single-Token Experts ────────────► Zero-Transfer CPU FFN (In-place RAM)
   │
   └─► Only tiny activation vectors traverse PCIe, eliminating weight thrashing.
```

### 1. Hot Cache in VRAM & RAM (Dynamic Expert Caching)
* **Problem:** Monolithic offloading locks inactive experts into VRAM, wasting capacity.
* **Solution:** Analyzes the binary GGUF geometry to extract non-expert tensors (Attention backbone + Routers + Embeddings = ~1.22 GB) and pins them permanently in VRAM. The remaining **~4.78 GB of free VRAM** operates as a dynamic **Segmented LRU (SLRU)** cache storing ~3,200–3,800 active expert slots.
* **Result:** **67.97% cache hit rate** verified on Groot-19B-A3B.

### 2. Split the Math (Async Pipelining)
* **Problem:** Cache misses stall the GPU while synchronously fetching weights across PCIe.
* **Solution:** Decouples computation and DMA data transfer into parallel CUDA streams. While the GPU compute cores process Dense Attention and resident hot experts, background DMA engines (`ggml_backend_tensor_set_async`) stream upcoming missing experts over PCIe concurrently without halting math execution.

### 3. Zero-Transfer CPU FFN (Asymmetric In-Place Execution)
* **Problem:** Moving multi-gigabyte FFN weight matrices over PCIe for single-token activations chokes the PCIe bus.
* **Solution:** Massive cold expert weights remain permanently pinned in system RAM (`-cmoe`). When an expert computes on the CPU, math is performed **in-place on host CPU cores**. Only tiny activation vectors ($d_{\text{model}}$) traverse the PCIe bus—**zero weight matrices cross the bus**.

---

## Live Benchmark Validation

Evaluated on **NVIDIA GeForce RTX 3050 6GB Laptop GPU + Host RAM**:

| Metric | Measured Value | Configuration |
| :--- | :--- | :--- |
| **Model** | Groot-19B-A3B (Q3_K_S) | 41 Layers, 128 Experts, Top-8 Active |
| **Prompt Processing (PP)** | **110.27 tokens/sec** | 508 tokens evaluated in 4.60s |
| **Generation (Decode)** | **22.17 tokens/sec** | 250 tokens generated with MTP |
| **VRAM Cache Hit Rate** | **67.97%** | 49,319 hits vs 23,238 misses |
| **MTP Acceptance Rate** | **66.93%** | Mean acceptance length: 3.00 tokens |
| **PCIe Weight Thrashing** | **0 MB** | Full Zero-Transfer CPU FFN |

---

## Project Structure

```
BeyondVRAM/
├── .gitignore               # Ignores model weights and binaries
├── README.md                # Project architecture & benchmarks
├── docs/
│   └── ARCHITECTURE.md      # Detailed systems specification
└── src/
    ├── __init__.py
    ├── header_parser.py     # Step 1: Binary GGUF geometry & VRAM budgeter
    ├── moe_cache.py         # Step 1: SLRU Hot/Cold Cache implementation
    ├── async_pipeline.py    # Step 2: Asynchronous DMA & compute stream orchestration
    └── zero_transfer_ffn.py # Step 3: In-place CPU FFN execution engine
```

---

## License

Apache-2.0 License.
