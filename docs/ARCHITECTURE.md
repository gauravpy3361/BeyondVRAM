# BeyondVRAM: Technical Architecture Specification

This document specifies the memory hierarchy, GGUF tensor budgeting, upstream `llama.cpp` Segmented-LRU (`llama_moe_cache`) runtime mechanics, and standalone C++/CUDA microbenchmark architectures evaluated in BeyondVRAM.

---

## 1. Target Model Geometry & Exact VRAM Budget (`Groot-19B-A3B`)

All measurements target `Groot-19B-A3B-261023.i1-Q3_K_S.gguf` (`qwen35moe` hybrid architecture, `7.82 GiB` on disk) running on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (`6,143 MiB` GDDR6, Compute Capability `8.6`, PCIe 4.0) paired with an Intel Core 5 210H CPU (`16 GB` DDR4, AVX2 + FMA).

### 1.1 Hybrid Model Topology (Parsed via `src/header_parser.py`)
- **Total MoE Blocks:** `41` (`40` main transformer blocks `blk.0..39` + `1` Multi-Token Prediction `nextn` block `blk.40`)
- **Experts per Block:** `128` routed experts (`5,248` total routed experts across 41 blocks) + `1` shared expert per block
- **Active Routing per Token:** `Top-8` (`6.25%` of routed experts active per token at `ubatch = 1`)
- **Hybrid Attention / Recurrent Split:**
  - `10` full-attention layers (`blk.3, 7, 11, 15, 19, 23, 27, 31, 35, 39`) with Grouped-Query Attention (`n_head = 16`, `n_head_kv = 2`, `d_k = d_v = 256`)
  - `31` recurrent / SSM layers (`251.5–256.0 MiB` fixed recurrent state buffer)

### 1.2 Exact Tensor & Memory Footprint
| Component | Exact Size | Location in BeyondVRAM (`-cmoe --moe-cache-mib 3300`) |
| :--- | :--- | :--- |
| **Non-Expert Backbone** (`token_embd`, norms, routers, shared experts, attention/SSM, `output`) | `1,036.93 MiB` (`1,087,300,608 B`) | Permanently pinned in GPU VRAM (`CUDA0`) |
| **Routed MoE Expert Tensors** (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` × 41 layers) | `6,765.00 MiB` (`7,093,616,640 B`) | Stored in host RAM (`CPU_Mapped`), cached dynamically in VRAM |
| **Single Quantized Expert** (`3 × [512 × 2048]` `Q3_K` matrices = `3 × 450,560 B`) | `1.289 MiB` (`1,351,680 B`) | Granularity of a single SLRU cache slot |
| **128K Context Buffer** (`-c 131072 -ctk q4_0 -ctv q4_0`) | `971.50 MiB` (`720.0 MiB` KV + `251.5 MiB` SSM) | Allocated in GPU VRAM (`CUDA0`) |
| **Graph Compute Buffer** | `471.00 MiB` | Allocated in GPU VRAM (`CUDA0`) |
| **Dynamic MoE SLRU Cache** (`--moe-cache-mib 3300`) | `3,300.00 MiB` (`2,560` expert slots = `48.8%` of all experts) | Allocated in GPU VRAM (`CUDA0`) |
| **Total GPU VRAM Used** | **`5,779.00 MiB / 6,143.00 MiB`** | Leaves `364 MiB` guardband (zero WDDM paging) |

---

## 2. Runtime Mechanics in Upstream `llama.cpp`

### 2.1 Static Placement Baselines (`-fit on` vs. `--n-cpu-moe N`)
Stock `llama.cpp` does not stream all offloaded layers over PCIe during decode; layers assigned to the CPU execute on the CPU.
1. **Stock Auto-Fit (`-fit on`):** `common/fit.cpp` reserves a default `1,024 MiB` free-VRAM margin (`--fit-target 1024`). On a 6 GB RTX 3050 at 128K context, Auto-Fit converts `35` layers of MoE experts to CPU (`blk.*.ffn_*_exps=CPU`), leaving only `7` layers of experts in VRAM (`3,653 MiB` total VRAM used, `1,076 MiB` left idle).
2. **Tuned Static `--n-cpu-moe 23` (`-fit off -ngl 99 -ncmoe 23`):** Reclaiming the idle margin allows `18` MoE layers to reside permanently in VRAM (`4,006 MiB` model buffer, `5,449 MiB` total VRAM used) while `23` MoE layers execute statically on the CPU. Because `18/41` (`43.9%`) of layers run on the GPU regardless of which experts are selected, this static configuration improves decode speed from `14.80 t/s` to `20.56 t/s` (MTP on) and `26.05 t/s` (MTP off).

### 2.2 Dynamic Segmented-LRU Expert Caching (`--moe-cache-mib`)
Even with tuned static offloading (`--n-cpu-moe 23`), each statically GPU-resident layer stores all `128` experts in VRAM (`165 MiB/layer`), even though a single token only activates `8` experts (`6.25%`).

With `--cpu-moe --moe-cache-mib 3300` (implemented in `llama.cpp`'s `src/llama-moe-cache.cpp`), `3,300 MiB` of VRAM is partitioned into `2,560` individual expert slots (`48.8%` of the model's `5,248` total experts) managed by a Segmented-LRU (SLRU) replacement policy:
- **Probation vs. Protected Segments:** Newly missed experts enter the **Probation** LRU list. A subsequent hit promotes the expert to the **Protected** LRU list (capped at `50%` of total capacity; overflow demotes the protected LRU tail back to Probation). Evictions draw from the Probation LRU tail first, preventing one-off burst activations from evicting core topic experts.
- **Small Micro-Batches (`ubatch <= 8`, Decode & MTP Verification):**
  - Selected expert IDs across the micro-batch are deduplicated (`uniq`) and sorted.
  - Cache hits remain in GPU VRAM (0 bytes transferred over PCIe).
  - Cache misses are uploaded asynchronously from pinned host memory (`CPU_Mapped`) to their assigned GPU SLRU slots via `ggml_backend_tensor_set_async`.
  - Hit/miss telemetry is reported at exit under `llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, uploaded = ... MiB`.
- **Large Batches (`n_tokens > 32`, Prompt Prefill):**
  - During prompt prefill, `get_slot_map()` returns `nullptr` so large prompts never thrash or evict the decode SLRU working set.
  - `copy_experts()` copies any experts already present in the GPU cache (`llama_moe_cache: large batches: N experts copied from the cache`) and executes remaining host experts on the CPU.

### 2.3 Interaction Between Speculative MTP (`draft-mtp`) and MoE Offloading
`Groot-19B-A3B` includes an auxiliary Multi-Token Prediction layer (`blk.40`, `nextn`). When `--spec-type draft-mtp` is enabled, decoding verifies micro-batches of `ubatch = 4` tokens:
- At `ubatch = 1` (`--spec-type none`), each layer routes `8` experts per step. With an **`88.86%`** SLRU hit rate (`58,237` hits / `7,298` misses over 200 tokens), the engine averages fewer than `1` cache miss per layer (`0.91` misses/layer/token), achieving **`33.83 tokens/sec`** (`+29.9%` faster than tuned static `--n-cpu-moe 23` at `26.05 t/s`).
- At `ubatch = 4` (`--spec-type draft-mtp`), 4 tokens touch ~20–24 unique experts per layer (`~2.5x–3x` more experts evaluated or transferred per step) while achieving a `34.7%–46.0%` draft acceptance rate (`~1.4–1.8` tokens accepted per step). Because MoE offloading and cache-miss costs scale with the number of unique experts touched in `ubatch`, `ubatch = 1` (`--spec-type none`) achieves higher net decode throughput on bandwidth-constrained consumer hardware.

---

## 3. Standalone C++/CUDA Microbenchmarks (`src/`)

In addition to the end-to-end `llama.cpp` benchmarks, this repository provides standalone C++/CUDA microbenchmarks to measure the underlying hardware primitives in isolation:

1. **Dual-Stream Compute/DMA Overlap (`src/async_pipeline.cu`):**
   - Models a layer executing dense attention + 7 resident GPU experts (`Stream 0`, `stream_compute`) while `Stream 1` (`stream_dma`) asynchronously fetches a `1.29 MiB` (`1,351,680 B`) cold expert from pinned host memory (`cudaMallocHost`) into double-buffered device staging memory.
   - Enforces true bidirectional `cudaEvent_t` data dependencies (`router_ready_event[slot]` and `dma_ready_event[slot]`) so `stream_dma` cannot start step `s` until `stream_compute` launches step `s`, and `cold_expert_swiglu_kernel` on `stream_compute` consumes the exact weights transferred by `stream_dma` (`0.00e+00` numerical diff vs. sequential execution).
   - **Measured Result (verified with NVIDIA Nsight Systems `nsys`):** Sequential execution takes **`0.464 ms/step`** (`0.264 ms` hot compute + `0.117 ms` HtoD DMA + `0.011 ms` cold expert + sync), whereas dual-stream pipelining completes in **`0.287 ms/step`** (**`38.1%` latency reduction / `1.61x` speedup**), hiding the entire `0.117 ms` PCIe transfer behind the `0.264 ms` GPU compute kernel.

2. **3-Matrix Quantized SwiGLU Cold-Expert Execution (`src/zero_transfer_ffn.cpp`):**
   - Compares three execution paths for a single `1.29 MiB` (`1,351,680 B`) quantized 3-matrix SwiGLU expert (`W_gate [512×2048]`, `W_up [512×2048]`, `W_down [2048×512]`, `d_model = 2048`, `d_ff = 512`), including full per-call GPU↔CPU activation transfer (`8 KiB` D2H + `8 KiB` H2D = `16 KiB`) and stream synchronization on the CPU path:
     - **Path 0 (GPU SLRU Cache Hit, 0 B PCIe):** **`0.052 ms / expert`** (`3.68x` faster than PCIe upload, `2.83x` faster than CPU in-place)
     - **Path A (PCIe 1.29 MiB Weight Upload + GPU SwiGLU):** **`0.193 ms / expert`** (`~0.141 ms` PCIe H2D + `0.052 ms` GPU compute)
     - **Path B (Zero-Transfer CPU AVX2 SwiGLU + 16 KiB PCIe Sync):** **`0.148 ms / expert`** (`0.099 ms` OpenMP AVX2 compute + `0.049 ms` GPU↔CPU activation transfer & sync; `1.30x` faster than Path A for a single uncached expert, `98.79%` less PCIe traffic).

3. **Reference C++ SLRU Cache & Unit Tests (`src/beyond_vram_cache.h`, `src/beyond_vram_cache.cpp`, `tests/test_cache.cpp`):**
   - Standalone C++17 reference implementation of the Segmented-LRU slot allocator with deterministic unit tests covering probation insertion, promotion to protected on re-reference, 50% protected cap demotion, and probation-first victim eviction.
