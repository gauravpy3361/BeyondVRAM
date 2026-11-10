# BeyondVRAM: Technical Architecture Specification

This document details the GGUF tensor geometry, GPU memory budgeting, Segmented-LRU (`llama_moe_cache`) runtime mechanics, Multi-Token Prediction (`MTP` vs. `Non-MTP`) batching behavior, and standalone C++/CUDA microbenchmarks in BeyondVRAM.

---

## 1. Target Model Geometry & Exact VRAM Budget (`Groot-19B-A3B`)

All measurements target `Groot-19B-A3B-261023.i1-Q3_K_S.gguf` (`qwen35moe` hybrid architecture, `7.82 GiB` on disk) running on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (`6,143 MiB` GDDR6, Compute Capability `8.6`, PCIe 4.0) paired with an Intel Core 5 210H CPU (`16 GB` DDR4, AVX2 + FMA).

### 1.1 Hybrid Model Topology (Parsed via [`src/header_parser.py`](../src/header_parser.py))
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

## 2. Default `llama.cpp` vs. BeyondVRAM Tuned

### 2.1 Why Default `llama.cpp` (`-fit on`) Leaves VRAM Underutilized
When running a model that exceeds GPU memory, default `llama.cpp` (`-fit on`, implemented in `common/fit.cpp`) makes placement decisions at **whole-layer granularity** while reserving a conservative `1,024 MiB` free-VRAM margin (`--fit-target 1024`):
- Each MoE layer's 128 routed experts occupy `165 MiB` and must be placed all-in-VRAM or all-in-RAM.
- On a 6 GB RTX 3050 at 128K context, `-fit on` keeps only `7` of `41` MoE layers on the GPU and pushes `35` layers to system RAM (`3,653 MiB` total VRAM used, leaving **`2,490 MiB` of GDDR6 unused**).
- Furthermore, within those `7` GPU-resident layers, `93.75%` of the stored experts (`120` of `128`) go unused on any given token.
- Result: **`14.80 tok/s`** decode speed.

### 2.2 BeyondVRAM Tuned: Dynamic Segmented-LRU Expert Caching (`--moe-cache-mib 3300`)
Instead of locking whole layers of mostly idle experts into VRAM, BeyondVRAM (`src/header_parser.py` and `beyondvram.py`) pins the `1,037 MiB` non-expert backbone and `128K` context in VRAM and allocates **`3,300 MiB`** to a dynamic Segmented-LRU expert pool (`2,560` individual expert slots across all `41` layers, covering `48.8%` of the model's `5,248` total experts):
- **Probation vs. Protected Segments:** Newly missed experts enter the **Probation** LRU list. A subsequent hit promotes the expert to the **Protected** LRU list (capped at `50%` of total capacity; overflow demotes the protected LRU tail back to Probation). Evictions draw from the Probation LRU tail first, preventing one-off burst activations from evicting core topic experts.
- **Small Micro-Batches (`ubatch <= 8`, Decode & MTP Verification):**
  - Selected expert IDs across the micro-batch are deduplicated (`uniq`) and sorted.
  - Cache hits execute directly in GPU VRAM (`0 B` transferred over PCIe).
  - Cache misses are uploaded asynchronously from pinned host memory (`CPU_Mapped`) into their assigned GPU SLRU slots via `ggml_backend_tensor_set_async`.
- **Large Batches (`n_tokens > 32`, Prompt Prefill):**
  - During prompt prefill, `get_slot_map()` returns `nullptr` so long prompts never flush or thrash the decode SLRU working set.

### 2.3 MTP (`draft-mtp`) vs. Non-MTP (`--spec-type none`)
`Groot-19B-A3B` includes an auxiliary Multi-Token Prediction layer (`blk.40`, `nextn`). Comparing both modes reveals a critical hardware interaction between speculative batching and MoE expert caching:
- **MTP On (`--spec-type draft-mtp`, `ubatch = 4`):** Verifying 4 speculative tokens per step touches **~20–24 unique experts per layer** (`~2.5x–3x` more experts evaluated or transferred per step) for a `34.7%–46.0%` acceptance rate (`~1.4–1.8` tokens accepted per step). With the `3,300 MiB` SLRU cache (`89.77%` hit rate), throughput reaches **`23.42 tok/s`** (`+58.2%` vs. Default `llama.cpp`), and **`24.02 tok/s`** (`+62.3%`) in Hybrid mode (`10` static layers + `1,650 MiB` cache).
- **Non-MTP (`--spec-type none`, `ubatch = 1`):** Each layer routes exactly **`8` experts per step**. With an **`88.86%`** SLRU hit rate (`58,237` hits / `7,298` misses over 200 tokens), the GPU averages fewer than `1` cache miss per layer (`0.91` misses/layer/token), reaching **`33.83 tok/s`** (**`2.29x` / `+128.6%` faster than Default `llama.cpp`**).

---

## 3. Standalone C++/CUDA Hardware Microbenchmarks (`src/`)

To isolate why cache hits and asynchronous transfers govern end-to-end throughput, BeyondVRAM includes three standalone C++/CUDA programs:

1. **Dual-Stream Compute/DMA Overlap ([`src/async_pipeline.cu`](../src/async_pipeline.cu)):**
   - Models a layer executing dense attention + 7 resident GPU experts (`Stream 0`, `stream_compute`) while `Stream 1` (`stream_dma`) asynchronously fetches a `1.29 MiB` (`1,351,680 B`) cold expert from pinned host memory (`cudaMallocHost`) into double-buffered device staging memory.
   - Enforces bidirectional `cudaEvent_t` data dependencies (`router_ready_event[slot]` and `dma_ready_event[slot]`) so `cold_expert_swiglu_kernel` on `stream_compute` consumes the exact weights transferred by `stream_dma` (`0.00e+00` numerical diff vs. sequential execution).
   - **Measured Result (verified with NVIDIA Nsight Systems `nsys`):** Sequential execution takes **`0.464 ms/step`** (`0.264 ms` hot compute + `0.117 ms` HtoD DMA + `0.011 ms` cold expert + sync), whereas dual-stream pipelining completes in **`0.287 ms/step`** (**`38.1%` latency reduction / `1.61x` speedup**), hiding the `0.117 ms` PCIe transfer behind the `0.264 ms` GPU compute kernel.

2. **3-Matrix Quantized SwiGLU Cold-Expert Execution ([`src/zero_transfer_ffn.cpp`](../src/zero_transfer_ffn.cpp)):**
   - Compares three execution paths for a single `1.29 MiB` (`1,351,680 B`) quantized 3-matrix SwiGLU expert (`W_gate [512×2048]`, `W_up [512×2048]`, `W_down [2048×512]`, `d_model = 2048`, `d_ff = 512`), including full per-call GPU↔CPU activation transfer (`8 KiB` D2H + `8 KiB` H2D = `16 KiB`) and stream synchronization on the CPU path:
     - **GPU SLRU Cache Hit (`0 B` PCIe):** **`0.052 ms / expert`** (`3.68x` faster than PCIe upload, `2.83x` faster than CPU in-place)
     - **Cold Miss via PCIe `1.29 MiB` Weight Upload + GPU SwiGLU:** **`0.193 ms / expert`** (`~0.141 ms` PCIe H2D + `0.052 ms` GPU compute)
     - **Cold Miss via CPU AVX2 SwiGLU + `16 KiB` Activation Sync:** **`0.148 ms / expert`** (`0.099 ms` OpenMP AVX2 compute + `0.049 ms` GPU↔CPU activation transfer & sync)

3. **Reference C++ SLRU Cache & Unit Tests ([`src/beyond_vram_cache.h`](../src/beyond_vram_cache.h), [`src/beyond_vram_cache.cpp`](../src/beyond_vram_cache.cpp), [`tests/test_cache.cpp`](../tests/test_cache.cpp)):**
   - Standalone C++17 implementation of the Segmented-LRU slot allocator with deterministic unit tests covering probation insertion, promotion to protected on re-reference, 50% protected cap demotion, and probation-first victim eviction.
