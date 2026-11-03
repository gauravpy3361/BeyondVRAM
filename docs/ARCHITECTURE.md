# BeyondVRAM: Technical Architecture Specification

## Part 1: The Core 3 System Pillars

BeyondVRAM replaces monolithic layer-by-layer offloading with fine-grained, asynchronous hardware orchestration designed for consumer hardware (e.g. 4GB–8GB laptop GPUs).

### 1. Hot Cache in VRAM & RAM (SLRU Dynamic Expert Caching)
- **Problem Formulation:** Standard GGUF offloading allocates memory layer-by-layer. For a 41-layer, 128-expert MoE model, each layer contains 128 experts, but a single token only routes to Top-8 experts. Storing entire layers in VRAM wastes 93.75% of capacity on dormant experts.
- **Architectural Solution:**
  1. Identify non-expert backbone tensors: Self-Attention ($Q, K, V, O$), Attention Norms, Gating Routers, Embeddings, and MTP Heads. Pin 100% of these in VRAM (~1.22 GB total).
  2. Reserve remaining free VRAM (~4.0 GB on a 6GB card) as a dynamic Segmented Least Recently Used (SLRU) cache pool.
  3. Divide the cache into fixed-size slots corresponding to individual expert FFN weights (~1.25 MB each).
  4. Maintain a probation segment (for newly activated experts) and a protected segment (for experts activated multiple times), preventing thrashing under shifting token distributions.

### 2. Split the Math (Async Compute & DMA Pipelining)
- **Problem Formulation:** Traditional inference operates sequentially:
  $$\text{Wait for PCIe Transfer} \longrightarrow \text{Compute Attention} \longrightarrow \text{Compute FFN}$$
  When an expert misses the VRAM cache, the GPU compute cores stall at 0% utilization while waiting for host memory transfers.
- **Architectural Solution:**
  1. Decouple execution across independent CUDA streams:
     - **Stream 0 (Compute):** Executes dense Attention, RoPE, and resident VRAM expert GEMMs.
     - **Stream 1 (DMA Copy):** Issues non-blocking asynchronous host-to-device transfers (`cudaMemcpyAsync` / `ggml_backend_tensor_set_async`).
  2. Exploit lookahead routing: The router determines upcoming expert assignments early in the layer graph, triggering DMA transfers across PCIe in parallel with active attention math.
  3. The transfer latency is hidden behind compute cycles, approaching zero visible PCIe delay.

### 3. Zero-Transfer CPU FFN (Asymmetric In-Place Execution)
- **Problem Formulation:** Shuffling multi-gigabyte FFN matrices across a PCIe 3.0/4.0 x4/x8 mobile bus destroys throughput (dropping speeds to 1–2 tokens/sec).
- **Architectural Solution:**
  1. MoE expert weights remain resident in system RAM.
  2. For cold expert activations that do not justify VRAM caching, computation executes **in-place on host CPU cores** using vectorized SIMD (AVX2 / AVX-512).
  3. **Zero weights travel over PCIe.** Only the tiny 1D token activation vector ($d_{\text{model}}$—a few kilobytes) is transferred from CPU to GPU over PCIe and accumulated into the residual stream.

---

## Part 2: Paired Architectural Extensions

### 4. Aligned Binary Chunks
- Restructuring model layout so each expert's weights are aligned to 4KB / 2MB memory page boundaries for zero-copy OS paging and direct virtual memory mapping.

### 5. Speculative Routing ($N+2$ Lookahead)
- Auxiliary lightweight router predictors determine layer $L+2$ expert selection while layer $L$ is executing, doubling the DMA prefetch window.

### 6. Multi-Token Prediction (MTP) Speculative Decoding
- Conditioned auxiliary transformer heads (`blk.40.nextn`) predict future tokens ($k=1, 2, 3$) in a single verification pass, achieving 66%+ draft acceptance without needing a separate draft model.
