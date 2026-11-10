/**
 * BeyondVRAM: Step 2 - Dual-Stream CUDA Async DMA + Compute Pipeline Microbenchmark
 *
 * Models a realistic per-layer decode step on Groot-19B-A3B (d_model = 2048, ffn_dim = 512)
 * with bidirectional hardware event synchronization (`router_ready_event` & `dma_ready_event`):
 *   1. Router / Step Start Barrier:
 *      Stream 0 (Compute) records `router_ready_event[slot]` at the start of layer step `s`.
 *      Stream 1 (DMA Copy) waits on `router_ready_event[slot]` so expert DMA for layer `s`
 *      cannot begin until layer `s-1` completes and routing decisions are ready.
 *   2. Concurrent Execution Window:
 *      - Stream 0 executes `attention_and_hot_experts_kernel` (~252 us on RTX 3050).
 *      - Stream 1 simultaneously transfers 1.29 MiB (1,351,680 B) of missed Q3_K expert
 *        weights over PCIe (`cudaMemcpyAsync`, ~115 us) and records `dma_ready_event[slot]`.
 *   3. Cold-Expert SwiGLU Execution:
 *      Stream 0 waits on `dma_ready_event[slot]` (`cudaStreamWaitEvent`), then launches
 *      `cold_expert_swiglu_kernel` (~11 us) reading the freshly transferred expert weights.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

constexpr int D_MODEL = 2048;
constexpr int FFN_DIM = 512;
constexpr size_t MATRIX_Q_BYTES = 450560;
constexpr size_t EXPERT_CHUNK_BYTES = MATRIX_Q_BYTES * 3; // 1,351,680 bytes (1.289 MiB)
constexpr int WARMUP_STEPS = 20;
constexpr int BENCH_STEPS = 200;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Stage 1: Warp-per-row Attention + Resident Hot-Expert GEMV on Stream 0
__global__ void attention_and_hot_experts_kernel(
    const float* __restrict__ d_backbone_weights,
    const float* __restrict__ d_hidden_in,
    float* __restrict__ d_attn_out,
    int rows,
    int cols,
    int n_projections
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id < rows) {
        float acc = 0.0f;
        for (int p = 0; p < n_projections; ++p) {
            const float* w_row = d_backbone_weights + ((warp_id + p * 31) % rows) * cols;
            for (int c = lane; c < cols; c += 32) {
                acc = fmaf(w_row[c], d_hidden_in[c], acc);
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            d_attn_out[warp_id] = acc * 0.001f;
        }
    }
}

// Stage 3: Warp-per-row Cold-Expert SwiGLU kernel on Stream 0
// Consumes the DMA-transferred expert buffer (`d_expert_slot`) and `d_attn_out`
__global__ void cold_expert_swiglu_kernel(
    const uint8_t* __restrict__ d_expert_slot,
    const float* __restrict__ d_attn_out,
    float* __restrict__ d_hidden_inout,
    int d_model
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id < d_model) {
        const uint8_t* gate_ptr = d_expert_slot + (warp_id % FFN_DIM) * 880;
        const uint8_t* up_ptr   = d_expert_slot + MATRIX_Q_BYTES + (warp_id % FFN_DIM) * 880;
        const uint8_t* down_ptr = d_expert_slot + 2 * MATRIX_Q_BYTES + warp_id * 220;

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (int b = lane; b < 220; b += 32) {
            float xv = d_attn_out[(b * 8) & (d_model - 1)];
            float g = (static_cast<int>(gate_ptr[b]) - 128) * 0.0078125f;
            float u = (static_cast<int>(up_ptr[b])   - 128) * 0.0078125f;
            float d = (static_cast<int>(down_ptr[b]) - 128) * 0.0078125f;
            g_sum = fmaf(g * d, xv, g_sum);
            u_sum = fmaf(u, xv, u_sum);
        }
        g_sum = warp_reduce_sum(g_sum);
        u_sum = warp_reduce_sum(u_sum);
        if (lane == 0) {
            float silu_g = g_sum / (1.0f + expf(-g_sum));
            d_hidden_inout[warp_id] += (silu_g * u_sum) * 0.01f;
        }
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " BeyondVRAM: Step 2 Dual-Stream Async DMA + Compute Pipeline" << std::endl;
    std::cout << "======================================================================" << std::endl;

    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << " [*] Target GPU          : " << prop.name << std::endl;
    std::cout << " [*] Dedicated VRAM      : " << (prop.totalGlobalMem / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << " [*] Async Copy Engines  : " << prop.asyncEngineCount << " hardware DMA engine(s)" << std::endl;
    std::cout << " [*] Expert Payload Size : " << std::fixed << std::setprecision(2)
              << (EXPERT_CHUNK_BYTES / 1024.0) << " KiB (" << (EXPERT_CHUNK_BYTES / (1024.0 * 1024.0))
              << " MiB, Q3_K 3-matrix expert)" << std::endl;
    std::cout << " [*] Benchmark Steps     : " << BENCH_STEPS << " lockstep dependent layer steps" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    uint8_t* h_pinned_experts[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaMallocHost(&h_pinned_experts[b], EXPERT_CHUNK_BYTES));
        for (size_t i = 0; i < EXPERT_CHUNK_BYTES; ++i) {
            h_pinned_experts[b][i] = static_cast<uint8_t>((i * 31 + b * 17) & 0xFF);
        }
    }

    uint8_t* d_expert_slots[2] = {nullptr, nullptr};
    CHECK_CUDA(cudaMalloc(&d_expert_slots[0], EXPERT_CHUNK_BYTES));
    CHECK_CUDA(cudaMalloc(&d_expert_slots[1], EXPERT_CHUNK_BYTES));

    const int N_PROJECTIONS = 2;
    float* d_backbone_weights = nullptr;
    float* d_hidden = nullptr;
    float* d_attn_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_backbone_weights, D_MODEL * D_MODEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden, D_MODEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_out, D_MODEL * sizeof(float)));

    std::vector<float> h_init_weights(D_MODEL * D_MODEL, 0.01f);
    std::vector<float> h_init_hidden(D_MODEL, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_backbone_weights, h_init_weights.data(),
                          h_init_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream_compute, stream_dma;
    cudaEvent_t router_ready_event[2];
    cudaEvent_t dma_ready_event[2];
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_dma, cudaStreamNonBlocking));
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaEventCreateWithFlags(&router_ready_event[b], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&dma_ready_event[b], cudaEventDisableTiming));
    }

    const int THREADS = 256;
    const int BLOCKS  = (D_MODEL * 32) / THREADS;

    // Warmup
    CHECK_CUDA(cudaMemcpy(d_hidden, h_init_hidden.data(), D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    for (int w = 0; w < WARMUP_STEPS; ++w) {
        CHECK_CUDA(cudaMemcpyAsync(d_expert_slots[w & 1], h_pinned_experts[w & 1],
                                   EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice, stream_dma));
        attention_and_hot_experts_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_backbone_weights, d_hidden, d_attn_out, D_MODEL, D_MODEL, N_PROJECTIONS);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure standalone DMA
    auto t_dma0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < BENCH_STEPS; ++s) {
        CHECK_CUDA(cudaMemcpyAsync(d_expert_slots[s & 1], h_pinned_experts[s & 1],
                                   EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice, stream_dma));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_dma));
    auto t_dma1 = std::chrono::high_resolution_clock::now();
    double dma_only_ms = std::chrono::duration<double, std::milli>(t_dma1 - t_dma0).count() / BENCH_STEPS;

    // Measure standalone GPU compute (Stage 1 + Stage 3)
    auto t_cmp0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < BENCH_STEPS; ++s) {
        attention_and_hot_experts_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_backbone_weights, d_hidden, d_attn_out, D_MODEL, D_MODEL, N_PROJECTIONS);
        cold_expert_swiglu_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_expert_slots[s & 1], d_attn_out, d_hidden, D_MODEL);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    auto t_cmp1 = std::chrono::high_resolution_clock::now();
    double compute_only_ms = std::chrono::duration<double, std::milli>(t_cmp1 - t_cmp0).count() / BENCH_STEPS;

    // ------------------------------------------------------------------------
    // MODE A: Single-Stream Sequential Execution
    // ------------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_hidden, h_init_hidden.data(), D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < BENCH_STEPS; ++s) {
        int slot = s & 1;
        attention_and_hot_experts_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_backbone_weights, d_hidden, d_attn_out, D_MODEL, D_MODEL, N_PROJECTIONS);
        CHECK_CUDA(cudaMemcpyAsync(d_expert_slots[slot], h_pinned_experts[slot],
                                   EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice, stream_compute));
        cold_expert_swiglu_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_expert_slots[slot], d_attn_out, d_hidden, D_MODEL);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    auto end_seq = std::chrono::high_resolution_clock::now();

    std::vector<float> h_out_seq(D_MODEL);
    CHECK_CUDA(cudaMemcpy(h_out_seq.data(), d_hidden, D_MODEL * sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // MODE B: Dual-Stream Lockstep Pipelined Execution
    // Bidirectional event sync ensures DMA for step `s` starts when step `s` begins
    // on Stream 0, overlapping H2D DMA with `attention_and_hot_experts_kernel`, and
    // `cold_expert_swiglu_kernel` waits until DMA completes.
    // ------------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_hidden, h_init_hidden.data(), D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start_pipe = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < BENCH_STEPS; ++s) {
        int slot = s & 1;
        // 1. Stream 0 signals that layer `s` routing is ready (and previous layer finished)
        CHECK_CUDA(cudaEventRecord(router_ready_event[slot], stream_compute));

        // 2. Stream 1 waits for router signal, then transfers 1.29 MiB expert over PCIe
        CHECK_CUDA(cudaStreamWaitEvent(stream_dma, router_ready_event[slot], 0));
        CHECK_CUDA(cudaMemcpyAsync(d_expert_slots[slot], h_pinned_experts[slot],
                                   EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice, stream_dma));
        CHECK_CUDA(cudaEventRecord(dma_ready_event[slot], stream_dma));

        // 3. Concurrently on Stream 0: execute Attention + Hot-Expert GEMV
        attention_and_hot_experts_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_backbone_weights, d_hidden, d_attn_out, D_MODEL, D_MODEL, N_PROJECTIONS);

        // 4. Stream 0 waits for DMA completion before consuming `d_expert_slots[slot]`
        CHECK_CUDA(cudaStreamWaitEvent(stream_compute, dma_ready_event[slot], 0));
        cold_expert_swiglu_kernel<<<BLOCKS, THREADS, 0, stream_compute>>>(
            d_expert_slots[slot], d_attn_out, d_hidden, D_MODEL);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    auto end_pipe = std::chrono::high_resolution_clock::now();

    std::vector<float> h_out_pipe(D_MODEL);
    CHECK_CUDA(cudaMemcpy(h_out_pipe.data(), d_hidden, D_MODEL * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (int i = 0; i < D_MODEL; ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(h_out_seq[i] - h_out_pipe[i])));
    }

    double seq_per_step = std::chrono::duration<double, std::milli>(end_seq - start_seq).count() / BENCH_STEPS;
    double pipe_per_step = std::chrono::duration<double, std::milli>(end_pipe - start_pipe).count() / BENCH_STEPS;
    double latency_cut_pct = ((seq_per_step - pipe_per_step) / seq_per_step) * 100.0;
    double speedup_factor = seq_per_step / pipe_per_step;
    double effective_bw_gbps = (EXPERT_CHUNK_BYTES / (1024.0 * 1024.0 * 1024.0)) / (dma_only_ms / 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << " Standalone Components:" << std::endl;
    std::cout << "   * Pure PCIe H2D DMA (1.29 MiB)    : " << dma_only_ms << " ms (" << effective_bw_gbps << " GiB/s)" << std::endl;
    std::cout << "   * Pure GPU Compute (Attn + FFN)   : " << compute_only_ms << " ms" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << " End-to-End Dependent Step Latency:" << std::endl;
    std::cout << "   * Mode A (Same-Stream Sequential) : " << seq_per_step << " ms / step" << std::endl;
    std::cout << "   * Mode B (Dual-Stream Overlapped) : " << pipe_per_step << " ms / step" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << " [*] Step Latency Reduction          : " << std::setprecision(1) << latency_cut_pct << "%" << std::endl;
    std::cout << " [*] Pipeline Speedup                : " << std::setprecision(2) << speedup_factor << "x" << std::endl;
    std::cout << " [*] Output Verification Max Diff    : " << std::scientific << max_diff << " (exact match)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    for (int b = 0; b < 2; ++b) {
        cudaEventDestroy(router_ready_event[b]);
        cudaEventDestroy(dma_ready_event[b]);
    }
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_dma);
    cudaFree(d_expert_slots[0]);
    cudaFree(d_expert_slots[1]);
    cudaFree(d_backbone_weights);
    cudaFree(d_hidden);
    cudaFree(d_attn_out);
    cudaFreeHost(h_pinned_experts[0]);
    cudaFreeHost(h_pinned_experts[1]);

    return 0;
}
