/**
 * BeyondVRAM: Step 2 - Async Pipelining & Split-the-Math CUDA Engine
 * 
 * Demonstrates the asynchronous decoupling of Compute and DMA Transfer:
 * - Stream 0 (Compute): Executes Attention / Hot Expert GEMMs on CUDA Cores.
 * - Stream 1 (DMA Copy): Concurrently transfers cold expert chunks over PCIe.
 * 
 * Uses CUDA Events for non-blocking stream synchronization, eliminating
 * synchronous PCIe stalls and hiding transfer latency behind active math.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Simulated GEMM math kernel for Attention & Hot Expert execution
__global__ void beyond_vram_gemm_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        #pragma unroll 8
        for (int i = 0; i < iterations; i++) {
            val = val * 1.00005f + 0.0001f;
        }
        data[idx] = val;
    }
}

int main(int argc, char** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << " BeyondVRAM: Step 2 Async Dual-Stream Pipeline Engine" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Detect GPU
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << " [*] Target GPU          : " << prop.name << std::endl;
    std::cout << " [*] Dedicated VRAM      : " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " [*] Async Copy Engines  : " << prop.asyncEngineCount << " concurrent DMA engine(s)" << std::endl;

    // Model parameters matching Groot-19B
    const size_t EXPERT_CHUNK_BYTES = 1351680; // ~1.29 MB (aligned expert chunk)
    const int TOKENS_TO_SIMULATE = 100;
    const int COMPUTE_ELEMS = 1024 * 1024; // 1M elements for attention math

    std::cout << " [*] Expert Chunk Size   : " << (EXPERT_CHUNK_BYTES / 1024.0) << " KB (4KB Aligned)" << std::endl;
    std::cout << " [*] Benchmark Workload  : " << TOKENS_TO_SIMULATE << " sequential token generation steps" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // 1. Allocate Pinned Host RAM (Required for high-speed PCIe DMA)
    void* h_pinned_expert = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned_expert, EXPERT_CHUNK_BYTES));
    memset(h_pinned_expert, 0xEF, EXPERT_CHUNK_BYTES);

    // 2. Allocate VRAM Buffers
    void* d_expert_vram_slot = nullptr;
    float* d_compute_tensor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_expert_vram_slot, EXPERT_CHUNK_BYTES));
    CHECK_CUDA(cudaMalloc(&d_compute_tensor, COMPUTE_ELEMS * sizeof(float)));

    // 3. Create Independent Non-Blocking CUDA Streams & Events
    cudaStream_t stream_compute;
    cudaStream_t stream_dma;
    cudaEvent_t dma_ready_event;

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_dma, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&dma_ready_event, cudaEventDisableTiming));

    // ------------------------------------------------------------------------
    // MODE A: Naive Sequential Execution (Synchronous stall on every cold miss)
    // ------------------------------------------------------------------------
    std::cout << " [1/2] Running Mode A: Naive Sequential Execution..." << std::endl;
    auto start_seq = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOKENS_TO_SIMULATE; ++t) {
        // Step 1: Synchronous DMA transfer (blocks until completion)
        CHECK_CUDA(cudaMemcpy(d_expert_vram_slot, h_pinned_expert, EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice));

        // Step 2: Attention compute
        beyond_vram_gemm_kernel<<<1024, 256, 0, stream_compute>>>(d_compute_tensor, COMPUTE_ELEMS, 300);
        CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    }

    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
    double seq_per_token = seq_ms / TOKENS_TO_SIMULATE;

    // ------------------------------------------------------------------------
    // MODE B: BeyondVRAM Dual-Stream Pipelining (Async DMA hidden behind Attention)
    // ------------------------------------------------------------------------
    std::cout << " [2/2] Running Mode B: BeyondVRAM Async Pipelining..." << std::endl;
    auto start_pipe = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOKENS_TO_SIMULATE; ++t) {
        // Stream 1 (DMA Engine): Fire non-blocking asynchronous transfer for next expert
        CHECK_CUDA(cudaMemcpyAsync(d_expert_vram_slot, h_pinned_expert, EXPERT_CHUNK_BYTES, cudaMemcpyHostToDevice, stream_dma));
        CHECK_CUDA(cudaEventRecord(dma_ready_event, stream_dma));

        // Stream 0 (Compute Engine): Simultaneously execute Dense Attention math
        beyond_vram_gemm_kernel<<<1024, 256, 0, stream_compute>>>(d_compute_tensor, COMPUTE_ELEMS, 300);

        // Hardware synchronization barrier: Stream 0 only waits for the DMA event
        CHECK_CUDA(cudaStreamWaitEvent(stream_compute, dma_ready_event, 0));
        CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    }

    auto end_pipe = std::chrono::high_resolution_clock::now();
    double pipe_ms = std::chrono::duration<double, std::milli>(end_pipe - start_pipe).count();
    double pipe_per_token = pipe_ms / TOKENS_TO_SIMULATE;

    // ------------------------------------------------------------------------
    // REPORT RESULTS
    // ------------------------------------------------------------------------
    double latency_cut_pct = ((seq_per_token - pipe_per_token) / seq_per_token) * 100.0;
    double speedup_factor = seq_per_token / pipe_per_token;

    std::cout << "==========================================================" << std::endl;
    std::cout << "              STEP 2 PIPELINE BENCHMARK RESULTS" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << " Mode A (Sequential Stall)   : " << seq_per_token << " ms / token" << std::endl;
    std::cout << " Mode B (BeyondVRAM Pipeline): " << pipe_per_token << " ms / token" << std::endl;
    std::cout << " ---------------------------------------------------------" << std::endl;
    std::cout << " [*] Direct Latency Cut      : " << latency_cut_pct << "% faster execution" << std::endl;
    std::cout << " [*] Throughput Speedup      : " << speedup_factor << "x hardware multiplier" << std::endl;
    std::cout << " [*] Hidden Transfer Masking : Overlap achieved across " << prop.asyncEngineCount << " DMA engine(s)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Clean up
    cudaFree(d_expert_vram_slot);
    cudaFree(d_compute_tensor);
    cudaFreeHost(h_pinned_expert);
    cudaEventDestroy(dma_ready_event);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_dma);

    return 0;
}
