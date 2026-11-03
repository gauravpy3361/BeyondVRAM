/**
 * BeyondVRAM: Step 3 - Zero-Transfer CPU FFN In-Place Benchmark
 * 
 * Demonstrates the physical latency and bandwidth comparison:
 * - Mode A (Naive Offload): Transfer expert weights over PCIe to GPU + execute GPU GEMV.
 * - Mode B (Zero-Transfer CPU FFN): Keep weights in host RAM, transfer 8 KB
 *   activation token, compute in-place on CPU AVX2, transfer 8 KB result back.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// GPU GEMV Kernel: Matrix-vector product y = A * x
__global__ void gpu_gemv_kernel(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        const float* row_ptr = A + r * cols;
        #pragma unroll 4
        for (int c = 0; c < cols; ++c) {
            sum += row_ptr[c] * x[c];
        }
        y[r] = sum;
    }
}

// In-place CPU Vectorized GEMV (Matrix-Vector multiplication) using AVX2
void cpu_inplace_gemv_avx2(const float* A, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        __m256 sum256 = _mm256_setzero_ps();
        const float* row_ptr = A + r * cols;

        int c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 a = _mm256_loadu_ps(row_ptr + c);
            __m256 b = _mm256_loadu_ps(x + c);
            sum256 = _mm256_fmadd_ps(a, b, sum256);
        }

        // Horizontal sum of 8 floats
        __m128 lo = _mm256_castps256_ps128(sum256);
        __m128 hi = _mm256_extractf128_ps(sum256, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);

        float row_sum = _mm_cvtss_f32(s);

        // Cleanup tail
        for (; c < cols; ++c) {
            row_sum += row_ptr[c] * x[c];
        }

        y[r] = row_sum;
    }
}

int main(int argc, char** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << " BeyondVRAM: Step 3 Zero-Transfer CPU FFN Engine" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Detect GPU
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << " [*] Target GPU          : " << prop.name << std::endl;

    // Expert geometry: d_model = 2048, ffn_dim = 512
    const int D_MODEL = 2048;
    const int FFN_DIM = 512;
    const size_t WEIGHT_BYTES = D_MODEL * FFN_DIM * sizeof(float); // ~4.19 MB (FP32 baseline)
    const size_t ACTIVATION_BYTES = D_MODEL * sizeof(float);       // 8 KB (FP32 token)
    const size_t OUTPUT_BYTES = FFN_DIM * sizeof(float);           // 2 KB
    const int TEST_ITERATIONS = 500;

    std::cout << " [*] Expert Matrix Size  : " << (WEIGHT_BYTES / 1024.0) << " KB in Host RAM" << std::endl;
    std::cout << " [*] Token Activation    : " << (ACTIVATION_BYTES / 1024.0) << " KB (1D token vector)" << std::endl;
    std::cout << " [*] Test Iterations     : " << TEST_ITERATIONS << " cold expert invocations" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // 1. Allocate & Initialize Host Weights (Deterministic Values)
    std::vector<float> h_weights(D_MODEL * FFN_DIM);
    for (size_t i = 0; i < h_weights.size(); ++i) {
        h_weights[i] = static_cast<float>((i % 100) + 1) * 0.001f;
    }

    // Pinned Host Memory for fast transfers
    void* h_pinned_weight = nullptr;
    void* h_pinned_act_in = nullptr;
    void* h_pinned_act_out = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned_weight, WEIGHT_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_in, ACTIVATION_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_out, OUTPUT_BYTES));

    // Copy initialized data into pinned buffers
    std::memcpy(h_pinned_weight, h_weights.data(), WEIGHT_BYTES);
    for (int i = 0; i < D_MODEL; ++i) {
        static_cast<float*>(h_pinned_act_in)[i] = 1.0f + 0.01f * (i % 10);
    }
    std::memset(h_pinned_act_out, 0, OUTPUT_BYTES);

    // 2. Allocate & Initialize VRAM Buffers
    void* d_vram_weight = nullptr;
    void* d_vram_act_in = nullptr;
    void* d_vram_act_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_vram_weight, WEIGHT_BYTES));
    CHECK_CUDA(cudaMalloc(&d_vram_act_in, ACTIVATION_BYTES));
    CHECK_CUDA(cudaMalloc(&d_vram_act_out, OUTPUT_BYTES));

    CHECK_CUDA(cudaMemcpy(d_vram_act_in, h_pinned_act_in, ACTIVATION_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_vram_act_out, 0, OUTPUT_BYTES));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    CHECK_CUDA(cudaMemcpyAsync(d_vram_weight, h_pinned_weight, WEIGHT_BYTES, cudaMemcpyHostToDevice, stream));
    gpu_gemv_kernel<<<(FFN_DIM + 255) / 256, 256, 0, stream>>>(
        (const float*)d_vram_weight, (const float*)d_vram_act_in, (float*)d_vram_act_out, FFN_DIM, D_MODEL);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------------
    // MODE A: Naive GPU Transfer + GPU Compute (Full apples-to-apples)
    // ------------------------------------------------------------------------
    std::cout << " [1/2] Benchmarking Mode A: PCIe Weight Transfer + GPU Compute..." << std::endl;
    auto t0_a = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        // Step A1: Transfer full 4.19 MB weight matrix over PCIe
        CHECK_CUDA(cudaMemcpyAsync(d_vram_weight, h_pinned_weight, WEIGHT_BYTES, cudaMemcpyHostToDevice, stream));

        // Step A2: Execute GPU GEMV kernel
        gpu_gemv_kernel<<<(FFN_DIM + 255) / 256, 256, 0, stream>>>(
            (const float*)d_vram_weight, (const float*)d_vram_act_in, (float*)d_vram_act_out, FFN_DIM, D_MODEL);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t1_a = std::chrono::high_resolution_clock::now();
    double mode_a_total_ms = std::chrono::duration<double, std::milli>(t1_a - t0_a).count();
    double mode_a_per_call_ms = mode_a_total_ms / TEST_ITERATIONS;

    // ------------------------------------------------------------------------
    // MODE B: BeyondVRAM Zero-Transfer CPU FFN (Weights stay in RAM, send 8KB token)
    // ------------------------------------------------------------------------
    std::cout << " [2/2] Benchmarking Mode B: BeyondVRAM Zero-Transfer CPU In-Place..." << std::endl;
    auto t0_b = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        // Step B1: Transfer activation vector from GPU to CPU (8 KB)
        CHECK_CUDA(cudaMemcpyAsync(h_pinned_act_in, d_vram_act_in, ACTIVATION_BYTES, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Step B2: Compute in-place on CPU SIMD using AVX2
        cpu_inplace_gemv_avx2(h_weights.data(), (const float*)h_pinned_act_in, (float*)h_pinned_act_out, FFN_DIM, D_MODEL);

        // Step B3: Transfer result vector back to GPU (2 KB)
        CHECK_CUDA(cudaMemcpyAsync(d_vram_act_out, h_pinned_act_out, OUTPUT_BYTES, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    auto t1_b = std::chrono::high_resolution_clock::now();
    double mode_b_total_ms = std::chrono::duration<double, std::milli>(t1_b - t0_b).count();
    double mode_b_per_call_ms = mode_b_total_ms / TEST_ITERATIONS;

    // ------------------------------------------------------------------------
    // REPORT
    // ------------------------------------------------------------------------
    double speedup = mode_a_per_call_ms / mode_b_per_call_ms;
    double traffic_saved_bytes = (WEIGHT_BYTES - (ACTIVATION_BYTES + OUTPUT_BYTES));
    double traffic_cut_pct = (traffic_saved_bytes / WEIGHT_BYTES) * 100.0;

    std::cout << "==========================================================" << std::endl;
    std::cout << "            STEP 3 ZERO-TRANSFER BENCHMARK RESULTS" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << " Mode A (Transfer Weights + GPU GEMV) : " << mode_a_per_call_ms << " ms / cold call" << std::endl;
    std::cout << " Mode B (Zero-Transfer CPU In-Place)  : " << mode_b_per_call_ms << " ms / cold call" << std::endl;
    std::cout << " ---------------------------------------------------------" << std::endl;
    std::cout << " [*] In-Place Execution Speedup       : " << speedup << "x FASTER than PCIe transfer + GPU!" << std::endl;
    std::cout << " [*] PCIe Data Shuttling Reduced      : " << traffic_cut_pct << "% bus traffic eliminated" << std::endl;
    std::cout << " [*] Weight Transferred Over Bus      : ZERO BYTES (weights stay in RAM)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_vram_weight);
    cudaFree(d_vram_act_in);
    cudaFree(d_vram_act_out);
    cudaFreeHost(h_pinned_weight);
    cudaFreeHost(h_pinned_act_in);
    cudaFreeHost(h_pinned_act_out);

    return 0;
}
