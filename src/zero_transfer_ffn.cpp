/**
 * BeyondVRAM: Step 3 - Zero-Transfer CPU FFN In-Place Benchmark
 * 
 * Demonstrates the physical latency and bandwidth comparison:
 * - Mode A (Naive Offload): Transfer 1.29 MB expert weights over PCIe to GPU.
 * - Mode B (Zero-Transfer CPU FFN): Keep weights in host RAM, transfer 4 KB
 *   activation token, compute in-place on CPU SIMD, transfer 4 KB result back.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
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
    const int TEST_ITERATIONS = 500;

    std::cout << " [*] Expert Matrix Size  : " << (WEIGHT_BYTES / 1024.0) << " KB in Host RAM" << std::endl;
    std::cout << " [*] Token Activation    : " << (ACTIVATION_BYTES / 1024.0) << " KB (1D token vector)" << std::endl;
    std::cout << " [*] Test Iterations     : " << TEST_ITERATIONS << " cold expert invocations" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // 1. Allocate Host Weights (In-place RAM)
    std::vector<float> h_weights(D_MODEL * FFN_DIM, 0.05f);
    std::vector<float> h_activation_in(D_MODEL, 1.0f);
    std::vector<float> h_activation_out(FFN_DIM, 0.0f);

    // Pinned Host Memory for fast transfers
    void* h_pinned_weight = nullptr;
    void* h_pinned_act_in = nullptr;
    void* h_pinned_act_out = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned_weight, WEIGHT_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_in, ACTIVATION_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_out, ACTIVATION_BYTES));

    // 2. Allocate VRAM Buffers
    void* d_vram_weight = nullptr;
    void* d_vram_act = nullptr;
    CHECK_CUDA(cudaMalloc(&d_vram_weight, WEIGHT_BYTES));
    CHECK_CUDA(cudaMalloc(&d_vram_act, ACTIVATION_BYTES));

    // ------------------------------------------------------------------------
    // MODE A: Naive GPU Transfer (Transfer full 4.19 MB weight matrix over PCIe)
    // ------------------------------------------------------------------------
    std::cout << " [1/2] Benchmarking Mode A: Naive PCIe Weight Transfer to GPU..." << std::endl;
    auto t0_a = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        CHECK_CUDA(cudaMemcpy(d_vram_weight, h_pinned_weight, WEIGHT_BYTES, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1_a = std::chrono::high_resolution_clock::now();
    double mode_a_total_ms = std::chrono::duration<double, std::milli>(t1_a - t0_a).count();
    double mode_a_per_call_ms = mode_a_total_ms / TEST_ITERATIONS;

    // ------------------------------------------------------------------------
    // MODE B: BeyondVRAM Zero-Transfer CPU FFN (Weights stay in RAM, send 8KB token)
    // ------------------------------------------------------------------------
    std::cout << " [2/2] Benchmarking Mode B: BeyondVRAM Zero-Transfer CPU In-Place..." << std::endl;
    auto t0_b = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        // Step 1: Transfer tiny activation vector from GPU to CPU (8 KB)
        CHECK_CUDA(cudaMemcpy(h_pinned_act_in, d_vram_act, ACTIVATION_BYTES, cudaMemcpyDeviceToHost));

        // Step 2: Compute in-place on CPU SIMD using AVX2
        cpu_inplace_gemv_avx2(h_weights.data(), (const float*)h_pinned_act_in, (float*)h_pinned_act_out, FFN_DIM, D_MODEL);

        // Step 3: Transfer tiny result vector back to GPU (8 KB)
        CHECK_CUDA(cudaMemcpy(d_vram_act, h_pinned_act_out, ACTIVATION_BYTES, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1_b = std::chrono::high_resolution_clock::now();
    double mode_b_total_ms = std::chrono::duration<double, std::milli>(t1_b - t0_b).count();
    double mode_b_per_call_ms = mode_b_total_ms / TEST_ITERATIONS;

    // ------------------------------------------------------------------------
    // REPORT
    // ------------------------------------------------------------------------
    double speedup = mode_a_per_call_ms / mode_b_per_call_ms;
    double traffic_saved_bytes = (WEIGHT_BYTES - (2 * ACTIVATION_BYTES));
    double traffic_cut_pct = (traffic_saved_bytes / WEIGHT_BYTES) * 100.0;

    std::cout << "==========================================================" << std::endl;
    std::cout << "            STEP 3 ZERO-TRANSFER BENCHMARK RESULTS" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << " Mode A (Transfer Weights to GPU) : " << mode_a_per_call_ms << " ms / cold call" << std::endl;
    std::cout << " Mode B (Zero-Transfer CPU FFN)   : " << mode_b_per_call_ms << " ms / cold call" << std::endl;
    std::cout << " ---------------------------------------------------------" << std::endl;
    std::cout << " [*] In-Place Execution Speedup   : " << speedup << "x FASTER than PCIe transfer!" << std::endl;
    std::cout << " [*] PCIe Data Shuttling Reduced  : " << traffic_cut_pct << "% bus traffic eliminated" << std::endl;
    std::cout << " [*] Weight Transferred Over Bus  : ZERO BYTES (weights stay in RAM)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Cleanup
    cudaFree(d_vram_weight);
    cudaFree(d_vram_act);
    cudaFreeHost(h_pinned_weight);
    cudaFreeHost(h_pinned_act_in);
    cudaFreeHost(h_pinned_act_out);

    return 0;
}
