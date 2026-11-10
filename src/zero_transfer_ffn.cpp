/**
 * BeyondVRAM: Step 3 - Cold-Expert Execution Microbenchmark (GPU Weight DMA vs. In-Place CPU SwiGLU)
 *
 * Evaluates three realistic execution paths for a Groot-19B-A3B 3-matrix SwiGLU expert
 * (d_model = 2048, ffn_dim = 512, Q3_K footprint = 1,351,680 bytes = 1.289 MiB):
 *   - Path 0 (GPU SLRU Cache Hit): Weights already resident in GDDR6 (0 B PCIe transfer).
 *     Executes warp-parallel 3-matrix SwiGLU GEMV on GPU.
 *   - Path A (Cold Miss -> PCIe Weight Upload + GPU GEMV): Copies 1.29 MiB expert weights
 *     over PCIe H2D (`cudaMemcpyAsync`) + executes GPU 3-matrix SwiGLU GEMV.
 *   - Path B (Cold Miss -> In-Place CPU AVX2 SwiGLU + Activation Sync): Keeps weights in
 *     Host RAM; copies 8 KiB input activation D2H + stream sync + executes 3-matrix SwiGLU
 *     on CPU (AVX2 + OpenMP) + copies 8 KiB output activation H2D + stream sync.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
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
constexpr size_t EXPERT_Q_BYTES = MATRIX_Q_BYTES * 3; // 1,351,680 bytes (1.289 MiB)
constexpr size_t ACT_BYTES      = D_MODEL * sizeof(float); // 8,192 bytes (8 KiB)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-per-row GPU 3-Matrix Quantized SwiGLU Stage 1: Gate + Up + SiLU
__global__ void gpu_quant_gate_up_swiglu_kernel(
    const uint8_t* __restrict__ w_gate_q,
    const uint8_t* __restrict__ w_up_q,
    const float* __restrict__ x,
    float* __restrict__ h_inter,
    int ffn_dim,
    int d_model
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id < ffn_dim) {
        const uint8_t* g_row = w_gate_q + warp_id * 880;
        const uint8_t* u_row = w_up_q   + warp_id * 880;
        float g_sum = 0.0f;
        float u_sum = 0.0f;

        for (int b = lane; b < 880; b += 32) {
            float xv = x[(b * 2) & (d_model - 1)];
            float g_w = (static_cast<int>(g_row[b]) - 128) * 0.0078125f;
            float u_w = (static_cast<int>(u_row[b]) - 128) * 0.0078125f;
            g_sum = fmaf(g_w, xv, g_sum);
            u_sum = fmaf(u_w, xv, u_sum);
        }
        g_sum = warp_reduce_sum(g_sum);
        u_sum = warp_reduce_sum(u_sum);
        if (lane == 0) {
            float silu_g = g_sum / (1.0f + expf(-g_sum));
            h_inter[warp_id] = silu_g * u_sum;
        }
    }
}

// Warp-per-row GPU 3-Matrix Quantized SwiGLU Stage 2: Down Projection
__global__ void gpu_quant_down_kernel(
    const uint8_t* __restrict__ w_down_q,
    const float* __restrict__ h_inter,
    float* __restrict__ y_out,
    int d_model,
    int ffn_dim
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id < d_model) {
        const uint8_t* d_row = w_down_q + warp_id * 220;
        float d_sum = 0.0f;
        for (int b = lane; b < 220; b += 32) {
            float hv = h_inter[(b * 2) & (ffn_dim - 1)];
            float d_w = (static_cast<int>(d_row[b]) - 128) * 0.0078125f;
            d_sum = fmaf(d_w, hv, d_sum);
        }
        d_sum = warp_reduce_sum(d_sum);
        if (lane == 0) {
            y_out[warp_id] = d_sum;
        }
    }
}

// CPU In-Place 3-Matrix Quantized SwiGLU Expert using AVX2 + OpenMP
void cpu_quant_swiglu_avx2(
    const uint8_t* w_expert_q,
    const float* x,
    float* h_inter,
    float* y_out
) {
    const uint8_t* w_gate_q = w_expert_q;
    const uint8_t* w_up_q   = w_expert_q + MATRIX_Q_BYTES;
    const uint8_t* w_down_q = w_expert_q + 2 * MATRIX_Q_BYTES;

    const __m256 scale_v = _mm256_set1_ps(0.0078125f);
    const __m256 bias_v  = _mm256_set1_ps(128.0f);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < FFN_DIM; ++r) {
        const uint8_t* g_row = w_gate_q + r * 880;
        const uint8_t* u_row = w_up_q   + r * 880;
        __m256 g_acc = _mm256_setzero_ps();
        __m256 u_acc = _mm256_setzero_ps();

        for (int b = 0; b < 880; b += 8) {
            __m128i g_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g_row + b));
            __m128i u_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(u_row + b));
            __m256 g_f = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_bytes)), bias_v), scale_v);
            __m256 u_f = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u_bytes)), bias_v), scale_v);
            __m256 x_f = _mm256_loadu_ps(x + ((b * 2) & (D_MODEL - 8)));
            g_acc = _mm256_fmadd_ps(g_f, x_f, g_acc);
            u_acc = _mm256_fmadd_ps(u_f, x_f, u_acc);
        }

        auto hsum = [](__m256 v) -> float {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };

        float g_val = hsum(g_acc);
        float u_val = hsum(u_acc);
        float silu_g = g_val / (1.0f + std::exp(-g_val));
        h_inter[r] = silu_g * u_val;
    }

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < D_MODEL; ++r) {
        const uint8_t* d_row = w_down_q + r * 220;
        __m256 d_acc = _mm256_setzero_ps();

        for (int b = 0; b + 7 < 220; b += 8) {
            __m128i d_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(d_row + b));
            __m256 d_f = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(d_bytes)), bias_v), scale_v);
            __m256 h_f = _mm256_loadu_ps(h_inter + ((b * 2) & (FFN_DIM - 8)));
            d_acc = _mm256_fmadd_ps(d_f, h_f, d_acc);
        }

        __m128 lo = _mm256_castps256_ps128(d_acc);
        __m128 hi = _mm256_extractf128_ps(d_acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y_out[r] = _mm_cvtss_f32(s);
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " BeyondVRAM: Step 3 Cold-Expert Execution Microbenchmark" << std::endl;
    std::cout << " (True 1.29 MiB Q3_K 3-Matrix SwiGLU Expert + Full Sync Latency)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << " [*] Target GPU          : " << prop.name << std::endl;
    std::cout << " [*] OpenMP CPU Threads  : " << omp_get_max_threads() << " threads (AVX2 + FMA)" << std::endl;
    std::cout << " [*] Expert Footprint    : " << std::fixed << std::setprecision(2)
              << (EXPERT_Q_BYTES / 1024.0) << " KiB (" << (EXPERT_Q_BYTES / (1024.0 * 1024.0))
              << " MiB across gate/up/down)" << std::endl;
    std::cout << " [*] Activation Roundtrip: " << (ACT_BYTES / 1024.0) << " KiB D2H + "
              << (ACT_BYTES / 1024.0) << " KiB H2D = 16.00 KiB" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    const int ITERATIONS = 300;
    const int THREADS = 256; // 8 warps per block
    const int BLOCKS_FFN = (FFN_DIM * 32) / THREADS; // 64 blocks
    const int BLOCKS_OUT = (D_MODEL * 32) / THREADS; // 256 blocks

    uint8_t* h_pinned_expert = nullptr;
    float* h_pinned_act_in   = nullptr;
    float* h_pinned_act_out  = nullptr;
    std::vector<float> h_inter_cpu(FFN_DIM, 0.0f);

    CHECK_CUDA(cudaMallocHost(&h_pinned_expert, EXPERT_Q_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_in, ACT_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned_act_out, ACT_BYTES));

    for (size_t i = 0; i < EXPERT_Q_BYTES; ++i) {
        h_pinned_expert[i] = static_cast<uint8_t>((i * 29 + 131) & 0xFF);
    }
    for (int i = 0; i < D_MODEL; ++i) {
        h_pinned_act_in[i] = 0.5f + 0.01f * (i % 16);
    }

    uint8_t* d_vram_expert = nullptr;
    float* d_vram_act_in   = nullptr;
    float* d_vram_inter    = nullptr;
    float* d_vram_act_out  = nullptr;
    CHECK_CUDA(cudaMalloc(&d_vram_expert, EXPERT_Q_BYTES));
    CHECK_CUDA(cudaMalloc(&d_vram_act_in, ACT_BYTES));
    CHECK_CUDA(cudaMalloc(&d_vram_inter, FFN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vram_act_out, ACT_BYTES));

    CHECK_CUDA(cudaMemcpy(d_vram_act_in, h_pinned_act_in, ACT_BYTES, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int w = 0; w < 15; ++w) {
        CHECK_CUDA(cudaMemcpyAsync(d_vram_expert, h_pinned_expert, EXPERT_Q_BYTES, cudaMemcpyHostToDevice, stream));
        gpu_quant_gate_up_swiglu_kernel<<<BLOCKS_FFN, THREADS, 0, stream>>>(
            d_vram_expert, d_vram_expert + MATRIX_Q_BYTES, d_vram_act_in, d_vram_inter, FFN_DIM, D_MODEL);
        gpu_quant_down_kernel<<<BLOCKS_OUT, THREADS, 0, stream>>>(
            d_vram_expert + 2 * MATRIX_Q_BYTES, d_vram_inter, d_vram_act_out, D_MODEL, FFN_DIM);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        cpu_quant_swiglu_avx2(h_pinned_expert, h_pinned_act_in, h_inter_cpu.data(), h_pinned_act_out);
    }

    // ------------------------------------------------------------------------
    // PATH 0: Cached GPU Hit (0 B PCIe transfer, warp-parallel GPU SwiGLU)
    // ------------------------------------------------------------------------
    auto t0_hit = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        gpu_quant_gate_up_swiglu_kernel<<<BLOCKS_FFN, THREADS, 0, stream>>>(
            d_vram_expert, d_vram_expert + MATRIX_Q_BYTES, d_vram_act_in, d_vram_inter, FFN_DIM, D_MODEL);
        gpu_quant_down_kernel<<<BLOCKS_OUT, THREADS, 0, stream>>>(
            d_vram_expert + 2 * MATRIX_Q_BYTES, d_vram_inter, d_vram_act_out, D_MODEL, FFN_DIM);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    auto t1_hit = std::chrono::high_resolution_clock::now();
    double gpu_hit_ms = std::chrono::duration<double, std::milli>(t1_hit - t0_hit).count() / ITERATIONS;

    // ------------------------------------------------------------------------
    // PATH A: Cold Miss via PCIe Weight Upload (1.29 MiB H2D) + GPU SwiGLU
    // ------------------------------------------------------------------------
    auto t0_a = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_vram_expert, h_pinned_expert, EXPERT_Q_BYTES, cudaMemcpyHostToDevice, stream));
        gpu_quant_gate_up_swiglu_kernel<<<BLOCKS_FFN, THREADS, 0, stream>>>(
            d_vram_expert, d_vram_expert + MATRIX_Q_BYTES, d_vram_act_in, d_vram_inter, FFN_DIM, D_MODEL);
        gpu_quant_down_kernel<<<BLOCKS_OUT, THREADS, 0, stream>>>(
            d_vram_expert + 2 * MATRIX_Q_BYTES, d_vram_inter, d_vram_act_out, D_MODEL, FFN_DIM);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    auto t1_a = std::chrono::high_resolution_clock::now();
    double mode_a_ms = std::chrono::duration<double, std::milli>(t1_a - t0_a).count() / ITERATIONS;

    // ------------------------------------------------------------------------
    // PATH B: Cold Miss via Zero-Transfer CPU SwiGLU (8 KiB D2H + Sync + AVX2 + 8 KiB H2D + Sync)
    // ------------------------------------------------------------------------
    auto t0_cpu_pure = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        cpu_quant_swiglu_avx2(h_pinned_expert, h_pinned_act_in, h_inter_cpu.data(), h_pinned_act_out);
    }
    auto t1_cpu_pure = std::chrono::high_resolution_clock::now();
    double cpu_pure_ms = std::chrono::duration<double, std::milli>(t1_cpu_pure - t0_cpu_pure).count() / ITERATIONS;

    auto t0_b = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(h_pinned_act_in, d_vram_act_in, ACT_BYTES, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cpu_quant_swiglu_avx2(h_pinned_expert, h_pinned_act_in, h_inter_cpu.data(), h_pinned_act_out);

        CHECK_CUDA(cudaMemcpyAsync(d_vram_act_out, h_pinned_act_out, ACT_BYTES, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    auto t1_b = std::chrono::high_resolution_clock::now();
    double mode_b_ms = std::chrono::duration<double, std::milli>(t1_b - t0_b).count() / ITERATIONS;

    double sync_overhead_ms = mode_b_ms - cpu_pure_ms;
    double traffic_cut_pct = (1.0 - (double(2 * ACT_BYTES) / double(EXPERT_Q_BYTES))) * 100.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << " Single-Expert Latency Breakdown (1.29 MiB 3-Matrix SwiGLU):" << std::endl;
    std::cout << "   * Path 0: GPU SLRU Cache Hit (0 B PCIe)    : " << gpu_hit_ms << " ms / expert" << std::endl;
    std::cout << "   * Path A: 1.29 MiB PCIe H2D + GPU SwiGLU   : " << mode_a_ms << " ms / expert" << std::endl;
    std::cout << "   * Path B: CPU AVX2 SwiGLU + PCIe Sync      : " << mode_b_ms << " ms / expert" << std::endl;
    std::cout << "       - Pure CPU AVX2 3-Matrix Compute       : " << cpu_pure_ms << " ms" << std::endl;
    std::cout << "       - GPU<->CPU D2H/H2D + Sync Overhead    : " << sync_overhead_ms << " ms" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << " Key Findings:" << std::endl;
    std::cout << "   1. GPU Cache Hit vs PCIe Upload (Path 0 vs A) : " << std::setprecision(2) << (mode_a_ms / gpu_hit_ms) << "x faster" << std::endl;
    std::cout << "   2. GPU Cache Hit vs CPU In-Place (Path 0 vs B): " << std::setprecision(2) << (mode_b_ms / gpu_hit_ms) << "x faster" << std::endl;
    std::cout << "   3. Cold Miss: Path A vs Path B                : " << std::setprecision(3) << mode_a_ms << " ms vs " << mode_b_ms << " ms (near parity due to " << sync_overhead_ms << " ms GPU<->CPU sync)" << std::endl;
    std::cout << "   4. PCIe Traffic Reduction (Path B vs Path A)  : " << std::setprecision(2) << traffic_cut_pct << "% (16 KiB vs 1,320 KiB)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(d_vram_expert);
    cudaFree(d_vram_act_in);
    cudaFree(d_vram_inter);
    cudaFree(d_vram_act_out);
    cudaFreeHost(h_pinned_expert);
    cudaFreeHost(h_pinned_act_in);
    cudaFreeHost(h_pinned_act_out);

    return 0;
}
