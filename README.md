# BeyondVRAM

MoE models are big but only use a few experts per token, so most of their weights sit idle, yet they still have to live somewhere. On a 6 GB GPU, that "somewhere" is slow system RAM.

**Core insight:** expert use is skewed. A cache holding only **49%** of all experts serves about **89%** of expert requests from VRAM, so an SLRU (segmented LRU) expert cache works.

**Result:** **33.8 tok/s** decoding a 19B MoE model (Groot-19B-A3B, `Q3_K_S`) with a 128K context on a 6 GB RTX 3050 laptop GPU. That is 2.3x stock `llama.cpp` and 30% faster than a hand-tuned static baseline.

![Benchmark results](docs/assets/benchmark_results.png)

---

## Results

Decode speed (median tok/s), 128K context reserved, ~900-token prompt, 200 generated tokens, greedy (`--temp 0`), 2–3 runs each.

| Setup | MTP on | MTP off | Cache hit rate |
|---|---:|---:|---:|
| Stock `llama.cpp` (`-fit on`) | 14.8 | – | – |
| Hand-tuned static (`--n-cpu-moe 23`) | 20.6 | 26.1 | – |
| BeyondVRAM, 3300 MiB cache | 23.4 | **33.8** | ~89% |
| BeyondVRAM, hybrid (static + 1650 MiB cache) | **24.0** | – | 75% |

**What beats what, and why the baselines matter:** stock `-fit on` leaves about 1 GB of VRAM unused, so it is a weak baseline. Tuning it by hand already gives +39%. The cache is only interesting because it beats the tuned baseline too (+14–17% with MTP on, +30% with MTP off).

**Long generation (3 prompts × 1000 tokens, default sampling):**

| Workload | Stock `llama.cpp` | BeyondVRAM (3300 MiB) | Decode Speedup | SLRU Hit Rate |
|---|---:|---:|---:|---:|
| Q1: Distributed AI | 14.1 tok/s | 23.6 tok/s | +67.4% | 87.8% |
| Q2: AgentOS Kernel | 14.8 tok/s | 24.8 tok/s | +66.8% | 90.2% |
| Q3: Compiler & PTX | 18.5 tok/s | 22.4 tok/s | +21.5% | 89.0% |
| **Suite Mean** | **15.8 tok/s** | **23.6 tok/s** | **+49.3%** | **89.0%** |

Total wall-clock time dropped from 267 s to 206 s (-61 s). Q3 showed lower relative gains because default sampling led to different speculative drafting trajectories. Raw telemetry is in [`benchmarks/`](benchmarks/).

---

## How it works

**The model.** Groot-19B-A3B has 41 MoE layers with 128 experts each (5,248 in total). Each token uses only 8 experts per layer. About 1 GB (attention, routers) is needed every token. The other 6.6 GB is experts, and each expert is 1.29 MiB.

**Why stock `llama.cpp` leaves VRAM idle.** `-fit on` decides per layer: a layer's experts are either all in VRAM or all in RAM. It also keeps a 1 GB safety margin. On this card it fits only 7 layers' experts and runs the rest on the CPU. Most of the VRAM that holds experts is holding ones that are rarely used.

**How the SLRU cache fills the gap.** Instead of whole layers, we keep the backbone and the 128K context in VRAM (~2.5 GB) and use ~3.3 GB as a pool of individual expert slots (2,560 slots).
- Expert in the pool → used directly on the GPU.
- Expert missing → copied from RAM into a slot.
- New experts go into a "probation" segment. A second hit promotes them to a "protected" segment, so one-off experts don't push out the regulars.
- Prompt prefill does not touch the cache, so a long prompt cannot flush it.

**The locality observation (the thesis).** With slots for only 48.8% of experts, the measured hit rate is 89% (88.9% with MTP off, 89.8% with MTP on). That is only possible if a small set of experts handles most of the requests. We observed this through the hit rate in `llama.cpp` logs, not by analyzing per-expert statistics directly.

The cache is the `--moe-cache-mib` feature already in upstream `llama.cpp`. This repo measures it against fair baselines and adds a memory calculator and microbenchmarks.

---

## Hardware & setup

| | |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 6 GB VRAM (PCIe 4.0) |
| CPU | Intel Core 5 210H, 12 threads, AVX2 |
| RAM | 16 GB DDR4 (single channel) |
| OS | Windows 11 |
| Software | `llama.cpp` `b1-ac8e169`, CUDA 13.2, MSVC 19.44, Nsight Systems 2025.6.3 |
| Model | `Groot-19B-A3B-261023.i1-Q3_K_S.gguf` (7.82 GiB) |

VRAM use with the full cache: 5,779 of 6,143 MiB. This leaves a small margin so Windows does not spill into shared memory.

---

## Microbenchmark findings

These are small standalone CUDA/C++ tests of the hardware behaviour behind the cache. They are separate from the end-to-end numbers above.

**1. Copy/compute overlap** ([`src/async_pipeline.cu`](src/async_pipeline.cu)). Fetching a 1.29 MiB expert over PCIe on one CUDA stream while the GPU computes on another cut a step from **0.464 ms to 0.287 ms (1.61x)**. The compute kernel really uses the copied expert, and the outputs match the sequential version exactly.

![Nsight timeline](docs/assets/nsys_timeline.png)

**2. Cold expert: CPU vs GPU** ([`src/zero_transfer_ffn.cpp`](src/zero_transfer_ffn.cpp)). One full 3-matrix expert at its real size, including the cost of moving activations and syncing:

| Path | Time per expert |
|---|---:|
| GPU, expert already in cache | 0.052 ms |
| CPU (AVX2), send only 16 KiB of activations | 0.148 ms |
| Copy 1.29 MiB to GPU, then compute | 0.193 ms |

A cache hit is fastest by far. For a miss, running on the CPU is a little faster than copying the expert, but about a third of that time is GPU↔CPU sync. This is why the hit rate drives the end-to-end speed.

---

## Related work

MoE offloading and expert caching have been explored by [Mixtral-Offloading](https://arxiv.org/abs/2312.17238) (LRU caching with prefetching), [Fiddler](https://arxiv.org/abs/2402.07033) (computing cold experts on the CPU instead of moving weights), [MoE-Infinity](https://arxiv.org/abs/2401.14361), and [KTransformers](https://github.com/kvcache-ai/ktransformers). BeyondVRAM evaluates how these principles—specifically Segmented-LRU caching and CPU/GPU split execution—behave in upstream `llama.cpp` under consumer mobile constraints (6 GB VRAM, PCIe 4.0 x4, single-channel RAM).

---

## Requirements

- **Operating System:** Windows 10/11 (PowerShell).
- **Inference Engine:** Upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp) with `--moe-cache-mib` support (`b1-ac8e169` or newer).
- **Python:** Python 3.10+ with `matplotlib` and `numpy`.
- **C++/CUDA Toolchain (optional):** MSVC 19.40+ (Visual Studio 2022) and CUDA Toolkit 12.0+ with `nvcc` (only needed for microbenchmarks).
- **Model File:** A GGUF MoE model (benchmarked on `Groot-19B-A3B-261023.i1-Q3_K_S.gguf`, 7.82 GiB).

---

## Quick start

> [!NOTE]
> Currently Windows-only (MSVC + CUDA Toolkit required for C++/CUDA tests).

```powershell
git clone https://github.com/gauravpy3361/BeyondVRAM.git
cd BeyondVRAM

# 1. How big can the cache be on your GPU?
python src/header_parser.py --model "C:\path\to\model.gguf" --vram 6144 --ctx 131072

# 2. Run with the cache (use the cache size from step 1)
llama-cli.exe -m "C:\path\to\model.gguf" -c 131072 -fa on -ctk q4_0 -ctv q4_0 -cmoe --moe-cache-mib 3300 --spec-type none
```

Needs a `llama.cpp` build that has `--moe-cache-mib`. `--spec-type none` turns MTP off, which was fastest in our tests.

---

## Reproduce the benchmarks

```powershell
$env:LLAMA_CLI_PATH = "C:\path\to\llama-cli.exe"
$env:BEYONDVRAM_MODEL = "C:\path\to\model.gguf"

python src/run_controlled_ablation.py     # main table, ~10 min, writes benchmarks/ablation_results.json
python src/run_full_3x1k_std_bench.py     # long-generation run, stock
python src/run_full_3x1k_bench.py         # long-generation run, with cache
python src/plot_benchmarks.py             # redraws docs/assets/benchmark_results.png
```

Close other GPU apps first and let the laptop cool between runs. Results will differ on other hardware.

---

## Build the C++/CUDA tests

Open an "x64 Native Tools Command Prompt for VS" with CUDA installed. Change `sm_86` to match your GPU (RTX 3050 = `sm_86`).

```powershell
cl /EHsc /O2 /std:c++17 /I src tests\test_cache.cpp src\beyond_vram_cache.cpp /Fe:bin\test_cache.exe
nvcc -O3 -arch=sm_86 src\async_pipeline.cu -o bin\async_pipeline.exe
nvcc -O3 -x cu -arch=sm_86 -Xcompiler "/O2 /openmp /arch:AVX2" src\zero_transfer_ffn.cpp -o bin\bench_zero_transfer.exe

.\bin\test_cache.exe
.\bin\async_pipeline.exe
.\bin\bench_zero_transfer.exe
```

Optional profiling: `nsys profile --trace=cuda -o bin\async_pipeline_trace .\bin\async_pipeline.exe`, then `python src/plot_nsys_timeline.py`.

---

## Project structure

```text
BeyondVRAM/
├── README.md
├── LICENSE
├── docs/
│   ├── ARCHITECTURE.md              # detailed memory and cache notes
│   └── assets/                      # charts used here
├── benchmarks/
│   ├── ablation_results.json        # main comparison
│   ├── results_3x1k*.json           # long-generation runs
│   ├── nsys_trace_summary.json      # Nsight data
│   └── prompts/                     # benchmark prompts
├── src/
│   ├── header_parser.py             # reads GGUF, computes VRAM budget
│   ├── run_controlled_ablation.py   # main benchmark runner
│   ├── run_full_3x1k_*.py           # long-generation runners
│   ├── plot_*.py                    # chart scripts
│   ├── async_pipeline.cu            # copy/compute overlap test
│   ├── zero_transfer_ffn.cpp        # CPU vs GPU cold-expert test
│   ├── beyond_vram_cache.{h,cpp}    # C++ SLRU reference implementation
│   └── moe_cache.py, async_pipeline.py, zero_transfer_ffn.py, benchmark_comparator.py
│                                    # simple Python simulators / models (not measurements)
└── tests/test_cache.cpp             # cache unit tests
```

---

## Limitations & honest caveats

- **We did not write the cache.** The SLRU cache is `--moe-cache-mib` in upstream `llama.cpp`. The C++ cache in this repo is a standalone reference with unit tests and is not used in the benchmarks.
- **Small sample.** One machine, one model, one prompt for the main table, and 2–3 runs per setting. The reported ranges are small, but this is not a statistical study.
- **The headline number needs MTP off.** With MTP on (the default for this model) the best result is 24.0 tok/s. 33.8 tok/s is with `--spec-type none`.
- **The baseline matters.** The 2.3x figure is against stock `-fit on`. Against a hand-tuned static setup, the gain is +30% (MTP off) or +14–17% (MTP on).
- **Prefill is slower with the cache** (about 220–260 tok/s vs 310–350 tok/s for static). The cache optimizes generation, not prompt reading.
- **Locality is inferred, not measured directly.** It comes from hit rates in the logs on a few technical prompts. Other kinds of text may behave differently.
- **Long-generation test is noisy.** Default random sampling changed how many speculative tokens were accepted, so the +21% to +67% spread is partly noise.
- **128K means reserved, not used.** The tests used ~900-token prompts and up to 1000 generated tokens. Behaviour with a truly full 128K context was not measured.
- **Microbenchmarks are not end-to-end.** They test single experts in isolation and do not include the real `llama.cpp` scheduling.
- **Python files marked as simulators are models,** not measurements. The synthetic hit rate in `moe_cache.py` is a made-up distribution.
- **Hardware specific.** Results are for a laptop with single-channel RAM and PCIe 4.0 x4. A desktop GPU or dual-channel RAM would shift the numbers.

License: [Apache-2.0](LICENSE)
