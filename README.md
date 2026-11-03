# BeyondVRAM

Run a 19B Mixture-of-Experts model with a 128K context on a **6 GB laptop GPU**, and make it faster by keeping the most-used experts in VRAM.

This repo has the benchmarks, a GGUF memory calculator, and small CUDA/C++ experiments. The inference itself is done by upstream [llama.cpp](https://github.com/ggml-org/llama.cpp).

![Benchmark results](docs/assets/benchmark_results.png)

---

## 1. Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 6 GB VRAM (PCIe 4.0) |
| CPU | Intel Core 5 210H, 12 threads, AVX2 |
| RAM | 16 GB DDR4 (single channel) |
| OS | Windows 11 |
| Software | llama.cpp `b1-ac8e169`, CUDA 13.2, MSVC 19.44 |

## 2. Model and how it runs

**Model:** Groot-19B-A3B, `Q3_K_S` GGUF (7.8 GB). It is a Mixture-of-Experts model. It has 41 layers with 128 experts each, but only 8 experts are used per token (about 2.4B active parameters out of 19.4B).

**Standard llama.cpp (`-fit on`):** it guesses how many layers fit on the GPU and runs the rest on the CPU. On this card it only fits the experts of 7 layers, and the rest run on the CPU.

**With BeyondVRAM:**
- The small, always-needed parts (attention, routers, 1 GB) stay in VRAM.
- All experts stay in system RAM.
- About 3.3 GB of VRAM is used as a **cache of recently used experts**. If an expert is in the cache, the GPU uses it directly. If not, it is loaded from RAM.
- Cache policy is SLRU (a smarter LRU that keeps experts used more than once).

The cache itself is the `--moe-cache-mib` option already in llama.cpp. This repo measures it against fair baselines.

## 3. Benchmarks

Setup: 128K context reserved, ~900-token prompt, greedy decoding (`--temp 0`), 2–3 runs each. Numbers are the median decode speed.

**Main comparison (speculative MTP on):**

| Setup | Decode speed | Cache hit rate |
|---|---:|---:|
| Standard llama.cpp (`-fit on`) | 14.8 tok/s | – |
| Hand-tuned static (`--n-cpu-moe 23`) | 20.6 tok/s | – |
| BeyondVRAM, full cache (3300 MiB) | 23.4 tok/s | 89.8% |
| BeyondVRAM, hybrid (static + 1650 MiB cache) | 24.0 tok/s | 75.4% |

**Best result (MTP off):**

| Setup | Decode speed | Cache hit rate |
|---|---:|---:|
| Hand-tuned static (`--n-cpu-moe 23`) | 26.1 tok/s | – |
| BeyondVRAM, full cache (3300 MiB) | **33.8 tok/s** | 88.9% |

**Long generation (3 prompts, 1000 tokens each, default sampling):**

| | Standard | BeyondVRAM |
|---|---:|---:|
| Average decode | 15.8 tok/s | 23.6 tok/s |
| Total time | 267 s | 206 s |

Raw data is in [`benchmarks/`](benchmarks/).

**GPU overlap test.** In a small CUDA test, copying an expert over PCIe while the GPU computes cut one step from 0.464 ms to 0.287 ms (1.61x). It was profiled with Nsight Systems:

![Nsight timeline](docs/assets/nsys_timeline.png)

## 4. What the results mean

- **Standard `-fit on` is not a fair baseline.** It leaves about 1 GB of VRAM unused. Tuning it by hand (`--n-cpu-moe 23`) already gives +39%. Our cache has to beat that, and it does, by 14–30%.
- **Why the cache helps.** About 89% of expert requests are found in VRAM, so most of the work avoids slow RAM and PCIe.
- **Why MTP off is faster here.** MTP checks 4 tokens at once, which touches many more experts, and only about 35–45% of its guesses are accepted. On this hardware the extra expert loading costs more than it gains.
- **Prefill is slower with the cache** (about 220–260 tok/s vs 310–350 tok/s for static). The cache is tuned for generation speed, not prompt reading.
- **Limits.** One machine, one model, and few repeats. The 3×1K test used random sampling, so its per-prompt gains vary (+21% to +67%). Treat these as measurements on this setup, not general claims.
- **Cold expert test.** For an expert not in VRAM, running it on the CPU (0.148 ms) is a bit faster than copying it to the GPU (0.193 ms). A cache hit on the GPU is fastest (0.052 ms).
- **Related work:** [Mixtral-Offloading](https://arxiv.org/abs/2312.17238), [Fiddler](https://arxiv.org/abs/2402.07033), [MoE-Infinity](https://arxiv.org/abs/2401.14361), [KTransformers](https://github.com/kvcache-ai/ktransformers) explore the same idea.

More detail is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 5. Project structure

```text
BeyondVRAM/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md          # detailed memory and cache notes
│   └── assets/                  # charts used in this README
├── benchmarks/
│   ├── ablation_results.json    # main comparison (temp 0, repeats)
│   ├── results_3x1k*.json       # long-generation runs
│   ├── nsys_trace_summary.json  # Nsight data
│   └── prompts/                 # benchmark prompts
├── src/
│   ├── header_parser.py         # reads GGUF, calculates VRAM budget
│   ├── run_controlled_ablation.py   # runs the main comparison
│   ├── run_full_3x1k_*.py       # runs the long-generation tests
│   ├── plot_*.py                # make the charts
│   ├── async_pipeline.cu        # CUDA copy/compute overlap test
│   ├── zero_transfer_ffn.cpp    # CPU vs GPU cold-expert test
│   ├── beyond_vram_cache.*      # C++ SLRU cache reference
│   └── *.py (moe_cache, ...)    # simple simulators / models
└── tests/test_cache.cpp         # cache unit tests
```

## 6. Download and use

**Requirements:** Windows, Python 3.10+, `matplotlib`, a recent llama.cpp build with `--moe-cache-mib`, and the GGUF model. CUDA Toolkit + MSVC are only needed for the C++/CUDA tests.

```powershell
git clone https://github.com/gauravpy3361/BeyondVRAM.git
cd BeyondVRAM
```

**Check how much cache fits your GPU:**
```powershell
python src/header_parser.py --model "C:\path\to\model.gguf" --vram 6144 --ctx 131072
```

**Run the model with the expert cache:**
```powershell
llama-cli.exe -m model.gguf -c 131072 -fa on -ctk q4_0 -ctv q4_0 -cmoe --moe-cache-mib 3300 --spec-type none
```

**Reproduce the benchmarks** (set `LLAMA_CLI_PATH` and `BEYONDVRAM_MODEL` first):
```powershell
python src/run_controlled_ablation.py
python src/plot_benchmarks.py
```

**Build and run the C++/CUDA tests** (from a "x64 Native Tools" prompt):
```powershell
cl /EHsc /O2 /std:c++17 /I src tests\test_cache.cpp src\beyond_vram_cache.cpp /Fe:bin\test_cache.exe
nvcc -O3 -arch=sm_86 src\async_pipeline.cu -o bin\async_pipeline.exe
nvcc -O3 -x cu -arch=sm_86 -Xcompiler "/O2 /openmp /arch:AVX2" src\zero_transfer_ffn.cpp -o bin\bench_zero_transfer.exe
.\bin\test_cache.exe; .\bin\async_pipeline.exe; .\bin\bench_zero_transfer.exe
```
Use `-arch=sm_XX` that matches your GPU (RTX 3050 = `sm_86`).

License: [Apache-2.0](LICENSE)
