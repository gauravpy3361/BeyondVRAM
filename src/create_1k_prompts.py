import os

prompts_dir = r"C:\Users\gaurav\bench_1k_prompts"
os.makedirs(prompts_dir, exist_ok=True)

p1 = """You are a distinguished Principal AI Systems Architect, Supercomputing Fellow, and Distributed Infrastructure Lead with extensive industry expertise designing exascale training and inference platforms for multi-trillion-parameter foundation models.

Please author an exhaustive, authoritative, mathematically rigorous, and deeply technical treatise analyzing the physical, microarchitectural, algorithmic, and networking foundations of 3D Parallelism, Ultra-Scale Collective Fabrics, and Disaggregated Inference Serving for Frontier Artificial Intelligence Systems.

In your treatise, thoroughly address each of the foundational architectural domains below with academic rigor:

1. 3D Parallelism Breakdown & Mathematical Formulations:
- Tensor Parallelism (TP): Rigorously define column-parallel and row-parallel linear transformations in the Megatron-LM formulation. Formulate the exact forward and backward matrix multiplications for multi-head self-attention and MLP blocks. Trace the precise collective communication primitives required per Transformer block, identifying where All-Reduce and All-Gather operations occur. Detail the exact activation memory overhead per GPU. Mathematically demonstrate why Tensor Parallelism is strictly constrained within a single multi-GPU node (the NVLink/NVSwitch crossbar domain) due to ultra-low latency requirements and severe PCIe bisectional bandwidth degradation across InfiniBand clusters.
- Pipeline Parallelism (PP): Formulate the 1F1B (One-Forward-One-Backward) schedule, comparing its bubble fraction directly against naive GPipe and contemporary interleaved schedules such as Dual-Pipe and ZB-1P (Zero-Bubble One-Pipeline). Formulate the mathematical equation for the pipeline bubble ratio as a function of the number of pipeline stages (P) and micro-batches (M). Detail peak activation memory pressure under 1F1B, activation stashing mechanics, and how inter-stage point-to-point peer-to-peer CUDA IPC communications are double-buffered to hide latency.
- Context Parallelism (CP) & Sequence Parallelism (SP): Analyze Ring-Attention, DeepSpeed Ulysses, and block-wise distributed self-attention over extreme sequence lengths spanning 256k to 1M tokens. Detail the all-to-all communication patterns required to transpose sequence partitions across head dimensions, quantifying the communication volume per token relative to hidden dimension d_model.
- Data Parallelism (DP) & Fully Sharded Data Parallel (FSDP / ZeRO): Formulate the mathematical memory footprints of ZeRO Stage 1 (Optimizer State Partitioning), ZeRO Stage 2 (Gradient Partitioning), and ZeRO Stage 3 (Parameter Sharding). Trace the exact communication volume per training step, explaining why ZeRO-3 communication volume equals 3x standard data parallelism.

2. Distributed Interconnect & Network Fabric Physics:
- Physical Topologies & Fabrics: Analyze bandwidth, bisectional bandwidth, packet latency, and signal integrity realities across dense multi-node compute clusters. Rigorously contrast Fat-Tree non-blocking Clos topologies with DragonFly+ and multi-dimensional Torus architectures. Explain the physical differences between 800 Gbps InfiniBand (Quantum-2 / NDR) and RoCEv2 (RDMA over Converged Ethernet), detailing PFC (Priority Flow Control) deadlocks, RDMA Go-Back-N packet drops, and ECN (Explicit Congestion Notification) marking under network incast congestion.
- Hardware-Offloaded In-Network Reductions: Detail the algorithmic mechanics of Ring All-Reduce vs. Tree All-Reduce (Recursive Doubling). Explain how NVIDIA SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) performs floating-point tensor reductions directly inside physical switch silicon ASICs, cutting host CPU interrupts and PCIe bus traversal.

3. Inference Serving, Memory Walls, and MoE Disaggregation:
- Roofline Model of Distributed Decoding: Mathematically formulate Williams' Roofline Model for distributed generative decoding. Explain why single-batch autoregressive generation (GEMV) is memory-bandwidth-bound, whereas prompt prefill (GEMM) is compute-bound.
- Disaggregated Serving (Split Prefill and Decode): Architect a cluster-level serving system separating dedicated Prefill nodes (high compute, dense tensor cores) from dedicated Decode nodes (high bandwidth, disaggregated memory). Detail the KV-cache network transfer protocols over RDMA required to stream KV states from prefillers to decoders with minimal time-to-first-token (TTFT) degradation.
- Sparse Mixture-of-Experts (MoE) All-to-All Dispatch: Formulate the distributed routing problem when 128+ experts are distributed across thousands of GPUs. Detail the token dispatch All-to-All communication bottleneck, router drop policies, capacity factors, and expert load balancing auxiliary losses (e.g. switch routing loss, router z-loss).

Structure your treatise with rigorous academic precision, incorporating exact mathematical formulations, matrix dimensions, communication complexity equations, and concrete hardware numbers throughout. Your analysis must be exhaustive, comprehensive, and authoritative."""

p2 = """You are a Principal Operating Systems Architect and Autonomous Agent Systems Fellow specializing in the design, runtime verification, sandboxed isolation, and high-concurrency orchestration of autonomous multi-agent operating systems.

Please provide an exhaustive, authoritative, deeply technical systems architecture blueprint for an enterprise-scale Autonomous Agent Operating System (AgentOS) designed to supervise, schedule, isolate, and orchestrate thousands of concurrent, autonomous reasoning agents executing complex multi-step software engineering, data analysis, and infrastructure remediation workflows.

In your architectural blueprint, thoroughly address each of the following core operating system domains in complete technical depth:

1. Kernel Architecture, Process Abstractions, and Lifecycle Management:
- The Agent Control Block (ACB): Define the core "Agent Process" data structure abstraction inside the OS kernel. Detail the exact state transition diagram (Ready, Executing, Waiting_for_Tool, Blocked_on_Human, Paged_Out, Terminated, Deadlocked). Detail the registers, stack frames, scratchpad memory states, capability tokens, and cryptographic identity keys tracked inside each ACB.
- Preemptive Agent Scheduling: Design a priority-based, fair-share scheduling algorithm (e.g., Multi-Level Feedback Queue with Token-Budget Decay) that schedules compute turns, context-window allowances, and API rate limits across heterogeneous agents. Formulate how priority inversion is prevented when a high-priority supervisor agent waits on a dependency produced by a low-priority research worker agent.
- Inter-Agent Communication (IAC) & Message Bus: Architect an asynchronous, high-throughput message-passing fabric. Contrast direct memory-mapped ring buffers for local co-located agents with zero-copy distributed message brokers (gRPC over shared memory, eBPF socket filters, or Kafka/NATS streams). Detail the message serialization format, type-safe schema contracts, and backpressure mechanisms when workers overwhelm supervisor queues.

2. Hierarchical Multi-Agent Orchestration & DAG Task Execution:
- Dynamic Task Decomposition: Formulate the mathematical and algorithmic engine used by the Supervisor Agent to decompose arbitrary natural language requests into Directed Acyclic Graphs (DAGs) of discrete subtasks. Formulate cycle detection, topological sorting, dynamic parallel branch execution, and runtime graph mutation when intermediate tasks fail or yield unexpected discovery tokens.
- Deadlock Detection & Consensus in Swarms: Formalize the deadlock conditions (Mutual Exclusion, Hold and Wait, No Preemption, Circular Wait) in autonomous multi-agent interactions when agents compete for exclusive workspace locks, database tables, or API quotas. Implement a Banker's Algorithm or dynamic dependency graph wait-for graph cycle analyzer to resolve distributed agent deadlocks.

3. Memory Disaggregation, Context Compaction, and Virtual Paging:
- Hierarchical Agent Memory Architecture: Detail the multi-tiered memory hierarchy spanning L1 Working Context (active prompt/response window), L2 Persistent Session Scratchpad (key-value vector memory), and L3 Long-Term Cold Storage (distributed vector graph database with episodic memory indexing).
- Dynamic Context Paging & Working Set Maintenance: Design an algorithm for virtual context management when cumulative conversation and tool transcripts exceed model context limits. Detail semantic summarization, AST-aware code compaction, token eviction policies (e.g., Attention-Weighted LRU), and lossless compression of tool outputs.

4. Sandboxed Tool Execution, Security Enclaves, and Safety Verifiers:
- Execution Sandboxing: Detail the isolation primitives required for untrusted code execution (Linux cgroups v2, seccomp-BPF filters, Firecracker microVMs, or gVisor secure containers). Specify strict limits on CPU cycle quotas, network egress filtering (allowlisted domains), ephemeral filesystem snapshots with copy-on-write (CoW), and memory ceilings.
- Deterministic Verification Gates: Detail the automated policy enforcement layer (static AST analysis, secret detection, dynamic invariant checks) that verifies code patches and terminal operations before committing mutations to production branches.

Provide clean, modular, production-ready architectural pseudocode and rigorous systems specifications throughout your treatise."""

p3 = """You are a Distinguished Compiler Engineer and High-Performance Deep Learning Microkernel Architect specializing in low-level GPU code generation, polyhedral loop optimization, LLVM IR transformations, and hand-tuned CUDA/PTX microkernels for modern neural network accelerators.

Please provide an exhaustive, authoritative, deeply technical engineering treatise detailing the compilation, mathematical formulation, memory tiling, and microarchitectural assembly optimization of a unified, high-performance fused attention and low-bit quantized Mixture-of-Experts (MoE) kernel.

In your treatise, thoroughly address each of the following low-level optimization domains in complete technical depth:

1. Compiler Intermediate Representations (IR) & Lowering Pipelines:
- Polyhedral Loop Optimization & MLIR Dialects: Trace the lowering path of a multi-head sparse MoE Transformer block from high-level PyTorch / FX Graph down to MLIR (Multi-Level Intermediate Representation). Detail the transformations across the Torch dialect, LinAlg (Linear Algebra) dialect, Affine dialect, GPU dialect, and finally LLVM IR / NVVM.
- Fusion Passes & Memory Traffic Elimination: Formulate the mathematical graph-rewriting rules that detect and fuse memory-bound elementwise operations (RoPE rotary embedding application, RMSNorm, SwiGLU gating activation, and dequantization scaling) directly into the producer and consumer GEMM/GEMV loops. Quantify the exact global memory read/write traffic eliminated by this fusion using Williams' Roofline Model.

2. Fused FlashAttention Forward Pass Microarchitecture:
- Mathematical Derivation of Online Softmax: Rigorously derive the online softmax normalization algorithm (Milakov & Gimelshein / Welford technique) that computes attention across sequence blocks without materializing the intermediate N x N attention matrix. Formulate the running maximum m_i and running sum l_i recurrence relations, detailing the dynamic rescaling of intermediate accumulated values P_ij V_j.
- Shared Memory Tiling & Bank Conflict Elimination: Design the exact 2D tile layout (B_r x d and B_c x d) inside GPU shared memory (SRAM) for Ampere / Hopper architectures (Tensor Cores). Show how shared memory pad offsets and swizzled address bit layouts prevent 32-way shared memory bank conflicts during warp-synchronous cooperative loading.
- Asynchronous Data Movement & Pipeline Stages: Formulate the use of CUDA 11/12 asynchronous copy instructions (cp.async on Ampere, TMA / Tensor Memory Accelerator on Hopper) to double-buffer and triple-buffer global-to-shared memory transfers, achieving 100% overlap between arithmetic Tensor Core execution (mma.sync / wgmma) and global memory latency.

3. High-Performance Quantized MoE GEMV / GEMM Microkernels:
- Low-Bit Quantization Packing & Unpacking: Formulate the microarchitectural mechanics of on-the-fly dequantization for uniform 4-bit (INT4 / Q4_0 / Q4_K) and 8-bit floating-point (FP8 E4M3 / E5M2) weight representations. Detail how SIMD shift, bitwise AND masks, and fast integer-to-float conversions are scheduled inside GPU warp registers to prevent register pressure spilling to local DRAM memory.
- Asymmetric Zero-Transfer & SIMD CPU Dispatch: Contrast the GPU warp execution path with host CPU SIMD implementations (AVX-512 / AVX2 with FMA). Detail how unrolled inner product loops, memory prefetch intrinsics (_mm_prefetch), and cache-line alignment ensure that single-shot cold expert FFN computations execute at host DDR memory bus saturation without stalling CPU cores.

Incorporate rigorous mathematical formulations, hardware register allocation trade-offs, and concrete assembly/CUDA pseudocode snippets throughout your treatise."""

with open(os.path.join(prompts_dir, "prompt_1k_q1_distributed.txt"), "w", encoding="utf-8") as f:
    f.write(p1.strip())

with open(os.path.join(prompts_dir, "prompt_1k_q2_agent_os.txt"), "w", encoding="utf-8") as f:
    f.write(p2.strip())

with open(os.path.join(prompts_dir, "prompt_1k_q3_compiler_simd.txt"), "w", encoding="utf-8") as f:
    f.write(p3.strip())

print("Prompts written successfully to", prompts_dir)
