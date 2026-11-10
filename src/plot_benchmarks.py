"""
BeyondVRAM: Benchmark & Architecture Visualizations
Generates two high-resolution charts from recorded benchmark telemetry:
  1. `docs/assets/benchmark_results.png`:
     - Left: Controlled Decode Throughput — Default llama.cpp vs. BeyondVRAM Tuned (MTP & Non-MTP)
     - Right: Long-Generation Suite (3 Prompts x 1,000 Tokens)
  2. `docs/assets/vram_and_locality.png`:
     - Left: 6 GB VRAM Budget Utilization — Default llama.cpp (-fit on) vs. BeyondVRAM Tuned
     - Right: Hardware Expert Execution Latency & MTP vs. Non-MTP Routing Breakdown
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def render_benchmark_results(ablation, suite_cache, suite_std):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.4, 5.8), dpi=200, facecolor="#0d1117",
        gridspec_kw={"width_ratios": [1.22, 1.0]}
    )
    fig.subplots_adjust(left=0.21, right=0.97, top=0.80, bottom=0.15, wspace=0.24)

    fig.text(
        0.5, 0.935,
        "BeyondVRAM Benchmarks — Groot-19B-A3B (Q3_K_S, 128K Context) on 6 GB RTX 3050 Laptop GPU",
        ha="center", va="center", color="#f0f6fc", fontsize=13.5, fontweight="bold"
    )
    fig.text(
        0.5, 0.885,
        "Left: Default vs. BeyondVRAM Tuned (MTP On vs. Non-MTP, --temp 0)   |   Right: 3x (1K Prompt x 1K Decode) Sustained Suite",
        ha="center", va="center", color="#8b949e", fontsize=10
    )

    # -------------------------------------------------------------------------
    # Panel 1: Controlled Throughput Comparison (Default vs BeyondVRAM, MTP/Non-MTP)
    # -------------------------------------------------------------------------
    ax1.set_facecolor("#161b22")

    order = [
        "autofit_default",
        "slru_cache_3300",
        "hybrid_ncmoe31_c1650",
        "slru_cache_3300_no_mtp",
    ]
    cfg_map = {c["id"]: c for c in ablation}
    ordered_cfgs = [cfg_map[k] for k in order]

    y_labels = [
        "Default llama.cpp (-fit on)\n[MTP On | 3,653 MiB VRAM]",
        "BeyondVRAM Tuned (3,300 MiB)\n[MTP On | 89.8% Cache Hit]",
        "BeyondVRAM Hybrid (1,650 MiB)\n[MTP On | 75.4% Cache Hit]",
        "BeyondVRAM Tuned (3,300 MiB)\n[Non-MTP, ubatch=1 | 88.9% Hit]",
    ]
    medians = [c["tg_median_tps"] for c in ordered_cfgs]
    err_low = [c["tg_median_tps"] - c["tg_min_tps"] for c in ordered_cfgs]
    err_high = [c["tg_max_tps"] - c["tg_median_tps"] for c in ordered_cfgs]

    colors = [
        "#6e7681",  # Default llama.cpp
        "#1f6feb",  # BeyondVRAM Tuned (MTP On)
        "#388bfd",  # BeyondVRAM Hybrid (MTP On)
        "#2ea043",  # BeyondVRAM Tuned (Non-MTP)
    ]

    y_pos = np.array([0.0, 1.0, 2.0, 3.15])
    bars = ax1.barh(
        y_pos, medians, height=0.54, color=colors, edgecolor="#f0f6fc", linewidth=0.7, alpha=0.93,
        xerr=[err_low, err_high], error_kw={"ecolor": "#f0f6fc", "capsize": 3.5, "elinewidth": 1.1}
    )

    ax1.axhline(2.58, color="#30363d", linestyle="--", linewidth=1.1)
    ax1.text(
        54.8, 2.66, "Non-MTP (--spec-type none, ubatch=1)",
        color="#7ee787", fontsize=8.4, ha="right", va="bottom", fontweight="bold"
    )
    ax1.text(
        54.8, 2.48, "MTP On (--spec-type draft-mtp, ubatch=4)",
        color="#79c0ff", fontsize=8.4, ha="right", va="top", fontweight="bold"
    )

    base_default = cfg_map["autofit_default"]["tg_median_tps"]

    for idx, (bar, c) in enumerate(zip(bars, ordered_cfgs)):
        val = c["tg_median_tps"]
        if c["id"] == "autofit_default":
            tag = f"{val:.1f} tok/s  (Default Baseline)"
            txt_color = "#c9d1d9"
        else:
            mult = val / base_default
            pct = (mult - 1.0) * 100.0
            tag = f"{val:.1f} tok/s  (+{pct:.0f}% / {mult:.2f}x vs Default)"
            txt_color = "#7ee787" if c["id"] == "slru_cache_3300_no_mtp" else "#f0f6fc"

        ax1.text(
            val + err_high[idx] + 0.7, bar.get_y() + bar.get_height() / 2.0,
            tag, va="center", ha="left", color=txt_color, fontsize=8.7, fontweight="bold"
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, color="#e6edf3", fontsize=8.9, fontweight="bold")
    ax1.set_xlim(0, 56.0)
    ax1.set_ylim(-0.5, 3.75)
    ax1.set_xlabel("Decode Throughput (tokens / sec, median ± range)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax1.set_title("Controlled Benchmark: Default vs. BeyondVRAM (MTP & Non-MTP)", color="#f0f6fc", fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", colors="#8b949e", labelsize=9)
    ax1.grid(axis="x", color="#21262d", linestyle="--", linewidth=0.7)
    ax1.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Panel 2: 3x (1K Prompt x 1K Decode) Long-Generation Suite
    # -------------------------------------------------------------------------
    ax2.set_facecolor("#161b22")

    categories = [
        "Q1: Distributed AI\n(87.8% Cache Hit)",
        "Q2: AgentOS Kernel\n(90.2% Cache Hit)",
        "Q3: Compiler & PTX\n(89.0% Cache Hit)",
        "Suite Mean\n(89.0% Cache Hit)",
    ]
    std_vals = [r["tg_speed"] for r in suite_std["results"]] + [suite_std["summary"]["avg_tg"]]
    cache_vals = [r["tg_speed"] for r in suite_cache["results"]] + [suite_cache["summary"]["avg_tg"]]

    x = np.arange(len(categories))
    width = 0.35

    b_std = ax2.bar(
        x - width / 2, std_vals, width, label="Default llama.cpp (-fit on)",
        color="#6e7681", edgecolor="#f0f6fc", linewidth=0.6, alpha=0.9
    )
    b_cache = ax2.bar(
        x + width / 2, cache_vals, width, label="BeyondVRAM Tuned (3,300 MiB SLRU Cache)",
        color="#1f6feb", edgecolor="#f0f6fc", linewidth=0.6, alpha=0.93
    )

    for bar in b_std:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.4,
            f"{h:.1f}", ha="center", va="bottom", color="#c9d1d9", fontsize=8.6, fontweight="bold"
        )

    for idx, bar in enumerate(b_cache):
        h = bar.get_height()
        gain = (h / std_vals[idx] - 1.0) * 100.0
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.4,
            f"{h:.1f}\n(+{gain:.0f}%)", ha="center", va="bottom", color="#79c0ff", fontsize=8.6, fontweight="bold"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, color="#e6edf3", fontsize=8.5)
    ax2.set_ylim(0, 31.5)
    ax2.set_ylabel("Sustained Decode Throughput (tokens / sec)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax2.set_title("Long-Generation Suite: 3x (1K Prompt x 1K Decode, MTP On)", color="#f0f6fc", fontsize=11, fontweight="bold", pad=10)
    ax2.tick_params(axis="y", colors="#8b949e", labelsize=9)
    ax2.grid(axis="y", color="#21262d", linestyle="--", linewidth=0.7)
    ax2.set_axisbelow(True)

    leg = ax2.legend(loc="upper right", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5)
    for text in leg.get_texts():
        text.set_color("#f0f6fc")

    out_path = os.path.join("docs", "assets", "benchmark_results.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved benchmark chart to {out_path}")


def render_vram_and_locality():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.4, 5.5), dpi=200, facecolor="#0d1117",
        gridspec_kw={"width_ratios": [1.12, 1.0]}
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.78, bottom=0.16, wspace=0.34)

    fig.text(
        0.5, 0.93,
        "Architecture Breakdown — How BeyondVRAM Reclaims Idle VRAM & Minimizes PCIe Stalls",
        ha="center", va="center", color="#f0f6fc", fontsize=13.5, fontweight="bold"
    )
    fig.text(
        0.5, 0.875,
        "Left: 6,143 MiB GDDR6 Memory Partitioning at 128K Context   |   Right: Single-Expert Execution Latency (1.29 MiB Q3_K SwiGLU)",
        ha="center", va="center", color="#8b949e", fontsize=10
    )

    # -------------------------------------------------------------------------
    # Panel 1: Stacked Horizontal Bar of 6,143 MiB VRAM Allocation
    # -------------------------------------------------------------------------
    ax1.set_facecolor("#161b22")

    y_pos = np.array([0, 1])
    y_labels = [
        "Default llama.cpp\n(-fit on)\n14.8 tok/s",
        "BeyondVRAM Tuned\n(3,300 MiB Cache)\n33.8 tok/s (Non-MTP)",
    ]

    backbone = np.array([1037, 1037])
    ctx_comp = np.array([1461, 1442])
    static_moe = np.array([1155, 0])
    slru_cache = np.array([0, 3300])
    unused_or_guard = np.array([2490, 364])

    h = 0.46
    ax1.barh(y_pos, backbone, height=h, color="#3fb950", edgecolor="#0d1117", linewidth=1.0, label="Backbone (Attn/Router/Shared)")
    ax1.barh(y_pos, ctx_comp, left=backbone, height=h, color="#8957e5", edgecolor="#0d1117", linewidth=1.0, label="128K Context + Compute")
    ax1.barh(y_pos, static_moe, left=backbone + ctx_comp, height=h, color="#d29922", edgecolor="#0d1117", linewidth=1.0, label="Static MoE (7/41 Layers Only)")
    ax1.barh(y_pos, slru_cache, left=backbone + ctx_comp + static_moe, height=h, color="#1f6feb", edgecolor="#0d1117", linewidth=1.0, label="SLRU Expert Cache (2,560 Slots, 89% Hit)")
    ax1.barh(
        y_pos, unused_or_guard, left=backbone + ctx_comp + static_moe + slru_cache,
        height=h, color="#21262d", edgecolor="#6e7681", linewidth=0.9, hatch="//", label="Unused VRAM / WDDM Guardband"
    )

    ax1.text(518, 0, "Backbone\n1,037 MiB", ha="center", va="center", color="#0d1117", fontsize=7.8, fontweight="bold")
    ax1.text(1037 + 730, 0, "128K Ctx\n1,461 MiB", ha="center", va="center", color="white", fontsize=7.8, fontweight="bold")
    ax1.text(2498 + 577, 0, "7 Layers\n1,155 MiB", ha="center", va="center", color="#0d1117", fontsize=7.8, fontweight="bold")
    ax1.text(
        3653 + 1245, 0, "2,490 MiB Unused VRAM\n(35 MoE Layers Forced to RAM)",
        ha="center", va="center", color="#ff7b72", fontsize=7.9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor="#ff7b72", alpha=0.92, linewidth=0.8)
    )

    ax1.text(518, 1, "Backbone\n1,037 MiB", ha="center", va="center", color="#0d1117", fontsize=7.8, fontweight="bold")
    ax1.text(1037 + 721, 1, "128K Ctx\n1,442 MiB", ha="center", va="center", color="white", fontsize=7.8, fontweight="bold")
    ax1.text(2479 + 1650, 1, "3,300 MiB Dynamic SLRU Cache\n(2,560 Hot Expert Slots -> ~89% VRAM Hit Rate)", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, color="#e6edf3", fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, 6350)
    ax1.set_ylim(-0.55, 1.65)
    ax1.set_xlabel("GPU Memory Allocation (MiB / 6,143 MiB Total GDDR6)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax1.set_title("6 GB VRAM Budget: Default Auto-Fit vs. Dynamic SLRU Cache", color="#f0f6fc", fontsize=10.8, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", colors="#8b949e", labelsize=9)
    ax1.grid(axis="x", color="#21262d", linestyle="--", linewidth=0.7)
    ax1.set_axisbelow(True)

    leg1 = ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=7.6)
    for text in leg1.get_texts():
        text.set_color("#c9d1d9")

    # -------------------------------------------------------------------------
    # Panel 2: Single-Expert Latency & MTP vs Non-MTP Routing Cost
    # -------------------------------------------------------------------------
    ax2.set_facecolor("#161b22")

    paths = [
        "Cold Miss: PCIe + GPU\n(1.29 MiB H2D Copy)",
        "Cold Miss: CPU AVX2\n(+ 16 KiB Sync)",
        "Hot Hit: GPU Cache\n(0 B PCIe, GDDR6)",
    ]
    latencies = [0.193, 0.148, 0.052]
    p_colors = ["#da3633", "#d29922", "#2ea043"]

    yp = np.arange(len(paths))
    bars2 = ax2.barh(yp, latencies, height=0.48, color=p_colors, edgecolor="#f0f6fc", linewidth=0.7, alpha=0.92)

    labels2 = [
        "0.193 ms / expert  (3.7x slower — PCIe bottleneck)",
        "0.148 ms / expert  (2.8x slower — CPU + sync)",
        "0.052 ms / expert  (Fast Path — 88.9% of lookups!)",
    ]
    for bar, lbl, col in zip(bars2, labels2, ["#ff7b72", "#e3b341", "#7ee787"]):
        w = bar.get_width()
        ax2.text(
            w + 0.006, bar.get_y() + bar.get_height() / 2.0,
            lbl, va="center", ha="left", color=col, fontsize=8.4, fontweight="bold"
        )

    callout_text = (
        "Why Non-MTP (ubatch=1) Reaches 33.8 tok/s (2.29x):\n"
        "• MTP On (ubatch=4): Verifying 4 draft tokens touches ~20–24 unique\n"
        "  experts/layer per step -> 2.5–3x more expert lookups & misses (23.4 tok/s).\n"
        "• Non-MTP (ubatch=1): Routes 8 experts/layer. At 88.9% SLRU hit rate,\n"
        "  7.1 experts/layer run at 0.052 ms (<1 miss/layer) -> 33.8 tok/s."
    )
    ax2.text(
        0.03, 0.95, callout_text, transform=ax2.transAxes,
        va="top", ha="left", color="#e6edf3", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117", edgecolor="#1f6feb", linewidth=1.1)
    )

    ax2.set_yticks(yp)
    ax2.set_yticklabels(paths, color="#e6edf3", fontsize=8.6, fontweight="bold")
    ax2.set_xlim(0, 0.39)
    ax2.set_ylim(-0.45, 3.35)
    ax2.set_xlabel("Measured Latency per 3-Matrix SwiGLU Expert (ms, lower is better)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax2.set_title("Hardware Microbenchmark: Cache Hit vs. Cold Miss & MTP Cost", color="#f0f6fc", fontsize=10.8, fontweight="bold", pad=10)
    ax2.tick_params(axis="x", colors="#8b949e", labelsize=9)
    ax2.grid(axis="x", color="#21262d", linestyle="--", linewidth=0.7)
    ax2.set_axisbelow(True)

    out_path = os.path.join("docs", "assets", "vram_and_locality.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved architecture chart to {out_path}")


def main():
    with open(os.path.join("benchmarks", "ablation_results.json"), "r", encoding="utf-8") as f:
        ablation = json.load(f)["configs"]
    with open(os.path.join("benchmarks", "results_3x1k.json"), "r", encoding="utf-8") as f:
        suite_cache = json.load(f)
    with open(os.path.join("benchmarks", "results_3x1k_std.json"), "r", encoding="utf-8") as f:
        suite_std = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": "#30363d",
        "axes.linewidth": 1.0,
    })

    render_benchmark_results(ablation, suite_cache, suite_std)
    render_vram_and_locality()


if __name__ == "__main__":
    main()
