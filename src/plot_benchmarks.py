"""
BeyondVRAM: Benchmark Results Visualization
Renders a clean, two-panel chart (`docs/assets/benchmark_results.png`) from:
  1. `benchmarks/ablation_results.json` (Controlled Deterministic Ablation, --temp 0)
  2. `benchmarks/results_3x1k.json` & `benchmarks/results_3x1k_std.json` (3x 1K-Prompt x 1K-Decode Suite)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


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

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 6.2), dpi=200, facecolor="#0d1117",
        gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.14, wspace=0.26)

    fig.text(
        0.5, 0.94,
        "BeyondVRAM — Groot-19B-A3B (Q3_K_S, 128K Context) on 6 GB RTX 3050 Laptop GPU",
        ha="center", va="center", color="#f0f6fc", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5, 0.895,
        "Left: Controlled Deterministic Ablation (--temp 0, 916 Prompt -> 200 Decode)   |   Right: 3x (1K Prompt x 1K Decode) Long-Generation Suite",
        ha="center", va="center", color="#8b949e", fontsize=10
    )

    # -------------------------------------------------------------------------
    # Panel 1: Controlled Deterministic Ablation (Horizontal Bar Chart)
    # -------------------------------------------------------------------------
    ax1.set_facecolor("#161b22")

    order = [
        "autofit_default",
        "static_ncmoe23",
        "slru_cache_3300",
        "hybrid_ncmoe31_c1650",
        "static_ncmoe23_no_mtp",
        "slru_cache_3300_no_mtp",
    ]
    cfg_map = {c["id"]: c for c in ablation}
    ordered_cfgs = [cfg_map[k] for k in order]

    y_labels = [
        "Stock Auto-Fit (-fit on)\n[MTP On, 35 CPU MoE]",
        "Tuned Static (--n-cpu-moe 23)\n[MTP On, No Cache]",
        "Full SLRU Cache (3,300 MiB)\n[MTP On, 89.8% Hit]",
        "Hybrid (-ncmoe 31 + 1,650 MiB)\n[MTP On, 75.4% Hit]",
        "Tuned Static (--n-cpu-moe 23)\n[MTP Off, ubatch=1, No Cache]",
        "Full SLRU Cache (3,300 MiB)\n[MTP Off, ubatch=1, 88.9% Hit]",
    ]
    medians = [c["tg_median_tps"] for c in ordered_cfgs]
    err_low = [c["tg_median_tps"] - c["tg_min_tps"] for c in ordered_cfgs]
    err_high = [c["tg_max_tps"] - c["tg_median_tps"] for c in ordered_cfgs]

    colors = [
        "#6e7681",  # Stock Auto-Fit
        "#d29922",  # Tuned Static (MTP On)
        "#1f6feb",  # Full SLRU (MTP On)
        "#388bfd",  # Hybrid (MTP On)
        "#e3b341",  # Tuned Static (MTP Off)
        "#2ea043",  # Full SLRU (MTP Off)
    ]

    y_pos = np.arange(len(ordered_cfgs))
    bars = ax1.barh(
        y_pos, medians, height=0.62, color=colors, edgecolor="#f0f6fc", linewidth=0.6, alpha=0.92,
        xerr=[err_low, err_high], error_kw={"ecolor": "#f0f6fc", "capsize": 3, "elinewidth": 1.1}
    )

    ax1.axhline(3.5, color="#30363d", linestyle="--", linewidth=1.0)
    ax1.text(
        48.5, 3.60, "MTP Off (ubatch=1)", color="#7ee787", fontsize=8.5, ha="right", va="bottom", fontweight="bold"
    )
    ax1.text(
        48.5, 3.38, "MTP On (draft-mtp)", color="#79c0ff", fontsize=8.5, ha="right", va="top", fontweight="bold"
    )

    base_mtp_on = cfg_map["static_ncmoe23"]["tg_median_tps"]
    base_mtp_off = cfg_map["static_ncmoe23_no_mtp"]["tg_median_tps"]

    for idx, (bar, c) in enumerate(zip(bars, ordered_cfgs)):
        val = c["tg_median_tps"]
        if c["id"] == "autofit_default":
            tag = f"{val:.2f} t/s (Stock Baseline)"
        elif c["id"] == "static_ncmoe23":
            tag = f"{val:.2f} t/s (+38.9% vs Stock)"
        elif c["id"] in ("slru_cache_3300", "hybrid_ncmoe31_c1650"):
            pct = (val / base_mtp_on - 1.0) * 100.0
            tag = f"{val:.2f} t/s (+{pct:.1f}% vs Static)"
        elif c["id"] == "static_ncmoe23_no_mtp":
            tag = f"{val:.2f} t/s (Static ubatch=1)"
        else:
            pct = (val / base_mtp_off - 1.0) * 100.0
            tag = f"{val:.2f} t/s (+{pct:.1f}% vs Static)"

        ax1.text(
            val + err_high[idx] + 0.55, bar.get_y() + bar.get_height() / 2.0,
            tag, va="center", ha="left", color="#f0f6fc", fontsize=8.5, fontweight="bold"
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, color="#c9d1d9", fontsize=8.8)
    ax1.set_xlim(0, 50.0)
    ax1.set_xlabel("Sustained Decode Throughput (tokens / sec, median ± range)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax1.set_title("Controlled Ablation: Static Offload vs. Dynamic SLRU Cache", color="#f0f6fc", fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", colors="#8b949e", labelsize=9)
    ax1.grid(axis="x", color="#21262d", linestyle="--", linewidth=0.7)
    ax1.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Panel 2: 3x (1K Prompt x 1K Decode) Long-Generation Suite
    # -------------------------------------------------------------------------
    ax2.set_facecolor("#161b22")

    categories = [
        "Q1: Distributed AI\n(916 PP / 1K TG)",
        "Q2: AgentOS Kernel\n(819 PP / 1K TG)",
        "Q3: Compiler & PTX\n(816 PP / 1K TG)",
        "Suite Mean\n(3,000 Decode Tok)",
    ]
    std_vals = [r["tg_speed"] for r in suite_std["results"]] + [suite_std["summary"]["avg_tg"]]
    cache_vals = [r["tg_speed"] for r in suite_cache["results"]] + [suite_cache["summary"]["avg_tg"]]
    hit_rates = [r["cache_rate"] for r in suite_cache["results"]] + [suite_cache["summary"]["avg_cache"]]

    x = np.arange(len(categories))
    width = 0.35

    b_std = ax2.bar(
        x - width / 2, std_vals, width, label="Stock llama.cpp (-fit on, 0% Cache)",
        color="#6e7681", edgecolor="#f0f6fc", linewidth=0.6, alpha=0.9
    )
    b_cache = ax2.bar(
        x + width / 2, cache_vals, width, label="BeyondVRAM (3,300 MiB SLRU Cache)",
        color="#1f6feb", edgecolor="#f0f6fc", linewidth=0.6, alpha=0.92
    )

    for bar in b_std:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.4,
            f"{h:.1f}", ha="center", va="bottom", color="#c9d1d9", fontsize=8.5, fontweight="bold"
        )

    for idx, bar in enumerate(b_cache):
        h = bar.get_height()
        gain = (h / std_vals[idx] - 1.0) * 100.0
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.4,
            f"{h:.1f}\n(+{gain:.0f}%)", ha="center", va="bottom", color="#79c0ff", fontsize=8.5, fontweight="bold"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, color="#c9d1d9", fontsize=8.8)
    ax2.set_ylim(0, 31.0)
    ax2.set_ylabel("Sustained Decode Throughput (tokens / sec)", color="#c9d1d9", fontsize=9.5, fontweight="bold")
    ax2.set_title("Long-Generation Suite: 3x (1K Prompt x 1K Decode)", color="#f0f6fc", fontsize=11, fontweight="bold", pad=10)
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


if __name__ == "__main__":
    main()
