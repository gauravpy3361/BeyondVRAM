"""
Extracts hardware CUPTI kernel and PCIe DMA timestamps from the Nsight Systems
SQLite export (`bin/async_pipeline_trace.sqlite`) and renders a clean timeline
comparison (`docs/assets/nsys_timeline.png`) + JSON summary (`benchmarks/nsys_trace_summary.json`).
"""

import os
import json
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SQLITE_PATH = os.path.join("bin", "async_pipeline_trace.sqlite")
OUT_IMG_PATH = os.path.join("docs", "assets", "nsys_timeline.png")
OUT_JSON_PATH = os.path.join("benchmarks", "nsys_trace_summary.json")


def extract_trace_windows(sqlite_path: str):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    kernels = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """
    ).fetchall()

    memcpys = cur.execute(
        """
        SELECT m.start, m.end, m.streamId, m.bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.bytes = 1351680
        ORDER BY m.start
        """
    ).fetchall()

    # Mode A (Same-Stream Sequential on Stream 13): steps 196..199 -> kernels[812:820], memcpys[416:420]
    a_k8 = kernels[812:820]
    a_m4 = memcpys[416:420]
    a_base = a_k8[0][0]

    mode_a_events = []
    for k in a_k8:
        mode_a_events.append({
            "type": "attn" if "attention" in k[3] else "ffn",
            "stream": k[2],
            "start_us": round((k[0] - a_base) / 1e3, 2),
            "end_us": round((k[1] - a_base) / 1e3, 2),
            "dur_us": round((k[1] - k[0]) / 1e3, 2),
        })
    for m in a_m4:
        mode_a_events.append({
            "type": "dma",
            "stream": m[2],
            "start_us": round((m[0] - a_base) / 1e3, 2),
            "end_us": round((m[1] - a_base) / 1e3, 2),
            "dur_us": round((m[1] - m[0]) / 1e3, 2),
        })

    # Mode B (Dual-Stream Lockstep Overlapped on Streams 13 & 14): steps 196..199 -> kernels[1212:1220], memcpys[616:620]
    b_k8 = kernels[1212:1220]
    b_m4 = memcpys[616:620]
    b_base = b_k8[0][0]

    mode_b_events = []
    for k in b_k8:
        mode_b_events.append({
            "type": "attn" if "attention" in k[3] else "ffn",
            "stream": k[2],
            "start_us": round((k[0] - b_base) / 1e3, 2),
            "end_us": round((k[1] - b_base) / 1e3, 2),
            "dur_us": round((k[1] - k[0]) / 1e3, 2),
        })
    for m in b_m4:
        mode_b_events.append({
            "type": "dma",
            "stream": m[2],
            "start_us": round((m[0] - b_base) / 1e3, 2),
            "end_us": round((m[1] - b_base) / 1e3, 2),
            "dur_us": round((m[1] - m[0]) / 1e3, 2),
        })

    return {"mode_a_sequential": mode_a_events, "mode_b_pipelined": mode_b_events}


def render_timeline(data):
    os.makedirs(os.path.dirname(OUT_IMG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    plt.rcParams["font.sans-serif"] = "Segoe UI"
    plt.rcParams["axes.edgecolor"] = "#333942"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), dpi=180, sharex=True)
    fig.patch.set_facecolor("#14171c")

    colors = {
        "attn": "#2b8cbe",  # CUDA Compute Kernel (Attention + Hot Experts)
        "dma":  "#e68a00",  # PCIe HtoD DMA Copy (1.29 MiB Cold Expert)
        "ffn":  "#31a354",  # Cold Expert SwiGLU Kernel
    }

    # --- Panel 1: Mode A (Sequential on Stream 13) ---
    ax1.set_facecolor("#1b1f26")
    ax1.set_title(
        "Mode A — Single-Stream Sequential (Stream 13): 1.29 MiB PCIe HtoD DMA Stalls GPU Compute (~456 µs / step)",
        color="#e6edf3", fontsize=11, fontweight="bold", loc="left", pad=8
    )
    for ev in data["mode_a_sequential"]:
        y = 1.0 if ev["type"] != "dma" else 0.25
        w = ev["end_us"] - ev["start_us"]
        rect = mpatches.Rectangle(
            (ev["start_us"], y - 0.24), w, 0.48,
            facecolor=colors[ev["type"]], edgecolor="#0d1117", linewidth=1.0
        )
        ax1.add_patch(rect)
        if w > 80:
            label = f"Attn+Hot\n{ev['dur_us']:.0f} µs" if ev["type"] == "attn" else f"H2D DMA\n{ev['dur_us']:.0f} µs"
            ax1.text(
                ev["start_us"] + w / 2, y, label,
                ha="center", va="center", color="white", fontsize=8, fontweight="bold"
            )

    ax1.set_yticks([0.25, 1.0])
    ax1.set_yticklabels(["Stream 13\n[Memcpy H2D]", "Stream 13\n[GPU Compute]"], color="#c9d1d9", fontsize=9)
    ax1.set_ylim(-0.2, 1.45)
    ax1.grid(axis="x", color="#2d333b", linestyle="--", linewidth=0.7, alpha=0.8)
    ax1.tick_params(colors="#8b949e")

    # --- Panel 2: Mode B (Dual-Stream Lockstep Overlapped) ---
    ax2.set_facecolor("#1b1f26")
    ax2.set_title(
        "Mode B — Dual-Stream Lockstep Pipeline (Streams 13 & 14): 1.29 MiB HtoD DMA Overlaps Attention Compute (~278 µs / step, 1.61x Faster)",
        color="#58a6ff", fontsize=11, fontweight="bold", loc="left", pad=8
    )
    for ev in data["mode_b_pipelined"]:
        y = 1.0 if ev["stream"] == 13 else 0.25
        w = ev["end_us"] - ev["start_us"]
        rect = mpatches.Rectangle(
            (ev["start_us"], y - 0.24), w, 0.48,
            facecolor=colors[ev["type"]], edgecolor="#0d1117", linewidth=1.0
        )
        ax2.add_patch(rect)
        if w > 80:
            label = f"Attn+Hot\n{ev['dur_us']:.0f} µs" if ev["type"] == "attn" else f"Async H2D\n{ev['dur_us']:.0f} µs"
            ax2.text(
                ev["start_us"] + w / 2, y, label,
                ha="center", va="center", color="white", fontsize=8, fontweight="bold"
            )

    ax2.set_yticks([0.25, 1.0])
    ax2.set_yticklabels(["Stream 14\n[Async DMA H2D]", "Stream 13\n[GPU Compute]"], color="#c9d1d9", fontsize=9)
    ax2.set_ylim(-0.2, 1.45)
    ax2.set_xlim(-20, 1850)
    ax2.set_xlabel("Hardware Execution Timeline (µs) — Captured via NVIDIA Nsight Systems 2025.6 (CUPTI SQLite Trace on RTX 3050 6GB)", color="#c9d1d9", fontsize=9.5)
    ax2.grid(axis="x", color="#2d333b", linestyle="--", linewidth=0.7, alpha=0.8)
    ax2.tick_params(colors="#8b949e")

    legend_patches = [
        mpatches.Patch(color=colors["attn"], label="Attention + Hot Experts Compute (Stream 13, ~256–264 µs)"),
        mpatches.Patch(color=colors["dma"],  label="PCIe 4.0 HtoD Cold-Expert Weight DMA (1.29 MiB, ~113–116 µs)"),
        mpatches.Patch(color=colors["ffn"],  label="Cold-Expert SwiGLU Compute (Stream 13, ~11–16 µs, runs after cudaStreamWaitEvent)"),
    ]
    fig.legend(
        handles=legend_patches, loc="upper center", ncol=3,
        frameon=True, facecolor="#1b1f26", edgecolor="#333942",
        labelcolor="#e6edf3", fontsize=8.8, bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUT_IMG_PATH, dpi=180, facecolor=fig.get_facecolor())
    print(f"Saved Nsight timeline chart to {OUT_IMG_PATH}")


if __name__ == "__main__":
    trace_data = extract_trace_windows(SQLITE_PATH)
    render_timeline(trace_data)
