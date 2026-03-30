"""
Generate comparison charts for the multi-provider LLM evaluation.
Produces PNG images for slides and screenshots.

Usage:
    pip install matplotlib
    python generate_charts.py

Output:
    data/benchmark/chart_accuracy.png
    data/benchmark/chart_response_time.png
    data/benchmark/chart_combined.png
"""

import json
import glob
import os

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving files
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    exit(1)


def load_eval_data():
    """Load all eval_*.json files."""
    eval_files = sorted(glob.glob("data/benchmark/eval_*.json"))
    if not eval_files:
        print("No eval_*.json files found. Run evaluations first.")
        return None, None

    providers = []
    data_all = []
    for path in eval_files:
        with open(path) as f:
            data = json.load(f)
        provider = data.get("provider", os.path.basename(path).replace("eval_", "").replace(".json", ""))
        providers.append(provider.capitalize())
        data_all.append(data)

    return providers, data_all


def chart_accuracy(providers, data_all):
    """Bar chart comparing answer correctness across providers."""
    accuracies = []
    for data in data_all:
        if data.get("answer"):
            accuracies.append(data["answer"]["accuracy"] * 100)
        else:
            accuracies.append(0)

    if all(a == 0 for a in accuracies):
        print("  Skipping accuracy chart — no answer data yet (run eval without --quick)")
        return

    colors = ["#4285F4", "#34A853", "#EA4335", "#FBBC05"]  # Google-style palette
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(providers, accuracies, color=colors[:len(providers)], width=0.6, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_ylabel("Answer Correctness (%)", fontsize=12)
    ax.set_title("Answer Correctness by LLM Provider", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    plt.tight_layout()
    out = "data/benchmark/chart_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def chart_response_time(providers, data_all):
    """Bar chart comparing average response time across providers."""
    times = []
    for data in data_all:
        if data.get("answer"):
            times.append(data["answer"]["avg_time_seconds"])
        else:
            times.append(0)

    if all(t == 0 for t in times):
        print("  Skipping response time chart — no answer data yet")
        return

    colors = ["#4285F4", "#34A853", "#EA4335", "#FBBC05"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(providers, times, color=colors[:len(providers)], width=0.6, edgecolor="white", linewidth=1.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{t:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_ylabel("Avg Response Time (seconds)", fontsize=12)
    ax.set_title("Average Response Time by LLM Provider", fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    plt.tight_layout()
    out = "data/benchmark/chart_response_time.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def chart_combined(providers, data_all):
    """Combined dashboard: classification + retrieval + answer accuracy + time."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#4285F4", "#34A853", "#EA4335", "#FBBC05"][:len(providers)]

    # 1. Classification accuracy (same for all)
    class_acc = [d["classification"]["accuracy"] * 100 for d in data_all]
    axes[0].bar(providers, class_acc, color=colors, width=0.6)
    axes[0].set_title("Classification\nAccuracy", fontweight="bold")
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(class_acc):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=11)

    # 2. Retrieval precision (same for all)
    ret_prec = []
    for d in data_all:
        ret_prec.append(d["retrieval"]["topic_precision"] * 100 if d.get("retrieval") else 0)
    axes[1].bar(providers, ret_prec, color=colors, width=0.6)
    axes[1].set_title("Retrieval Topic\nPrecision", fontweight="bold")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(ret_prec):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)

    # 3. Answer correctness (differs per LLM)
    ans_acc = []
    for d in data_all:
        ans_acc.append(d["answer"]["accuracy"] * 100 if d.get("answer") else 0)
    axes[2].bar(providers, ans_acc, color=colors, width=0.6)
    axes[2].set_title("Answer\nCorrectness", fontweight="bold")
    axes[2].set_ylim(0, 110)
    for i, v in enumerate(ans_acc):
        label = f"{v:.1f}%" if v > 0 else "N/A"
        axes[2].text(i, max(v, 2) + 1, label, ha="center", fontweight="bold", fontsize=11)

    # 4. Response time (differs per LLM)
    resp_time = []
    for d in data_all:
        resp_time.append(d["answer"]["avg_time_seconds"] if d.get("answer") else 0)
    axes[3].bar(providers, resp_time, color=colors, width=0.6)
    axes[3].set_title("Avg Response\nTime (s)", fontweight="bold")
    for i, v in enumerate(resp_time):
        label = f"{v:.1f}s" if v > 0 else "N/A"
        axes[3].text(i, max(v, 0.1) + 0.1, label, ha="center", fontweight="bold", fontsize=11)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10, rotation=15)

    fig.suptitle("PSLE Math Study Bot — Multi-Provider LLM Comparison",
                 fontsize=15, fontweight="bold", y=1.02)

    # Add annotation
    fig.text(0.5, -0.04,
             "Note: Classification and retrieval use local embeddings — identical across providers.\n"
             "Only answer generation differs between LLMs.",
             ha="center", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    out = "data/benchmark/chart_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("Generating comparison charts...")
    providers, data_all = load_eval_data()
    if providers:
        chart_accuracy(providers, data_all)
        chart_response_time(providers, data_all)
        chart_combined(providers, data_all)
        print("\nDone! Charts saved to data/benchmark/")
        print("Use these in your slides or screenshot them.")