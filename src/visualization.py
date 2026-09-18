import csv

def load_results(csv_file):
    with csv_file.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def plot_results(csv_file, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_results(csv_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranks = [8, 16, 32]
    rank_f1 = [float(data[name]["f1"]) for name in ("rank8", "baseline", "rank32")]
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, rank_f1, "o-", color="steelblue", linewidth=2, markersize=8)
    for x, y in zip(ranks, rank_f1):
        plt.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    plt.xlabel("LoRA rank")
    plt.ylabel("F1")
    plt.title("F1 vs LoRA rank (QLoRA, lr=5e-5)")
    plt.xticks(ranks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_rank_f1.png", dpi=150)
    plt.close()

    rates = ["1e-5", "5e-5", "1e-4"]
    rate_f1 = [float(data[name]["f1"]) for name in ("lr1e5", "baseline", "lr1e4")]
    plt.figure(figsize=(6, 4))
    plt.plot(rates, rate_f1, "s-", color="coral", linewidth=2, markersize=8)
    for x, y in zip(rates, rate_f1):
        plt.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    plt.xlabel("Learning rate")
    plt.ylabel("F1")
    plt.title("F1 vs Learning rate (QLoRA, rank=16)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_lr_f1.png", dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    names = ["QLoRA (4-bit)", "LoRA (BF16)"]
    f1_values = [float(data["baseline"]["f1"]), float(data["lora"]["f1"])]
    ax1.bar(names, f1_values, width=0.4, color=["steelblue", "coral"])
    ax1.set_ylabel("F1")
    ax1.set_ylim(0.8, 0.87)
    for i, value in enumerate(f1_values):
        ax1.text(i, value + 0.002, f"{value:.4f}", ha="center")
    ax2 = ax1.twinx()
    memory = [25.9, 27.8]
    ax2.plot([0, 1], memory, "D--", color="green", markersize=8)
    ax2.set_ylabel("GPU Memory (GB)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    plt.title("QLoRA vs LoRA: F1 & GPU Memory")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_quant_compare.png", dpi=150)
    plt.close()
