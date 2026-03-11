"""
analyse_bridge_receivers.py
============================
Compares RX nodes separately per TX node for bridge environment.
Each receiver is kept as its own series — NOT mixed together.

File format expected: RX{A}_bridge_{B}_clean.csv
  → rx_node = A, env = bridge, tx_node = B

Outputs (in analysis_bridge/):
  - stats_per_tx_rx.csv              raw stats per (tx, rx) pair
  - plot_timeseries_tx{X}.png        time series: all RX nodes for TX=X
  - plot_distributions_tx{X}.png     violin plots: RSSI distribution per RX for TX=X
  - plot_diff_tx{X}.png              RSSI fluctuation (diff) per RX for TX=X
  - plot_overview_heatmap.png        heatmap: mean RSSI for all (tx, rx) pairs
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_bridge")
FILE_PATTERN = "*_clean.csv"

NODE_COLORS = {
    "A": "#2563EB",
    "B": "#16A34A",
    "C": "#DC2626",
    "D": "#D97706",
    "E": "#7C3AED",
}

# ── Load ───────────────────────────────────────────────────────────────────────

def load_all(data_dir):
    records = []
    csv_files = glob.glob(os.path.join(data_dir, FILE_PATTERN))

    if not csv_files:
        print(f"[ERROR] No files matching '{FILE_PATTERN}' in {data_dir}")
        return pd.DataFrame()

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        parts = filename.replace("_clean.csv", "").split("_")
        # Expected: RXA_bridge_B  →  ["RXA", "bridge", "B"]
        if len(parts) < 3:
            print(f"[WARN] Skipping {filename} — unexpected format")
            continue

        rx_node = parts[0].replace("RX", "")
        env_id  = parts[1]
        tx_node = parts[2]

        try:
            df = pd.read_csv(filepath, skiprows=1,
                             names=["node_id", "timestamp", "rssi", "lqi"])
            df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
            df["lqi"]  = pd.to_numeric(df["lqi"],  errors="coerce")
            df.dropna(subset=["rssi"], inplace=True)
            df["rx_node"] = rx_node
            df["tx_node"] = tx_node
            df["env"]     = env_id
            df["file"]    = filename
            df["sample_idx"] = range(len(df))
            records.append(df)
            print(f"  ✓ TX={tx_node}  RX={rx_node}  {len(df):>6} samples   "
                  f"RSSI: {df['rssi'].mean():.1f} ± {df['rssi'].std():.2f} dBm")
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(df):
    rows = []
    for (tx, rx), gdf in df.groupby(["tx_node", "rx_node"]):
        rssi = gdf["rssi"].values
        diff = np.diff(rssi)
        rows.append({
            "tx_node":        tx,
            "rx_node":        rx,
            "n_samples":      len(rssi),
            "mean_rssi":      round(np.mean(rssi), 3),
            "median_rssi":    round(np.median(rssi), 3),
            "std_rssi":       round(np.std(rssi), 3),
            "min_rssi":       round(np.min(rssi), 3),
            "max_rssi":       round(np.max(rssi), 3),
            "range_rssi":     round(np.max(rssi) - np.min(rssi), 3),
            "mean_abs_diff":  round(np.mean(np.abs(diff)), 3),
            "std_diff":       round(np.std(diff), 3),
            "max_abs_diff":   round(np.max(np.abs(diff)), 3),
            "skewness":       round(stats.skew(rssi), 3),
            "kurtosis":       round(stats.kurtosis(rssi), 3),
            "mean_lqi":       round(gdf["lqi"].mean(), 3),
            "std_lqi":        round(gdf["lqi"].std(), 3),
        })
    return pd.DataFrame(rows).sort_values(["tx_node", "rx_node"])


# ── Plots per TX ───────────────────────────────────────────────────────────────

def plot_timeseries_per_tx(df, output_dir, n_samples=600):
    """
    For each TX node: one plot with one subplot per RX node.
    Each RX node shown as its own separate time series — NOT overlaid.
    """
    for tx, tx_df in df.groupby("tx_node"):
        rx_nodes = sorted(tx_df["rx_node"].unique())
        n = len(rx_nodes)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), squeeze=False)
        fig.suptitle(f"RSSI Time Series — TX Node {tx}  |  Each Receiver Separately",
                     fontsize=13, fontweight="bold")

        for i, rx in enumerate(rx_nodes):
            ax = axes[i][0]
            rx_df = tx_df[tx_df["rx_node"] == rx].head(n_samples)
            color = NODE_COLORS.get(rx, "#888888")

            ax.plot(rx_df["sample_idx"].values, rx_df["rssi"].values,
                    color=color, linewidth=0.8, alpha=0.9)

            # Moving average
            if len(rx_df) > 20:
                ma = pd.Series(rx_df["rssi"].values).rolling(20, center=True).mean()
                ax.plot(rx_df["sample_idx"].values, ma.values,
                        color="black", linewidth=1.5, linestyle="--",
                        alpha=0.6, label="Moving avg (20)")

            mean_val = rx_df["rssi"].mean()
            ax.axhline(mean_val, color=color, linewidth=1, linestyle=":",
                       alpha=0.7, label=f"Mean: {mean_val:.1f} dBm")

            ax.set_title(f"RX Node {rx}  (TX={tx})", fontsize=11, fontweight="bold",
                         color=color)
            ax.set_ylabel("RSSI (dBm)", fontsize=10)
            ax.set_xlabel("Sample Index", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_facecolor("#F8FAFC")

            # Stats annotation
            std_val = rx_df["rssi"].std()
            ax.text(0.01, 0.05,
                    f"μ={mean_val:.1f}  σ={std_val:.2f}  "
                    f"n={len(rx_df)}",
                    transform=ax.transAxes, fontsize=9,
                    color="gray", va="bottom")

        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_timeseries_tx{tx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {path}")


def plot_distributions_per_tx(df, output_dir):
    """
    For each TX node: violin + box plot showing RSSI distribution
    for each RX node side by side.
    """
    for tx, tx_df in df.groupby("tx_node"):
        rx_nodes = sorted(tx_df["rx_node"].unique())
        if not rx_nodes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"RSSI Distribution per Receiver — TX Node {tx}",
                     fontsize=13, fontweight="bold")

        data_per_rx = [tx_df[tx_df["rx_node"] == rx]["rssi"].values for rx in rx_nodes]
        colors      = [NODE_COLORS.get(rx, "#888888") for rx in rx_nodes]

        # Violin plot
        ax = axes[0]
        vp = ax.violinplot(data_per_rx, positions=range(len(rx_nodes)),
                           showmedians=True, showextrema=True)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)
        ax.set_xticks(range(len(rx_nodes)))
        ax.set_xticklabels([f"RX {rx}" for rx in rx_nodes], fontsize=11)
        ax.set_ylabel("RSSI (dBm)", fontsize=11)
        ax.set_title("Violin Plot", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8FAFC")

        # Box plot
        ax = axes[1]
        bp = ax.boxplot(data_per_rx, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels([f"RX {rx}" for rx in rx_nodes], fontsize=11)
        ax.set_ylabel("RSSI (dBm)", fontsize=11)
        ax.set_title("Box Plot", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8FAFC")

        # Add mean annotations
        for i, (rx, data) in enumerate(zip(rx_nodes, data_per_rx)):
            ax.text(i + 1, np.min(data) - 0.5,
                    f"μ={np.mean(data):.1f}\nσ={np.std(data):.2f}",
                    ha="center", va="top", fontsize=8.5, color="gray")

        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_distributions_tx{tx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {path}")


def plot_diff_per_tx(df, output_dir):
    """
    For each TX node: plot the differentiated RSSI series
    for each RX node in separate subplots.
    """
    for tx, tx_df in df.groupby("tx_node"):
        rx_nodes = sorted(tx_df["rx_node"].unique())
        n = len(rx_nodes)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), squeeze=False)
        fig.suptitle(f"RSSI First-Order Difference (Δ RSSI) — TX Node {tx}",
                     fontsize=13, fontweight="bold")

        for i, rx in enumerate(rx_nodes):
            ax = axes[i][0]
            rx_df = tx_df[tx_df["rx_node"] == rx]
            diff  = np.diff(rx_df["rssi"].values)
            color = NODE_COLORS.get(rx, "#888888")

            ax.plot(diff, color=color, linewidth=0.7, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

            ax.set_title(f"RX Node {rx}  —  std(Δ)={np.std(diff):.3f}  "
                         f"mean|Δ|={np.mean(np.abs(diff)):.3f}",
                         fontsize=10, fontweight="bold", color=color)
            ax.set_ylabel("Δ RSSI (dBm)", fontsize=9)
            ax.set_xlabel("Sample", fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_facecolor("#F8FAFC")

        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_diff_tx{tx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {path}")


def plot_overview_heatmap(df, output_dir):
    """
    Overview heatmap: rows = TX nodes, cols = RX nodes.
    Shows mean RSSI and std dev for each (TX, RX) pair.
    """
    mean_pivot = df.groupby(["tx_node", "rx_node"])["rssi"].mean().unstack()
    std_pivot  = df.groupby(["tx_node", "rx_node"])["rssi"].std().unstack()
    diff_pivot = df.groupby(["tx_node", "rx_node"])["rssi"].apply(
        lambda x: np.std(np.diff(x.values))
    ).unstack()

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(mean_pivot) * 1.5 + 2)))
    fig.suptitle("Overview: All TX × RX Combinations — Bridge Environment",
                 fontsize=13, fontweight="bold")

    for ax, pivot, title, cmap, fmt in zip(
        axes,
        [mean_pivot, std_pivot, diff_pivot],
        ["Mean RSSI (dBm)", "Std Dev RSSI (dBm)", "Std of ΔRSSI (dBm)"],
        ["RdYlGn", "YlOrRd", "YlOrRd"],
        [".1f", ".2f", ".3f"]
    ):
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                    ax=ax, linewidths=0.8,
                    cbar_kws={"label": title},
                    annot_kws={"size": 11})
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("RX Node", fontsize=10)
        ax.set_ylabel("TX Node", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_overview_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Bridge Receiver Analysis")
    print(f"  Data:   {args.data}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    print("Loading data...")
    df = load_all(args.data)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"\nTotal samples: {len(df):,}")
    print(f"TX Nodes: {sorted(df['tx_node'].unique())}")
    print(f"RX Nodes: {sorted(df['rx_node'].unique())}")
    print(f"Pairs:    {sorted(df.groupby(['tx_node','rx_node']).groups.keys())}")

    # Stats CSV
    print("\nComputing statistics...")
    stats_df = compute_stats(df)
    path_csv = os.path.join(args.output, "stats_per_tx_rx.csv")
    stats_df.to_csv(path_csv, index=False)
    print(f"  ✓ {path_csv}")
    print("\n" + stats_df[["tx_node", "rx_node", "n_samples",
                            "mean_rssi", "std_rssi", "std_diff"]].to_string(index=False))

    # Plots
    print("\nGenerating plots...")
    plot_timeseries_per_tx(df, args.output)
    plot_distributions_per_tx(df, args.output)
    plot_diff_per_tx(df, args.output)
    plot_overview_heatmap(df, args.output)

    print(f"\n{'='*60}")
    print(f"  Done! Outputs saved to: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()