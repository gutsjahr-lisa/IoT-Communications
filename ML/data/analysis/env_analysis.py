"""
analyse_environments.py
=======================
Compares RSSI statistics across deployment environments and nodes.
Outputs:
  - stats_environments.csv   per-environment statistics
  - stats_nodes.csv          per-node statistics
  - stats_full.csv           per-file statistics
  - plot_rssi_distributions.png
  - plot_rssi_timeseries.png
  - plot_rssi_heatmap.png
  - plot_node_comparison.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
FILE_PATTERN = "*_clean.csv"

# Expected filename format: RXA_forest_B_clean.csv
# parts: [RXA, park, B, clean]

COLORS = {
    "A": "#2563EB",
    "B": "#16A34A",
    "C": "#DC2626",
    "D": "#D97706",
    "E": "#7C3AED",
}

ENV_COLORS = {
    "park":   "#16A34A",
    "bridge": "#2563EB",
    "garden": "#84CC16",
    "forest": "#065F46",
    "river":  "#0EA5E9",
    "lake":   "#6366F1",
}

# ── Load Data ──────────────────────────────────────────────────────────────────

def load_all(data_dir):
    records = []
    csv_files = glob.glob(os.path.join(data_dir, FILE_PATTERN))

    if not csv_files:
        print(f"[ERROR] No files matching '{FILE_PATTERN}' found in {data_dir}")
        return pd.DataFrame()

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").replace("_clean", "").split("_")
        if len(parts) < 3:
            print(f"[WARN] Skipping {filename} — unexpected format")
            continue

        rx_node = parts[0].replace("RX", "")   # e.g. "A"
        env_id  = parts[1]                       # e.g. "park"
        tx_node = parts[2]                       # e.g. "B"

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
            records.append(df)
            print(f"  ✓ {filename:40s}  {len(df):>6} rows  RSSI: {df['rssi'].mean():.1f} dBm")
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(df, group_col):
    """Compute RSSI statistics grouped by group_col."""
    rows = []
    for group_val, gdf in df.groupby(group_col):
        rssi = gdf["rssi"].values
        diff = np.diff(rssi)

        # Basic stats
        row = {
            group_col:        group_val,
            "n_samples":      len(rssi),
            "mean_rssi":      np.mean(rssi),
            "median_rssi":    np.median(rssi),
            "std_rssi":       np.std(rssi),
            "min_rssi":       np.min(rssi),
            "max_rssi":       np.max(rssi),
            "range_rssi":     np.max(rssi) - np.min(rssi),
            # Fluctuation (after diff)
            "mean_abs_diff":  np.mean(np.abs(diff)),
            "std_diff":       np.std(diff),
            "max_abs_diff":   np.max(np.abs(diff)),
            # Distribution shape
            "skewness":       stats.skew(rssi),
            "kurtosis":       stats.kurtosis(rssi),
            # LQI
            "mean_lqi":       gdf["lqi"].mean(),
            "std_lqi":        gdf["lqi"].std(),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    # Round for readability
    float_cols = result.select_dtypes(include="float").columns
    result[float_cols] = result[float_cols].round(3)
    return result


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_distributions(df, output_dir):
    """Violin + box plots of RSSI per environment and per node."""
    envs  = sorted(df["env"].unique())
    nodes = sorted(df["tx_node"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("RSSI Distribution by Environment and Transmitter Node",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Left: by environment ──
    ax = axes[0]
    env_data  = [df[df["env"] == e]["rssi"].values for e in envs]
    env_clrs  = [ENV_COLORS.get(e, "#888888") for e in envs]
    vp = ax.violinplot(env_data, positions=range(len(envs)), showmedians=True, showextrema=True)
    for i, (body, color) in enumerate(zip(vp["bodies"], env_clrs)):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, fontsize=11)
    ax.set_ylabel("RSSI (dBm)", fontsize=11)
    ax.set_title("By Environment", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F8FAFC")

    # ── Right: by TX node ──
    ax = axes[1]
    node_data = [df[df["tx_node"] == n]["rssi"].values for n in nodes]
    node_clrs = [COLORS.get(n, "#888888") for n in nodes]
    vp = ax.violinplot(node_data, positions=range(len(nodes)), showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], node_clrs):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([f"Node {n}" for n in nodes], fontsize=11)
    ax.set_ylabel("RSSI (dBm)", fontsize=11)
    ax.set_title("By Transmitter Node", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F8FAFC")

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_rssi_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_timeseries(df, output_dir, n_samples=500):
    """Plot RSSI time series for each TX node per environment."""
    envs  = sorted(df["env"].unique())
    nodes = sorted(df["tx_node"].unique())

    fig, axes = plt.subplots(len(envs), 1, figsize=(16, 4 * len(envs)), squeeze=False)
    fig.suptitle("RSSI Time Series (first 500 samples per node per environment)",
                 fontsize=13, fontweight="bold")

    for i, env in enumerate(envs):
        ax = axes[i][0]
        env_df = df[df["env"] == env]
        for node in nodes:
            node_df = env_df[env_df["tx_node"] == node].head(n_samples)
            if len(node_df) == 0:
                continue
            ax.plot(node_df["rssi"].values,
                    label=f"TX Node {node}",
                    color=COLORS.get(node, "#888888"),
                    alpha=0.8, linewidth=0.8)
        ax.set_title(f"Environment: {env.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("RSSI (dBm)")
        ax.set_xlabel("Sample")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#F8FAFC")

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_rssi_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_heatmap(df, output_dir):
    """Heatmap of mean RSSI: environments vs TX nodes."""
    pivot = df.groupby(["env", "tx_node"])["rssi"].mean().unstack(fill_value=np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(pivot) * 1.2 + 2)))
    fig.suptitle("Mean RSSI and Std Dev — Environment × Transmitter Node",
                 fontsize=13, fontweight="bold")

    # Mean RSSI heatmap
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                ax=axes[0], linewidths=0.5, cbar_kws={"label": "Mean RSSI (dBm)"})
    axes[0].set_title("Mean RSSI (dBm)", fontweight="bold")
    axes[0].set_xlabel("TX Node")
    axes[0].set_ylabel("Environment")

    # Std dev heatmap
    pivot_std = df.groupby(["env", "tx_node"])["rssi"].std().unstack(fill_value=np.nan)
    sns.heatmap(pivot_std, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=axes[1], linewidths=0.5, cbar_kws={"label": "Std Dev (dBm)"})
    axes[1].set_title("RSSI Std Dev (dBm)", fontweight="bold")
    axes[1].set_xlabel("TX Node")
    axes[1].set_ylabel("Environment")

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_rssi_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_diff_stats(df, output_dir):
    """Bar chart comparing fluctuation (std of diff) per node and environment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RSSI Fluctuation (Std of First-Order Difference)",
                 fontsize=13, fontweight="bold")

    # By TX node
    node_stats = df.groupby("tx_node")["rssi"].apply(lambda x: np.std(np.diff(x.values)))
    axes[0].bar(node_stats.index,
                node_stats.values,
                color=[COLORS.get(n, "#888") for n in node_stats.index],
                edgecolor="white", linewidth=0.5)
    axes[0].set_title("By Transmitter Node", fontweight="bold")
    axes[0].set_xlabel("TX Node")
    axes[0].set_ylabel("Std of ΔRSSI (dBm)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_facecolor("#F8FAFC")
    for i, (node, val) in enumerate(node_stats.items()):
        axes[0].text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # By environment
    env_stats = df.groupby("env")["rssi"].apply(lambda x: np.std(np.diff(x.values)))
    colors_env = [ENV_COLORS.get(e, "#888") for e in env_stats.index]
    axes[1].bar(env_stats.index, env_stats.values,
                color=colors_env, edgecolor="white", linewidth=0.5)
    axes[1].set_title("By Environment", fontweight="bold")
    axes[1].set_xlabel("Environment")
    axes[1].set_ylabel("Std of ΔRSSI (dBm)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_facecolor("#F8FAFC")
    for i, (env, val) in enumerate(env_stats.items()):
        axes[1].text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_node_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse RSSI data across environments.")
    parser.add_argument("--data",   default=DATA_DIR,   help="Path to data folder")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  IoT RSSI Analysis")
    print(f"  Data:   {args.data}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # Load
    print("Loading data...")
    df = load_all(args.data)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"\nTotal samples loaded: {len(df):,}")
    print(f"Environments: {sorted(df['env'].unique())}")
    print(f"TX Nodes:     {sorted(df['tx_node'].unique())}")
    print(f"RX Nodes:     {sorted(df['rx_node'].unique())}")

    # Stats
    print("\nComputing statistics...")
    stats_env  = compute_stats(df, "env")
    stats_node = compute_stats(df, "tx_node")
    stats_file = compute_stats(df, "file")

    # Save CSVs
    path_env  = os.path.join(args.output, "stats_environments.csv")
    path_node = os.path.join(args.output, "stats_nodes.csv")
    path_file = os.path.join(args.output, "stats_full.csv")

    stats_env.to_csv(path_env,   index=False)
    stats_node.to_csv(path_node, index=False)
    stats_file.to_csv(path_file, index=False)

    print(f"\n  ✓ {path_env}")
    print(f"  ✓ {path_node}")
    print(f"  ✓ {path_file}")

    # Print to terminal
    print("\n── Statistics by Environment ──────────────────────────")
    print(stats_env[["env", "n_samples", "mean_rssi", "std_rssi",
                      "range_rssi", "mean_abs_diff", "std_diff"]].to_string(index=False))

    print("\n── Statistics by TX Node ───────────────────────────────")
    print(stats_node[["tx_node", "n_samples", "mean_rssi", "std_rssi",
                       "range_rssi", "mean_abs_diff", "std_diff"]].to_string(index=False))

    # Plots
    print("\nGenerating plots...")
    plot_distributions(df, args.output)
    plot_timeseries(df, args.output)
    plot_heatmap(df, args.output)
    plot_diff_stats(df, args.output)

    print(f"\n{'='*60}")
    print(f"  Done! All outputs saved to: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()