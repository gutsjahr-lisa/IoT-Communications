import pandas as pd
import numpy as np
import os
import glob


def load_and_preprocess(data_dir, frame_size=100, overlap=0.5):
    """
    Loading all csv logs, differentiates, normalizes and segmentation operations.
    Re-Naming: node_A_env_forest.csv
    """
    frames_all = []
    labels_env = []
    labels_node = []

    csv_files = glob.glob(os.path.join(data_dir, "*_clean.csv"))

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        print(f"[preprocessing]   Lade: {filename}")

        # Format: RXA_park_B.csv
        # parts = ["RXA", "park", "B"]
        parts = filename.replace(".csv", "").split("_")
        env_id = parts[1]  # "park"
        tx_node_id = parts[2]  # "B", "C", "D"

        df = pd.read_csv(filepath)
        df.columns = ["node_id", "timestamp", "rssi", "lqi"]
        rssi = df["rssi"].values.astype(float)

        # 1. Differentiation, so focus will be on the change in the link quality alone
        y = np.diff(rssi)

        # 2. Normalization to [0, 1], to prevent from bias
        y_min, y_max = y.min(), y.max()
        if y_max - y_min == 0:
            continue
        z = (y - y_min) / (y_max - y_min)

        # 3. Segmentation from time series into model inputs with Overlap
        step = int(frame_size * (1 - overlap))
        for start in range(0, len(z) - frame_size, step):
            frame = z[start:start + frame_size]
            frames_all.append(frame)
            labels_env.append(env_id)
            labels_node.append(tx_node_id)

    X = np.array(frames_all, dtype=np.float32)
    return X, np.array(labels_env), np.array(labels_node)
