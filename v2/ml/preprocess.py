"""
ml/preprocess.py
----------------
Loads the k3s final_labeled_dataset.csv, cleans it, normalizes features,
builds per-(run, node) sliding windows, and saves train/val/test splits.

Handles:
  - Dead eBPF sensor detection (all eBPF columns = 0 for an entire run+node)
  - log1p transform on network byte columns (heavy-tailed distributions)
  - Run-aware train/test split (no data leakage between runs)
  - Autoencoder windows (normal-only) AND evaluation windows (all labels)
  - Flat feature arrays for the Random Forest classifier

Output files
------------
dataset/processed/
    X_train.npy        normal-only train windows  (N, SEQ_LEN, 12)
    X_val.npy          normal-only val windows    (N, SEQ_LEN, 12)
    X_test_all.npy     ALL test windows           (N, SEQ_LEN, 12)
    y_test_all.npy     test labels (string array)
    X_clf_train.npy    flat features for RF train  (N, 12*4)
    y_clf_train.npy    labels for RF train
    X_clf_test.npy     flat features for RF test   (N, 12*4)
    y_clf_test.npy     labels for RF test
    feature_names.txt  one feature name per line
ml/
    scaler.pkl         fitted MinMaxScaler
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Configuration ────────────────────────────────────────────────────────────
# Resolve paths relative to this script's location so they work regardless of CWD
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_V2_DIR      = os.path.dirname(_SCRIPT_DIR)          # infra-proj/v2
_ROOT_DIR    = os.path.dirname(_V2_DIR)               # infra-proj
RAW_FILE     = os.path.join(_ROOT_DIR, "k3s-monitoring-setup", "final_labeled_dataset.csv")
OUT_DIR      = os.path.join(_V2_DIR, "dataset", "processed")
SCALER_PATH  = os.path.join(_V2_DIR, "ml", "scaler.pkl")

FEATURES = [
    "avg_cpu", "avg_mem",
    "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out",
    "net_drop_rate",
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]

# Network byte columns get log1p transform (heavy-tailed)
BYTE_COLS = [
    "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out",
]

# eBPF columns — used for dead-sensor detection
EBPF_COLS = [
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]

SEQ_LEN      = 30      # 30 × 10s = 5 min window
VAL_SPLIT    = 0.15    # 15% of normal training windows for validation
TEST_FRAC    = 0.20    # 20% of runs held out for testing


# ── Step 1 : Load & validate ─────────────────────────────────────────────────
def load_dataset(path):
    print(f"  Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")

    # Ensure numeric features are actually numeric
    for col in FEATURES:
        if col not in df.columns:
            print(f"  [ERROR] Missing column: {col}")
            sys.exit(1)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURES)
    if len(df) < before:
        print(f"  [WARN] Dropped {before - len(df)} rows with NaN features")

    print(f"  Labels: {dict(df['label'].value_counts())}")
    print(f"  Nodes:  {dict(df['node'].value_counts())}")
    print(f"  Runs:   {df['run_id'].nunique()}")

    return df


# ── Step 2 : Detect dead eBPF sensors ────────────────────────────────────────
def detect_dead_sensors(df):
    """Find (run_id, node) combos where ALL eBPF columns sum to 0.
    This means Tetragon was dead/crashed for that entire run on that node.

    Returns (df_full, df_clean):
      - df_full:  original data KEPT INTACT (for LSTM Autoencoder)
                  → the AE learns that all-zero eBPF = normal, so it won't
                    produce false positives if Tetragon fails in production.
      - df_clean: dead-sensor rows REMOVED (for Random Forest classifier)
                  → the RF needs complete features to learn fault signatures;
                    dead-sensor anomaly rows have missing eBPF signal.
    """
    grouped = df.groupby(["run_id", "node"])[EBPF_COLS].sum()
    dead = grouped[grouped.sum(axis=1) == 0].index

    if len(dead) == 0:
        print("  No dead eBPF sensors found — all data is clean.")
        return df, df

    print(f"  Found {len(dead)} dead sensor (run, node) combos:")
    for run_id, node in dead:
        count = len(df[(df["run_id"] == run_id) & (df["node"] == node)])
        print(f"    Run {run_id} / {node}: {count} rows")

    df_indexed = df.set_index(["run_id", "node"])
    df_clean = df_indexed.drop(index=dead).reset_index()
    dropped = len(df) - len(df_clean)

    print(f"\n  Strategy (hybrid):")
    print(f"    LSTM Autoencoder : KEEPING all {len(df)} rows (including dead sensors)")
    print(f"                      → model learns all-zero eBPF = valid normal pattern")
    print(f"                      → won't false-alarm if Tetragon fails in production")
    print(f"    Random Forest    : DROPPING {dropped} dead-sensor rows ({dropped/len(df)*100:.1f}%)")
    print(f"                      → classifier needs complete eBPF features for fault detection")

    return df, df_clean


# ── Step 3 : Per-group sliding windows ───────────────────────────────────────
def make_windows(df, features, seq_len, scaler=None, label_col="label"):
    """Build sliding windows within each (run_id, node) group.
    Returns (windows, labels) where labels is the majority label per window."""
    windows = []
    labels  = []
    group_counts = {}

    for (run_id, node), grp in df.groupby(["run_id", "node"]):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        vals = grp[features].values.astype(np.float32)

        if scaler is not None:
            vals = scaler.transform(vals)

        lbls = grp[label_col].values
        n_win = max(0, len(vals) - seq_len + 1)

        for i in range(n_win):
            windows.append(vals[i : i + seq_len])
            # Window label: if ANY timestep is anomalous, take that anomaly label
            win_labels = lbls[i : i + seq_len]
            unique = set(win_labels)
            if len(unique) == 1 and "normal" in unique:
                labels.append("normal")
            else:
                anomalies = [l for l in unique if l != "normal"]
                labels.append(anomalies[0] if anomalies else "normal")

        group_counts[f"{run_id[:8]}/{node}"] = n_win

    print(f"\n  Windows per (run, node)  [showing first 10]:")
    for i, (k, v) in enumerate(sorted(group_counts.items())):
        if i < 10:
            print(f"    {k}: {v}")
    if len(group_counts) > 10:
        print(f"    ... and {len(group_counts) - 10} more groups")

    return np.array(windows, dtype=np.float32), np.array(labels)


# ── Step 4 : Flatten windows for classifier ──────────────────────────────────
def flatten_windows(X):
    """Convert (N, seq, features) → (N, features*4) using mean/std/min/max."""
    stats = []
    for i in range(X.shape[2]):
        feat = X[:, :, i]
        stats.append(np.mean(feat, axis=1, keepdims=True))
        stats.append(np.std(feat,  axis=1, keepdims=True))
        stats.append(np.min(feat,  axis=1, keepdims=True))
        stats.append(np.max(feat,  axis=1, keepdims=True))
    return np.hstack(stats).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 1  Loading k3s labeled dataset")
    print("=" * 60)
    df = load_dataset(RAW_FILE)

    print("\n" + "=" * 60)
    print("STEP 2  log1p transform on network byte columns")
    print("=" * 60)
    for col in BYTE_COLS:
        df[col] = np.log1p(df[col])
    for col in BYTE_COLS:
        print(f"  {col}: max -> {df[col].max():.4f}")

    print("\n" + "=" * 60)
    print("STEP 3  Detecting dead eBPF sensors")
    print("=" * 60)
    df_full, df_clean = detect_dead_sensors(df)
    # df_full  → used for LSTM autoencoder (keeps dead-sensor rows)
    # df_clean → used for RF classifier (drops dead-sensor rows)

    # Quick stats on eBPF zero rates (informational)
    print("\n  eBPF zero-rate per column (% of rows that are 0):")
    for col in EBPF_COLS:
        zero_pct = (df_full[col] == 0).mean() * 100
        print(f"    {col:30s}  {zero_pct:5.1f}% zeros")
    print("  (Zeros in normal rows are EXPECTED — these features spike during attacks)")

    print("\n" + "=" * 60)
    print("STEP 4  Run-aware train/test split")
    print("=" * 60)
    unique_runs = df_full["run_id"].unique()
    train_runs, test_runs = train_test_split(
        unique_runs, test_size=TEST_FRAC, random_state=42
    )
    print(f"  Total runs: {len(unique_runs)}")
    print(f"  Train runs: {len(train_runs)}")
    print(f"  Test runs:  {len(test_runs)}")

    # LSTM autoencoder uses df_full (includes dead-sensor data)
    df_train_full = df_full[df_full["run_id"].isin(train_runs)].copy()
    df_test_full  = df_full[df_full["run_id"].isin(test_runs)].copy()
    train_normal  = df_train_full[df_train_full["label"] == "normal"]

    # Classifier uses df_clean (dead sensors removed)
    df_train_clean = df_clean[df_clean["run_id"].isin(train_runs)].copy()
    df_test_clean  = df_clean[df_clean["run_id"].isin(test_runs)].copy()

    print(f"\n  LSTM splits (full data, incl. dead sensors):")
    print(f"    Train: {len(df_train_full)} rows, normal: {len(train_normal)}")
    print(f"    Test:  {len(df_test_full)} rows")
    print(f"  RF splits (cleaned, dead sensors removed):")
    print(f"    Train: {len(df_train_clean)} rows")
    print(f"    Test:  {len(df_test_clean)} rows")

    print("\n" + "=" * 60)
    print("STEP 5  Fitting MinMaxScaler on normal training data")
    print("=" * 60)
    scaler = MinMaxScaler()
    # Fit on ALL normal training data (including dead-sensor rows)
    # This teaches the scaler that zero-eBPF is within normal range
    scaler.fit(train_normal[FEATURES].values)

    print("  Scaler ranges (from normal training data):")
    for feat, mn, mx in zip(FEATURES, scaler.data_min_, scaler.data_max_):
        print(f"    {feat:30s}  [{mn:.4f} .. {mx:.4f}]")

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Scaler saved → {SCALER_PATH}")

    print("\n" + "=" * 60)
    print("STEP 6  Building sliding windows")
    print("=" * 60)

    # 6a. Training windows (normal only, for autoencoder)
    print("\n  --- Normal training windows (for LSTM Autoencoder) ---")
    X_normal, y_normal = make_windows(
        train_normal, FEATURES, SEQ_LEN, scaler=scaler
    )
    print(f"  Total normal windows: {len(X_normal)}")

    # Split normal into train/val
    cut = int(len(X_normal) * (1 - VAL_SPLIT))
    X_train = X_normal[:cut]
    X_val   = X_normal[cut:]
    print(f"  AE Train: {len(X_train)}  |  AE Val: {len(X_val)}")

    # 6b. Test windows for LSTM (ALL labels, full data)
    print("\n  --- Test windows (all labels, for LSTM evaluation) ---")
    X_test_all, y_test_all = make_windows(
        df_test_full, FEATURES, SEQ_LEN, scaler=scaler
    )
    print(f"  Total test windows: {len(X_test_all)}")
    from collections import Counter
    test_dist = Counter(y_test_all)
    for k, v in sorted(test_dist.items()):
        print(f"    {k:35s}: {v}")

    # 6c. Classifier data (CLEANED — dead sensors removed)
    print("\n  --- Classifier training windows (cleaned data, for RF) ---")
    X_clf_all, y_clf_all = make_windows(
        df_train_clean, FEATURES, SEQ_LEN, scaler=scaler
    )
    print(f"  Total classifier train windows: {len(X_clf_all)}")
    clf_dist = Counter(y_clf_all)
    for k, v in sorted(clf_dist.items()):
        print(f"    {k:35s}: {v}")

    # Classifier test also uses cleaned data
    X_clf_test_seq, y_clf_test = make_windows(
        df_test_clean, FEATURES, SEQ_LEN, scaler=scaler
    )
    print(f"  Classifier test windows: {len(X_clf_test_seq)}")

    print("\n" + "=" * 60)
    print("STEP 7  Flattening windows for classifier")
    print("=" * 60)
    X_clf_train_flat = flatten_windows(X_clf_all)
    X_clf_test_flat  = flatten_windows(X_clf_test_seq)
    print(f"  Classifier train shape: {X_clf_train_flat.shape}")
    print(f"  Classifier test shape:  {X_clf_test_flat.shape}")

    print("\n" + "=" * 60)
    print("STEP 8  Saving processed arrays")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Autoencoder data
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
    print(f"  X_train     → {X_train.shape}  ({X_train.nbytes/1024:.1f} KB)")
    print(f"  X_val       → {X_val.shape}    ({X_val.nbytes/1024:.1f} KB)")

    # Evaluation data
    np.save(os.path.join(OUT_DIR, "X_test_all.npy"), X_test_all)
    np.save(os.path.join(OUT_DIR, "y_test_all.npy"), y_test_all)
    print(f"  X_test_all  → {X_test_all.shape}")
    print(f"  y_test_all  → {y_test_all.shape}")

    # Classifier data
    np.save(os.path.join(OUT_DIR, "X_clf_train.npy"), X_clf_train_flat)
    np.save(os.path.join(OUT_DIR, "y_clf_train.npy"), y_clf_all)
    np.save(os.path.join(OUT_DIR, "X_clf_test.npy"),  X_clf_test_flat)
    np.save(os.path.join(OUT_DIR, "y_clf_test.npy"),  y_clf_test)
    print(f"  X_clf_train → {X_clf_train_flat.shape}")
    print(f"  y_clf_train → {y_clf_all.shape}")
    print(f"  X_clf_test  → {X_clf_test_flat.shape}")
    print(f"  y_clf_test  → {y_clf_test.shape}")

    # Feature names
    feat_path = os.path.join(OUT_DIR, "feature_names.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(FEATURES))
    print(f"  Features    → {feat_path}")

    print("\n" + "=" * 60)
    print("✓  Preprocessing complete.")
    print("   Next:  python ml/train_lstm.py     (LSTM Autoencoder)")
    print("          python ml/train_classifier.py (Random Forest)")
    print("=" * 60)


if __name__ == "__main__":
    main()
