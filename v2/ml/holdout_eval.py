"""
ml/holdout_eval.py
------------------
AE-2 Fault-Type Holdout Evaluation.

Retrains the Random Forest classifier IN-MEMORY with one fault type
('security_suspicious_network') excluded from training labels.
Then evaluates detection rates for:
  - AE only    (unchanged -- AE never trained on labeled faults)
  - RF holdout (retrained without the held-out fault)
  - Hybrid     (AE OR RF-holdout)

This demonstrates that:
  1. RF drops to near-0% recall on the held-out fault  (supervised model's weakness)
  2. AE maintains its recall on the same fault          (unsupervised generalization)
  3. Hybrid stays effective via the AE brain            (why the dual-brain design matters)

Run AFTER: preprocess.py, train_lstm.py  (RF is retrained here in-memory)
Usage:
    cd v2
    python ml/holdout_eval.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

# ── Paths (same as evaluate_hybrid.py) ──────────────────────────────────────
DATASET_PATH   = os.path.join("..", "k3s-monitoring-setup", "final_labeled_dataset.csv")
MODEL_PATH     = "ml/lstm_model.pth"
SCALER_PATH    = "ml/scaler.pkl"
THRESHOLD_PATH = "ml/threshold.txt"
CLF_TRAIN_X    = "dataset/processed/X_clf_train.npy"  # flat RF features
CLF_TRAIN_Y    = "dataset/processed/y_clf_train.npy"  # labels
RESULTS_PATH   = "ml/holdout_eval_results.json"

HOLDOUT_FAULT  = "security_suspicious_network"   # fault EXCLUDED from RF training

# ── Feature config (must match preprocess.py) ────────────────────────────────
FEATURES = [
    "avg_cpu", "avg_mem", "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out", "net_drop_rate",
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]
BYTE_COLS = ["net_bytes_in", "net_bytes_out",
             "net_internal_bytes_in", "net_internal_bytes_out"]
SEQ_LEN        = 30
SPARSE_INDICES = [6, 7, 8, 9, 10, 11]

AE_WEIGHTS = torch.tensor([
    5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    20.0, 20.0, 50.0, 10.0, 20.0
])

# ── LSTM AE definition (must match train_lstm.py) ────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, sparse_indices=None):
        super().__init__()
        self.sparse_indices = sparse_indices
        self.encoder_lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent  = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden  = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm   = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer   = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)
        context = enc_out.mean(dim=1)  # Mean Pooling
        latent  = self.hidden2latent(context)
        h_dec   = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(h_dec)
        recon   = self.output_layer(dec_out)
        
        # sparse_branch removed
        return recon


def flatten_window(seq):
    """(SEQ_LEN, F) → (F*4,) mean/std/min/max, clipped to [0,1] first."""
    seq = np.clip(seq, 0.0, 1.0)
    stats = []
    for i in range(seq.shape[1]):
        col = seq[:, i]
        stats.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
    return np.array(stats, dtype=np.float32)


def main():
    print("=" * 75)
    print("  AE-2 FAULT-TYPE HOLDOUT EVALUATION")
    print(f"  Held-out fault (excluded from RF training): {HOLDOUT_FAULT}")
    print("=" * 75)

    # 1. Check paths
    for path, name in [(MODEL_PATH, "LSTM"), (SCALER_PATH, "Scaler"),
                       (THRESHOLD_PATH, "Threshold"),
                       (CLF_TRAIN_X, "RF train X"), (CLF_TRAIN_Y, "RF train y")]:
        if not os.path.exists(path):
            print(f"  ERROR: Missing {name} at {path}")
            return

    # 2. Load LSTM AE artifacts
    scaler    = joblib.load(SCALER_PATH)
    threshold = float(open(THRESHOLD_PATH).read().strip())
    ae_model  = LSTMAutoencoder(len(FEATURES), 64, 8, sparse_indices=SPARSE_INDICES)
    ae_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    ae_model.eval()
    print(f"\n  LSTM AE threshold : {threshold:.6f}")

    # 3. Retrain RF WITHOUT the holdout fault
    X_clf = np.load(CLF_TRAIN_X)
    y_clf = np.load(CLF_TRAIN_Y)

    # Exclude windows labelled as the holdout fault
    keep = y_clf != HOLDOUT_FAULT
    X_train_hold = X_clf[keep]
    y_train_hold = y_clf[keep]

    n_removed = int(np.sum(~keep))
    print(f"\n  RF training data:")
    print(f"    Original  : {len(y_clf)} windows")
    print(f"    Removed   : {n_removed} '{HOLDOUT_FAULT}' windows")
    print(f"    Remaining : {len(y_train_hold)} windows")

    # Check if enough samples to train
    from collections import Counter
    label_counts = Counter(y_train_hold)
    print(f"    Label dist: {dict(label_counts)}")

    print(f"\n  Training RF holdout model (200 trees) ...")
    rf_holdout = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_holdout.fit(X_train_hold, y_train_hold)
    print("  RF holdout trained.")

    # 4. Load full dataset for evaluation
    print(f"\n  Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df = df[df["node"] != "k3s-wk1"]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES)
    for col in BYTE_COLS:
        df[col] = np.log1p(df[col])
    print(f"  Rows: {len(df)}, Runs: {df['run_id'].nunique()}, Nodes: {df['node'].nunique()}")

    # 5. Simulate per (run, node) group
    print(f"\n{'='*75}")
    print(f"  SIMULATING: AE-only vs RF-Holdout vs Hybrid-Holdout")
    print(f"{'='*75}")

    modes = ["ae_only", "rf_holdout", "hybrid_holdout"]
    total_metrics    = {m: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for m in modes}
    fault_detection  = {m: defaultdict(lambda: {"detected": 0, "total": 0}) for m in modes}

    groups    = df.groupby(["run_id", "node"])
    n_groups  = len(groups)
    processed = 0

    for (run_id, node), grp in groups:
        grp    = grp.sort_values("timestamp").reset_index(drop=True)
        vals   = grp[FEATURES].values.astype(np.float32)
        labels = grp["label"].values

        n_windows = max(0, len(vals) - SEQ_LEN + 1)
        if n_windows == 0:
            continue

        for i in range(n_windows):
            window     = vals[i:i+SEQ_LEN]
            win_labels = labels[i:i+SEQ_LEN]

            unique = set(win_labels)
            if len(unique) == 1 and "normal" in unique:
                true_label = "normal"
            else:
                anomalies  = [l for l in unique if l != "normal"]
                true_label = anomalies[0] if anomalies else "normal"

            # Skip transition windows
            if true_label == "transition":
                continue

            is_true_anomaly = (true_label != "normal")
            seq_scaled      = scaler.transform(window)

            # AE inference
            seq_t = torch.FloatTensor(seq_scaled).unsqueeze(0)
            with torch.no_grad():
                recon  = ae_model(seq_t)
                w      = AE_WEIGHTS.to(recon.device)
                sq_err = ((recon - seq_t) ** 2) * w
                ae_loss = torch.mean(sq_err, dim=(1, 2))[0].item()
            ae_flags = ae_loss > threshold

            # RF holdout inference
            flat     = flatten_window(seq_scaled).reshape(1, -1)
            flat     = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
            rf_pred  = rf_holdout.predict(flat)[0]
            rf_flags = (rf_pred != "normal")

            predictions = {
                "ae_only":       ae_flags,
                "rf_holdout":    rf_flags,
                "hybrid_holdout": ae_flags or rf_flags,
            }

            for mode in modes:
                flagged = predictions[mode]
                m = total_metrics[mode]
                if is_true_anomaly and flagged:
                    m["tp"] += 1
                elif is_true_anomaly and not flagged:
                    m["fn"] += 1
                elif not is_true_anomaly and flagged:
                    m["fp"] += 1
                else:
                    m["tn"] += 1

                if is_true_anomaly:
                    fault_detection[mode][true_label]["total"] += 1
                    if flagged:
                        fault_detection[mode][true_label]["detected"] += 1

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{n_groups} groups...")

    print(f"  Processed {processed}/{n_groups} groups. Done.\n")

    # 6. Print results
    def calc_metrics(m):
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        total = tp + fp + tn + fn
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {"prec": prec, "rec": rec, "f1": f1, "fpr": fpr,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    results = {m: calc_metrics(total_metrics[m]) for m in modes}

    print("=" * 75)
    print("  OVERALL COMPARISON  (RF trained WITHOUT security_suspicious_network)")
    print("=" * 75)
    print(f"\n  {'Metric':<20s} {'AE Only':>15s} {'RF Holdout':>12s} {'Hybrid':>10s}")
    print(f"  {'-'*60}")
    for metric_name, key in [("Precision", "prec"), ("Recall", "rec"),
                              ("F1 Score", "f1"), ("FP Rate", "fpr")]:
        print(f"  {metric_name:<20s} {results['ae_only'][key]:>10.1%}"
              f" {results['rf_holdout'][key]:>12.1%}"
              f" {results['hybrid_holdout'][key]:>10.1%}")

    # Per-fault detection focused on the holdout fault
    print(f"\n{'='*75}")
    print(f"  PER-FAULT DETECTION  (key: does the held-out fault get caught?)")
    print(f"{'='*75}")
    print(f"  {'Fault':<35s} {'AE Only':>9s} {'RF(hold)':>10s} {'Hybrid':>8s} {'Windows':>8s}")
    print(f"  {'-'*72}")

    all_faults = sorted(set(
        list(fault_detection["ae_only"].keys()) +
        list(fault_detection["rf_holdout"].keys())
    ))
    for fault in all_faults:
        ae_d = fault_detection["ae_only"][fault]
        rf_d = fault_detection["rf_holdout"][fault]
        hy_d = fault_detection["hybrid_holdout"][fault]
        total = hy_d["total"]
        if total == 0:
            continue
        ae_r = ae_d["detected"] / total * 100
        rf_r = rf_d["detected"] / total * 100
        hy_r = hy_d["detected"] / total * 100
        marker = "  ← HELD-OUT" if fault == HOLDOUT_FAULT else ""
        print(f"  {fault:<35s} {ae_r:>5.1f}% {rf_r:>5.1f}% {hy_r:>7.1f}%  {total:>6d}{marker}")

    print(f"\n  KEY FINDING:")
    ho_ae  = fault_detection["ae_only"][HOLDOUT_FAULT]
    ho_rf  = fault_detection["rf_holdout"][HOLDOUT_FAULT]
    ho_hy  = fault_detection["hybrid_holdout"][HOLDOUT_FAULT]
    tot    = ho_hy["total"]
    if tot > 0:
        print(f"    AE detects {HOLDOUT_FAULT}: "
              f"{ho_ae['detected']}/{tot} = {ho_ae['detected']/tot*100:.1f}%")
        print(f"    RF detects {HOLDOUT_FAULT}: "
              f"{ho_rf['detected']}/{tot} = {ho_rf['detected']/tot*100:.1f}%  "
              f"(RF never saw this fault in training)")
        print(f"    Hybrid:    {HOLDOUT_FAULT}: "
              f"{ho_hy['detected']}/{tot} = {ho_hy['detected']/tot*100:.1f}%  "
              f"(AE compensates for RF's blind spot)")

    # Save
    save_results = {
        "holdout_fault": HOLDOUT_FAULT,
        "comparison": {m: {k: float(v) for k, v in results[m].items()} for m in modes},
        "per_fault": {
            fault: {
                mode: {
                    "detected": fault_detection[mode][fault]["detected"],
                    "total":    fault_detection[mode][fault]["total"],
                    "rate": (fault_detection[mode][fault]["detected"] /
                             fault_detection[mode][fault]["total"]
                             if fault_detection[mode][fault]["total"] > 0 else 0)
                }
                for mode in modes
            }
            for fault in all_faults
            if fault_detection["hybrid_holdout"][fault]["total"] > 0
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")
    print(f"\n{'='*75}")
    print(f"  Done!")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
