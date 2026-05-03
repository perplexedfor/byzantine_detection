"""
ml/evaluate_hybrid.py
---------------------
Simulates the hybrid operator (LSTM AE + RF) against the k3s labeled dataset.

Replays the time-series data through the trust scoring system and measures:
  - Per-node trust score trajectories
  - Time-to-detect for each fault injection
  - Cordon accuracy (would the operator have cordoned the right nodes?)
  - Overall detection metrics for the hybrid vs AE-only vs RF-only

Run AFTER:  preprocess.py, train_lstm.py, train_classifier.py

Usage:
    cd v2
    python ml/evaluate_hybrid.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import json
from collections import defaultdict

# â”€â”€ Paths (relative to v2/) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASET_PATH   = os.path.join("..", "k3s-monitoring-setup", "final_labeled_dataset.csv")
MODEL_PATH     = "ml/lstm_model.pth"
SCALER_PATH    = "ml/scaler.pkl"
THRESHOLD_PATH = "ml/threshold.txt"
RF_MODEL_PATH  = "ml/rf_binary.pkl"
RF_MULTI_PATH  = "ml/rf_multiclass.pkl"
RESULTS_PATH   = "ml/hybrid_evaluation_results.json"

# Feature config (must match preprocess.py)
FEATURES = [
    "avg_cpu", "avg_mem", "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out", "net_drop_rate",
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]
BYTE_COLS = ["net_bytes_in", "net_bytes_out",
             "net_internal_bytes_in", "net_internal_bytes_out"]
BYTE_IDX = [2, 3, 4, 5]
SEQ_LEN = 30
SPARSE_INDICES = [6, 7, 8, 9, 10, 11]

# Trust config (must match operator/main.py exactly)
# - RF handles infrastructure (low-confidence single-model — low penalty)
# - AE handles security/novelty (high-confidence novel signal — severe penalty)
TRUST_DECAY_RF   = -2.0    # RF flags only (main.py: TRUST_DECAY_RF_ONLY)
TRUST_DECAY_AE   = -15.0   # AE flags only (security/zero-day novelty)
TRUST_DECAY_BOTH = -25.0   # Both flag     (catastrophic)
TRUST_REWARD     = +4.0    # Both normal   (main.py: TRUST_REWARD_NORMAL)
TRUST_INITIAL    = 100.0
TRUST_CORDON     = 40.0    # Cordon  threshold (main.py: TRUST_CORDON_BELOW)
TRUST_UNCORDON   = 60.0    # Uncordon threshold (main.py: TRUST_UNCORDON_ABOVE)
GRACE_PERIOD_TICKS = 5     # First N inferences per group — no trust penalty

# AE feature weights (must match train_lstm.py)
AE_WEIGHTS = torch.tensor([
    5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,   # cpu/mem boosted
    20.0, 20.0,                          # exec counts (boosted)
    50.0, 10.0,                        # security (tmp_exec massively boosted)
    20.0                               # mining_port
])


# â”€â”€ Model Definition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, sparse_indices=None):
        super().__init__()
        self.sparse_indices = sparse_indices
        self.encoder_lstm  = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)
        context = enc_out.mean(dim=1)  # Mean Pooling
        latent = self.hidden2latent(context)
        h_dec = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(h_dec)
        recon = self.output_layer(dec_out)
        
        # sparse_branch removed
        return recon


def flatten_window(seq):
    """Flatten (SEQ_LEN, features) → (features*4,) using mean/std/min/max.
    Input must already be clipped to [0, 1] (MinMaxScaler training range)
    to prevent out-of-distribution std/min/max during condition transitions.
    """
    stats = []
    for i in range(seq.shape[1]):
        col = seq[:, i]
        stats.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
    return np.array(stats, dtype=np.float32)


# â”€â”€ Trust Simulator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TrustSimulator:
    """Simulates the hybrid trust system for a single (run, node) group.
    Mirrors HybridTrustTracker + grace period logic in operator/main.py.
    """
    def __init__(self, name):
        self.name = name
        self.score = TRUST_INITIAL
        self.inference_count = 0   # tracks grace period (mirrors NODE_INFERENCE_COUNT)
        self.history = []  # list of dicts per tick

    def update(self, timestamp, label, ae_flags, rf_flags, rf_label="normal"):
        self.inference_count += 1
        in_grace = self.inference_count <= GRACE_PERIOD_TICKS

        if in_grace:
            # Grace period: log but do NOT penalise trust (mirrors main.py)
            delta = 0.0
            reason = f"GRACE ({self.inference_count}/{GRACE_PERIOD_TICKS})"
        elif ae_flags and rf_flags:
            delta = TRUST_DECAY_BOTH
            reason = "BOTH"
        elif rf_flags:
            delta = TRUST_DECAY_RF
            reason = "RF_ONLY"
        elif ae_flags:
            delta = TRUST_DECAY_AE
            reason = "AE_ONLY"
        else:
            delta = TRUST_REWARD
            reason = "NORMAL"

        self.score = max(0.0, min(TRUST_INITIAL, self.score + delta))
        self.history.append({
            "ts": timestamp, "score": self.score, "label": label,
            "ae": ae_flags, "rf": rf_flags, "rf_label": rf_label,
            "in_grace": in_grace, "reason": reason,
            "cordoned": self.score < TRUST_CORDON,
        })
        return self.score, in_grace


# â”€â”€ Main Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    print("=" * 75)
    print("  HYBRID OPERATOR SIMULATION â€” Byzantine Fault Detection")
    print("=" * 75)

    # 1. Load all artifacts
    for path, name in [(MODEL_PATH, "LSTM"), (SCALER_PATH, "Scaler"),
                        (THRESHOLD_PATH, "Threshold"), (RF_MODEL_PATH, "RF Binary")]:
        if not os.path.exists(path):
            print(f"  ERROR: Missing {name} at {path}")
            return

    scaler = joblib.load(SCALER_PATH)
    threshold = float(open(THRESHOLD_PATH).read().strip())
    rf_model = joblib.load(RF_MODEL_PATH)

    ae_model = LSTMAutoencoder(len(FEATURES), 64, 8, sparse_indices=SPARSE_INDICES)
    ae_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    ae_model.eval()

    # Try loading multi-class RF too
    rf_multi = None
    if os.path.exists(RF_MULTI_PATH):
        rf_multi = joblib.load(RF_MULTI_PATH)

    print(f"\n  LSTM AE threshold : {threshold:.6f}")
    print(f"  RF binary model   : {type(rf_model).__name__}")
    print(f"  RF multi model    : {'loaded' if rf_multi else 'not found'}")

    # 2. Load dataset
    print(f"\n  Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # --- MATCH ONLINE SCOPE ---
    # The live operator (main.py) explicitly skips the control-plane node (k3s-wk1)
    # We remove it here so our offline metrics reflect the actual worker-only environment.
    df = df[df["node"] != "k3s-wk1"]

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES)

    # Apply log1p on byte columns (match preprocess.py)
    for col in BYTE_COLS:
        df[col] = np.log1p(df[col])

    print(f"  Rows: {len(df)}, Runs: {df['run_id'].nunique()}, "
          f"Nodes: {df['node'].nunique()}")

    # 3. Simulate per (run, node) group
    print(f"\n{'='*75}")
    print(f"  SIMULATING HYBRID OPERATOR")
    print(f"{'='*75}")

    # Metrics accumulators for 3 modes: hybrid, ae_only, rf_only
    modes = ["hybrid", "ae_only", "rf_only"]
    total_metrics = {m: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for m in modes}
    fault_detection = {m: defaultdict(lambda: {"detected": 0, "total": 0}) for m in modes}
    cordon_results = {"correct_cordons": 0, "false_cordons": 0,
                      "total_anomaly_windows": 0, "total_normal_windows": 0}
    # Detection latency: fault_type -> list of window-counts from fault start to first flag
    # Each entry is an int (ticks) or None (fault episode not detected at all)
    fault_latency = defaultdict(list)  # fault_type -> [ticks, ...]
    cordon_latency = defaultdict(list)

    groups = df.groupby(["run_id", "node"])
    n_groups = len(groups)
    processed = 0

    for (run_id, node), grp in groups:
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        vals = grp[FEATURES].values.astype(np.float32)
        labels = grp["label"].values

        n_windows = max(0, len(vals) - SEQ_LEN + 1)
        if n_windows == 0:
            continue

        trust_sim = TrustSimulator(f"{run_id[:8]}/{node}")

        # Per-group latency tracking state
        _lat_fault      = None   # active fault type being tracked
        _lat_start_tick = None   # window index when that fault first appeared
        _lat_detected   = False  # whether it has been flagged yet this episode
        _lat_cordoned   = False  # whether it has been cordoned yet this episode

        for i in range(n_windows):
            window = vals[i:i+SEQ_LEN]
            win_labels = labels[i:i+SEQ_LEN]
            ts = grp["timestamp"].iloc[i+SEQ_LEN-1]

            # Determine ground truth label for this window
            unique = set(win_labels)
            if len(unique) == 1 and "normal" in unique:
                true_label = "normal"
            else:
                anomalies = [l for l in unique if l != "normal"]
                true_label = anomalies[0] if anomalies else "normal"

            is_true_anomaly = (true_label != "normal")

            # Scale (mirrors preprocess_sequence in main.py)
            seq_scaled = scaler.transform(window)

            # --- LSTM AE inference ---
            seq_t = torch.FloatTensor(seq_scaled).unsqueeze(0)
            with torch.no_grad():
                recon = ae_model(seq_t)
                w = AE_WEIGHTS.to(recon.device)
                sq_err = ((recon - seq_t) ** 2) * w
                ae_loss = torch.mean(sq_err, dim=(1,2))[0].item()
            ae_flags = ae_loss > threshold

            # --- RF inference ---
            # Clip to [0,1] before flattening: MinMaxScaler was fit on normal training
            # data so live/fault values can exceed the range, producing std/min/max
            # the RF never trained on.  Mirrors the np.clip fix in main.py run_inference().
            seq_clipped = np.clip(seq_scaled, 0.0, 1.0)
            flat = flatten_window(seq_clipped).reshape(1, -1)
            flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
            rf_pred = rf_model.predict(flat)[0]
            rf_flags = (rf_pred != "normal")

            rf_label = "anomaly" if rf_flags else "normal"
            if rf_multi is not None:
                rf_label = rf_multi.predict(flat)[0]

            # --- Update trust (hybrid) — mirrors HybridTrustTracker.update() ---
            _, in_grace = trust_sim.update(ts, true_label, ae_flags, rf_flags, rf_label)

            # --- Accumulate metrics for all 3 modes (skip grace period ticks) ---
            if not in_grace:
                for mode in modes:
                    if mode == "hybrid":
                        flagged = ae_flags or rf_flags
                    elif mode == "ae_only":
                        flagged = ae_flags
                    else:  # rf_only
                        flagged = rf_flags

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

            # --- Detection latency tracking (hybrid only) ---
            hybrid_flagged = ae_flags or rf_flags
            if is_true_anomaly:
                if _lat_fault != true_label:
                    # A new fault episode started (or fault type switched)
                    # Close out the previous episode if it was never detected
                    if _lat_fault is not None:
                        if not _lat_detected: fault_latency[_lat_fault].append(None)
                        if not _lat_cordoned: cordon_latency[_lat_fault].append(None)
                    _lat_fault      = true_label
                    _lat_start_tick = i
                    _lat_detected   = False
                    _lat_cordoned   = False

                if not _lat_detected and hybrid_flagged:
                    latency_ticks = i - _lat_start_tick
                    fault_latency[_lat_fault].append(latency_ticks)
                    _lat_detected = True
                    
                if not _lat_cordoned and trust_sim.score < TRUST_CORDON:
                    cordon_ticks = i - _lat_start_tick
                    cordon_latency[_lat_fault].append(cordon_ticks)
                    _lat_cordoned = True
            else:
                # Returned to normal — close out any open fault episode
                if _lat_fault is not None:
                    if not _lat_detected: fault_latency[_lat_fault].append(None)
                    if not _lat_cordoned: cordon_latency[_lat_fault].append(None)
                _lat_fault      = None
                _lat_start_tick = None
                _lat_detected   = False
                _lat_cordoned   = False

        # Check if trust dropped below cordon threshold
        is_legitimately_cordoned = False
        
        for entry in trust_sim.history:
            # Update state machine for hysteresis (simulate K8s node state)
            if entry["score"] < TRUST_CORDON:
                if entry["label"] != "normal":
                    is_legitimately_cordoned = True
            elif entry["score"] > TRUST_UNCORDON:
                is_legitimately_cordoned = False

            if entry["label"] != "normal":
                cordon_results["total_anomaly_windows"] += 1
                if entry["cordoned"]:
                    cordon_results["correct_cordons"] += 1
            else:
                cordon_results["total_normal_windows"] += 1
                # To be a false cordon, the node must be cordoned AND not currently recovering from a legit anomaly
                if entry["cordoned"] and not is_legitimately_cordoned:
                    cordon_results["false_cordons"] += 1

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{n_groups} groups...")

    print(f"  Processed {processed}/{n_groups} groups. Done.\n")

    # 4. Print results
    def calc_metrics(m):
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        total = tp + fp + tn + fn
        acc  = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "fpr": fpr,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    print("=" * 75)
    print("  COMPARISON: Dual-Brain Hybrid vs Individual Models")
    print("=" * 75)
    print(f"\n  {'Metric':<20s} {'Security (AE)':>15s} {'Infra (RF)':>15s} {'Hybrid':>10s}")
    print(f"  {'-'*65}")

    results = {}
    for mode in modes:
        results[mode] = calc_metrics(total_metrics[mode])

    for metric_name, key in [("Accuracy", "acc"), ("Precision", "prec"),
                               ("Recall", "rec"), ("F1 Score", "f1"),
                               ("FP Rate", "fpr")]:
        ae_val = results["ae_only"][key]
        rf_val = results["rf_only"][key]
        hy_val = results["hybrid"][key]
        print(f"  {metric_name:<20s} {ae_val:>10.1%} {rf_val:>10.1%} {hy_val:>10.1%}")

    # Confusion matrix for hybrid
    h = results["hybrid"]
    print(f"\n  Hybrid Confusion Matrix:")
    print(f"    TP={h['tp']}  FP={h['fp']}")
    print(f"    FN={h['fn']}  TN={h['tn']}")

    # Per-fault detection for hybrid
    print(f"\n{'='*75}")
    print(f"  PER-FAULT DETECTION RATES (Dual-Brain Hybrid)")
    print(f"{'='*75}")
    print(f"  {'Fault':<35s} {'AE(Sec)':>8s} {'RF(Infra)':>10s} {'Hybrid':>8s} {'Total':>6s}")
    print(f"  {'-'*72}")

    all_faults = sorted(set(
        list(fault_detection["hybrid"].keys()) +
        list(fault_detection["ae_only"].keys()) +
        list(fault_detection["rf_only"].keys())
    ))

    for fault in all_faults:
        ae_d = fault_detection["ae_only"][fault]
        rf_d = fault_detection["rf_only"][fault]
        hy_d = fault_detection["hybrid"][fault]
        total = hy_d["total"]
        if total == 0:
            continue
        ae_r = ae_d["detected"] / total * 100
        rf_r = rf_d["detected"] / total * 100
        hy_r = hy_d["detected"] / total * 100
        print(f"  {fault:<35s} {ae_r:>5.1f}% {rf_r:>5.1f}% {hy_r:>7.1f}% {total:>6d}")

    # Trust-based cordon results
    print(f"\n{'='*75}")
    print(f"  TRUST-BASED CORDONING (trust < {TRUST_CORDON})")
    print(f"{'='*75}")
    c = cordon_results
    print(f"  Anomaly windows where node was cordoned : {c['correct_cordons']}/{c['total_anomaly_windows']}")
    if c['total_anomaly_windows'] > 0:
        print(f"  Cordon effectiveness                    : {c['correct_cordons']/c['total_anomaly_windows']*100:.1f}%")
    print(f"  Normal windows where node was cordoned  : {c['false_cordons']}/{c['total_normal_windows']}")
    if c['total_normal_windows'] > 0:
        print(f"  False cordon rate                       : {c['false_cordons']/c['total_normal_windows']*100:.2f}%")

    # Detection latency per fault type
    print(f"\n{'='*95}")
    print(f"  LATENCY: Detection (Model Flags) vs Remediation (Operator Cordons) [window-ticks × 10s = seconds]")
    print(f"{'='*95}")
    print(f"  {'Fault':<35s} | {'Episodes':>8s} | {'Detect(Mean)':>12s} | {'Cordon(Mean)':>12s} | {'Delta':>8s}")
    print(f"  {'-'*95}")
    latency_summary = {}
    for fault in sorted(fault_latency.keys()):
        episodes = fault_latency[fault]
        cordon_eps = cordon_latency[fault]
        
        detected = [t for t in episodes if t is not None]
        cordoned = [t for t in cordon_eps if t is not None]
        
        missed = len(episodes) - len(detected)
        
        detect_mean = np.mean(detected) * 10 if detected else None
        cordon_mean = np.mean(cordoned) * 10 if cordoned else None
        
        det_str = f"{detect_mean:.0f}s" if detect_mean is not None else "N/A"
        cor_str = f"{cordon_mean:.0f}s" if cordon_mean is not None else "N/A"
        
        if detect_mean is not None and cordon_mean is not None:
            delta_str = f"+{cordon_mean - detect_mean:.0f}s"
        else:
            delta_str = "N/A"
            
        print(f"  {fault:<35s} | {len(episodes):>8d} | {det_str:>12s} | {cor_str:>12s} | {delta_str:>8s}")
        
        latency_summary[fault] = {
            "episodes": len(episodes), "detected": len(detected), "missed": missed,
            "detect_mean_s": float(detect_mean) if detect_mean is not None else None,
            "cordon_mean_s": float(cordon_mean) if cordon_mean is not None else None,
        }

    # Save results
    save_results = {
        "comparison": {m: {k: float(v) for k, v in results[m].items()} for m in modes},
        "per_fault_hybrid": {
            f: {"detected": int(d["detected"]), "total": int(d["total"]),
                "rate": float(d["detected"]/d["total"]) if d["total"] > 0 else 0.0}
            for f, d in fault_detection["hybrid"].items()
        },
        "cordon_results": {k: int(v) for k, v in cordon_results.items()},
        "detection_latency": latency_summary,
        "config": {
            "trust_decay_rf_only": TRUST_DECAY_RF,
            "trust_decay_ae_only": TRUST_DECAY_AE,
            "trust_decay_both": TRUST_DECAY_BOTH,
            "trust_reward_normal": TRUST_REWARD,
            "trust_cordon_below": TRUST_CORDON,
            "trust_uncordon_above": TRUST_UNCORDON,
            "grace_period_ticks": GRACE_PERIOD_TICKS,
            "ae_threshold": threshold,
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved â†’ {RESULTS_PATH}")

    print(f"\n{'='*75}")
    print(f"  Done!")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
