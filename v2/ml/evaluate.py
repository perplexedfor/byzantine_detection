"""
ml/evaluate.py
--------------
Evaluates the trained LSTM Autoencoder against REAL anomaly data
from the k3s labeled dataset (not synthetic).

Reports:
  - Normal data: false positive rate
  - Per-fault-type detection rates
  - Overall precision, recall, F1, confusion matrix
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

MODEL_PATH      = "ml/lstm_model.pth"
SCALER_PATH     = "ml/scaler.pkl"
THRESHOLD_PATH  = "ml/threshold.txt"
TEST_DATA_PATH  = "dataset/processed/X_test_all.npy"
TEST_LABEL_PATH = "dataset/processed/y_test_all.npy"
VAL_DATA_PATH   = "dataset/processed/X_val.npy"

# Feature weights (must match train_lstm.py)
FEATURE_WEIGHTS = torch.tensor([
    5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,   # cpu/mem boosted, net_drop 2x
    5.0, 5.0,                          # exec counts
    10.0, 10.0,                        # security
    20.0,                              # mining_port
])

# ── Model Definition (must match train_lstm.py) ──────────────────────────────
SPARSE_INDICES = [6, 7, 8, 9, 10, 11]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, sparse_indices=None):
        super().__init__()
        self.sparse_indices = sparse_indices
        self.encoder_lstm  = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer  = nn.Linear(hidden_dim, input_dim)

        if sparse_indices:
            n_sparse = len(sparse_indices)
            self.sparse_branch = nn.Sequential(
                nn.Linear(n_sparse, 4),
                nn.ReLU(),
                nn.Linear(4, n_sparse)
            )

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent  = self.hidden2latent(h_n[-1])
        h_dec   = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(h_dec)
        recon = self.output_layer(dec_out)

        if self.sparse_indices:
            sparse_in = x[:, :, self.sparse_indices]
            recon[:, :, self.sparse_indices] += self.sparse_branch(sparse_in)
        return recon
# class LSTMAutoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, latent_dim=16, sparse_indices=None):
#         super().__init__()
#         self.sparse_indices = sparse_indices
#         self.encoder_lstm  = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
#         self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.output_layer  = nn.Linear(hidden_dim, input_dim)
        
#         if sparse_indices:
#             n_sparse = len(sparse_indices)
#             self.sparse_branch = nn.Sequential(
#                 nn.Linear(n_sparse, 4),
#                 nn.ReLU(),
#                 nn.Linear(4, n_sparse)
#             )

#     def forward(self, x):
#         enc_out, _ = self.encoder_lstm(x)
#         context = enc_out.mean(dim=1)   # <-- Using Mean Pooling for comparison
#         latent  = self.hidden2latent(context)

#         h_dec   = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
#         dec_out, _ = self.decoder_lstm(h_dec)
#         recon = self.output_layer(dec_out)
        
#         if self.sparse_indices:
#             sparse_in = x[:, :, self.sparse_indices]
#             recon[:, :, self.sparse_indices] += self.sparse_branch(sparse_in)

#         return recon


def get_errors(model, X_np, weights=None):
    """Compute per-window reconstruction error (MAE)."""
    model.eval()
    with torch.no_grad():
        t     = torch.FloatTensor(X_np)
        recon = model(t)
        w = FEATURE_WEIGHTS.to(recon.device)
        sq_err = ((recon - t) ** 2) * w
        return torch.mean(sq_err, dim=(1, 2)).numpy()


def binary_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr, tp, fp, tn, fn


def main():
    # ── Check files exist ─────────────────────────────────────────────────────
    required = [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, TEST_DATA_PATH,
                TEST_LABEL_PATH, VAL_DATA_PATH]
    for p in required:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}")
            print("       Run preprocess.py then train_lstm.py first.")
            return

    # ── Load artifacts ────────────────────────────────────────────────────────
    threshold = float(open(THRESHOLD_PATH).read().strip())
    scaler    = joblib.load(SCALER_PATH)
    n_features = len(scaler.data_min_)

    model = LSTMAutoencoder(input_dim=n_features, latent_dim=8, sparse_indices=SPARSE_INDICES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"Model loaded  |  Features = {n_features}  |  Threshold = {threshold:.6f}\n")

    # ── 1. Normal validation data (from training split — should NOT be flagged)
    print("=" * 65)
    print("1. NORMAL VALIDATION DATA (should NOT be flagged)")
    print("=" * 65)
    X_val = np.load(VAL_DATA_PATH)
    errors_val = get_errors(model, X_val, FEATURE_WEIGHTS)
    pred_val   = (errors_val > threshold).astype(int)

    print(f"   Windows tested      : {len(X_val)}")
    print(f"   MAE  mean           : {errors_val.mean():.6f}")
    print(f"   MAE  std            : {errors_val.std():.6f}")
    print(f"   MAE  max            : {errors_val.max():.6f}")
    print(f"   Wrongly flagged     : {pred_val.sum()} / {len(X_val)}")
    print(f"   False Positive Rate : {pred_val.mean()*100:.2f}%")

    # ── 2. Test data (real anomalies from k3s runs) ───────────────────────────
    X_test = np.load(TEST_DATA_PATH)
    y_test = np.load(TEST_LABEL_PATH)
    errors_test = get_errors(model, X_test, FEATURE_WEIGHTS)
    pred_test   = (errors_test > threshold).astype(int)

    # Separate normal vs anomaly
    is_normal  = (y_test == "normal")
    is_anomaly = ~is_normal

    # Binary labels: 0 = normal, 1 = anomaly
    y_binary = is_anomaly.astype(int)

    # ── 2a. Normal test windows ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("2. NORMAL TEST DATA (unseen runs — should NOT be flagged)")
    print("=" * 65)
    n_norm = is_normal.sum()
    n_norm_flagged = pred_test[is_normal].sum()
    print(f"   Windows tested      : {n_norm}")
    print(f"   MAE  mean           : {errors_test[is_normal].mean():.6f}")
    print(f"   Wrongly flagged     : {n_norm_flagged} / {n_norm}")
    print(f"   False Positive Rate : {n_norm_flagged/n_norm*100:.2f}%" if n_norm > 0 else "   N/A")

    # ── 2b. Per-fault-type detection ──────────────────────────────────────────
    fault_types = sorted(set(y_test[is_anomaly]))

    print()
    print("=" * 65)
    print("3. PER-FAULT-TYPE DETECTION RATES (real anomalies)")
    print("=" * 65)
    print(f"   {'Fault Type':<35s} {'Detected':>8s} {'Total':>6s} {'Rate':>7s}  {'Mean MAE':>9s}")
    print("   " + "-" * 70)

    total_detected = 0
    total_anomaly = 0

    for fault in fault_types:
        mask = (y_test == fault)
        n_total = mask.sum()
        n_detected = pred_test[mask].sum()
        mean_err = errors_test[mask].mean()
        rate = n_detected / n_total * 100 if n_total > 0 else 0
        total_detected += n_detected
        total_anomaly += n_total
        print(f"   {fault:<35s} {n_detected:>8d} {n_total:>6d} {rate:>6.1f}%  {mean_err:>9.6f}")

    overall_rate = total_detected / total_anomaly * 100 if total_anomaly > 0 else 0
    print("   " + "-" * 70)
    print(f"   {'TOTAL':<35s} {total_detected:>8d} {total_anomaly:>6d} {overall_rate:>6.1f}%")

    # ── 4. Overall binary metrics ─────────────────────────────────────────────
    precision, recall, f1, fpr, tp, fp, tn, fn = binary_metrics(y_binary, pred_test)

    print()
    print("=" * 65)
    print("4. OVERALL BINARY METRICS (normal vs anomaly)")
    print("=" * 65)
    print(f"   TP (anomaly flagged correctly)  : {tp}")
    print(f"   FP (normal flagged wrongly)     : {fp}")
    print(f"   TN (normal passed correctly)    : {tn}")
    print(f"   FN (anomaly missed)             : {fn}")
    print()
    print(f"   Precision  : {precision*100:.1f}%   (of flagged, how many are real)")
    print(f"   Recall     : {recall*100:.1f}%   (of real anomalies, how many caught)")
    print(f"   F1 Score   : {f1*100:.1f}%   (harmonic mean of P and R)")
    print(f"   FP Rate    : {fpr*100:.2f}%  (normal windows wrongly flagged)")
    print()

    if f1 >= 0.85:
        print("   -> Excellent — model is production-ready")
    elif f1 >= 0.70:
        print("   -> Good but could improve — consider tuning threshold")
    else:
        print("   -> Needs improvement — check threshold or add more data")


if __name__ == "__main__":
    main()
