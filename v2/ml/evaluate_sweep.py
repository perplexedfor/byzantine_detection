"""
ml/evaluate_sweep.py
---------------------
Performs a threshold sweep for the LSTM Autoencoder using 4 different
reconstruction error scoring methods (MAE, MSE, Top-3, Max).

Reports:
  - Best F1 limit directly on Test Set (optimistic theoretical max)
  - Recall and F1 at fixed False Positive Rates (FPR = 0.5%, 1%, 2%) calculated
    from the purely normal Validation Set.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH      = "ml/lstm_model.pth"
TEST_DATA_PATH  = "dataset/processed/X_test_all.npy"
TEST_LABEL_PATH = "dataset/processed/y_test_all.npy"
VAL_DATA_PATH   = "dataset/processed/X_val.npy"
SCORES_OUT_PATH = "ml/test_scores.csv"

# Feature weights (must match train_lstm.py)
FEATURE_WEIGHTS = torch.tensor([
    5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,   # cpu/mem boosted, net_drop 2x
    5.0, 5.0,                            # exec counts
    10.0, 10.0,                          # security
    20.0,                                # mining_port
])

SPARSE_INDICES = [6, 7, 8, 9, 10, 11]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Definition ─────────────────────────────────────────────────────────
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

# ── Score Computation ────────────────────────────────────────────────────────
def get_scores(model, X_np, method="mae", weights=None):
    model.eval()
    results = []
    w = weights.to(device) if weights is not None else 1.0
    
    # Process in batches to avoid memory issues with larger datasets
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            t = torch.FloatTensor(X_np[i:i+batch_size]).to(device)
            recon = model(t)
            diff = recon - t
            
            # Application of the scoring choices:
            # 1. Aggregate error per signal (mean over time) -> shape (Batch, Features)
            if method == "mse":
                sq_diff = (diff ** 2) * w
                per_signal = torch.mean(sq_diff, dim=1)
                batch_scores = torch.mean(per_signal, dim=1).cpu().numpy()
            else:
                abs_diff = torch.abs(diff) * w
                per_signal = torch.mean(abs_diff, dim=1)
                
                if method == "mae":
                    # Mean over 12 features
                    batch_scores = torch.mean(per_signal, dim=1).cpu().numpy()
                elif method == "top3":
                    # Mean of largest 3 feature errors
                    top3_vals, _ = torch.topk(per_signal, 3, dim=1)
                    batch_scores = torch.mean(top3_vals, dim=1).cpu().numpy()
                elif method == "max":
                    # Max over 12 features
                    batch_scores = torch.max(per_signal, dim=1)[0].cpu().numpy()
            
            results.extend(batch_scores)
            
    return np.array(results)

def find_fpr_threshold(normal_scores, target_fpr):
    """
    Returns the threshold that ensures FPR <= target_fpr.
    Uses purely normal validation data.
    """
    sorted_scores = np.sort(normal_scores)[::-1]
    allowed_fps = int(len(sorted_scores) * target_fpr)
    
    if allowed_fps == 0:
        return sorted_scores[0] + 1e-6
    elif allowed_fps >= len(sorted_scores):
        return sorted_scores[-1] - 1e-6
    else:
        return sorted_scores[allowed_fps]

def evaluate_threshold(preds, y_true):
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return prec, rec, f1

# ── Main Evaluator ───────────────────────────────────────────────────────────
def main():
    required = [TEST_DATA_PATH, TEST_LABEL_PATH, VAL_DATA_PATH]
    for p in required:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}. Ensure prepocess.py has been executed.")
            return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file {MODEL_PATH} not found. Run train_lstm.py first.")
        return

    print("Loading datasets...")
    X_val = np.load(VAL_DATA_PATH)
    X_test = np.load(TEST_DATA_PATH)
    y_test_raw = np.load(TEST_LABEL_PATH)
    
    # Binary conversion
    is_anomaly = (y_test_raw != "normal")
    y_test_binary = is_anomaly.astype(int)
    
    n_features = X_val.shape[-1]
    
    print(f"Loaded: Val={X_val.shape[0]} windows, Test={X_test.shape[0]} windows")
    print(f"Test Class Balance: {sum(y_test_binary)} Anomalies, {len(y_test_binary)-sum(y_test_binary)} Normal")
    
    model = LSTMAutoencoder(input_dim=n_features, sparse_indices=SPARSE_INDICES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # Init storage for test_scores.csv 
    test_scores_df = pd.DataFrame({
        "y_true": y_test_binary,
        "fault_type": y_test_raw
    })
    
    methods = ["mae", "mse", "top3", "max"]
    results_records = []
    
    print("\nStarting Threshold Sweep (Evaluating scoring mechanisms)...")
    
    for method in methods:
        val_scores = get_scores(model, X_val, method=method, weights=FEATURE_WEIGHTS)
        test_scores = get_scores(model, X_test, method=method, weights=FEATURE_WEIGHTS)
        
        test_scores_df[f"score_{method}"] = test_scores
        
        # 1) BEST F1 THRESHOLD (Sweeping across test set to find bound limit)
        percentiles = np.linspace(0, 100, 500)
        thresholds = np.percentile(test_scores, percentiles)
        
        best_f1 = 0
        best_f1_th = 0
        
        for th in thresholds:
            preds = (test_scores > th).astype(int)
            f1 = f1_score(y_test_binary, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_th = th
                
        # 2) FIXED FPR THRESHOLDS (Calculated fairly on normal Validation Set)
        th_05 = find_fpr_threshold(val_scores, 0.005)
        th_10 = find_fpr_threshold(val_scores, 0.01)
        th_20 = find_fpr_threshold(val_scores, 0.02)
        
        # Determine metrics on test set at 1% FPR for main table
        preds_10 = (test_scores > th_10).astype(int)
        prec_10, rec_10, f1_10 = evaluate_threshold(preds_10, y_test_binary)
        
        # Recall at other FPRs
        preds_05 = (test_scores > th_05).astype(int)
        _, rec_05, _ = evaluate_threshold(preds_05, y_test_binary)
        
        preds_20 = (test_scores > th_20).astype(int)
        _, rec_20, _ = evaluate_threshold(preds_20, y_test_binary)
        
        # Store for display
        results_records.append({
            "Score Type": method.upper() if method != "top3" else "Top-3",
            "Best Threshold": f"{best_f1_th:.4f}",
            "Precision": f"{prec_10*100:.2f}%",     
            "Recall": f"{rec_10*100:.2f}%",         
            "F1": f"{f1_10*100:.2f}%",              
            "Best F1 (Test Bound)": f"{best_f1*100:.2f}%",
            "Recall @ 0.5% FPR": f"{rec_05*100:.2f}%",
            "Recall @ 1% FPR": f"{rec_10*100:.2f}%",
            "Recall @ 2% FPR": f"{rec_20*100:.2f}%"
        })

    print("\n" + "="*115)
    print("RESULTS REPORT (Metrics calculated at Validation FPR = 1% unless specified)")
    print("="*115)
    
    df_results = pd.DataFrame(results_records)
    # Ensure layout aligns symmetrically
    cols = ["Score Type", "Best Threshold", "Precision", "Recall", "F1", 
            "Best F1 (Test Bound)", "Recall @ 0.5% FPR", "Recall @ 1% FPR", "Recall @ 2% FPR"]
    print(df_results[cols].to_string(index=False))
    
    print("="*115)
    
    # Ensure output dir exists before saving
    os.makedirs(os.path.dirname(SCORES_OUT_PATH), exist_ok=True)
    test_scores_df.to_csv(SCORES_OUT_PATH, index=False)
    print(f"\nSaved per-window metrics to {SCORES_OUT_PATH}")


if __name__ == "__main__":
    main()
