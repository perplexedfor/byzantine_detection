"""
ml/train_classifier.py
----------------------
Trains a Random Forest classifier for Byzantine fault detection
as a NON-deep-learning baseline comparison to the LSTM Autoencoder.

Uses the same preprocessed data from  ml/preprocess.py:
  - Sliding windows are flattened into tabular features (mean/std/min/max)
  - Same run-aware train/test split (no leakage)

Two modes:
  1. Binary:     normal vs anomaly
  2. Multi-class: normal vs each fault type

Run  ml/preprocess.py  FIRST to generate the classifier data files.

Saves:
    ml/rf_binary.pkl       â€” binary Random Forest model
    ml/rf_multiclass.pkl   â€” multi-class Random Forest model
"""
import numpy as np
import os
import joblib
import json
from collections import Counter

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROCESSED_DIR   = "dataset/processed"
BINARY_PATH     = "ml/rf_binary.pkl"
MULTI_PATH      = "ml/rf_multiclass.pkl"
RESULTS_PATH    = "ml/classifier_results.json"

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def classification_report(y_true, y_pred, labels=None):
    """Manual classification report (avoids sklearn.metrics import issues)."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    rows = []
    for label in labels:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        support = int(np.sum(y_true == label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

        rows.append({
            "label": label, "precision": precision, "recall": recall,
            "f1": f1, "support": support
        })

    return rows


def print_report(rows, title="Classification Report"):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"  {'Label':<35s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'Support':>8s}")
    print(f"  {'-'*68}")

    total_support = sum(r["support"] for r in rows)
    weighted_p = weighted_r = weighted_f1 = 0.0

    for r in rows:
        print(f"  {r['label']:<35s} {r['precision']:>6.1%} {r['recall']:>7.1%} "
              f"{r['f1']:>6.1%} {r['support']:>8d}")
        w = r["support"] / total_support if total_support > 0 else 0
        weighted_p  += r["precision"] * w
        weighted_r  += r["recall"] * w
        weighted_f1 += r["f1"] * w

    print(f"  {'-'*68}")
    print(f"  {'weighted avg':<35s} {weighted_p:>6.1%} {weighted_r:>7.1%} "
          f"{weighted_f1:>6.1%} {total_support:>8d}")

    accuracy = sum(r["support"] * r["recall"] for r in rows) / total_support if total_support > 0 else 0
    print(f"\n  Accuracy: {accuracy:.1%} ({int(accuracy*total_support)}/{total_support})")

    return accuracy, weighted_f1


def confusion_matrix_print(y_true, y_pred, labels):
    """Print a simple confusion matrix."""
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            mat[label_idx[t]][label_idx[p]] += 1

    # Print
    max_len = max(len(l) for l in labels)
    header = " " * (max_len + 2) + "  ".join(f"{l[:7]:>7s}" for l in labels)
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {header}")
    for i, label in enumerate(labels):
        row_str = "  ".join(f"{mat[i][j]:>7d}" for j in range(n))
        print(f"  {label:<{max_len+2}s}{row_str}")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    from sklearn.ensemble import RandomForestClassifier

    # 1. Load preprocessed data
    print("=" * 75)
    print("  RANDOM FOREST CLASSIFIER â€” Byzantine Fault Detection")
    print("=" * 75)

    files = {
        "X_train": os.path.join(PROCESSED_DIR, "X_clf_train.npy"),
        "y_train": os.path.join(PROCESSED_DIR, "y_clf_train.npy"),
        "X_test":  os.path.join(PROCESSED_DIR, "X_clf_test.npy"),
        "y_test":  os.path.join(PROCESSED_DIR, "y_clf_test.npy"),
    }

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"ERROR: Missing {path}")
            print("       Run  python ml/preprocess.py  first.")
            return

    X_train = np.load(files["X_train"])
    y_train = np.load(files["y_train"])
    X_test  = np.load(files["X_test"])
    y_test  = np.load(files["y_test"])

    print(f"\n  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"\n  Train label distribution:")
    for k, v in sorted(Counter(y_train).items()):
        print(f"    {k:<35s}: {v}")
    print(f"\n  Test label distribution:")
    for k, v in sorted(Counter(y_test).items()):
        print(f"    {k:<35s}: {v}")

    # Replace NaN/Inf (edge case from log1p or std of constant windows)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1.0, neginf=0.0)

    results = {}

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PART 1: BINARY CLASSIFICATION (normal vs anomaly)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print("\n" + "=" * 75)
    print("  PART 1: BINARY CLASSIFICATION (normal vs anomaly)")
    print("=" * 75)

    y_train_bin = np.where(y_train == "normal", "normal", "anomaly")
    y_test_bin  = np.where(y_test  == "normal", "normal", "anomaly")

    print(f"\n  Train: normal={np.sum(y_train_bin=='normal')}, "
          f"anomaly={np.sum(y_train_bin=='anomaly')}")
    print(f"  Test:  normal={np.sum(y_test_bin=='normal')}, "
          f"anomaly={np.sum(y_test_bin=='anomaly')}")

    rf_binary = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",    # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )

    print("\n  Training binary Random Forest (200 trees, balanced weights)...")
    rf_binary.fit(X_train, y_train_bin)

    y_pred_bin = rf_binary.predict(X_test)
    report_bin = classification_report(y_test_bin, y_pred_bin, ["normal", "anomaly"])
    acc_bin, f1_bin = print_report(report_bin, "Binary Classification Report")
    confusion_matrix_print(y_test_bin, y_pred_bin, ["normal", "anomaly"])

    os.makedirs(os.path.dirname(BINARY_PATH), exist_ok=True)
    joblib.dump(rf_binary, BINARY_PATH)
    print(f"\n  Saved binary model â†’ {BINARY_PATH}")

    results["binary"] = {"accuracy": float(acc_bin), "weighted_f1": float(f1_bin)}

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PART 2: MULTI-CLASS CLASSIFICATION (normal + each fault type)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print("\n" + "=" * 75)
    print("  PART 2: MULTI-CLASS CLASSIFICATION (fault-type identification)")
    print("=" * 75)

    all_labels = sorted(set(y_train) | set(y_test))
    print(f"\n  Classes ({len(all_labels)}): {all_labels}")

    rf_multi = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\n  Training multi-class Random Forest (300 trees, balanced weights)...")
    rf_multi.fit(X_train, y_train)

    y_pred_multi = rf_multi.predict(X_test)
    report_multi = classification_report(y_test, y_pred_multi, all_labels)
    acc_multi, f1_multi = print_report(report_multi, "Multi-Class Classification Report")
    confusion_matrix_print(y_test, y_pred_multi, all_labels)

    joblib.dump(rf_multi, MULTI_PATH)
    print(f"\n  Saved multi-class model â†’ {MULTI_PATH}")

    results["multiclass"] = {"accuracy": float(acc_multi), "weighted_f1": float(f1_multi)}

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PART 3: FEATURE IMPORTANCE
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print("\n" + "=" * 75)
    print("  PART 3: TOP FEATURE IMPORTANCES (binary model)")
    print("=" * 75)

    # Feature names: 11 features Ã— 4 stats = 44 features
    feature_names_path = os.path.join(PROCESSED_DIR, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            base_features = f.read().strip().split("\n")
    else:
        base_features = [f"feature_{i}" for i in range(11)]

    stat_names = ["mean", "std", "min", "max"]
    full_names = [f"{feat}_{stat}" for feat in base_features for stat in stat_names]

    importances = rf_binary.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<5s} {'Feature':<40s} {'Importance':>10s}")
    print(f"  {'-'*58}")
    for rank, idx in enumerate(indices[:15]):
        name = full_names[idx] if idx < len(full_names) else f"feature_{idx}"
        print(f"  {rank+1:<5d} {name:<40s} {importances[idx]:>10.4f}")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PART 4: COMPARISON SUMMARY
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print("\n" + "=" * 75)
    print("  COMPARISON: LSTM Autoencoder vs Random Forest")
    print("=" * 75)

    # Load LSTM results if available
    lstm_threshold_path = "ml/threshold.txt"
    lstm_test_data = os.path.join(PROCESSED_DIR, "X_test_all.npy")
    lstm_test_labels = os.path.join(PROCESSED_DIR, "y_test_all.npy")

    if all(os.path.exists(p) for p in [lstm_threshold_path, "ml/lstm_model.pth",
                                        lstm_test_data, lstm_test_labels]):
        try:
            import torch
            from train_lstm import LSTMAutoencoder, FEATURE_WEIGHTS

            threshold = float(open(lstm_threshold_path).read().strip())
            n_feat = len(base_features)
            model = LSTMAutoencoder(n_feat, 64, 8, sparse_indices=[6,7,8,9,10,11])
            model.load_state_dict(torch.load("ml/lstm_model.pth", weights_only=True))
            model.eval()

            X_t = np.load(lstm_test_data)
            y_t = np.load(lstm_test_labels)

            with torch.no_grad():
                t = torch.FloatTensor(X_t)
                recon = model(t)
                w = FEATURE_WEIGHTS.to(recon.device)
                sq_err = ((recon - t) ** 2) * w
                errors = torch.mean(sq_err, dim=(1, 2)).numpy()
            y_true_bin = np.where(y_t == "normal", 0, 1)
            y_pred_bin_lstm = (errors > threshold).astype(int)

            tp = np.sum((y_pred_bin_lstm == 1) & (y_true_bin == 1))
            fp = np.sum((y_pred_bin_lstm == 1) & (y_true_bin == 0))
            fn = np.sum((y_pred_bin_lstm == 0) & (y_true_bin == 1))
            tn = np.sum((y_pred_bin_lstm == 0) & (y_true_bin == 0))

            lstm_prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            lstm_rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
            lstm_f1   = 2*lstm_prec*lstm_rec/(lstm_prec+lstm_rec) if (lstm_prec+lstm_rec) > 0 else 0
            lstm_acc  = (tp+tn)/(tp+fp+tn+fn)

            print(f"\n  {'Metric':<25s} {'LSTM AE':>12s} {'Random Forest':>14s}")
            print(f"  {'-'*55}")
            print(f"  {'Accuracy':<25s} {lstm_acc:>12.1%} {acc_bin:>14.1%}")
            print(f"  {'Precision':<25s} {lstm_prec:>12.1%} "
                  f"{report_bin[1]['precision']:>14.1%}")
            print(f"  {'Recall':<25s} {lstm_rec:>12.1%} "
                  f"{report_bin[1]['recall']:>14.1%}")
            print(f"  {'F1 Score':<25s} {lstm_f1:>12.1%} "
                  f"{report_bin[1]['f1']:>14.1%}")
            print(f"  {'Multi-class F1':<25s} {'N/A':>12s} {f1_multi:>14.1%}")
            print(f"\n  Note: LSTM AE is unsupervised (anomaly detection by reconstruction error)")
            print(f"        RF is supervised (needs labeled training data)")

            results["lstm_ae"] = {
                "accuracy": float(lstm_acc), "precision": float(lstm_prec),
                "recall": float(lstm_rec), "f1": float(lstm_f1)
            }

        except Exception as e:
            print(f"\n  Could not load LSTM model for comparison: {e}")
            print(f"  Run  python ml/train_lstm.py  first, then re-run this script.")
    else:
        print(f"\n  LSTM model not found. Run train_lstm.py first for comparison.")
        print(f"  Showing Random Forest results only:")
        print(f"\n  {'Metric':<25s} {'Random Forest':>14s}")
        print(f"  {'-'*42}")
        print(f"  {'Binary Accuracy':<25s} {acc_bin:>14.1%}")
        print(f"  {'Binary F1':<25s} {f1_bin:>14.1%}")
        print(f"  {'Multi-class Accuracy':<25s} {acc_multi:>14.1%}")
        print(f"  {'Multi-class F1':<25s} {f1_multi:>14.1%}")

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved â†’ {RESULTS_PATH}")

    print("\n" + "=" * 75)
    print("  Done!")
    print("=" * 75)


if __name__ == "__main__":
    main()
