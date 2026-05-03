import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RF_MODEL_PATH = "ml/rf_binary.pkl"
OUTPUT_IMAGE  = "ml/feature_importance.png"

FEATURES = [
    "avg_cpu", "avg_mem", "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out", "net_drop_rate",
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]
STATS = ["mean", "std", "min", "max"]

# 1. Load the model
if not os.path.exists(RF_MODEL_PATH):
    # Fallback to absolute path if running from subfolder
    RF_MODEL_PATH = os.path.join("v2", RF_MODEL_PATH)

rf = joblib.load(RF_MODEL_PATH)

# 2. Map indices to names (e.g., 0 -> avg_cpu_mean)
feat_names = []
for f in FEATURES:
    for s in STATS:
        feat_names.append(f"{f}_{s}")

# 3. Get top 10
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

top_names = [feat_names[i] for i in indices]
top_scores = [importances[i] for i in indices]

# 4. Plot
plt.figure(figsize=(10, 6))
plt.barh(top_names[::-1], top_scores[::-1], color='skyblue', edgecolor='navy')
plt.xlabel('Importance Score')
plt.title('Top 10 Features for Byzantine Fault Detection (Random Forest)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 5. Save and Show
plt.savefig(OUTPUT_IMAGE)
print(f"✅ Feature importance chart saved to: {OUTPUT_IMAGE}")
print("\nTop 10 Features:")
for name, score in zip(top_names, top_scores):
    print(f"  {name:30s}: {score:.4f}")
