import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR) # infra-proj
INPUT_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_aligned.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_normalized.csv")
MODEL_DIR = os.path.join(PROJ_DIR, "model")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.json")

numeric_cols = [
    'avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out',
    'net_internal_bytes_in', 'net_internal_bytes_out', 
    'exec_count', 'unique_process_count', 'tmp_exec_count',
    'outbound_connect_count', 'mining_port_count'
]

byte_cols = ["net_bytes_in", "net_bytes_out", 
             "net_internal_bytes_in", "net_internal_bytes_out"]

def normalize():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    df[byte_cols] = np.log1p(df[byte_cols])  
    
    unique_runs = df['run_id'].unique()
    print(f"Total unique runs found: {len(unique_runs)}")
    
    # Run-aware split: 80% train, 20% test
    # We use a fixed random state for reproducibility
    #train test split code unique splits test size random state
    train_runs, test_runs = train_test_split(unique_runs, test_size=0.2, random_state=42)
    print(f"Training runs: {len(train_runs)}")
    print(f"Testing runs: {len(test_runs)}")
    
    # Mark the runs in the dataset so Phase 3/4 knows the split
    df['split'] = 'test'
    df.loc[df['run_id'].isin(train_runs), 'split'] = 'train'
    
    # Identify normal data in training runs for fitting the scaler
    train_normal_mask = (df['split'] == 'train') & (df['label'] == 'normal')
    train_normal_df = df[train_normal_mask]
    
    print(f"Fitting scaler on {len(train_normal_df)} normal rows from {len(train_runs)} training runs...")
    
    scaler = MinMaxScaler()
    scaler.fit(train_normal_df[numeric_cols])
    
    # Save scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    
    # Save feature order
    with open(FEATURE_ORDER_PATH, 'w') as f:
        json.dump(numeric_cols, f)

    print(f"Saved feature order to {FEATURE_ORDER_PATH}")
    
    # Transform the entire dataset (train and test)
    print(f"Applying normalization to all {len(df)} rows...")
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Save the normalized dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved normalized dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    normalize()
