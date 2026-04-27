import pandas as pd
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_normalized.csv")
OUT_DIR = os.path.join(PROJ_DIR, "dataset", "processed")
MODEL_DIR = os.path.join(PROJ_DIR, "model")

WINDOW_SIZE = 30
# Design constraint: Features < 15
numeric_cols = [
    'avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out',
    'net_internal_bytes_in', 'net_internal_bytes_out', 
    'net_drop_rate',
    'exec_count', 'unique_process_count', 'tmp_exec_count',
    'outbound_connect_count', 'mining_port_count'
]
# We have 12 features

def create_sequences():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save window_size.json
    with open(os.path.join(MODEL_DIR, "window_size.json"), 'w') as f:
        json.dump(WINDOW_SIZE, f)
    print(f"Saved window_size.json to {MODEL_DIR}")

    X_train, y_train_meta = [], []
    X_test, y_test_meta = [], []

    grouped = df.groupby(['run_id', 'node'])
    
    for (run_id, node), group in grouped:
        group = group.sort_values('timestamp').reset_index(drop=True)
        split_val = group['split'].iloc[0]
        
        vals = group[numeric_cols].values.astype(np.float32)
        labels = group['label'].values
        # fault_type is basically the label if it's anomalous, else 'normal'
        
        # We need at least WINDOW_SIZE elements to form 1 window
        n_win = max(0, len(vals) - WINDOW_SIZE + 1)
        
        for i in range(n_win):
            win_X = vals[i : i + WINDOW_SIZE]
            
            # Extract all labels for the current window
            win_labels = labels[i : i + WINDOW_SIZE]
            
            # Strict Normal Logic:
            # If all timesteps are 'normal', the window is 'normal'.
            # If any timestep is anomalous, the window takes that anomaly's label.
            unique_labels = np.unique(win_labels)
            
            if len(unique_labels) == 1 and unique_labels[0] == 'normal':
                win_label = 'normal'
            else:
                # Filter out 'normal' to find what the anomaly is
                anomalies = [l for l in unique_labels if l != 'normal']
                win_label = anomalies[0] if anomalies else 'normal'
                
            fault_type = win_label if win_label != 'normal' else 'none'
            
            meta = [run_id, win_label, fault_type]
            
            if split_val == 'train':
                X_train.append(win_X)
                y_train_meta.append(meta)
            else:
                X_test.append(win_X)
                y_test_meta.append(meta)
                
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    # string arrays for meta
    y_train_meta = np.array(y_train_meta, dtype=str)
    y_test_meta = np.array(y_test_meta, dtype=str)
    
    # Save arrays
    train_x_path = os.path.join(OUT_DIR, "X_train.npy")
    train_y_path = os.path.join(OUT_DIR, "y_train_meta.npy")
    test_x_path = os.path.join(OUT_DIR, "X_test.npy")
    test_y_path = os.path.join(OUT_DIR, "y_test_meta.npy")
    
    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train_meta)
    np.save(test_x_path, X_test)
    np.save(test_y_path, y_test_meta)
    
    print(f"Created Train sequences: {X_train.shape[0]} shape={X_train.shape}")
    print(f"Created Test sequences: {X_test.shape[0]} shape={X_test.shape}")
    print(f"Saved processed sequences to {OUT_DIR}")

if __name__ == "__main__":
    create_sequences()
