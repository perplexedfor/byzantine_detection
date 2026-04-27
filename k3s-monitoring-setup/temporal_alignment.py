import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_cleaned.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_aligned.csv")

# 10 second scrape interval requirement
EXPECTED_INTERVAL = 10
MAX_GAP_FILL = 3  # Forward fill up to 30 seconds of missing data

numeric_cols = [
    'avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out',
    'net_internal_bytes_in', 'net_internal_bytes_out', 
    'net_drop_rate',
    'exec_count', 'unique_process_count', 'tmp_exec_count',
    'outbound_connect_count', 'mining_port_count'
]

categorical_cols = ['label', 'syscall_feature_vector', 'fault_order_hash', 'intensity_seed']

def align_and_clean():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Initial shape: {df.shape}")

    # Ensure timestamp is integer
    df['timestamp'] = df['timestamp'].astype(float).astype(int)

    # Some nodes might have jitter. Round timestamp to nearest 10 to align.
    df['rounded_timestamp'] = (df['timestamp'] / 10).round() * 10

    # Process group by run_id and node
    aligned_dfs = []
    
    grouped = df.groupby(['run_id', 'node'])
    total_corrupted_dropped = 0

    for (run_id, node), group in grouped:
        group = group.sort_values('rounded_timestamp').drop_duplicates('rounded_timestamp')
        
        if len(group) < 2:
            continue
            
        min_ts = group['rounded_timestamp'].min()
        max_ts = group['rounded_timestamp'].max()
        
        # Create expected regular index
        expected_index = np.arange(min_ts, max_ts + EXPECTED_INTERVAL, EXPECTED_INTERVAL)
        
        # Reindex
        group = group.set_index('rounded_timestamp')
        group_aligned = group.reindex(expected_index)
        
        # Forward fill up to MAX_GAP_FILL
        # For numeric metrics
        for col in numeric_cols:
            if col in group_aligned.columns:
                group_aligned[col] = group_aligned[col].ffill(limit=MAX_GAP_FILL)
        
        # For categorical properties like label, run_id, node
        for col in categorical_cols:
            if col in group_aligned.columns:
                group_aligned[col] = group_aligned[col].ffill(limit=MAX_GAP_FILL).bfill(limit=MAX_GAP_FILL)
        
        # If last_successful_scrape_age_sec is > 30, it might signify corrupt data but we handle drops based on NaNs in cpu
        if 'last_successful_scrape_age_sec' in group_aligned.columns:
            group_aligned['last_successful_scrape_age_sec'] = group_aligned['last_successful_scrape_age_sec'].ffill(limit=MAX_GAP_FILL)
            
        # Identify non-filled remaining NaNs (corrupted intervals) 
        # that are larger than MAX_GAP_FILL
        before_drop = len(group_aligned)
        group_aligned = group_aligned.dropna(subset=['avg_cpu']) # If cpu is NaN, gap was too large
        after_drop = len(group_aligned)
        
        total_corrupted_dropped += (before_drop - after_drop)
        
        group_aligned['timestamp'] = group_aligned.index.astype(int)
        aligned_dfs.append(group_aligned.reset_index(drop=True))

    print(f"Dropped {total_corrupted_dropped} rows due to large gaps (corrupted intervals).")
    
    final_df = pd.concat(aligned_dfs, ignore_index=True)
    
    # Restore original column order but drop rounded_timestamp (now index and timestamp handled)
    orig_cols = [c for c in df.columns if c != 'rounded_timestamp' and c in final_df.columns]
    final_df = final_df[orig_cols]
    
    print(f"Final aligned dataset shape: {final_df.shape}")
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved aligned dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    align_and_clean()
