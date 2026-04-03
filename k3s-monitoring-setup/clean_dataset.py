import pandas as pd
import os

# Ensure we run in the correct directory regardless of where it's called from
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset.csv")
CLEANED_FILE = os.path.join(SCRIPT_DIR, "final_labeled_dataset_cleaned.csv")

def main():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found. Ensure you run this after label_dataset.py")
        return

    print(f"Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Columns that represent Tetragon eBPF metrics
    ebpf_columns = [
        'exec_count', 
        'unique_process_count', 
        'tmp_exec_count', 
        'outbound_connect_count', 
        'mining_port_count'
    ]
    
    # Ensure all columns exist just in case
    ebpf_columns = [col for col in ebpf_columns if col in df.columns]

    print(f"Original dataset shape: {df.shape[0]} rows")
    print("Identifying dead eBPF sensors (0 total events for an entire run on a specific node)...")
    
    # 1. Group the data by run_id AND node
    # 2. Sum the eBPF metric columns for that specific combination
    # If the sum of all eBPF columns for a specific (run_id, node) is exactly 0.0, 
    # it means Tetragon failed to capture a single event for hours.
    grouped_sums = df.groupby(['run_id', 'node'])[ebpf_columns].sum() #vertical sum
    
    # Create a mask: True if the total sum across all those columns is 0
    dead_sensors = grouped_sums[grouped_sums.sum(axis=1) == 0].index #horizontal sum
    
    if len(dead_sensors) == 0:
        print("No dead eBPF sensors found! All nodes successfully reported eBPF events in every run.")
        df.to_csv(CLEANED_FILE, index=False)
        print(f"Copied dataset to {CLEANED_FILE}")
        return
        
    print(f"\nFound {len(dead_sensors)} dead sensor combinations to drop:")
    for run_id, node in dead_sensors:
        print(f"  - Run ID: {run_id} | Dead Node: {node}")
        
    # Drop the dead rows by setting the index to (run_id, node), dropping the dead ones, and resetting
    df_indexed = df.set_index(['run_id', 'node'])
    df_cleaned = df_indexed.drop(index=dead_sensors).reset_index()
    
    # Reorder columns to match original
    df_cleaned = df_cleaned[df.columns]
    
    dropped_count = len(df) - len(df_cleaned)
    print(f"\nDropped {dropped_count} useless rows where Tetragon was dead.")
    print(f"Cleaned dataset shape: {df_cleaned.shape[0]} rows")
    
    df_cleaned.to_csv(CLEANED_FILE, index=False)
    print(f"Cleaned dataset saved successfully to {CLEANED_FILE}")

if __name__ == "__main__":
    main()
