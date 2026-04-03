import csv
import datetime
import os
import shutil

import glob

DATASET_DIR = "../dataset"
OUTPUT_FILE = "final_labeled_dataset.csv"

# We now handle transitions explicitly in scenario_runner, so padding is 0 to avoid drift
START_PADDING_SEC = 0
END_PADDING_SEC = 0

# Using pure epoch timestamps now instead of isoformats

def process_run(run_id, metrics_file, labels_file, writer):
    print(f"\n--- Processing Run ID: {run_id} ---")
    
    # 1. Load the scenario time windows into memory for this specific run
    scenarios = []
    fault_order_hash = ""
    intensity_seed = ""
    with open(labels_file, mode='r') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            if not fault_order_hash and 'fault_order_hash' in row:
                fault_order_hash = row['fault_order_hash']
                intensity_seed = row['intensity_seed']
                
            original_start = int(row['start_time'])
            original_end = int(row['end_time'])
            
            # Apply padding to effectively shift the anomaly window later
            padded_start = original_start + START_PADDING_SEC
            padded_end = original_end + END_PADDING_SEC
            
            scenarios.append({
                "start": padded_start,
                "end": padded_end,
                "label": row['label'],
                "target_node": row.get('target_node', 'all')
            })
            
    print(f"Loaded {len(scenarios)} labeling time windows.")
    
    labeled_count = 0
    total_count = 0
    
    # 2. Iterate through the collected metrics and label them
    with open(metrics_file, mode='r') as mf:
        reader = csv.DictReader(mf)
        for row in reader:
            total_count += 1
            
            try:
                row_epoch = int(row['timestamp'])
            except ValueError:
                row_epoch = 0
            
            # Default label
            new_label = "normal"
            
            # Check against all scenario time windows
            for s in scenarios:
                if s['start'] <= row_epoch <= s['end']:
                    if s['target_node'] == 'all' or row.get('node') == s['target_node']:
                        new_label = s['label']
                    break
                    
            # Skip transition rows entirely to keep dataset clean
            if new_label == "transition":
                continue
                
            row['label'] = new_label
            if new_label != "normal":
                labeled_count += 1
                
            # Embed metadata details
            row['run_id'] = run_id
            row['fault_order_hash'] = fault_order_hash
            row['intensity_seed'] = intensity_seed
            
            writer.writerow(row)
            
    return total_count, labeled_count

def main():
    print(f"Scanning {DATASET_DIR} for execution runs...")
    
    label_files = glob.glob(os.path.join(DATASET_DIR, "scenario_labels_*.csv"))
    if not label_files:
        print("No run datasets found!")
        return
        
    runs = []
    for lf in label_files:
        # Extract run_id from filename scenario_labels_{run_id}.csv
        run_id = os.path.basename(lf).replace("scenario_labels_", "").replace(".csv", "")
        mf = os.path.join(DATASET_DIR, f"node_metrics_{run_id}.csv")
        
        if os.path.exists(mf):
            runs.append((run_id, mf, lf))
        else:
            print(f"Warning: Missing metrics file for run {run_id}. Skipping.")
            
    if not runs:
        print("No complete matching pairs found.")
        return
        
    print(f"Found {len(runs)} complete runs to merge.")
    
    grand_total_count = 0
    grand_labeled_count = 0
    
    # Open the final output file once
    with open(OUTPUT_FILE, mode='w', newline='') as of:
        # We need to peek at the first metrics file to get the headers
        with open(runs[0][1], 'r') as peek:
            reader = csv.DictReader(peek)
            fieldnames = list(reader.fieldnames)
            if 'label' not in fieldnames:
                fieldnames.insert(1, 'label')
            # Add the metadata columns
            for col in ['run_id', 'fault_order_hash', 'intensity_seed',
                        'net_internal_bytes_in', 'net_internal_bytes_out',
                        'last_successful_scrape_age_sec']:
                if col not in fieldnames:
                    fieldnames.append(col)
                    
            writer = csv.DictWriter(of, fieldnames=fieldnames)
            writer.writeheader()
            
            for run_id, metrics_file, labels_file in runs:
                t_count, l_count = process_run(run_id, metrics_file, labels_file, writer)
                grand_total_count += t_count
                grand_labeled_count += l_count
                
    print(f"\n=================================================")
    print(f"Aggregated Labeling Complete! Wrote combined metrics -> {OUTPUT_FILE}")
    print(f"Total Rows: {grand_total_count}")
    print(f"Anomalous (Non-Normal) Rows: {grand_labeled_count}")
    print(f"=================================================")

if __name__ == '__main__':
    main()
