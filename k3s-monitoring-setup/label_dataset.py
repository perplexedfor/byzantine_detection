import csv
import datetime
import os
import shutil

METRICS_FILE = "node_metrics.csv"
LABELS_FILE = "../workloads/scenario_labels.csv"
OUTPUT_FILE = "final_labeled_dataset.csv"

# Time adjustment padding in seconds
# We pad the start time to account for delayed effects (e.g. memory leak takes time to manifest)
# We pad the end time to account for cluster recovery time
START_PADDING_SEC = 10
END_PADDING_SEC = 15

def parse_isoformat(timestamp_str):
    """Parse ISO8601 timestamp string back to a datetime object."""
    # Handle the 'Z' (UTC) suffix if present or microseconds
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1]
    
    # Split fractional seconds to parse main part, if there are fractional seconds
    if '.' in timestamp_str:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")

def main():
    print(f"Reading metrics from {METRICS_FILE}...")
    print(f"Reading time windows from {LABELS_FILE}...")
    print(f"Applying Time Adjustments: +{START_PADDING_SEC}s to Start, +{END_PADDING_SEC}s to End (Recovery)")
    
    if not os.path.exists(METRICS_FILE):
        print(f"Error: {METRICS_FILE} not found. Ensure collect_baseline.py has run.")
        return
        
    if not os.path.exists(LABELS_FILE):
        print(f"Error: {LABELS_FILE} not found. Ensure scenario_runner.py has completed.")
        return

    # 1. Load the scenario time windows into memory with PADDING applied
    scenarios = []
    with open(LABELS_FILE, mode='r') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            original_start = int(row['start_time'])
            original_end = int(row['end_time'])
            
            # Apply padding to effectively shift the anomaly window later
            padded_start = original_start + START_PADDING_SEC
            padded_end = original_end + END_PADDING_SEC
            
            scenarios.append({
                "start": padded_start,
                "end": padded_end,
                "label": row['label']
            })
            
    print(f"Loaded {len(scenarios)} labeling time windows.")
    
    labeled_count = 0
    total_count = 0
    
    # 2. Iterate through the collected metrics and label them
    with open(METRICS_FILE, mode='r') as mf, open(OUTPUT_FILE, mode='w', newline='') as of:
        reader = csv.DictReader(mf)
        
        # We need to maintain the same header order, but ensure 'label' is there
        fieldnames = list(reader.fieldnames)
        if 'label' not in fieldnames:
            fieldnames.insert(1, 'label')
            
        writer = csv.DictWriter(of, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            total_count += 1
            
            # The timestamp in node_metrics.csv is isoformat
            row_dt = parse_isoformat(row['timestamp'])
            row_epoch = int(row_dt.timestamp())
            
            # Default label
            new_label = "normal"
            
            # Check against all scenario time windows
            for s in scenarios:
                if s['start'] <= row_epoch <= s['end']:
                    new_label = s['label']
                    break
                    
            row['label'] = new_label
            if new_label != "normal":
                labeled_count += 1
                
            writer.writerow(row)
            
    print(f"\nLabeling Complete! Wrote combined metrics -> {OUTPUT_FILE}")
    print(f"Total Rows: {total_count}")
    print(f"Anomalous (Non-Normal) Rows: {labeled_count}")

if __name__ == '__main__':
    main()
