import subprocess
import time
import datetime
import csv
import os

# Define paths to workloads based on current directory structure
WORKLOADS_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
FAULTS_DIR = os.path.join(WORKLOADS_BASE_DIR, "faults")
SECURITY_DIR = os.path.join(WORKLOADS_BASE_DIR, "security")

# Define the scenario timeline: (duration_seconds, label_name, workload_file_path)
# None for workload_file_path means "Normal Baseline" where nothing extra is deployed. 
# We assume run_normal_baseline.sh is already running in the background.
SCENARIOS = [
    (300, "normal", None), # 5 minutes of normal baseline
    
    # Controlled Faults (2 mins each)
    (120, "cpu_stress", os.path.join(FAULTS_DIR, "fault-cpu-stress.yaml")),
    (120, "memory_leak", os.path.join(FAULTS_DIR, "fault-memory-leak.yaml")),
    (120, "network_chaos", os.path.join(FAULTS_DIR, "fault-network-chaos.yaml")),
    (120, "crash_loop", os.path.join(FAULTS_DIR, "fault-crash-loop.yaml")),
    
    (120, "normal_recovery", None), # 2 mins to let cluster recover
    
    # Security Anomalies (2 mins each)
    (120, "security_tmp_exec", os.path.join(SECURITY_DIR, "security-tmp-exec.yaml")),
    (120, "security_high_process", os.path.join(SECURITY_DIR, "security-high-process.yaml")),
    (120, "security_suspicious_network", os.path.join(SECURITY_DIR, "security-suspicious-network.yaml")),
]

LABELS_OUTPUT_FILE = "scenario_labels.csv"

def run_kubectl(action, file_path):
    """Run k3s kubectl apply or delete"""
    if not file_path:
        return
        
    cmd = ["k3s", "kubectl", action, "-f", file_path]
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error {action}ing {file_path}: {e.stderr.decode()}")


def apply_workload(file_path):
    run_kubectl("apply", file_path)

def delete_workload(file_path):
    run_kubectl("delete", file_path)

def main():
    print("Starting Automated Scenario Runner")
    print("WARNING: Ensure that normal baseline workloads are already running (run_normal_baseline.sh)")
    print("Sleeping 5 seconds to abort if necessary...")
    time.sleep(5)
    
    with open(LABELS_OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_time", "end_time", "label", "workload"])
        
        for duration, label, filepath in SCENARIOS:
            start_time = time.time()
            start_time_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n==========================================")
            print(f"Starting Scenario: {label}")
            print(f"Time: {start_time_str}")
            print(f"Duration: {duration} seconds")
            
            if filepath:
                apply_workload(filepath)
                
            # Wait for the duration
            time.sleep(duration)
            
            end_time = time.time()
            end_time_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Scenario {label} completed at {end_time_str}")
            
            # Clean up workload
            if filepath:
                print(f"Cleaning up {filepath}...")
                delete_workload(filepath)
                # Small buffer to ensure cleanup before next scenario
                time.sleep(10) 
            
            # Write to CSV
            writer.writerow([int(start_time), int(end_time), label, os.path.basename(filepath) if filepath else "baseline"])
            # Flush to disk immediately in case script crashes
            file.flush()
            
    print(f"\nAll Scenarios finished! Time windows saved to {LABELS_OUTPUT_FILE}")
    print("You can use this file along with your collect_baseline.py to label your dataset.")

if __name__ == "__main__":
    main()
