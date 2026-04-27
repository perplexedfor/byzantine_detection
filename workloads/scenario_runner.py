import subprocess
import time
import datetime
import csv
import os
import random
import uuid
import hashlib
import json

# Define paths to workloads based on current directory structure
WORKLOADS_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
FAULTS_DIR = os.path.join(WORKLOADS_BASE_DIR, "faults")
SECURITY_DIR = os.path.join(WORKLOADS_BASE_DIR, "security")

# Define the anomaly scenarios (we will shuffle these and randomize duration)
ALL_ANOMALIES = [
    # Controlled Faults
    ("cpu_stress", os.path.join(FAULTS_DIR, "fault-cpu-stress.yaml")),
    ("memory_leak", os.path.join(FAULTS_DIR, "fault-memory-leak.yaml")),
    ("network_chaos", os.path.join(FAULTS_DIR, "fault-network-chaos.yaml")),
    ("crash_loop", os.path.join(FAULTS_DIR, "fault-crash-loop.yaml")),
    # Edge-specific Fault
    ("edge_network_flap", os.path.join(FAULTS_DIR, "fault-network-flap.yaml")),
    
    # Security Anomalies
    ("security_tmp_exec", os.path.join(SECURITY_DIR, "security-tmp-exec.yaml")),
    ("security_high_process", os.path.join(SECURITY_DIR, "security-high-process.yaml")),
    ("security_suspicious_network", os.path.join(SECURITY_DIR, "security-suspicious-network.yaml")),
]

# We will dynamically create the output dataset folder
# Use local disk (/tmp) during collection to avoid Shared Folder I/O stalls.
FINAL_DATASET_DIR = os.path.normpath(os.path.join(WORKLOADS_BASE_DIR, "../dataset"))
os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

if os.name != 'nt':
    # Avoid /tmp to prevent interfering with security anomaly detection (tmp-exec)
    COLLECTION_DIR = os.path.expanduser("~/k3s_data_collection")
else:
    COLLECTION_DIR = FINAL_DATASET_DIR

os.makedirs(COLLECTION_DIR, exist_ok=True)
DATASET_DIR = COLLECTION_DIR

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


def apply_workload(file_path, target_node="all", params=None):
    if not file_path:
        return
        
    if target_node != "all":
        with open(file_path, 'r') as f:
            content = f.read()
            
        if params:
            for key, value in params.items():
                content = content.replace(key, str(value))
            
        injection = f"      nodeSelector:\n        kubernetes.io/hostname: {target_node}\n      containers:"
        patched_content = content.replace("      containers:", injection)
        
        tmp_path = file_path + ".tmp.yaml"
        with open(tmp_path, 'w') as f:
            f.write(patched_content)
            
        run_kubectl("apply", tmp_path)
        os.remove(tmp_path)
    else:
        run_kubectl("apply", file_path)

def delete_workload(file_path):
    run_kubectl("delete", file_path)

def wait_for_nodes_ready(timeout_sec=300):
    """Wait until all nodes in the cluster are in the 'Ready' state."""
    print(f"Waiting for nodes to stabilize (timeout: {timeout_sec}s)...")
    start_wait = time.time()
    while time.time() - start_wait < timeout_sec:
        try:
            result = subprocess.run(
                ["k3s", "kubectl", "get", "nodes", "-o", "json"],
                capture_output=True, text=True, check=True
            )
            nodes = json.loads(result.stdout).get("items", [])
            all_ready = True
            for node in nodes:
                status = "NotReady"
                for cond in node.get("status", {}).get("conditions", []):
                    if cond.get("type") == "Ready":
                        status = "True" if cond.get("status") == "True" else "False"
                        break
                
                if status != "True":
                    print(f"  - Node {node['metadata']['name']} is still {status}...")
                    all_ready = False
            
            if all_ready:
                print("All nodes are Ready.")
                return True
        except Exception as e:
            print(f"Error checking node status: {e}")
        
        time.sleep(10)
    
    print("TIMEOUT: Nodes did not recover to Ready state.")
    return False

def main():
    print("Starting Automated Scenario Runner")
    print("WARNING: Ensure that normal baseline workloads are already running (run_normal_baseline.sh)")
    print("Sleeping 5 seconds to abort if necessary...")
    time.sleep(5)
    
    run_id = str(uuid.uuid4())[:8]
    intensity_seed = int(time.time() * 1000) % 10000
    random.seed(intensity_seed)
    
    print(f"Beginning Run ID: {run_id}")
    print(f"Intensity Seed: {intensity_seed}")
    
    # Start data collection process linked to this run
    collect_script = os.path.join(WORKLOADS_BASE_DIR, "../k3s-monitoring-setup/collect_baseline.py")
    print(f"Starting metric collection background process...")
    collect_process = subprocess.Popen(["python3", collect_script, run_id])
    
    LABELS_OUTPUT_FILE = os.path.join(DATASET_DIR, f"scenario_labels_{run_id}.csv")
    
    with open(LABELS_OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_time", "end_time", "label", "target_node", "run_id", "fault_order_hash", "intensity_seed"])
        
        # Build a randomized timeline for this execution run
        random.shuffle(ALL_ANOMALIES)
        
        # Create a hash of the fault order for tracking
        order_str = ",".join([a[0] for a in ALL_ANOMALIES])
        fault_order_hash = hashlib.md5(order_str.encode()).hexdigest()[:8]
        print(f"Fault Order Hash: {fault_order_hash}")
        
        # Always start with a solid 3-minute baseline block
        run_timeline = [(180, "normal", None)]
        
        # Add randomized anomalies to timeline
        for label, filepath in ALL_ANOMALIES:
            # Random duration: 120s +/- 20s
            duration = 120 + random.randint(-20, 20)
            run_timeline.append((duration, label, filepath))
            # Also insert a solid 2-minute normal baseline between every single anomaly 
            # to let the model see normal transitions
            run_timeline.append((120, "normal", None))

        for duration, label, filepath in run_timeline:
            start_time = time.time()
            start_time_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n==========================================")
            print(f"Starting Scenario: {label}")
            print(f"Time: {start_time_str}")
            print(f"Duration: {duration} seconds")
            
            target_node = "all"
            if filepath:
                # Fault-aware node affinity:
                # CPU/RAM-heavy faults → sw-wk3 only (sensor-gateway role is more stable
                # under high BPF event load; wk-2 Tetragon sidecar saturates first).
                # Light I/O faults → random wk2 or wk3 (safe for BPF ring buffer).
                # k3s-wk1 is excluded — it runs etcd + API server (control plane).
                HEAVY_FAULTS = {"cpu_stress", "memory_leak", "crash_loop", "security_high_process"}
                LIGHT_NODE_POOL = ["sw-wk2", "sw-wk3"]

                if label in HEAVY_FAULTS:
                    target_node = "sw-wk3"
                else:
                    target_node = random.choice(LIGHT_NODE_POOL)
                print(f"Target Node Selected: {target_node} (affinity: {'heavy→wk3' if label in HEAVY_FAULTS else 'light→random'})")  
                
                # Generate random parameters based on the label
                params = {}
                if label == "cpu_stress":
                    params = {
                        "{{CPU_WORKERS}}": random.randint(1, 2), # 1 or 2 cores
                        "{{CPU_LOAD}}": random.randint(60, 95),  # 60% to 95% load
                        "{{CPU_METHOD}}": random.choice(["all", "matrixprod", "sqrt", "ackermann"]),
                        "{{DURATION}}": f"{duration}s"
                    }
                elif label == "memory_leak":
                    params = {
                        "{{LEAK_RATE}}": random.choice([2, 3, 4, 5]) # MB per 2 seconds (REDUCED from [5,10,20,30])
                    }
                elif label == "network_chaos":
                    params = {
                        "{{DELAY}}": random.randint(50, 250), # 50ms to 250ms
                        "{{LOSS}}": random.randint(1, 10)     # 1% to 10%
                    }
                elif label == "crash_loop":
                    params = {
                        "{{CRASH_INTERVAL}}": random.randint(2, 10), # Crash every 2 to 10 seconds
                        "{{CRASH_WORKERS}}": random.randint(1, 2)
                    }
                elif label == "security_tmp_exec":
                    params = {
                        "{{EXEC_PATH}}": random.choice(["/tmp", "/dev/shm", "/var/tmp"]),
                        "{{EXEC_INTERVAL}}": random.randint(2, 10)
                    }
                elif label == "security_high_process":
                    params = {
                        "{{SPAWN_COUNT}}": random.randint(10, 25) # Capped: >25 forks/s overwhelms BPF ring buffer on wk-2
                    }
                elif label == "security_suspicious_network":
                    # Randomize destination ports to simulate beaconing vs mining
                    ports = ["3333", "4444", "5555", "8333", "14444", "6666", "7777"]
                    selected_ports = random.sample(ports, 2)
                    params = {
                        "{{PORT_1}}": selected_ports[0],
                        "{{PORT_2}}": selected_ports[1],
                        "{{CONNECT_INTERVAL}}": random.randint(3, 15)
                    }
                elif label == "edge_network_flap":
                    # Duration is already determined by the timeline; pass it through
                    # so fault-network-flap.yaml's tc netem runs for the right window.
                    params = {
                        "{{DURATION}}": f"{duration}s"
                    }
                
                print(f"Injecting Parameters: {params}")
                apply_workload(filepath, target_node, params)
                
            # Wait for the duration
            time.sleep(duration)
            
            end_time = time.time()
            end_time_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Scenario {label} completed at {end_time_str}")
            
            # Write primary scenario to CSV
            writer.writerow([int(start_time), int(end_time), label, target_node, run_id, fault_order_hash, intensity_seed])
            
            # Clean up workload
            if filepath:
                print(f"Cleaning up {filepath}...")
                delete_workload(filepath)
                
                # 60 second buffer to ensure cleanup finishes and memory is reclaimed
                # Record explicitly as 'transition' buffer
                # Wait for nodes to return to Ready state instead of a blind sleep.
                # This ensures RCU stalls or OOM scenarios have cleared.
                print("Verifying node health and waiting for recovery (recording as 'transition' buffer)...")
                trans_start = time.time()
                wait_for_nodes_ready(timeout_sec=300) 
                trans_end = time.time()
                writer.writerow([int(trans_start), int(trans_end), "transition", "all", run_id, fault_order_hash, intensity_seed])
            
            # Flush to disk immediately in case script crashes
            file.flush()
            
    print(f"\nAll Scenarios finished! Time windows saved to {LABELS_OUTPUT_FILE}")
    
    print("Terminating background metric collection process...")
    collect_process.terminate()
    collect_process.wait()
    
    # Sync labels and metrics from local collection dir to final project dataset folder
    if COLLECTION_DIR != FINAL_DATASET_DIR:
        print(f"Syncing collected data to {FINAL_DATASET_DIR}...")
        import shutil
        try:
            for f in os.listdir(COLLECTION_DIR):
                if f.startswith(f"scenario_labels_{run_id}") or f.startswith(f"node_metrics_{run_id}"):
                    src = os.path.join(COLLECTION_DIR, f)
                    dst = os.path.join(FINAL_DATASET_DIR, f)
                    shutil.copy2(src, dst)
                    print(f"  ✓ Copied: {f}")
        except Exception as e:
            print(f"Error syncing data: {e}")
            
    print(f"Successfully generated dataset segment for run {run_id}")

if __name__ == "__main__":
    main()
