import time
import csv
import json
import subprocess
import requests
import sys
import os
from datetime import datetime

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
TETRAGON_LOGS_CMD = [
    "k3s", "kubectl", "logs", "-n", "kube-system", "ds/tetragon", "-c", "export-stdout", "--tail=0", "-f"
]
COLLECTION_INTERVAL_SEC = 10 # 10 second aggregation window

WORKLOADS_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(WORKLOADS_BASE_DIR, "../dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Map Prometheus node IPs to actual node names from Tetragon
# Update this with your actual edge cluster IPs and hostnames if they differ
IP_TO_NODE_MAP = {
    "192.168.56.10": "k3s-wk1", # the server
    "192.168.56.11": "sw-wk2", # worker 1
    "192.168.56.12": "sw-wk3"  # worker 2
}

EXPECTED_NODES = len(IP_TO_NODE_MAP)  # Expected number of nodes in cluster

# State for Tetragon metrics per node
# Structure: { "node_name": { "exec_count": 0, "unique_process_count": set(), "tmp_exec_count": 0, "outbound_connect_count": 0, "mining_port_count": 0, "syscalls": {} } }
tetragon_state = {}

def get_prometheus_metric(query):
    """Fetch Prometheus metric. Returns tuple (data_dict, success_flag).
    
    Attempts to map instances to node names by:
    1. Using 'node' label if available (from Kubernetes service discovery)
    2. Falling back to IP_TO_NODE_MAP if node label not present
    3. Using raw instance IP as last resort
    """
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query", 
            params={'query': query},
            timeout=5
        )
        response.raise_for_status()
        if response.status_code != 200:
            print(f"Prometheus query failed with status {response.status_code}: {query}")
            return {}, False
        
        data = response.json()['data']['result']
        result_data = {}
        
        for item in data:
            # Try multiple strategies to identify the node
            instance_str = item['metric'].get('instance', '')
            instance_ip = instance_str.split(':')[0] if instance_str else ''
            
            # Strategy 1: Use 'node' label if available (preferred - from Kubernetes)
            node_name = item['metric'].get('node')
            
            # Strategy 2: Use '__meta_kubernetes_node_name' if present
            if not node_name:
                node_name = item['metric'].get('__meta_kubernetes_node_name')
            
            # Strategy 3: Map known IPs to node names
            if not node_name and instance_ip:
                node_name = IP_TO_NODE_MAP.get(instance_ip)
            
            # Strategy 4: Use the instance IP as fallback
            if not node_name:
                node_name = instance_ip if instance_ip else 'unknown'
            
            result_data[node_name] = float(item['value'][1])
        
        return result_data, True
    except requests.exceptions.Timeout:
        print(f"Timeout fetching Prometheus query '{query}'")
        return {}, False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error fetching Prometheus query '{query}': {e}")
        return {}, False
    except Exception as e:
        print(f"Error fetching Prometheus query '{query}': {e}")
        return {}, False

def collect_prometheus_metrics():
    """Collect Prometheus metrics. Returns dict with metrics and query_success flag."""
    # avg_cpu: 1 - avg idle CPU over the last 1m
    cpu_query = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) by (instance)'
    # avg_mem: (total - available) / total
    mem_query = '1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
    # net_bytes_in/out: network receive/transmit bytes rate over 1m for enp0s3 (NAT interface)
    net_in_query = 'rate(node_network_receive_bytes_total{device="enp0s3"}[1m])'
    net_out_query = 'rate(node_network_transmit_bytes_total{device="enp0s3"}[1m])'

    cpu_data, cpu_ok = get_prometheus_metric(cpu_query)
    mem_data, mem_ok = get_prometheus_metric(mem_query)
    net_in_data, net_in_ok = get_prometheus_metric(net_in_query)
    net_out_data, net_out_ok = get_prometheus_metric(net_out_query)
    
    # All queries must succeed
    query_success = cpu_ok and mem_ok and net_in_ok and net_out_ok

    return {
        'avg_cpu': cpu_data,
        'avg_mem': mem_data,
        'net_bytes_in': net_in_data,
        'net_bytes_out': net_out_data,
        'query_success': query_success
    }

def map_prom_metrics_to_nodes(prom_metrics):
    mapped_metrics = {
        'avg_cpu': {},
        'avg_mem': {},
        'net_bytes_in': {},
        'net_bytes_out': {}
    }
    
    for metric_name, node_data in prom_metrics.items():
        if metric_name == 'query_success':
            continue  # Skip the query_success flag
        for ip, value in node_data.items():
            # Apply mapping if IP is known, otherwise fallback to the IP string
            node_name = IP_TO_NODE_MAP.get(ip, ip)
            mapped_metrics[metric_name][node_name] = value

    return mapped_metrics

def detect_impossible_values(prom_metrics):
    """Check for sudden impossible values (instant zeros in CPU, Memory, Network).
    
    Returns True if impossible values detected (indicates monitoring gap).
    """
    # Check for sudden zeros across all metrics
    metrics_to_check = ['avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out']
    
    for metric_name in metrics_to_check:
        metric_data = prom_metrics.get(metric_name, {})
        for node, value in metric_data.items():
            # Sudden jump to exactly 0.0 in CPU, Memory, or Network is suspicious
            # (excluding valid zero cases like no network traffic is difficult,
            # but combined with other checks this helps detect monitoring gaps)
            if value == 0.0 and metric_name in ['avg_cpu', 'avg_mem']:
                print(f"Warning: Impossible value detected - {metric_name} = 0.0 on {node}")
                return True
    
    return False

def validate_metrics(prom_metrics, all_nodes_expected):
    """Validate that metrics are complete and reasonable.
    
    Returns tuple (is_valid, is_transition).
    is_valid: True if metrics look good
    is_transition: True if we should mark this window as transition
    """
    
    # Check 1: Prometheus query failure
    if not prom_metrics.get('query_success', True):
        print("Transition: Prometheus query failed")
        return False, True
    
    # Check 2: Missing metrics for expected nodes
    cpu_metrics = prom_metrics.get('avg_cpu', {})
    if len(cpu_metrics) < all_nodes_expected:
        print(f"Transition: Missing metrics - expected {all_nodes_expected} nodes, got {len(cpu_metrics)}")
        return False, True
    
    # Check 3: Detect impossible values (sudden zeros)
    if detect_impossible_values(prom_metrics):
        print("Transition: Impossible values detected (monitoring gap)")
        return False, True
    
    return True, False

def process_tetragon_event(event_line):
    try:
        event = json.loads(event_line)
        node_name = event.get('node_name', 'unknown')
        
        if node_name not in tetragon_state:
            tetragon_state[node_name] = {
                'exec_count': 0,
                'unique_process_count': set(),
                'tmp_exec_count': 0,
                'outbound_connect_count': 0,
                'mining_port_count': 0,
                'syscall_feature_vector': {} # simplified for now
            }

        state = tetragon_state[node_name]

        # Handle process_exec events
        if 'process_exec' in event:
            proc = event['process_exec']['process']
            binary = proc.get('binary', '')
            state['exec_count'] += 1
            state['unique_process_count'].add(binary)

            # Check if executing from /tmp or /dev/shm
            if binary.startswith('/tmp/') or binary.startswith('/dev/shm/'):
                state['tmp_exec_count'] += 1
                
        # Handle process_kprobe events (for network connections, if configured in Tetragon policies)
        # Note: This requires a TracingPolicy to be active for connections (e.g. tcp_connect)
        elif 'process_kprobe' in event:
             kprobe = event['process_kprobe']
             function_name = kprobe.get('function_name', '')
             if function_name == 'tcp_connect':
                  state['outbound_connect_count'] += 1
                  # Extract port from sock_arg (defined by args type in the TracingPolicy)
                  args = kprobe.get('args', [])
                  for arg in args:
                      if 'sock_arg' in arg:
                          dport = arg['sock_arg'].get('dport')
                          # Check against common crypto-mining stratum/RPC ports
                          if dport in {3333, 4444, 5555, 6666, 7777, 8332, 8333, 14433, 14444}:
                              state['mining_port_count'] += 1

        # Very basic syscall tracking based on event type if kprobes are heavily used
        event_type = list(event.keys())[0] if event else "unknown"
        state['syscall_feature_vector'][event_type] = state['syscall_feature_vector'].get(event_type, 0) + 1

    except json.JSONDecodeError:
         pass # Ignore non-JSON lines
    except Exception as e:
         print(f"Error parsing event: {e}")

def main():
    if len(sys.argv) < 2:
        print("Error: Missing run_id argument. Usage: python collect_baseline.py <run_id>")
        sys.exit(1)
        
    run_id = sys.argv[1]
    OUTPUT_CSV = os.path.join(DATASET_DIR, f"node_metrics_{run_id}.csv")
    
    # 1. Start reading Tetragon logs in the background
    print("Starting Tetragon log stream...")
    tetragon_process = subprocess.Popen(
        TETRAGON_LOGS_CMD, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )

    # 2. Setup CSV Writer
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Define headers
        headers = [
            'timestamp', 'label', 'node', 
            'avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out',
            'exec_count', 'unique_process_count', 'tmp_exec_count', 
            'outbound_connect_count', 'mining_port_count', 'syscall_feature_vector'
        ]
        writer.writerow(headers)

        print(f"Collecting baseline metrics every {COLLECTION_INTERVAL_SEC} seconds. Writing to {OUTPUT_CSV}...")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                start_time = time.time()
                
                # Non-blocking read of Tetragon logs collected during this interval
                import select
                while select.select([tetragon_process.stdout], [], [], 0.0)[0]:
                    line = tetragon_process.stdout.readline()
                    if line:
                        process_tetragon_event(line)
                    else:
                        break

                # Collect from Prometheus and map IPs to Node names
                raw_prom_metrics = collect_prometheus_metrics()
                prom_metrics = map_prom_metrics_to_nodes(raw_prom_metrics)
                
                # Get all unique nodes we know about (from both sources)
                all_nodes = set(list(tetragon_state.keys()))
                for metric_dict in prom_metrics.values():
                    all_nodes.update(metric_dict.keys())

                # Use the exact same epoch timestamp format as the scenario_runner
                current_timestamp = int(time.time())
                
                # Check for transition conditions
                is_transition = False
                metrics_valid, metrics_transition = validate_metrics(raw_prom_metrics, EXPECTED_NODES)
                if metrics_transition:
                    is_transition = True
                
                # Mark as transition or normal baseline
                current_label = "transition" if is_transition else "normal" 

                # Write out row per node
                for node in all_nodes:
                    node_short = node # Note: Prom metrics are already mapped to 'node' names now

                    t_state = tetragon_state.get(node, {})
                    
                    row = [
                        current_timestamp,
                        current_label,
                        node,
                        prom_metrics['avg_cpu'].get(node_short, 0.0),
                        prom_metrics['avg_mem'].get(node_short, 0.0),
                        prom_metrics['net_bytes_in'].get(node_short, 0.0),
                        prom_metrics['net_bytes_out'].get(node_short, 0.0),
                        t_state.get('exec_count', 0),
                        len(t_state.get('unique_process_count', set())),
                        t_state.get('tmp_exec_count', 0),
                        t_state.get('outbound_connect_count', 0),
                        t_state.get('mining_port_count', 0),
                        json.dumps(t_state.get('syscall_feature_vector', {}))
                    ]
                    writer.writerow(row)
                    
                    # Reset Tetragon aggregation state for this node for the next window
                    if node in tetragon_state:
                         tetragon_state[node] = {
                             'exec_count': 0,
                             'unique_process_count': set(),
                             'tmp_exec_count': 0,
                             'outbound_connect_count': 0,
                             'mining_port_count': 0,
                             'syscall_feature_vector': {}
                         }
                
                # Flush to disk immediately
                file.flush()

                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, COLLECTION_INTERVAL_SEC - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping data collection.")
        finally:
            tetragon_process.terminate()

if __name__ == "__main__":
    main()
