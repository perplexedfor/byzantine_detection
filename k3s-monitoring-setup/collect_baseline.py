import time
import csv
import json
import subprocess
import requests
from datetime import datetime

# Configuration
PROMETHEUS_URL = "http://localhost:8080"
TETRAGON_LOGS_CMD = [
    "k3s", "kubectl", "logs", "-n", "kube-system", "-ds/tetragon", "-c", "export-stdout", "--tail=0", "-f"
]
OUTPUT_CSV = "node_metrics.csv"
COLLECTION_INTERVAL_SEC = 10 # 10 second aggregation window

# State for Tetragon metrics per node
# Structure: { "node_name": { "exec_count": 0, "unique_process_count": set(), "tmp_exec_count": 0, "outbound_connect_count": 0, "mining_port_count": 0, "syscalls": {} } }
tetragon_state = {}

def get_prometheus_metric(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        response.raise_for_status()
        data = response.json()['data']['result']
        # Return a dictionary of node -> value
        return {item['metric'].get('instance', '').split(':')[0]: float(item['value'][1]) for item in data}
    except Exception as e:
        print(f"Error fetching Prometheus query '{query}': {e}")
        return {}

def collect_prometheus_metrics():
    # avg_cpu: 1 - avg idle CPU over the last 1m
    cpu_query = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) by (instance)'
    # avg_mem: (total - available) / total
    mem_query = '1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
    # net_bytes_in/out: network receive/transmit bytes rate over 1m for eth0 (adjust interface if needed)
    net_in_query = 'rate(node_network_receive_bytes_total{device="eth0"}[1m])'
    net_out_query = 'rate(node_network_transmit_bytes_total{device="eth0"}[1m])'

    return {
        'avg_cpu': get_prometheus_metric(cpu_query),
        'avg_mem': get_prometheus_metric(mem_query),
        'net_bytes_in': get_prometheus_metric(net_in_query),
        'net_bytes_out': get_prometheus_metric(net_out_query),
    }

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

                # Collect from Prometheus
                prom_metrics = collect_prometheus_metrics()
                
                # Get all unique nodes we know about (from both sources)
                all_nodes = set(list(tetragon_state.keys()))
                for metric_dict in prom_metrics.values():
                    all_nodes.update(metric_dict.keys())

                current_timestamp = datetime.now().isoformat()
                
                # Currently collecting 'normal' baseline behavior
                current_label = "normal" 

                # Write out row per node
                for node in all_nodes:
                    node_short = node # Prometheus often drops the domain, ensure matches

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
