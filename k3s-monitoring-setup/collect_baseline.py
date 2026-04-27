import time
import csv
import json
import subprocess
import requests
import sys
import os
import threading
import queue
from datetime import datetime

# Configuration
PROMETHEUS_URL = "http://localhost:9090"

# Tetragon is a DaemonSet — one pod per node. We must stream logs from each
# node's pod individually, otherwise events from worker nodes are silently lost.
# Node name -> pod-specific log command built at runtime in start_tetragon_streams().
COLLECTION_INTERVAL_SEC = 10 # 10 second aggregation window

WORKLOADS_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Use a fast local disk (/tmp) for collection to avoid VirtualBox Shared Folder latency stalls.
# Files will be synced back to the project dataset folder on graceful exit.
FINAL_DATASET_DIR = os.path.normpath(os.path.join(WORKLOADS_BASE_DIR, "../dataset"))
os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

if os.name != 'nt':
    # Avoid /tmp, /dev/shm, and /var/tmp as Tetragon specifically monitors these for anomalies.
    # Using a hidden folder in the user's home directory is safe and fast.
    COLLECTION_DIR = os.path.expanduser("~/k3s_data_collection")
else:
    # On Windows host, just use the project directory (no shared folder overhead here)
    COLLECTION_DIR = FINAL_DATASET_DIR

os.makedirs(COLLECTION_DIR, exist_ok=True)
DATASET_DIR = COLLECTION_DIR

# Map Prometheus node IPs to actual node names from Tetragon
# Update this with your actual edge cluster IPs and hostnames if they differ
IP_TO_NODE_MAP = {
    "192.168.56.10": "k3s-wk1", # the server
    "192.168.56.11": "sw-wk2", # worker 1
    "192.168.56.12": "sw-wk3"  # worker 2
}

EXPECTED_NODES = len(IP_TO_NODE_MAP)  # Expected number of nodes in cluster

# State for Tetragon metrics per node — 5 numeric MVS signals + syscall_feature_vector
# Structure: { "node_name": { "exec_count": 0, "unique_process_count": set(),
#              "tmp_exec_count": 0, "outbound_connect_count": 0, "mining_port_count": 0,
#              "syscall_feature_vector": {} } }
# tetragon_state is wrapped in a single-element list so the swap-accumulator
# pattern in main() (tetragon_state[0] = {}) is visible to ALL threads.
# If it were a plain dict, rebinding the name inside main() would only update
# main's local namespace — the event_processor thread would still write into
# the old (now-stale) dict object. The list cell is shared by reference.
tetragon_state = [{}]  # tetragon_state[0] is the live accumulator dict
tetragon_lock = threading.Lock()  # Synchronize swap vs event_processor writes

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
            instance_str = item['metric'].get('instance', '')
            instance_ip = instance_str.split(':')[0] if instance_str else ''
            
            node_name = item['metric'].get('node')
            
            if not node_name:
                node_name = item['metric'].get('__meta_kubernetes_node_name')
        
            if not node_name and instance_ip:
                node_name = IP_TO_NODE_MAP.get(instance_ip)
            
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
    # net_bytes_in/out: external NAT interface (enp0s3)
    net_in_query = 'rate(node_network_receive_bytes_total{device="enp0s3"}[1m])'
    net_out_query = 'rate(node_network_transmit_bytes_total{device="enp0s3"}[1m])'
    # net_internal_bytes_in/out: host-only inter-node interface (enp0s8)
    # Edge simulation: captures inter-node link degradation from fault-network-flap
    net_internal_in_query = 'rate(node_network_receive_bytes_total{device="enp0s8"}[1m])'
    net_internal_out_query = 'rate(node_network_transmit_bytes_total{device="enp0s8"}[1m])'

    cpu_data, cpu_ok = get_prometheus_metric(cpu_query)
    mem_data, mem_ok = get_prometheus_metric(mem_query)
    net_in_data, net_in_ok = get_prometheus_metric(net_in_query)
    net_out_data, net_out_ok = get_prometheus_metric(net_out_query)
    net_int_in_data, net_int_in_ok = get_prometheus_metric(net_internal_in_query)
    net_int_out_data, net_int_ok = get_prometheus_metric(net_internal_out_query)

    # net_drop_rate: explicitly capture packet loss as a signal
    net_drop_query = 'rate(node_network_receive_drop_total{device="enp0s3"}[1m]) + rate(node_network_transmit_drop_total{device="enp0s3"}[1m])'
    net_drop_data, net_drop_ok = get_prometheus_metric(net_drop_query)

    # All primary queries must succeed
    query_success = cpu_ok and mem_ok and net_in_ok and net_out_ok and net_drop_ok

    return {
        'avg_cpu': cpu_data,
        'avg_mem': mem_data,
        'net_bytes_in': net_in_data,
        'net_bytes_out': net_out_data,
        'net_internal_bytes_in': net_int_in_data,
        'net_internal_bytes_out': net_int_out_data,
        'net_drop_rate': net_drop_data,
        'query_success': query_success
    }

def map_prom_metrics_to_nodes(prom_metrics):
    mapped_metrics = {
        'avg_cpu': {},
        'avg_mem': {},
        'net_bytes_in': {},
        'net_bytes_out': {},
        'net_internal_bytes_in': {},
        'net_internal_bytes_out': {},
        'net_drop_rate': {},
    }
    
    for metric_name, node_data in prom_metrics.items():
        if metric_name == 'query_success':
            continue  # Skip the query_success flag
        if metric_name not in mapped_metrics:
            continue  # Skip unknown keys
        for ip, value in node_data.items():
            # Apply mapping if IP is known, otherwise fallback to the IP string
            node_name = IP_TO_NODE_MAP.get(ip, ip)
            mapped_metrics[metric_name][node_name] = value

    return mapped_metrics

def detect_impossible_values(prom_metrics, silent=False):
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
                if not silent: print(f"Warning: Impossible value detected - {metric_name} = 0.0 on {node}")
                return True
    
    return False

def validate_metrics(prom_metrics, all_nodes_expected, silent=False):
    """Validate that metrics are complete and reasonable.
    
    Returns tuple (is_valid, is_transition).
    is_valid: True if metrics look good
    is_transition: True if we should mark this window as transition
    """
    
    # Check 1: Prometheus query failure
    if not prom_metrics.get('query_success', True):
        if not silent: print("Transition: Prometheus query failed")
        return False, True
    
    # Check 2: Missing metrics for expected nodes
    cpu_metrics = prom_metrics.get('avg_cpu', {})
    if len(cpu_metrics) < all_nodes_expected:
        if not silent: print(f"Transition: Missing metrics - expected {all_nodes_expected} nodes, got {len(cpu_metrics)}")
        return False, True
    
    # Check 3: Detect impossible values (sudden zeros)
    if detect_impossible_values(prom_metrics, silent=silent):
        if not silent: print("Transition: Impossible values detected (monitoring gap)")
        return False, True

    
    return True, False

def process_tetragon_event(event_line):
    try:
        event = json.loads(event_line)
        node_name = event.get('node_name', 'unknown')

        # Always dereference through the cell — tetragon_state[0] may have been
        # swapped to a fresh dict by the main loop's accumulator swap.
        current = tetragon_state[0]

        if node_name not in current:
            current[node_name] = {
                'exec_count': 0,
                'unique_process_count': set(),
                'tmp_exec_count': 0,
                'outbound_connect_count': 0,
                'mining_port_count': 0,
                'syscall_feature_vector': {},
            }

        state = current[node_name]

        # Handle process_exec events
        if 'process_exec' in event:
            proc = event['process_exec']['process']
            binary = proc.get('binary', '')
            state['exec_count'] += 1
            state['unique_process_count'].add(binary)

            # Check if executing from /tmp, /dev/shm, or /var/tmp
            if binary.startswith('/tmp/') or binary.startswith('/dev/shm/') or binary.startswith('/var/tmp/'):
                state['tmp_exec_count'] += 1
                
        # Handle process_kprobe events (for network connections, if configured in Tetragon policies)
        # Note: This requires a TracingPolicy to be active for connections (e.g. tcp_connect)
        if 'process_kprobe' in event:
             kprobe = event['process_kprobe']
             function_name = kprobe.get('function_name', '')
             if function_name == 'tcp_connect':
                  state['outbound_connect_count'] += 1
                  # Extract port from sock_arg (defined by args type in the TracingPolicy)
                  args = kprobe.get('args', [])
                  for arg in args:
                      # Tetragon may serialize sock type as 'sock_arg', 'sock',
                      # or expose 'dport' directly at the arg level depending on version.
                      sock = arg.get('sock_arg') or arg.get('sock') or {}
                      dport = sock.get('dport') or arg.get('dport')
                      # Check against common crypto-mining stratum/RPC ports
                      # Cast to int — Tetragon JSON may serialize dport as a string
                      try:
                          dport = int(dport) if dport is not None else None
                      except (ValueError, TypeError):
                          dport = None
                      if dport in {3333, 4444, 5555, 6666, 7777, 8332, 8333, 14433, 14444}:
                          state['mining_port_count'] += 1

        # Track event type frequency for Random Forest training
        # RF can use these after exploding the JSON into per-type numeric columns
        known_types = ['process_exec', 'process_kprobe', 'process_exit', 'process_tracepoint']
        event_type = "unknown"
        for t in known_types:
            if t in event:
                event_type = t
                break
        
        state['syscall_feature_vector'][event_type] = state['syscall_feature_vector'].get(event_type, 0) + 1


    except json.JSONDecodeError:
         pass # Ignore non-JSON lines
    except Exception as e:
         print(f"Error parsing event: {e}")

def event_processor(event_queue):
    """Background thread that continuously drains the queue and updates tetragon_state.
    
    Processing JSON events in a dedicated thread prevents bursts of data from
    blocking the main loop and causing missed Prometheus scrapes.
    """
    while True:
        try:
            # Block for a short time to keep CPU usage low but react quickly
            event_line = event_queue.get(timeout=1.0)
            with tetragon_lock:
                process_tetragon_event(event_line)
            event_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processor thread error: {e}")
            time.sleep(1)

# Global state for restarts and heartbeats
tetragon_restart_counts = {}
tetragon_next_retry_time = {} # node -> timestamp of when we can try again
tetragon_last_stabilized_at = {} # node -> timestamp of last successful restart (to reset backoff)
tetragon_pod_map = {} # node -> pod_name (for heartbeats)

def _send_heartbeats():
    """Trigger one heartbeat exec on every tracked node. Safe to call from any thread."""
    current_map = tetragon_pod_map.copy()
    for node, pod in current_map.items():
        try:
            # Trigger a local process start inside the Tetragon pod itself.
            # This will be picked up by Tetragon's own sensor and streamed out.
            subprocess.Popen(
                ["k3s", "kubectl", "exec", "-n", "kube-system", pod, "--", "/bin/echo", "tetragon-heartbeat"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except:
            pass

def tetragon_heartbeat_loop():
    """Periodically triggers a detectable event on each node.
    
    By running 'echo heartbeat' inside the Tetragon pod, we guarantee a
    'process_exec' event is generated every 30s. This ensures the 
    stall detector (45s threshold) doesn't fire when the cluster is idle.
    The FIRST heartbeat fires immediately so wk-2 gets its first event
    within seconds of startup (not after a 30s blind window).
    """
    print("[i] Starting Tetragon self-heartbeat loop (30s interval, immediate first beat)...")
    # Fire immediately — don't wait 30s before the first beat.
    # Without this, a stall on wk-2 during the first 45s would be
    # misidentified as an idle-stream stall and trigger an unnecessary restart.
    _send_heartbeats()
    while True:
        time.sleep(30)
        _send_heartbeats()

# Backoff settings
INITIAL_BACKOFF = 5
MAX_BACKOFF = 300 # 5 minutes
STABILITY_THRESHOLD = 120 # 2 minutes of uptime resets the backoff counter

# Stall detection threshold: if a stream produces no events for this many seconds
# while the subprocess is still alive, the kubelet log-stream has likely timed out
# (the wk-2 idle failure mode). proc.poll() won't catch this.
# 45s is safe: the heartbeat loop guarantees an event every 30s, so 45s allows
# one missed heartbeat before we declare a stall and restart.
# Defined at module level so _stderr_watcher (inside start_tetragon_streams) can
# reference it without it being passed as a parameter.
STREAM_STALL_THRESHOLD_SEC = 45

def start_tetragon_streams(event_queue, restart_node=None):
    """Start one Tetragon log stream per node and feed all events into a shared queue.

    Tetragon is a DaemonSet — each node has its own pod. 'kubectl logs ds/tetragon'
    only picks ONE pod at random, silently dropping events from all other nodes.
    Here we discover every tetragon pod and start a dedicated stream for each.
    A daemon thread per stream feeds lines into the shared queue so the main loop
    can drain all nodes' events atomically each collection interval.
    """
    try:
        result = subprocess.run(
            ["k3s", "kubectl", "get", "pods", "-n", "kube-system",
             "-l", "app.kubernetes.io/name=tetragon",
             "--no-headers", "-o", "custom-columns=NAME:.metadata.name,NODE:.spec.nodeName"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
    except Exception as e:
        print(f"Could not list Tetragon pods: {e}")
        return {}, {}

    if not lines:
        print("No Tetragon pods found. Falling back to ds/tetragon (single-node only).")
        pod_node_pairs = [("ds/tetragon", "unknown")]
    else:
        pod_node_pairs = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                pod_node_pairs.append((parts[0], parts[1]))

    processes = {}
    last_event_time = {}
    
    # Update global pod map for heartbeats
    for pod_name, node_name in pod_node_pairs:
        tetragon_pod_map[node_name] = pod_name
    
    # Filter for restart if requested
    targets = [p for p in pod_node_pairs if p[1] == restart_node] if restart_node else pod_node_pairs

    for pod_name, node_name in targets:
        cmd = [
            "k3s", "kubectl", "logs", "-n", "kube-system", pod_name,
            "-c", "export-stdout", "--tail=0", "-f"
        ]
        print(f"Starting Tetragon stream: pod={pod_name} node={node_name}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes[node_name] = proc
            last_event_time[node_name] = time.time()

            def _reader(p=proc, n=node_name):
                for line in p.stdout:
                    if line:
                        event_queue.put(line)
                        last_event_time[n] = time.time()

            def _stderr_watcher(p=proc, n=node_name):
                """Watch kubectl's stderr for connection errors that signal a dead stream.

                When the kubelet log-stream times out on VirtualBox (host I/O pressure),
                kubectl prints to stderr (e.g. 'connection reset by peer', 'EOF') but the
                process stays alive — proc.poll() returns None and the stdout reader blocks
                forever. Pumping stderr into last_event_time would mask the stall, so
                instead we log the error and let the stall detector do its job after 45s.
                We DO update last_event_time here with a small penalty so a single transient
                error doesn't immediately spike us to 45s-from-last-event: it resets the
                clock to (now - 30s), giving the stall detector 15 more seconds to confirm.
                """
                for err_line in p.stderr:
                    err_line = err_line.strip()
                    if err_line:
                        print(f"[WARN] Tetragon stream stderr [{n}]: {err_line}")
                        # Partial reset: don't count stderr noise as a live event,
                        # but give the stall detector extra time to avoid false restarts
                        # on transient API server hiccups.
                        if any(k in err_line.lower() for k in ("connection reset", "eof", "broken pipe",
                                                                 "timeout", "unable to connect")):
                            # Force stall detection within one extra interval (15s cushion)
                            now_penalty = time.time() - (STREAM_STALL_THRESHOLD_SEC - 15)
                            if last_event_time.get(n, 0) > now_penalty:
                                last_event_time[n] = now_penalty
                                print(f"[WARN] Stream error on {n} — accelerating stall detection.")

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            t_err = threading.Thread(target=_stderr_watcher, daemon=True)
            t_err.start()
        except Exception as e:
            print(f"Failed to start log stream for {pod_name}: {e}")

    return processes, last_event_time

def main():
    if len(sys.argv) < 2:
        print("Error: Missing run_id argument. Usage: python collect_baseline.py <run_id>")
        sys.exit(1)
        
    run_id = sys.argv[1]
    OUTPUT_CSV = os.path.join(DATASET_DIR, f"node_metrics_{run_id}.csv")
    
    # 1. Start one Tetragon log stream per node and drain into a shared queue.
    #    Using ds/tetragon only streams ONE pod's logs — security pods can land
    #    on any node, so we must explicitly tail every node's Tetragon pod.
    tetragon_event_queue = queue.Queue()
    tetragon_processes, tetragon_last_event_time = start_tetragon_streams(tetragon_event_queue)
    if not tetragon_processes:
        print("WARNING: No Tetragon pods found. Tetragon metrics will be 0.")
    
    # Start the background event processor thread
    processor_thread = threading.Thread(target=event_processor, args=(tetragon_event_queue,), daemon=True)
    processor_thread.start()

    # Start the heartbeat issuer thread to prevent idle stalls
    heartbeat_thread = threading.Thread(target=tetragon_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    # 2. Setup CSV Writer
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Define headers — 5 Tetragon numeric signals + syscall_feature_vector (for RF) + 6 Prometheus signals
        headers = [
            'timestamp', 'label', 'node',
            'avg_cpu', 'avg_mem', 'net_bytes_in', 'net_bytes_out',
            # Edge simulation: enp0s8 inter-node interface metrics
            'net_internal_bytes_in', 'net_internal_bytes_out',
            # Packet loss signal (packets dropped/sec)
            'net_drop_rate',
            # Edge simulation: seconds since last successful Prometheus scrape
            'last_successful_scrape_age_sec',
            # Tetragon numeric signals (used by both LSTM and RF)
            'exec_count', 'unique_process_count', 'tmp_exec_count',
            'outbound_connect_count', 'mining_port_count',
            # Tetragon event-type frequency map — JSON blob, explode for RF training:
            #   pd.json_normalize(df['syscall_feature_vector'].apply(json.loads)).fillna(0)
            'syscall_feature_vector',
        ]
        writer.writerow(headers)

        print(f"Collecting baseline metrics every {COLLECTION_INTERVAL_SEC} seconds. Writing to {OUTPUT_CSV}...")
        print("Press Ctrl+C to stop.")

        # Wait for Prometheus to stabilize (rates need multiple scrapes across all nodes to populate)
        print(f"\n[i] Waiting for Prometheus metrics to stabilize (expecting {EXPECTED_NODES} nodes)...")
        _stabilize_attempts = 0
        while _stabilize_attempts < 12: # wait up to ~2 minutes
            raw_prom_metrics = collect_prometheus_metrics()
            metrics_valid, _ = validate_metrics(raw_prom_metrics, EXPECTED_NODES, silent=True)
            if metrics_valid:
                print("[i] Prometheus metrics stable! Starting collection loop.\n")
                break
            time.sleep(10)
            _stabilize_attempts += 1

        if _stabilize_attempts >= 12:
            print("[!] Timeout waiting for Prometheus stabilization. Proceeding anyway, but expect transition rows.")

        # Track last successful Prometheus scrape time for the scrape-gap-age feature.
        # Initialized to now so the first row shows 0 (no gap yet).
        last_successful_scrape_time = time.time()

        try:
            while True:
                start_time = time.time()
                
                # ALARM: Check if any Tetragon eBPF stream has secretly died 
                # (e.g., due to ring buffer overflow or VM VM-resume timer breaks)
                dead_nodes = []
                for node_name, proc in tetragon_processes.items():
                    if proc.poll() is not None:
                        dead_nodes.append(node_name)
                
                if dead_nodes:
                    now = time.time()
                    for node in dead_nodes:
                        if now < tetragon_next_retry_time.get(node, 0):
                            continue # Still in backoff
                        
                        count = tetragon_restart_counts.get(node, 0)
                        backoff = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** count))
                        
                        print(f"\n[!!!] Tetragon stream DIED for {node}. (Failure #{count+1})")
                        print(f"      Attempting restart. If fails, will wait {backoff}s.")
                        
                        # Clean up old process if any
                        try:
                            tetragon_processes[node].terminate()
                        except:
                            pass
                            
                        new_proc, new_time = start_tetragon_streams(tetragon_event_queue, restart_node=node)
                        
                        if node in new_proc:
                            tetragon_processes.update(new_proc)
                            tetragon_last_event_time.update(new_time)
                            tetragon_restart_counts[node] = count + 1
                            tetragon_next_retry_time[node] = time.time() + backoff
                            tetragon_last_stabilized_at[node] = time.time()
                        else:
                            print(f"[!!!] Failed to restart {node}. Retrying in {backoff}s.")
                            tetragon_next_retry_time[node] = time.time() + backoff

                # STALL CHECK: detect streams that are alive (proc.poll()==None) but
                # producing no output — the wk-2 failure mode. This happens when the
                # export-stdout sidecar is CPU-throttled and can't drain the ring buffer.
                # (proc.poll() returns None for a stalled-but-alive process, so the
                # dead-stream detector above is completely blind to this.)
                now = time.time()
                stalled_nodes = [
                    node for node, proc in tetragon_processes.items()
                    if proc.poll() is None  # still alive
                    and (now - tetragon_last_event_time.get(node, now)) > STREAM_STALL_THRESHOLD_SEC
                ]
                if stalled_nodes:
                    now = time.time()
                    for node in stalled_nodes:
                        if now < tetragon_next_retry_time.get(node, 0):
                            continue # Still in backoff

                        count = tetragon_restart_counts.get(node, 0)
                        backoff = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** count))

                        print(f"\n[!!!] Tetragon stream STALLED on {node}. (Stall #{count+1})")
                        print(f"      Killing stalled stream and restarting. Backoff: {backoff}s.")
                        
                        # Kill the stalled process.
                        # Use SIGKILL (not SIGTERM) — on VirtualBox, kubectl blocks in a
                        # kernel pipe-read when the kubelet stream hangs and ignores SIGTERM.
                        try:
                            tetragon_processes[node].kill()  # SIGKILL
                            tetragon_processes[node].wait(timeout=3)
                        except Exception:
                            pass
                        
                        # Restart
                        new_proc, new_time = start_tetragon_streams(tetragon_event_queue, restart_node=node)
                        
                        if node in new_proc:
                           tetragon_processes.update(new_proc)
                           tetragon_last_event_time.update(new_time)
                           tetragon_restart_counts[node] = count + 1
                           tetragon_next_retry_time[node] = time.time() + backoff
                           tetragon_last_stabilized_at[node] = time.time()
                        else:
                           print(f"[!!!] Failed to restart STALLED node {node}. Retrying in {backoff}s.")
                           tetragon_next_retry_time[node] = time.time() + backoff
                           # Reset timer so we don't spam
                           tetragon_last_event_time[node] = now

                # HEARTBEAT CHECK: reset error counts if stream has been stable
                now = time.time()
                for node in list(tetragon_restart_counts.keys()):
                    if node not in dead_nodes and node not in stalled_nodes:
                        uptime = now - tetragon_last_stabilized_at.get(node, now)
                        if uptime > STABILITY_THRESHOLD and tetragon_restart_counts[node] > 0:
                            print(f"[i] Node {node} has been stable for {STABILITY_THRESHOLD}s. Resetting restart counter.")
                            tetragon_restart_counts[node] = 0
                            tetragon_next_retry_time[node] = 0

                # --- SWAP-ACCUMULATOR PATTERN ---
                # Atomically swap tetragon_state[0] to a brand-new empty dict.
                # The event_processor thread reads tetragon_state[0] on every event,
                # so it immediately starts writing new events into the fresh dict.
                # We hold the completed window's dict in `snapshot` (lock-free from
                # here on) and use it below — zero bleed from the new window.
                with tetragon_lock:
                    snapshot = tetragon_state[0]       # grab reference to completed-window dict
                    tetragon_state[0] = {}             # install fresh accumulator atomically

                # Collect from Prometheus and map IPs to Node names
                raw_prom_metrics = collect_prometheus_metrics()
                prom_metrics = map_prom_metrics_to_nodes(raw_prom_metrics)
                
                # Get all unique nodes we know about (Tetragon snapshot + Prometheus)
                all_nodes = set(snapshot.keys())
                for metric_dict in prom_metrics.values():
                    all_nodes.update(metric_dict.keys())

                # Use the exact same epoch timestamp format as the scenario_runner
                current_timestamp = int(time.time())
                
                # Update scrape-gap-age tracking
                metrics_valid, metrics_transition = validate_metrics(raw_prom_metrics, EXPECTED_NODES)
                if metrics_valid:
                    last_successful_scrape_time = time.time()
                scrape_age_sec = int(time.time() - last_successful_scrape_time)

                # Check for transition conditions
                is_transition = False
                if metrics_transition:
                    is_transition = True

                # Mark as transition or normal baseline
                current_label = "transition" if is_transition else "normal"

                # snapshot is the completed window — no lock needed, event_processor
                # is now writing into the new tetragon_state dict
                nodes_to_process = list(all_nodes)
                current_tetragon_data = {node: snapshot.get(node, {}) for node in nodes_to_process}

                for node in nodes_to_process:
                    node_short = node # Note: Prom metrics are already mapped to 'node' names now

                    t_state = current_tetragon_data.get(node, {})
                    
                    row = [
                        current_timestamp,
                        current_label,
                        node,
                        prom_metrics['avg_cpu'].get(node_short, 0.0),
                        prom_metrics['avg_mem'].get(node_short, 0.0),
                        prom_metrics['net_bytes_in'].get(node_short, 0.0),
                        prom_metrics['net_bytes_out'].get(node_short, 0.0),
                        # Edge simulation: inter-node link metrics (enp0s8)
                        prom_metrics['net_internal_bytes_in'].get(node_short, 0.0),
                        prom_metrics['net_internal_bytes_out'].get(node_short, 0.0),
                        prom_metrics['net_drop_rate'].get(node_short, 0.0),
                        # Edge simulation: scrape gap age (connectivity intermittency)
                        scrape_age_sec,
                        t_state.get('exec_count', 0),
                        len(t_state.get('unique_process_count', set())),
                        t_state.get('tmp_exec_count', 0),
                        t_state.get('outbound_connect_count', 0),
                        t_state.get('mining_port_count', 0),
                        json.dumps(t_state.get('syscall_feature_vector', {})),
                    ]
                    writer.writerow(row)
                
                # Flush to disk immediately
                file.flush()

                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, COLLECTION_INTERVAL_SEC - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping data collection.")
        finally:
            print("Cleaning up Tetragon streams...")
            for proc in tetragon_processes.values():
                proc.terminate()
            
            # Sync files from local collection dir to the final project dataset folder
            if COLLECTION_DIR != FINAL_DATASET_DIR:
                print(f"Syncing collected data to {FINAL_DATASET_DIR}...")
                import shutil
                try:
                    for f in os.listdir(COLLECTION_DIR):
                        if f.startswith(f"node_metrics_{run_id}"):
                            src = os.path.join(COLLECTION_DIR, f)
                            dst = os.path.join(FINAL_DATASET_DIR, f)
                            shutil.copy2(src, dst)
                            print(f"  ✓ Copied: {f}")
                except Exception as e:
                    print(f"Error syncing data: {e}")

if __name__ == "__main__":
    main()
