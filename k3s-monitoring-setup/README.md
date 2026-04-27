# K3s Monitoring Setup & ML Dataset Pipeline

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 0 — VirtualBox Cluster Setup](#2-phase-0--virtualbox-cluster-setup)
3. [Phase 1 — Prerequisites](#3-phase-1--prerequisites)
4. [Phase 2 — Install Monitoring Stack](#4-phase-2--install-monitoring-stack)
5. [Phase 3 — Build Custom Docker Images](#5-phase-3--build-custom-docker-images)
6. [Phase 4 — Node Role Labels](#6-phase-4--node-role-labels-edge-simulation)
7. [Phase 5 — Run the Data Collection Pipeline](#7-phase-5--run-the-data-collection-pipeline)
8. [Phase 6 — Label the Dataset](#8-phase-6--label-the-dataset)
9. [Phase 7 — Inference Deployment](#9-phase-7--inference-deployment)
10. [Verification & Health Checks](#10-verification--health-checks)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Architecture Overview

**Cluster topology** (VirtualBox VMs, host-only network `192.168.56.0/24`):

| Node | Hostname | IP | Role | RAM | Edge Role |
|---|---|---|---|---|---|
| Node 1 | `k3s-wk1` | `192.168.56.10` | Control Plane (Server) | 3 GB | `coordinator` |
| Node 2 | `sw-wk2` | `192.168.56.11` | Worker (Agent) | 2 GB | `compute` |
| Node 3 | `sw-wk3` | `192.168.56.12` | Worker (Agent) | 2 GB | `sensor-gateway` |

**Data flow:**
```
Workloads (Nginx/Redis/API) + Fault Injector
          ↓ metrics (10s interval)
      Prometheus (NodePort: 9090)
      Tetragon  (eBPF kernel logs — 5 MVS signals)
          ↓
    collect_baseline.py ──→ node_metrics_<run_id>.csv
    scenario_runner.py  ──→ scenario_labels_<run_id>.csv
          ↓
    label_dataset.py ──→ final_labeled_dataset.csv
```

**Network interfaces per VM:**
- `enp0s3` — NAT adapter (internet access)
- `enp0s8` — Host-only adapter (inter-node communication, static IP)

---

## 2. Phase 0 — VirtualBox Cluster Setup

> **Skip this phase if your cluster is already running.** Jump to [Phase 1](#3-phase-1--prerequisites).

### 2.1 Fix Static Network (All 3 VMs)

VirtualBox Ubuntu Server VMs reset networking on reboot if cloud-init is managing it. Fix it permanently.

**Run on every node:**

```bash
# Step 1: Disable cloud-init network management
sudo nano /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
```
Add this single line and save:
```
network: {config: disabled}
```

```bash
# Step 2: Remove the cloud-init netplan file
sudo rm /etc/netplan/50-cloud-init.yaml

# Step 3: Create a stable static IP config
sudo nano /etc/netplan/01-static.yaml
```

Paste the following — **change the IP address per node**:
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp0s3:
      dhcp4: true        # NAT interface — keep DHCP for internet
    enp0s8:
      dhcp4: no
      addresses:
        - 192.168.56.10/24   # Change to .11 on Node 2, .12 on Node 3
```

```bash
# Step 4: Apply and reboot
sudo netplan generate
sudo netplan apply
sudo reboot
```

After reboot, verify: `ip a` — the static IP should persist.

---

### 2.2 Install K3s on Control Plane (Node 1 only)

This setup uses a **lean K3s config** to disable add-ons that waste ~140 MB RAM and generate BPF noise in Tetragon. The config file must be in place **before** running the install script.

**Step 1 — Drop the lean server config in place:**

```bash
sudo mkdir -p /etc/rancher/k3s

# Copy from shared mount (adjust path to wherever this repo lives in your VM)
sudo cp /mnt/shared/k3s-monitoring-setup/k3s-server-config.yaml /etc/rancher/k3s/config.yaml
```

What this config disables (see `k3s-server-config.yaml` for full comments):

| Add-on | RAM saved | Why disabled |
|---|---|---|
| `traefik` | ~80 MB | Not needed — NodePort is used instead |
| `servicelb` | ~30 MB | Not needed — same reason |
| `metrics-server` | ~30 MB | Use Prometheus instead of `kubectl top` |

> **Note:** `kubectl top nodes/pods` will NOT work without `metrics-server`. Use Prometheus queries or `kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes` as an alternative.

**Step 2 — Install K3s:**

```bash
curl -sfL https://get.k3s.io | sh -s - server \
  --node-ip=192.168.56.10 \
  --disable traefik
```

> The `--disable traefik` flag is redundant if `k3s-server-config.yaml` is in place, but kept for clarity.

**Step 3 — Save the cluster join token for worker nodes:**
```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

---

### 2.3 Join Workers

**Node 2 (`192.168.56.11`):**
```bash
curl -sfL https://get.k3s.io | sh -s - agent \
  --server https://192.168.56.10:6443 \
  --token <token-from-above> \
  --node-ip=192.168.56.11
```

**Node 3 (`192.168.56.12`):**
```bash
curl -sfL https://get.k3s.io | sh -s - agent \
  --server https://192.168.56.10:6443 \
  --token <token-from-above> \
  --node-ip=192.168.56.12
```

---

### 2.4 Verify Cluster

From Node 1:
```bash
sudo k3s kubectl get nodes
```

Expected output — all nodes `Ready`:
```
NAME      STATUS   ROLES                  AGE   VERSION
k3s-wk1   Ready    control-plane,master   2m    v1.x.x
sw-wk2    Ready    <none>                 1m    v1.x.x
sw-wk3    Ready    <none>                 1m    v1.x.x
```

**Verify that disabled add-ons are NOT running:**
```bash
kubectl get pods -n kube-system
# Expected: NO traefik-* or svclb-* pods
kubectl get helmchart -n kube-system
# Expected: traefik and traefik-crd are NOT listed
```

---

## 3. Phase 1 — Prerequisites

Run all steps below **from Node 1** unless stated otherwise.

### 3.1 Configure kubectl Without sudo

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
source ~/.bashrc
```

Test: `kubectl get nodes` (no sudo required).

---

### 3.2 Install Helm

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
rm get_helm.sh
```

Test: `helm version`

---

### 3.3 Prevent Time Drift (All 3 Nodes)

> **Critical:** The LSTM model training depends on precise 10-second timestamps. VM time drift corrupts the dataset alignment between Prometheus scrapes and scenario labels.

**Run on every node (Node 1, 2, and 3):**
```bash
sudo apt update && sudo apt install -y systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-ntp true
```

Verify sync: `timedatectl status` — look for `System clock synchronized: yes`.

---

### 3.4 Install Python Dependencies (Node 1)

```bash
pip install requests
```

---


## 4. Phase 2 — Install Monitoring Stack

### 4.1 Install Prometheus (kube-prometheus-stack)

The custom `prometheus-values.yaml` configures:
- **10s scrape/evaluation interval** — matches the LSTM training sampling rate
- **Metric filtering** — only the 4 metric families used by `collect_baseline.py` are kept (reduces memory + TSDB pressure)
- **Memory limits** — Prometheus: 700Mi, Alertmanager: 200Mi, node-exporter: 80Mi

```bash
# Add repo and create namespace
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring

# Install with custom values
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f prometheus-values.yaml
```

Wait for Prometheus to become ready:
```bash
kubectl rollout status deployment/prometheus-kube-prometheus-operator -n monitoring
kubectl get pods -n monitoring
```

> **To apply config changes later** (e.g., after editing `prometheus-values.yaml`):
> ```bash
> helm upgrade prometheus prometheus-community/kube-prometheus-stack \
>   --namespace monitoring -f prometheus-values.yaml
> ```

---

### 4.2 Expose Prometheus via Port-Forward (with Auto-Restart Watchdog)

Data collection scripts access Prometheus at `http://localhost:9090`. `kubectl port-forward` is used to tunnel it locally, but the tunnel can crash silently — causing missed scrapes and gaps in your dataset. The watchdog script below monitors the tunnel and restarts it automatically.

**Step 1 — Create the watchdog script:**

```bash
cat > /mnt/shared/k3s-monitoring-setup/pf-watchdog.sh << 'EOF'
#!/bin/bash
# Port-forward watchdog for Prometheus
# Restarts the tunnel immediately if it crashes, so no data is lost.

NAMESPACE="monitoring"
SERVICE="svc/prometheus-kube-prometheus-prometheus"
LOCAL_PORT=9090
REMOTE_PORT=9090
LOGFILE="/tmp/pf-watchdog.log"
PIDFILE="/tmp/pf-watchdog.pid"

echo $$ > "$PIDFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started (PID $$)" | tee -a "$LOGFILE"

cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog stopping — killing port-forward..." | tee -a "$LOGFILE"
    kill "$PF_PID" 2>/dev/null
    rm -f "$PIDFILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting port-forward..." | tee -a "$LOGFILE"

    kubectl port-forward "$SERVICE" "$LOCAL_PORT:$REMOTE_PORT" -n "$NAMESPACE" \
        >> "$LOGFILE" 2>&1 &
    PF_PID=$!

    # Wait for the tunnel to be ready
    sleep 3

    # Confirm it's actually listening
    if ! ss -tlnp | grep -q ":$LOCAL_PORT"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: port-forward failed to bind — retrying in 5s" | tee -a "$LOGFILE"
        kill "$PF_PID" 2>/dev/null
        sleep 5
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tunnel UP on localhost:$LOCAL_PORT (PID $PF_PID)" | tee -a "$LOGFILE"

    # Block until port-forward dies
    wait "$PF_PID"
    EXIT_CODE=$?

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port-forward exited (code $EXIT_CODE) — restarting in 2s..." | tee -a "$LOGFILE"
    sleep 2
done
EOF

chmod +x /mnt/shared/k3s-monitoring-setup/pf-watchdog.sh
```

**Step 2 — Start the watchdog in a dedicated terminal (Terminal 0):**

> Open a new terminal session on Node 1 **before** starting data collection. Keep it running for the entire pipeline duration.

```bash
bash /mnt/shared/k3s-monitoring-setup/pf-watchdog.sh
```

You should see output like:
```
[2026-03-19 12:00:01] Watchdog started (PID 4521)
[2026-03-19 12:00:01] Starting port-forward...
[2026-03-19 12:00:04] Tunnel UP on localhost:9090 (PID 4529)
```

**Step 3 — Verify the tunnel is working:**

```bash
curl -s http://localhost:9090/api/v1/query?query=up | python3 -m json.tool | head -20
```

Expected: JSON response with `"status": "success"` and all node-exporter targets showing `"1"`.

**Step 4 — Monitor the watchdog log (optional, in another pane):**

```bash
tail -f /tmp/pf-watchdog.log
```

**Step 5 — Stop the watchdog** (after data collection is complete):

```bash
# Send SIGTERM to gracefully shut down both watchdog and port-forward
kill $(cat /tmp/pf-watchdog.pid)
```

> **Note:** `PROMETHEUS_URL` in `collect_baseline.py` should remain `http://localhost:9090` — no changes needed.

---

### 4.3 Install Tetragon (eBPF Kernel Tracing — MVS Config)

Tetragon provides kernel-level process execution and network connection events via eBPF. The custom `tetragon-values.yaml` is tuned to **minimum viable signals (MVS)** — only the 5 features that are actually used in the LSTM dataset.

**Tetragon signals collected:**

| Signal | CSV Column | What it captures | Used by |
|---|---|---|---|
| Process execution count | `exec_count` | Total new processes started per 10s window | LSTM + RF |
| Unique binaries | `unique_process_count` | Count of distinct executables seen | LSTM + RF |
| Temp-dir execution | `tmp_exec_count` | Executions from `/tmp` or `/dev/shm` (malware indicator) | LSTM + RF |
| Outbound TCP connections | `outbound_connect_count` | Total new TCP connects (`tcp_connect` kprobe) | LSTM + RF |
| Mining port connections | `mining_port_count` | TCP connects to ports 3333, 4444, 5555, 6666, 7777, 8332, 8333, 14433, 14444 | LSTM + RF |
| Event-type frequency map | `syscall_feature_vector` | JSON: `{"process_exec": N, "process_kprobe": M, ...}` per 10s window | RF only |

> **RF usage for `syscall_feature_vector`:** explode the JSON blob into per-type numeric columns before training:
> ```python
> syscall_df = pd.json_normalize(df["syscall_feature_vector"].apply(json.loads)).fillna(0)
> syscall_df.columns = [f"syscall_{c}" for c in syscall_df.columns]
> ```
> Particularly discriminative for `security_tmp_exec` and `security_suspicious_network` fault classes.

**Resource profile (per node):**

| Resource | Before (old config) | After (current config) |
|---|---|---|
| Memory request | 200 Mi | 128 Mi |
| Memory limit | 400 Mi | 256 Mi |
| CPU request | 100 m | 50 m |
| CPU limit | 300 m | 150 m |
| BPF execve_map | 32768 entries | **32768 entries** (unchanged — see note below) |
| BPF tcp_map | 32768 entries | 8192 entries |

> **Why `execve_map` stays at 32768:** The `security_high_process` fault spawns a burst of processes on `sw-wk2` (compute node). With a smaller map (e.g. 8192), the ring buffer overflows during that fault and the Tetragon log stream on wk-2 **silently dies** — this was the observed wk-2-only failure. It is a ring buffer overflow, not an OOM kill. `tcp_map` can safely be halved since only connection counts are recorded.

```bash
helm repo add cilium https://helm.cilium.io
helm repo update
helm install tetragon cilium/tetragon \
  --namespace kube-system \
  -f tetragon-values.yaml
```

Wait for Tetragon DaemonSet to roll out:
```bash
kubectl rollout status daemonset/tetragon -n kube-system
```

**To apply the MVS config to an existing Tetragon installation:**
```bash
helm upgrade tetragon cilium/tetragon \
  --namespace kube-system \
  -f tetragon-values.yaml
kubectl rollout status daemonset/tetragon -n kube-system
```

After upgrade, verify memory is within MVS limits:
```bash
kubectl top pods -n kube-system | grep tetragon
# Expected: each pod < 200Mi
```

---

### 4.4 Apply Tetragon Tracing Policy

This policy instructs Tetragon to trace outbound TCP connections — required for `outbound_connect_count` and `mining_port_count` in `collect_baseline.py`:

```bash
kubectl apply -f tcp-connect-policy.yaml
```

Verify the policy is active:
```bash
kubectl get tracingpolicy
```

---

## 5. Phase 3 — Build Custom Docker Images

Three custom images are required for fault injection. They must be built inside the VM and imported into k3s's containerd (k3s does **not** use the Docker daemon for scheduling).

> **All build commands run on Node 1** unless a fault targets a specific worker node. Import must happen on every node that will run the image.

---

### 5.1 `crash-loop-stress:latest`

Used by: `fault-crash-loop.yaml`, `background-pressure.yaml`

```bash
# Build context is in docker_builds/crash-loop-image/
cd k3s-monitoring-setup/docker_builds/crash-loop-image/

docker build -t crash-loop-stress:latest .

# Import into k3s containerd (run on EACH node that needs it)
docker save crash-loop-stress:latest | sudo k3s ctr images import -
```

Verify import:
```bash
sudo k3s ctr images list | grep crash-loop
# Expected: crash-loop-stress:latest
```

---

### 5.2 `suspicious-network:latest`

Used by: `fault-network-chaos.yaml`, `security-suspicious-network.yaml`

```bash
cd k3s-monitoring-setup/docker_builds/security-suspicious-image/

docker build -t suspicious-network:latest .
docker save suspicious-network:latest | sudo k3s ctr images import -
```

Verify:
```bash
sudo k3s ctr images list | grep suspicious
```

---

### 5.3 `security-tmp-exec:latest`

Used by: `security-tmp-exec.yaml`

The image contains a polymorphic payload script (`run.sh`) that simulates execution from `/tmp`, `/dev/shm`, and `/var/tmp` — traced by Tetragon.

```bash
cd k3s-monitoring-setup/docker_builds/security-tmp-exec-image/

docker build -t security-tmp-exec:latest .
docker save security-tmp-exec:latest | sudo k3s ctr images import -
```

Verify:
```bash
sudo k3s ctr images list | grep security-tmp
```

> **Important:** For faults injected on worker nodes, you must also import images on those nodes. Run the `docker save ... | sudo k3s ctr images import -` command on `sw-wk2` and `sw-wk3` as well (copy images via `scp` or repeat the build).

---

## 6. Phase 4 — Node Role Labels (Edge Simulation)

Workload YAMLs use `nodeSelector` to pin each service to a specific node, creating **heterogeneous per-node baselines** — a key characteristic of real edge clusters where different nodes do different jobs.

> **This is a one-time setup step. Run on Node 1:**

```bash
# Control plane — acts as the state broker / coordinator
kubectl label node k3s-wk1 edge-role=coordinator

# Worker 1 — acts as the data processing compute node
kubectl label node sw-wk2 edge-role=compute

# Worker 2 — acts as the HTTP sensor data ingestion gateway
kubectl label node sw-wk3 edge-role=sensor-gateway
```

Verify:
```bash
kubectl get nodes --show-labels | grep edge-role
```

| Workload | Pinned To | Reason |
|---|---|---|
| `redis-baseline` | `k3s-wk1` (coordinator) | State / message broker |
| `api-baseline` | `sw-wk2` (compute) | Data processing API |
| `nginx-baseline` | `sw-wk3` (sensor-gateway) | HTTP ingestion endpoint |
| `background-pressure` | `sw-wk2`, `sw-wk3` (workers only) | Workers have headroom; control plane is already >50% |

> **Warning:** If these labels are missing when you run `run_normal_baseline.sh`, the pinned deployments (`nginx`, `api`, `redis`) will stay in `Pending` state indefinitely.

---

## 7. Phase 5 — Run the Data Collection Pipeline

### Load Budget — Why Controlled Traffic Matters

The goal is **30–50k rows** (≈ 28–46 hours of collection at 3 nodes × 10s intervals). For the LSTM to reliably separate fault classes from the normal baseline, the normal class must be stable and low-CPU. The controlled traffic generator enforces this budget:

| Traffic Mode | Steady rate | Peak connections | Effect on normal class |
|---|---|---|---|
| **`traffic-generator.yaml`** (full load) | 40 req/s | c=30 burst, 2 replicas | CPU spikes contaminate normal class — avoid for dataset runs |
| **`traffic-generator-controlled.yaml`** ✅ | 5 req/s | c=2, 1 replica | Workers stay <10% CPU — fault signals are cleanly separable |

**Target CPU budget during normal baseline:**

| Node | Expected baseline CPU | Max acceptable |
|---|---|---|
| `k3s-wk1` (control plane) | 15–30% | 40% |
| `sw-wk2` (compute) | 5–12% | 20% |
| `sw-wk3` (sensor-gateway) | 5–12% | 20% |

If any node exceeds the max during a normal baseline window, the rows will be noisier and fault detection harder. Stop the pipeline, check for runaway pods, and restart.

---

Open **three separate terminal sessions** on Node 1 and run each step in order.

---

### Terminal 1 — Start Normal Baseline Workloads

```bash
cd workloads
bash run_normal_baseline.sh
```

This deploys: `background-pressure` (workers only) → `nginx-baseline` → `redis-baseline` → `api-baseline` → `cron-logger` → **controlled traffic generator** (see next step).

Wait for everything to be `Running`:
```bash
kubectl get pods -o wide
```

**Important — use the controlled traffic generator** for dataset collection:
```bash
# Apply controlled generator INSTEAD of the default traffic-generator.yaml
kubectl apply -f workloads/traffic-generator-controlled.yaml
```

Verify CPU stays within budget before starting collection:
```bash
# Wait 60s for load to stabilize, then check
sleep 60 && kubectl top nodes
# Expected: workers < 10% CPU
```

Leave this running. Move to Terminal 2.

---

### Terminal 2 — Start Telemetry Collection

This script continuously pulls Prometheus and Tetragon metrics every 10 seconds and writes them to `dataset/node_metrics_<run_id>.csv`.

```bash
cd workloads
python scenario_runner.py
```

> `scenario_runner.py` automatically launches `collect_baseline.py` as a background process, so you don't need to start it manually. It will:
> 1. Start `collect_baseline.py` in the background (linked to the run ID)
> 2. Run through all anomaly scenarios in randomized order
> 3. Terminate `collect_baseline.py` after all scenarios complete

**Scenario timeline (approximate):**

| Phase | Duration |
|---|---|
| Initial normal baseline | 3 min |
| Per anomaly: active fault | ~2 min |
| Per anomaly: transition buffer | 1 min |
| Between anomalies: normal baseline | 2 min |
| **Total (8 anomalies)** | **~45 min** |

The script writes two files to `dataset/`:
- `node_metrics_<run_id>.csv` — raw telemetry (Prometheus + Tetragon, **no syscall_feature_vector column**)
- `scenario_labels_<run_id>.csv` — ground-truth fault windows with timestamps

> **Anomaly types injected:** `cpu_stress`, `memory_leak`, `network_chaos`, `crash_loop`, `edge_network_flap`, `security_tmp_exec`, `security_high_process`, `security_suspicious_network`

**Dataset target:** Run the scenario pipeline **3–4 times** with different random seeds to reach 30–50k rows. Each run produces ~3k rows (3 nodes × ~1000 samples/45 min).

---

### Terminal 3 — Monitor Progress (Optional)

Watch pods appear and disappear as faults are injected:
```bash
watch -n 5 kubectl get pods -o wide
```

Watch collector output:
```bash
tail -f dataset/node_metrics_*.csv
```

Check row count progress:
```bash
# Count rows across all collected CSVs (subtract 1 header per file)
wc -l dataset/node_metrics_*.csv
```

---

### Stopping the Pipeline

If you need to stop mid-run:
```bash
# Stop scenario runner (Terminal 2): Ctrl+C

# Clean up any stuck fault pods
kubectl delete pods -l role=fault-injection --grace-period=0 --force

# Stop baseline workloads (Terminal 1)
cd workloads
bash stop_normal_baseline.sh
```

---

## 8. Phase 6 — Label the Dataset

After the scenario runner completes, you have raw telemetry and time-window labels in `dataset/`. The labeling script merges them into a single ML-ready CSV.

```bash
cd k3s-monitoring-setup
python label_dataset.py
```

**Output:** `final_labeled_dataset.csv`

### How Labeling Works

The script joins `node_metrics_<run_id>.csv` with `scenario_labels_<run_id>.csv` on timestamp ranges, with two time padding adjustments:

| Padding | Duration | Reason |
|---|---|---|
| **Start padding** | +10s | Lets slow anomalies (e.g. memory leaks) ramp up before labeling starts |
| **End padding** | +15s | Captures residual spikes during cluster recovery after fault deletion |

Labels in the final dataset:

| Label | Meaning |
|---|---|
| `normal` | Clean baseline activity |
| `cpu_stress` / `memory_leak` / ... | Active fault window |
| `edge_network_flap` | Inter-node link degradation fault |
| `security_tmp_exec` / `security_suspicious_network` / ... | Security anomaly window |
| `transition` | Monitoring gap or cluster instability — excluded from training |

---

## 9. Phase 7 — Inference Deployment

After offline training, deploy the inference pod to worker nodes only.

Label the worker nodes for inference targeting:
```bash
kubectl label node sw-wk2 role=inference
kubectl label node sw-wk3 role=inference
```

Deploy the inference pod:
```bash
kubectl apply -f workloads/inference-deployment.yaml
```

The inference pod:
- Queries Prometheus every 10s (same interval as training data)
- Maintains a sliding window of metrics
- Runs LSTM forward pass (CPU-only, `torch.no_grad()`, batch size 1)
- Emits Kubernetes Events when anomaly is detected or drift is flagged

Monitor inference events:
```bash
kubectl get events --field-selector reason=AnomalyDetected
```

---

## 10. Verification & Health Checks

### Check Cluster Health
```bash
kubectl get nodes
# Note: kubectl top nodes requires metrics-server (disabled in lean config)
# Use Prometheus instead: curl http://localhost:9090/api/v1/query?query=node_memory_MemAvailable_bytes
```

### Verify K3s Lean Config (Add-Ons Disabled)
```bash
kubectl get pods -n kube-system
# Expected: NO traefik-* or svclb-* pods

kubectl get helmchart -n kube-system
# Expected: traefik and traefik-crd are NOT listed
```

### Check Monitoring Stack
```bash
kubectl get pods -n monitoring
kubectl get pods -n kube-system | grep tetragon
```

### Check Tetragon Memory (MVS Config)
```bash
kubectl top pods -n kube-system | grep tetragon
# Expected: each tetragon pod < 200Mi (MVS config limit is 256Mi)
```

### Check Prometheus is Scraping
```bash
# Via port-forward (watchdog must be running)
curl http://localhost:9090/api/v1/query?query=up
```

Expected: all three node-exporter targets show `"value": [<ts>, "1"]`.

### Check Tetragon is Receiving Events
```bash
kubectl logs -n kube-system ds/tetragon -c export-stdout --tail=20
```

You should see JSON lines with `process_exec` or `process_kprobe` events.

### Check Node Labels
```bash
kubectl get nodes --show-labels | grep edge-role
```

### Check Background Pressure (Workers Only)
```bash
kubectl get pods -l app=background-pressure -o wide
# Expected: 2 pods — one on sw-wk2, one on sw-wk3
```

### Validate Dataset Output
```bash
# Check CSV headers — should include syscall_feature_vector at the end
head -1 dataset/node_metrics_*.csv

# Expected headers:
# timestamp,label,node,avg_cpu,avg_mem,net_bytes_in,net_bytes_out,
# net_internal_bytes_in,net_internal_bytes_out,last_successful_scrape_age_sec,
# exec_count,unique_process_count,tmp_exec_count,outbound_connect_count,mining_port_count,
# syscall_feature_vector

# Count rows per label
python3 -c "
import csv
from collections import Counter
with open('k3s-monitoring-setup/final_labeled_dataset.csv') as f:
    labels = [row['label'] for row in csv.DictReader(f)]
print(Counter(labels))
"

# Target: 30,000–50,000 total rows, normal class should be 40–60% of total
```

### Verify Controlled Traffic Load
```bash
# Confirm traffic generator is in controlled mode
kubectl get deployment traffic-generator -o jsonpath='{.spec.replicas}'
# Expected: 1 (controlled = 1 replica; full-load = 2 replicas)

kubectl get pods -l variant=controlled
# Expected: 1 pod running
```

---

## 11. Troubleshooting

### Pods stuck in `Pending`
```bash
kubectl describe pod <pod-name>
```
**Likely cause:** Missing `edge-role` node label. Re-run the label commands in [Phase 4](#6-phase-4--node-role-labels-edge-simulation).

### `collect_baseline.py` shows 0.0 for all metrics
Prometheus likely isn't reachable. Verify:
```bash
curl http://localhost:9090/api/v1/query?query=up
```
If NodePort is used, check the port mapping: `kubectl get svc -n monitoring`.

### Tetragon logs show no events
The `tcp-connect-policy.yaml` may not be applied. Run:
```bash
kubectl apply -f k3s-monitoring-setup/tcp-connect-policy.yaml
kubectl get tracingpolicy
```

### Tetragon stream silently dies or produces 0s on `sw-wk2` only

**Root cause: `export-stdout` sidecar CPU throttle, not OOM or ring buffer size.**

`sw-wk2` is the `compute` node — it runs `api-baseline` and receives `security_high_process` and `security_suspicious_network` fault injections. These generate hundreds of events/sec. When the export-stdout sidecar hits its CPU limit, it can't drain the BPF ring buffer fast enough → Tetragon drops events → stream output slows to zero.

**Critically: `kubectl logs -f` stays alive** (`proc.poll() == None`) so the original dead-stream detector was completely blind to this. The updated `collect_baseline.py` now also runs a **stall detector** that triggers if a stream produces no events for >60s:
```
[!!!] WARNING: Tetragon stream STALLED (alive but silent >60s) on: sw-wk2
[!!!] Likely cause: export-stdout sidecar CPU-throttled under fault burst.
```

**Three fixes applied (all already in the current config):**

| Fix | What changed | Effect |
|---|---|---|
| **1. Namespace-scoped TracingPolicy** | `tcp-connect-policy.yaml` added `podSelector` for `default` + `kube-system` only | Cuts ~60–70% of tcp_connect event volume by ignoring host processes (containerd, sshd, kubectl, etc.) |
| **2. Higher CPU limit** | `tetragon-values.yaml` CPU limit: `150m → 500m` | export-stdout sidecar has enough headroom to drain bursts without stalling |
| **3. Stall detector** | `collect_baseline.py` tracks `last_event_time` per node | Catches silent hangs that `proc.poll()` misses |

**Diagnosis:**
```bash
# Is the Tetragon pod alive? (alive ≠ stream working)
kubectl get pods -n kube-system -o wide | grep tetragon

# Is the stream actually producing events?
kubectl logs -n kube-system <tetragon-pod-on-wk2> -c export-stdout --tail=5
# If no output for >30s during active workloads → stalled.

# Check CPU throttle events for the Tetragon pod
kubectl describe pod -n kube-system <tetragon-pod-on-wk2> | grep -A5 "Limits\|Throttling"
```

**Recovery (if stall still occurs despite fixes):**
```bash
kubectl rollout restart daemonset/tetragon -n kube-system
kubectl rollout status daemonset/tetragon -n kube-system
# Wait 30s, then restart collect_baseline.py with the same run_id
```

---

### Tetragon pods using too much memory
If Tetragon pods exceed 256Mi after the MVS upgrade, the BPF ring buffers may still be holding old map sizes. Force a full rollout:
```bash
helm upgrade tetragon cilium/tetragon --namespace kube-system -f k3s-monitoring-setup/tetragon-values.yaml
kubectl rollout restart daemonset/tetragon -n kube-system
kubectl rollout status daemonset/tetragon -n kube-system
```

### Traefik or ServiceLB still running after lean install
The `k3s-server-config.yaml` must be in `/etc/rancher/k3s/config.yaml` **before** K3s is installed or restarted. If K3s was already installed:
```bash
# Verify the config is in place
cat /etc/rancher/k3s/config.yaml

# Restart K3s to apply the config (will briefly interrupt the cluster)
sudo systemctl restart k3s

# Verify add-ons are gone
kubectl get helmchart -n kube-system
```

### `edge_network_flap` fault pod fails to start
The pod needs `NET_ADMIN` capability and `hostNetwork: true`. If it's failing, check:
```bash
kubectl describe pod -l app=fault-network-flap
```
Common issue: `iproute2` installs slowly on first run — the pod may take 30–60s before `tc` is available. This is expected.

### Image pull errors (`ErrImageNeverPull`)
The custom images must be imported into k3s containerd on each node that will run them. See [Phase 3](#5-phase-3--build-custom-docker-images) for import commands. Verify with:
```bash
sudo k3s ctr images list | grep -E "crash-loop|suspicious|security-tmp"
```

### Clock drift causing dataset misalignment
```bash
timedatectl status    # Run on all nodes
```
If `System clock synchronized: no`, re-run the time sync setup in [Phase 1.3](#33-prevent-time-drift-all-3-nodes).

### Normal baseline CPU too high (>20% on workers before faults)
```bash
# Check which pods are consuming CPU
kubectl top pods -o wide

# If traffic-generator is in full-load mode, switch it:
kubectl delete deployment traffic-generator
kubectl apply -f workloads/traffic-generator-controlled.yaml

# Wait 60s, then re-check
sleep 60 && kubectl top nodes
```

---

*Last updated: 2026-04-11 | Cluster: k3s v1.x | Monitoring: kube-prometheus-stack + Tetragon (MVS)*
