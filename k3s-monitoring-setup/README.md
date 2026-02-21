# K3s Monitoring Setup & ML Dataset Pipeline

Since you have a 3-node K3s cluster ready, this guide covers deploying Prometheus/Tetragon, orchestrating workloads, and extracting a clean dataset for training the Isolation Forest/LSTM Model.

## Prerequisites
Ensure `helm` and `kubectl` are installed, and `KUBECONFIG` points to your 3-node VM cluster. Install NTP (`systemd-timesyncd`) on all VMs to prevent time drift.

---

## 1. Install Monitoring Stack

### 1.1 Prometheus (kube-prometheus-stack)
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
kubectl create namespace monitoring
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring -f prometheus-values.yaml
```

### 1.2 Tetragon (eBPF Security)
```bash
helm repo add cilium https://helm.cilium.io
helm install tetragon cilium/tetragon \
  --namespace kube-system -f tetragon-values.yaml
```

### 1.3 Apply Tetragon Tracing Policy (Required)
Forces Tetragon to trace outbound TCP connections for mining port detection.
```bash
kubectl apply -f tcp-connect-policy.yaml
```

---

## 2. Dataset Generation Pipeline (Execution Order)

To generate the final dataset, strictly follow this execution order across three terminals.

### Terminal 1: Start Normal Workloads
Deploy the standard background noise (Nginx, Redis, APIs).
```bash
cd workloads
bash run_normal_baseline.sh
```

### Terminal 2: Start Telemetry Collection
Start the python script to continuously pull Prometheus + Tetragon metrics and write them to `node_metrics.csv`.
*Make sure to Port-Forward Prometheus first:*
```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 8080:9090 -n monitoring
```
```bash
# In another tab
cd k3s-monitoring-setup
pip install requests
python collect_baseline.py
```
*Leave this running!*

### Terminal 3: Inject Anomalies (Scenario Runner)
While `collect_baseline.py` is safely gathering data in Terminal 2, run the automated scenario injector. This will deploy faults (CPU stress, OOM loops) and security anomalies (/tmp execution), logging their exact start/end timestamps to `scenario_labels.csv`.
```bash
cd workloads
python scenario_runner.py
```
*Wait for this script to finish (~30 mins).*
*Once finished, you can safely `Ctrl+C` terminate Terminal 2 (`collect_baseline.py`).*

---

## 3. Labeling the Dataset

You now have raw unlabelled telemetry (`node_metrics.csv`) and a master list of incident windows (`scenario_labels.csv`).

To combine them into a ML-ready format, use the labeling script:

```bash
cd k3s-monitoring-setup
python label_dataset.py
```

### Note on Time Adjustments (Padding)
`label_dataset.py` automatically implements "Time Padding" to prevent inconsistencies:
- **Start Padding (+10s)**: It ignores the first 10 seconds of a fault to allow time for the anomaly (like a slow memory leak) to actually manifest in the system metrics.
- **End Padding (+15s)**: It extends the anomaly label for 15 seconds after the fault is deleted, ensuring residual CPU/Network spikes during "cluster recovery" aren't mistakenly labeled as normal data.

**Output:** `final_labeled_dataset.csv` (Ready for ML Training).
