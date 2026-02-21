# Byzantine / Anomalous Node Detection System - Project Report

## 1. Project Overview
The objective of this project is to build a research-grade, multi-modal anomaly detection system for Kubernetes edge clusters. The system focuses on identifying "Byzantine" or anomalous nodes by combining standard system metrics with deep kernel-level security telemetry. When an anomaly is detected, a custom Kubernetes Operator takes automated remediation actions (such as cordoning the compromised node).

## 2. Technologies Used
- **Infrastructure & Orchestration:** Kubernetes (initially `Kind`, migrated to a 3-node `K3s` VM cluster for realistic edge simulation), Docker.
- **Monitoring & Telemetry:** 
  - **Prometheus** (`kube-prometheus-stack` via Helm) for system resource metrics (CPU, Memory, Network).
  - **Tetragon** (eBPF-based security observability via Helm) for deep kernel-level telemetry (process executions, network connections, system calls).
- **Machine Learning & AI:** Python, PyTorch, `scikit-learn`.
  - **Models:** Isolation Forest (Baseline), LSTM Neural Networks (v2 sequence modeling).
- **Automation & Enforcement:** Python-based Custom Kubernetes Operator (`kopf` / `kubernetes-client`), `Bash` scripting, `helm`, `kubectl`.

## 3. System Architecture
The architecture transitions from a local Kind cluster to a robust 3-node K3s cluster (1 Master, 2 Workers) running on Virtual Machines to accurately simulate an edge environment. The key architectural components include:

### 3.1 Telemetry Gathering (The Sensory Layer)
- **Prometheus Scrapers:** Continuously collect node-level metrics (e.g., `avg_cpu`, `avg_mem`, `net_bytes_in/out`).
- **Tetragon eBPF Agents:** Installed as DaemonSets on all nodes, intercepting kernel operations in real-time. This provides granular data such as `exec_count`, `unique_process_count`, execution from temporary directories (`/tmp`, `/dev/shm`), and outbound connections to suspicious ports.

### 3.2 Data Collection Pipeline (`collect_baseline.py` & Scenario Runners)
A structured pipeline to generate realistic datasets comprising:
1. **Normal Activity:** Web traffic (Nginx), DB operations (Redis), and background crons.
2. **Controlled Faults:** System anomalies like CPU stressing (`stress-ng`), memory leaks, and network packet loss (`tc`).
3. **Security Anomalies:** Execution of malicious binaries, high-volume process spawning, and reverse shell/mining port connections detected via customized Tetragon egress policies and `process_kprobe` events.

### 3.3 Intelligence Layer (ML Models)
- **V1 (Baseline):** An Isolation Forest model trained on tabular metric aggregates to detect statistical outliers without needing temporal context.
- **V2 (Advanced):** An LSTM (Long Short-Term Memory) model built with PyTorch, designed to catch sequential, slow-burn anomalies by understanding the time-series nature of node telemetry. 

### 3.4 Enforcement Layer (Kubernetes Operator)
A custom Operator that acts as the "brain" in the cluster:
- **Detection:** Ingests the 10-second aggregated windows of combined Prometheus and Tetragon features, feeding them into the ML models.
- **Action:** If the model flags a node as 'Byzantine' or compromised, the Operator interacts with the Kubernetes API to immediately cordon the node, preventing new pods from scheduling and initiating recovery protocols.
- **Current State:** The operator is being tested locally (outside the cluster via `kubeconfig` and port-forwarding) to bypass slow build cycles caused by heavy ML dependencies (PyTorch) in Docker images.

## 4. Summary of Work Completed
1. **Initial Prototyping:** Deployed the Kubernetes Operator in a local Kind cluster, complete with CRDs, verifying its ability to cordon/uncordon nodes based on manual triggers.
2. **Model Development:** Developed and successfully drop-tested the baseline Isolation Forest model using sample text logs. Proceeded to develop the v2 LSTM model.
3. **Infrastructure Upgrade:** Moved to a simulated 3-node K3s setup for reliable multi-node testing.
4. **Monitoring Stack Deployment:** Successfully deployed Prometheus and Tetragon via Helm into the K3s cluster, ensuring proper 3-node visibility.
5. **eBPF Policy Customization:** Designed and applied custom Kubernetes network egress policies and Tetragon configuration to capture specific process executions (`process_kprobe`) and filter noise.
6. **Data Pipeline Construction:** Drafted the `collect_baseline.py` script and defined a robust methodology for automated, time-window-based labeling (10-second aggregation windows) to differentiate between "normal", "faulty", and "security_incident" states.

## 5. Future Work
1. **Dataset Generation & Refinement:** 
   - Execute the complete data collection pipeline on the 3-node cluster to generate a comprehensive, labeled dataset (CSV/Parquet format).
   - Solidify the feature engineering phase (combining Prometheus tabular metrics with Tetragon event counts).
2. **Training the v2 LSTM Model:** 
   - Utilize the newly generated multi-modal dataset to train, validate, and tune the PyTorch LSTM model for high accuracy and low false-positive rates.
3. **Operator Integration & Optimization:** 
   - Integrate the trained LSTM model's inference logic directly into the Operator codebase (`main.py`).
   - Address the Docker image size issue. Strategies include using lighter base images (e.g., Alpine or distroless), optimizing PyTorch installations (CPU-only wheels), or utilizing ONNX runtime for efficient, lightweight model inference inside the cluster.
4. **End-to-End Cluster Testing:** 
   - Deploy the optimized Operator back into the K3s cluster and run a live Red-Team simulation (injecting faults and malicious scripts) to verify automated, real-time node cordoning.
5. **Visualization Dashboard:** 
   - Build out the `dashboard` module to provide a visual representation of node health scores, incoming telemetry, and automated operator actions in real-time.
