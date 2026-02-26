# Anomaly Detection and Fault Identification in Kubernetes Clusters

## Overall Project Structure
This work presents an end-to-end real-time anomaly detection and fault identification framework for Kubernetes clusters, integrating machine learning experimentation with practical deployment constraints.

The system is organized into three tightly coupled layers:

**1. Offline Research Layer**
Model development, controlled experimentation, and evaluation under realistic infrastructure conditions.

**2. Model Artifact Layer**
Reproducible packaging of trained models and preprocessing components ensuring consistency between offline training and online inference.

**3. Online Deployment Layer**
Real-time anomaly detection deployed within a resource-constrained k3s cluster, enabling live monitoring and fault flagging.

The following phases define the complete system pipeline.

---

## Phase 1 — Infrastructure Setup (Cluster Foundation)

### 1.1 k3s Cluster Deployment
The experimental environment consists of:
*   **1 master node**
*   **2 worker nodes**

Each VM specifications:
*   2 CPU cores
*   2 GB RAM
*   25 GB storage

Installed components:
*   k3s
*   metrics-server
*   Lightweight Prometheus stack
*(Note: Prometheus retention is limited to 1–2 days due to storage constraints.)*

### 1.2 Prometheus Configuration
Prometheus serves as the unified metrics source for both dataset generation and online inference.
Configuration requirements:
*   Fixed scrape interval (10 seconds)
*   Consistent temporal resolution
*   Collection of: CPU utilization, Memory usage, Network throughput, Pod restart counts, Node load metrics.
*(Critical Requirement: Offline training sampling frequency must exactly match the Prometheus scrape interval.)*

### 1.3 Automated Fault Injection
A scenario runner generates controlled fault executions. Each execution represents an independent experimental run and records:
*   `run_id`
*   `fault_type`
*   `intensity_seed`
*   start/end timestamps
*   execution metadata

Each run is stored independently (`dataset/run_<uuid>.csv`), enabling leakage-free evaluation.

---

## Phase 2 — Dataset Engineering

### 2.1 Data Collection Pipeline
For every experimental run:
1.  Faults are injected.
2.  Metrics are queried from Prometheus APIs.
3.  Time-series data is stored.
4.  Metadata annotations are attached.

### 2.2 Temporal Alignment & Cleaning
Processing ensures LSTM compatibility:
*   Missing value handling
*   Forward filling short gaps
*   Removal of corrupted intervals
*   Fixed timestep spacing

### 2.3 Feature Normalization
Feature scaling parameters are computed **only** using normal training data.
Saved artifact: `scaler.pkl`
*(Note: Online recomputation is strictly prohibited.)*

### 2.4 Sequence Construction
Sliding-window segmentation converts metrics into temporal samples: `(window_size, num_features)`
Design constraints (resource-aware):
*   Window size: 30–60 timesteps
*   Features < 15
*   Lightweight representations

Each sequence stores: `run_id`, `label`, `fault_type`

---

## Phase 3 — Model Training
Two complementary models are trained.

### 3.1 LSTM Autoencoder (Primary Model)
*Purpose:* Learn normal cluster dynamics and detect deviations.
*   **Training data:** Normal sequences only
*   **Architecture:** Single LSTM encoder–decoder (Hidden dimension: 32–64, Dropout ≈ 0.1)
*   **Loss:** Mean Squared Error
*   **Threshold Selection:** Computed from validation normal data (`threshold = mean + 3 × std`). Threshold remains fixed across all experiments and deployment.

### 3.2 LSTM Classifier (Baseline Model)
*Purpose:* Provide supervised fault identification baseline.
*   **Architecture:** Single LSTM layer, Fully connected classifier, Softmax output
*   **Training:** Standard cross-entropy, Weighted cross-entropy (imbalance-aware)

---

## Phase 4 — Reduced Structured Evaluation
To ensure feasibility, evaluation is restricted to six core experiments.

*   **AE-1 — Run-Based Generalization:** Train on normal data from 80% runs; Test on remaining runs. Evaluates anomaly detection across unseen executions.
*   **AE-2 — Fault-Type Holdout:** One representative fault type is excluded during training. Tests detection of unseen anomaly behavior.
*   **AE-3 — Online Cluster Validation:** Autoencoder deployed in k3s cluster. Measures detection latency, false positives, recovery detection time.
*   **CLF-1 — Unweighted Classifier:** Baseline supervised learning under natural class imbalance.
*   **CLF-2 — Weighted Cross-Entropy Classifier:** Evaluates imbalance mitigation effectiveness.
*   **CLF-3 — Fault Holdout Classifier:** Demonstrates limitation of supervised models under unseen faults.

---

## Phase 5 — Model Artifact Packaging
All deployment artifacts are exported to `model/`:
*   `ae_model.pt`
*   `classifier_model.pt`
*   `scaler.pkl`
*   `threshold.json`
*   `window_size.json`
*   `feature_order.json`

*(Feature ordering must exactly match Prometheus query output.)*

---

## Phase 6 — Online Deployment (k3s)

### 6.1 Lightweight Inference System
A lightweight inference pod performs real-time monitoring.
**Responsibilities:**
*   Query Prometheus periodically
*   Maintain sliding metric window
*   Execute inference
*   Emit Kubernetes Events

**Inference logic:**
```python
if reconstruction_error > threshold:
    anomaly_detected()
    classify_fault()
```

**Design optimized for low-resource nodes:**
*   CPU-only inference
*   Batch size = 1
*   `torch.no_grad()`
*   Model size < 5MB

### 6.2 Drift Monitoring Module
A drift monitoring module tracks reconstruction error statistics to detect long-term deviations in system behavior caused by workload evolution rather than transient faults.

**Drift Detection Methodology:**
Instead of checking only `reconstruction_error > threshold`, the system maintains a rolling window of recent errors online:
1.  **Maintain Error Buffer:** `last_errors = deque(maxlen=500)`
2.  **Compute Rolling Mean:** `rolling_mean = np.mean(last_errors)`
3.  **Compare Against Training Baseline:** If `rolling_mean > train_mean + 2*train_std`, drift is detected.

**Handling Detected Drift:**
When long-term drift is flagged, the operator emits a specific Kubernetes Event to mark a distribution shift (e.g., suggesting an alert storm risk and logging the variation). The system explicitly **does not** retrain automatically, preserving stability.

---

## Phase 7 — Online Experimental Validation
Live fault injections evaluate system behavior under deployment conditions.
Metrics recorded:
*   Detection latency
*   False positive rate
*   Resource overhead
*   Recovery identification time

---

## Phase 8 — Resource-Constrained Design Considerations
Given cluster limits (2GB RAM per node):
*   Training performed offline
*   Only inference deployed
*   Reduced Prometheus retention
*   Lightweight monitoring stack

---

## Phase 9 — Research Deliverables
Paper structure checklist:
- [ ] Problem Statement
- [ ] System Architecture
- [ ] Dataset Generation Framework
- [ ] Experimental Methodology
- [ ] Model Comparison
- [ ] Online Deployment
- [ ] Real-Time Evaluation
- [ ] Drift Monitoring Analysis
- [ ] Resource Analysis
- [ ] Conclusion

---

### 🚨 Experimental Constraints (Non-Negotiable)
1. Identical offline/online preprocessing
2. Fixed normalization parameters
3. Fixed anomaly threshold
4. No online retraining
5. Run-aware dataset splitting

### 🎯 Final Project Scope
This work delivers a resource-aware real-time anomaly detection and fault classification framework for Kubernetes clusters, combining structured generalization evaluation with live operator-based deployment.
