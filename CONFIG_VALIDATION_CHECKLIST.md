# Configuration Validation Checklist

## Critical Issues Found

### 1. ⚠️ **IP-to-Node Mapping Mismatch**
**File:** `collect_baseline.py` (lines 24-28)
```python
IP_TO_NODE_MAP = {
    "192.168.56.10": "k3s-wk1",
    "192.168.56.11": "sw-wk2",
    "192.168.56.12": "sw-wk3"
}
```

**Issue:** 
- Maps are hardcoded to static IPs
- But Prometheus returns **instance labels** which might be:
  - Pod IPs (different from node IPs)
  - Service IPs
  - Different format than expected

**What Prometheus Actually Returns:**
- Node-exporter runs in a pod on each node
- Instance label format: `<pod-ip>:9100`
- Pod IPs ≠ Node IPs
- Need to map pod IPs or use node names directly from labels

**FIX NEEDED:**
```python
# Instead of IP mapping, use node name directly from Prometheus labels:
# Prometheus should return: metric.node (from Kubernetes service discovery)
# Or update scrape config to include node names
```

---

### 2. ⚠️ **Prometheus Scrape Config Missing Multi-Node Discovery**
**File:** `v2/observability/prometheus-config.yaml` 

**Issue:**
- Uses `role: node` kubernetes_sd_configs ✅
- Should discover all nodes automatically
- **BUT:** Need to verify label mapping includes node names

**What to Check:**
```bash
# In Prometheus UI, check if all 3 nodes appear:
http://localhost:9090/targets

# Verify each node-exporter instance is scraped
# Should see entries like:
# - node-exporter:9100 (on k3s-wk1)
# - node-exporter:9100 (on sw-wk2)
# - node-exporter:9100 (on sw-wk3)
```

---

### 3. ⚠️ **Instance Label Parsing Wrong**
**File:** `collect_baseline.py` (lines 73-79)

**Current Logic:**
```python
result_data = {item['metric'].get('instance', '').split(':')[0]: ...}
```

**Problem:**
- Assumes instance=`<ip>:<port>`
- Splits by `:` to get IP
- But IP might be pod IP, not in IP_TO_NODE_MAP

**Better Approach:**
```python
# Use node label directly from Prometheus:
node_name = item['metric'].get('node', 'unknown')  # Kubernetes node name
instance_ip = item['metric'].get('instance', '').split(':')[0]
```

---

### 4. ✅ **Prometheus Connection String**
**File:** `collect_baseline.py` (line 12)
```python
PROMETHEUS_URL = "http://localhost:9090"
```

**Status:** Correct for local setup with port-forward
**Prerequisite:** Must run `./pf-prometheus.sh` before running scenario

---

### 5. ✅ **Tetragon DaemonSet**
**File:** `monitor/daemonset.yaml`

**Status:** ✅ Correct
- No nodeSelector → runs on all nodes
- Captures events from all pods

---

### 6. ✅ **Node-Exporter DaemonSet**
**File:** `v2/observability/node-exporter.yaml`

**Status:** ✅ Correct
- DaemonSet (no nodeSelector)
- Runs on all nodes
- Properly mounts host filesystems

---

## Pre-Run Checklist

Before running `scenario_runner.py`, verify:

### Step 1: Prometheus Running & Accessible
```bash
# 1. Start port-forward
./pf-prometheus.sh

# 2. Verify in another terminal
curl http://localhost:9090/api/v1/targets

# 3. Check the JSON response - should show:
# - node-exporter targets from all 3 nodes
# - status should be "up"
```

### Step 2: Verify All Nodes Have Metrics
```bash
# In Prometheus UI: http://localhost:9090/graph

# Query to test multi-node collection:
up{job="node-exporter"}

# Should return 3 entries (one per node)
# If only 1 entry → node-exporter not running on other nodes
```

### Step 3: Fix Instance Label Mapping
**CRITICAL:** Update `collect_baseline.py` to use node names instead of IPs:

```python
# Current (broken):
instance_ip = item['metric'].get('instance', '').split(':')[0]
node_name = IP_TO_NODE_MAP.get(instance_ip, ip)

# Better (use Kubernetes labels):
node_name = item['metric'].get('node', 
                 item['metric'].get('__meta_kubernetes_node_name', 'unknown'))
```

### Step 4: Verify Tetragon Running on All Nodes
```bash
kubectl get pods -n kube-system | grep tetragon

# Should return 3 tetragon pods (one per node)
# If only 1 → DaemonSet not working on other nodes
```

### Step 5: Verify TCP-Connect Policy Loaded
```bash
kubectl get tracingpolicies -A | grep tcp-connect

# Should return the tcp-connect policy
# Verify in Tetragon logs:
kubectl logs -n kube-system -l app=tetragon -c export-stdout --tail=20
```

---

## Scenario Runner Configuration

**File:** `workloads/scenario_runner.py`

**Status:** ✅ Good
- Random node selection per fault ✅
- Proper padding/transition handling ✅
- Correct metric collection calling ✅

---

## Data Collection Configuration

**File:** `k3s-monitoring-setup/collect_baseline.py`

**Issues Found:**
1. ❌ Hard-coded IP to node mapping
2. ❌ Instance label parsing assumes node IPs
3. ⚠️ No fallback for unmapped IPs
4. ✅ Proper exception handling for Prometheus failures
5. ✅ Proper Tetragon event parsing

---

## Recommended Fixes

### Priority 1 (Critical - Must Fix)

**1. Update Instance Label Parsing in collect_baseline.py**

Replace lines 73-79:
```python
# OLD (broken):
def get_prometheus_metric(query):
    ...
    result_data = {item['metric'].get('instance', '').split(':')[0]: float(item['value'][1]) for item in data}
    return result_data

# NEW (fixed):
def get_prometheus_metric(query):
    ...
    result_data = {}
    for item in data:
        # Try to get node name directly from Prometheus labels
        node_ip = item['metric'].get('instance', '').split(':')[0]
        # First try: use explicit node label from scrape config
        node_name = item['metric'].get('node')
        # Fallback: map by IP
        if not node_name:
            node_name = IP_TO_NODE_MAP.get(node_ip, node_ip)
        result_data[node_name] = float(item['value'][1])
    return result_data, True
```

**2. Add Node Label to Prometheus Scrape Config**

In `v2/observability/prometheus-config.yaml`, update relabel_configs:
```yaml
relabel_configs:
- source_labels: [__meta_kubernetes_node_name]
  target_label: node
  action: replace
```

This ensures each scrape has a `node` label with the actual node name.

### Priority 2 (Important - To Verify)

**1. Verify Prometheus Targets Before Running**

```bash
# Check JSON from targets endpoint
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job == "node-exporter") | .labels'

# Should show all 3 nodes
```

**2. Verify Tetragon Pods on All Nodes**

```bash
kubectl get pods -n kube-system -o wide | grep tetragon | awk '{print $7}' | sort -u

# Should show 3 different nodes
```

---

## Manual Verification Commands

Run these BEFORE starting scenario_runner:

```bash
# 1. Check Tetragon on all nodes
echo "=== Tetragon DaemonSet Status ===" 
kubectl get daemonset tetragon -n kube-system
kubectl get pods -n kube-system -o wide -l app=tetragon

# 2. Check Node-Exporter if using v2 setup
echo "=== Node-Exporter Status ===" 
kubectl get daemonset node-exporter -n default
kubectl get pods -n default -o wide -l app=node-exporter

# 3. Check Prometheus targets
echo "=== Prometheus Targets ===" 
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'

# 4. Test Prometheus query
echo "=== Test Prometheus Query ===" 
curl -s 'http://localhost:9090/api/v1/query?query=up' | jq '.data.result | length'
```

---

## Summary

| Component | Status | Issue | Fix |
|-----------|--------|-------|-----|
| Prometheus | 🟡 Partial | Instance labels might not match IP map | Add node label to scrape config |
| Tetragon | ✅ Good | - | Verify pods on all 3 nodes before run |
| Node-Exporter | ✅ Good | - | Ensure running on all nodes |
| Scenario Runner | ✅ Good | - | Ready to use |
| Metric Collector | 🔴 Broken | Hard-coded IP mapping | Use Prometheus node labels instead |

---

## Next Steps

1. ✅ Fix instance label parsing in `collect_baseline.py`
2. ✅ Add node name to Prometheus scrape config
3. ✅ Run verification commands above
4. ✅ Check Prometheus targets show all 3 nodes
5. ✅ Check Tetragon pods on all 3 nodes
6. ✅ Run scenario_runner with validated setup
