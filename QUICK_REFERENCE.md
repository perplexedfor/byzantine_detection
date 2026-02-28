# Quick Reference: Pre-Run Checklist

**Run this IN ORDER before `python scenario_runner.py` on the VM**

## ✅ Quick Checklist (5 minutes)

### 1. Start Port-Forward (Terminal 1)
```bash
cd k3s-monitoring-setup
./pf-prometheus.sh
# Watch for: "Starting port-forward..."
# Keep this running!
```

### 2. Verify Setup (Terminal 2)
```bash
# Make the verification script executable (Windows → WSL via git bash if needed)
bash pre-run-verification.sh

# Should see:
# ✓ Cluster is accessible
# ✓ Tetragon running on all 3 nodes
# ✓ Prometheus accessible at http://localhost:9090
# ✓ Node-Exporter targets: 3 (or reports all nodes)
# ✓ All 3 nodes are Ready
```

### 3. Check Prometheus (Browser)
```
Visit: http://localhost:9090/targets
```
Look for section: **node-exporter**
- Should show **3 endpoints** (one per node)
- All should be **UP** (green)

If not all 3 are UP:
- ❌ Fix before continuing
- Tetragon/node-exporter not running on all nodes

### 4. Test Multi-Node Query (Browser)
```
1. Go to: http://localhost:9090/graph
2. Query: up{job="node-exporter"}
3. Click "Execute"
```
Should see: **3 results** (one per node)
- k3s-wk1
- sw-wk2
- sw-wk3

If not:
- ❌ Check Prometheus targets again
- Something not scraped properly

### 5. Start Baseline Workloads (Terminal 2)
```bash
cd workloads
bash run_normal_baseline.sh

# Wait for these to be Running:
# - nginx-deployment (3 pods)
# - redis-deployment (1 pod)
# - traffic-generator (1 pod)
# - api-deployment (1 pod)

# Verify:
kubectl get pods -n default | grep -E "nginx|redis|traffic|api"
```

### 6. Run Scenario Runner (Terminal 2)
```bash
cd workloads
python scenario_runner.py

# Watch for output:
# "Tetragon log stream started"
# "Starting Scenario: normal"
# "Tetragon logs collected" 
# Scenarios cycle through automatically ~20 min runtime
```

### 7. Monitor Progress (Terminal 2)
```bash
# In another terminal, watch the dataset grow:
watch -n 5 'wc -l k3s-monitoring-setup/node_metrics_*.csv'

# Or check final labeled dataset:
ls -lh k3s-monitoring-setup/final_labeled_dataset.csv
```

---

## ⚠️ Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| Only 1 node has metrics | Run verification script - Tetragon/node-exporter not on all nodes |
| Prometheus says "No data" | Start port-forward: `./pf-prometheus.sh` |
| Prometheus query shows 0 results | Check targets page - no scrapers up |
| Collection crashes with timeout | Port-forward terminated - restart it |
| Scenario runner hangs | Check baseline workloads are running |

---

## What's Fixed This Time? 

✅ **Multi-node metric collection** now works properly
- Prometheus labels include actual node names
- Collector understands all 3 nodes
- Dataset will have balanced data from all nodes

✅ **Prometheus scrape config** updated
- All nodes discoverable and labeled
- Instance labels properly mapped

✅ **collect_baseline.py** handles multiple node identification strategies
- Looks for explicit node label first
- Falls back to IP mapping if needed
- Graceful degradation if labels missing

**Result:** Next dataset run should have metrics from ALL THREE nodes, not just k3s-wk1

---

## Expected Outputs

After a successful run, you should have:

```
k3s-monitoring-setup/
├── node_metrics_<run-id>.csv          ← Raw metrics (30-sec collection windows)
│                                        Should have ~600-800 rows
│                                        3 entries per 10-sec window (one per node)
│
├── scenario_labels_<run-id>.csv        ← Scenario timing
│                                        All fault types should appear
│
└── final_labeled_dataset.csv           ← Final merge
                                        All nodes: k3s-wk1, sw-wk2, sw-wk3
                                        All labels: normal, transition, 
                                                    cpu_stress, memory_leak,
                                                    network_chaos, crash_loop, 
                                                    security_*
```

Check metrics are balanced:
```bash
# Should show ~3 nodes per timestamp:
head -20 k3s-monitoring-setup/node_metrics_<run-id>.csv | cut -d, -f3 | sort | uniq -c
```

---

## Troubleshooting Quick Links

| Problem | Look in |
|---------|----------|
| Prometheus connection fails | `CONFIG_VALIDATION_CHECKLIST.md` → Issue #4 |
| Only 1 node has metrics | `FIXES_APPLIED_SUMMARY.md` → Multi-Node Collection Broken |
| Tetragon not working | `CONFIG_VALIDATION_CHECKLIST.md` → Priority 2, Item 2 |
| High exec counts confusing | `CONFIG_VALIDATION_CHECKLIST.md` → Previous analysis |

---

## Time Estimates

- ⏱ Port-forward: instant (keep running)
- ⏱ Verification script: 2-3 minutes
- ⏱ Baseline workload startup: 3-5 minutes (let them stabilize)
- ⏱ Scenario runner full cycle: 15-20 minutes
- ⏱ Total time: **25-30 minutes**

---

## Success Criteria

After completing the run, the dataset should show:

✅ **3 nodes** with metrics (k3s-wk1, sw-wk2, sw-wk3)
✅ **Balanced distribution** (~33% metrics per node)
✅ **Multiple fault types** (cpu_stress, memory_leak, network_chaos, etc.)
✅ **Proper transitions** marked between faults
✅ **No zero-metrics** rows (if issue #1 is fully fixed)
✅ **exec_count varies** by fault type (higher during cpu_stress, security_tmp_exec, etc.)

If you see this → Run successful! 🎉

---

## Quick Commands

```bash
# Check Prometheus is accessible
curl http://localhost:9090/api/v1/query?query=up 2>/dev/null | head -50

# Count nodes in Prometheus data
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job=="node-exporter") | .labels.node' | sort -u

# Monitor dataset collection
tail -f k3s-monitoring-setup/node_metrics_*.csv | while IFS= read -r line; do echo "$(date): $line"; done

# Validate final dataset
python3 -c "
import csv
with open('k3s-monitoring-setup/final_labeled_dataset.csv') as f:
    reader = csv.DictReader(f)
    nodes = set()
    labels = set()
    for row in reader:
        nodes.add(row['node'])
        labels.add(row['label'])
    print(f'Nodes: {nodes}')
    print(f'Labels: {labels}')
    print(f'Rows: {row}')
"
```

---

## Emergency Commands

If something goes wrong:

```bash
# Stop everything
pkill -f "scenario_runner.py"
pkill -f "pf-prometheus.sh"
pkill -f "collect_baseline.py"

# Stop baseline workloads
cd workloads
bash stop_normal_baseline.sh

# Check what's still running
kubectl get pods -n default
kubectl get pods -n kube-system | grep -E "tetragon|prometheus"
```

---
