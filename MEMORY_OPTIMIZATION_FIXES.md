# Memory Optimization Fixes for Data Collection

## Problem
The data collection process was being hampered by **Out-of-Memory (OOM) kills** from the kernel. Python processes with PIDs 85030, 85209, 85230, etc. were being terminated due to excessive memory consumption when running baseline workloads, faults, and security anomalies concurrently.

## Root Causes
1. **Memory leak fault too aggressive**: Was leaking 5-30 MB/sec with only a 500Mi limit
2. **No resource limits**: Many workloads lacked memory requests/limits
3. **Concurrent resource pressure**: Multiple workloads running together without proper spacing
4. **Insufficient cleanup delays**: Only 30 seconds between scenarios, causing cumulative memory buildup

## Fixes Applied

### 1. Memory Leak Fault Optimization (`fault-memory-leak.yaml`)
- **Increased memory limit**: 500Mi → **1Gi** (double capacity)
- **Reduced leak rate**: [5, 10, 20, 30] MB/sec → **[2, 3, 4, 5] MB/2sec** (~60% reduction)
- **Slower allocation cadence**: 1 second → **2 seconds between allocations**
- **Added signal handling**: Graceful termination on SIGTERM
- **Added memory cap**: Will stop at 800MB instead of crashing at limit
- **Added memory requests**: 100Mi → **256Mi** (proper resource scheduling)

### 2. CPU Stress Fault (`fault-cpu-stress.yaml`)
- **Reduced memory limit**: 500Mi → **256Mi**
- **Increased memory request**: 100Mi → **128Mi**

### 3. Network Chaos Fault (`fault-network-chaos.yaml`)
- **Added memory request**: 64Mi
- **Added memory limit**: 128Mi

### 4. Crash Loop Fault (`fault-crash-loop.yaml`)
- **Added memory request**: 64Mi
- **Memory limit**: 256Mi (to test container restart behavior)

### 5. Security High-Process Workload (`security-high-process.yaml`)
- **Reduced process spawning**: [20-100] → **[10-40] processes**
- **Added memory request**: 128Mi
- **Added memory limit**: 256Mi

### 6. Security Tmp-Exec Workload (`security-tmp-exec.yaml`)
- **Added memory request**: 64Mi
- **Added memory limit**: 256Mi

### 7. Security Suspicious-Network Workload (`security-suspicious-network.yaml`)
- **Added memory request**: 64Mi
- **Added memory limit**: 128Mi

### 8. Scenario Runner Optimization (`scenario_runner.py`)
- **Reduced memory leak parameters**: [5, 10, 20, 30] → **[2, 3, 4, 5]** MB per allocation
- **Reduced process spawning**: [20-100] → **[10-40]** processes
- **Increased cleanup timeout**: 30 seconds → **60 seconds** for proper memory reclamation
- **Better resource scheduling**: Explicit cleanup windows allow kernel to reclaim memory

## Expected Improvements
✅ **60-70% reduction in peak memory consumption**  
✅ **Graceful memory leak simulation** without hitting hard limits  
✅ **Proper resource isolation** via Kubernetes resource limits  
✅ **Better data quality** - collectors won't be OOM-killed mid-execution  
✅ **More stable workload execution** with longer cleanup windows  

## Memory Budget Summary
```
Per-workload limits:
├── Memory Leak:         1Gi (was 500Mi)
├── CPU Stress:          256Mi (was 500Mi)
├── Network Chaos:       128Mi (new)
├── Crash Loop:          256Mi (new)
├── High-Process:        256Mi (new)
├── Tmp-Exec:            256Mi (new)
└── Suspicious-Network:  128Mi (new)

Total peak concurrent: ~2.5-3Gi (was hitting unlimited)
Baseline workloads: ~1-1.5Gi (nginx, redis, api, traffic-gen)
System overhead: ~500Mi
```

## Testing Recommendations
1. Monitor `/var/log/kern.log` or `dmesg` for OOM messages during runs
2. Check Kubernetes events: `k3s kubectl get events -A --sort-by='.lastTimestamp'`
3. Monitor node memory: `k3s kubectl top nodes` and `k3s kubectl top pods -A`
4. Run scenario_runner.py and verify it completes without killed processes
5. Verify data collection completes successfully and files are written

## Files Modified
- `workloads/faults/fault-memory-leak.yaml`
- `workloads/faults/fault-cpu-stress.yaml`
- `workloads/faults/fault-network-chaos.yaml`
- `workloads/faults/fault-crash-loop.yaml`
- `workloads/security/security-high-process.yaml`
- `workloads/security/security-tmp-exec.yaml`
- `workloads/security/security-suspicious-network.yaml`
- `workloads/scenario_runner.py`
