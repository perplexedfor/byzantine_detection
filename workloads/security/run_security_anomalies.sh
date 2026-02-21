#!/bin/bash
# script to apply security anomaly workloads one by one
# used for Step 4 (Scenario-Based Labeling)

echo "Starting Security Anomalies Workloads..."

echo "========================================="
echo "1. Simulating Malicious /tmp Execution..."
k3s kubectl apply -f security-tmp-exec.yaml
echo "Sleeping for 60 seconds to gather eBPF metrics..."
sleep 60
k3s kubectl delete -f security-tmp-exec.yaml

echo "========================================="
echo "2. Simulating High Process Volumes & Weird Names..."
k3s kubectl apply -f security-high-process.yaml
echo "Sleeping for 60 seconds to gather eBPF metrics..."
sleep 60
k3s kubectl delete -f security-high-process.yaml

echo "========================================="
echo "3. Simulating Suspicious Network Connections (3333, 4444)..."
k3s kubectl apply -f security-suspicious-network.yaml
echo "Sleeping for 60 seconds to gather eBPF metrics..."
sleep 60
k3s kubectl delete -f security-suspicious-network.yaml

echo "========================================="
echo "Security Anomaly workloads completed."
