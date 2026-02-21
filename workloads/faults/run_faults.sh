#!/bin/bash
# script to apply controlled fault workloads one by one
# used for Step 4 (Scenario-Based Labeling)

echo "Starting Controlled Faults Workloads..."

echo "========================================="
echo "1. Injecting CPU Stress Fault..."
k3s kubectl apply -f fault-cpu-stress.yaml
echo "Sleeping for 60 seconds to gather metrics..."
sleep 60
k3s kubectl delete -f fault-cpu-stress.yaml

echo "========================================="
echo "2. Injecting Memory Leak Fault..."
k3s kubectl apply -f fault-memory-leak.yaml
echo "Sleeping for 60 seconds to gather metrics..."
sleep 60
k3s kubectl delete -f fault-memory-leak.yaml

echo "========================================="
echo "3. Injecting Network Packet Loss and Delay Fault..."
k3s kubectl apply -f fault-network-chaos.yaml
echo "Sleeping for 60 seconds to gather metrics..."
sleep 60
k3s kubectl delete -f fault-network-chaos.yaml

echo "========================================="
echo "4. Injecting Pod Crash Loop Fault..."
k3s kubectl apply -f fault-crash-loop.yaml
echo "Sleeping for 60 seconds to gather metrics..."
sleep 60
k3s kubectl delete -f fault-crash-loop.yaml

echo "========================================="
echo "Controlled Fault workloads completed."
