#!/bin/bash
# script to stop and delete all normal activity workloads

echo "Stopping Normal Activity Workloads..."

# Stop traffic generator first to prevent new requests
echo "Stopping traffic generator..."
k3s kubectl delete -f traffic-generator.yaml --ignore-not-found

# Stop background tasks
echo "Stopping background tasks..."
k3s kubectl delete -f cron-logger.yaml --ignore-not-found

# Stop base services
echo "Stopping base services..."
k3s kubectl delete -f api-deployment.yaml --ignore-not-found
k3s kubectl delete -f redis-deployment.yaml --ignore-not-found
k3s kubectl delete -f nginx-deployment.yaml --ignore-not-found

echo "Normal baseline activity stopped."
