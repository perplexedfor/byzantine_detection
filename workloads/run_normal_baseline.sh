#!/bin/bash
# script to apply all normal activity workloads

echo "Starting Normal Activity Workloads..."

# Apply base services
k3s kubectl apply -f nginx-deployment.yaml
k3s kubectl apply -f redis-deployment.yaml
k3s kubectl apply -f api-deployment.yaml

# Apply background tasks
k3s kubectl apply -f cron-logger.yaml

# Wait for deployments to be ready before starting traffic
echo "Waiting for base services to become ready..."
k3s kubectl rollout status deployment/nginx-baseline
k3s kubectl rollout status deployment/redis-baseline
k3s kubectl rollout status deployment/api-baseline

# Start traffic generator
echo "Starting traffic generator..."
k3s kubectl apply -f traffic-generator.yaml

echo "Normal baseline activity running."
