#!/bin/bash
echo "================================="
echo "Starting Normal Activity Workloads"
echo "================================="

k3s kubectl apply -f nginx-deployment.yaml
k3s kubectl apply -f redis-deployment.yaml
k3s kubectl apply -f api-deployment.yaml

k3s kubectl apply -f cron-logger.yaml


echo "Waiting for services to become ready..."

k3s kubectl rollout status deployment/nginx-baseline
k3s kubectl rollout status deployment/redis-baseline
k3s kubectl rollout status deployment/api-baseline

echo "Allowing baseline stabilization..."
sleep 20

echo "Starting traffic generator..."
k3s kubectl apply -f traffic-generator.yaml

echo "================================="
echo "✅ Normal baseline activity running"
echo "================================="