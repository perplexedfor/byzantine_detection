echo "================================="
echo "Starting Normal Activity Workloads"
echo "================================="

# Step 0: Start background pressure DaemonSet FIRST
# This simulates edge node headroom compression (~200MB RAM per node)
# Must be running before any other workloads so their metrics reflect
# the compressed-headroom baseline.
echo "Applying background pressure (edge headroom simulation)..."
k3s kubectl apply -f background-pressure.yaml
k3s kubectl rollout status daemonset/background-pressure
echo "Background pressure running on all nodes."

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