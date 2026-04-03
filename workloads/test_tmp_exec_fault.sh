#!/bin/bash

# Ensure you are running this from the workloads directory
WORKLOADS_DIR="$(pwd)"
SECURITY_YAML="$WORKLOADS_DIR/security/security-tmp-exec.yaml"
COLLECT_SCRIPT="$WORKLOADS_DIR/../k3s-monitoring-setup/collect_baseline.py"

echo "================================================="
echo "1. Exporting and importing the Docker image to worker nodes"
echo "================================================="
# This fixes the ErrImagePull issue on the worker nodes!
docker save security-tmp-exec:latest -o tmp-exec.tar

echo "Copying to sw-wk2 and importing..."
# Replace with your actual ssh credentials if needed
scp tmp-exec.tar vagrant@192.168.56.11:/home/vagrant/
ssh vagrant@192.168.56.11 "sudo k3s ctr images import /home/vagrant/tmp-exec.tar"

echo "Copying to sw-wk3 and importing..."
scp tmp-exec.tar vagrant@192.168.56.12:/home/vagrant/
ssh vagrant@192.168.56.12 "sudo k3s ctr images import /home/vagrant/tmp-exec.tar"

# Setup cleanup on script exit
cleanup() {
    echo -e "\n================================================="
    echo "4. Cleaning up test..."
    echo "================================================="
    k3s kubectl delete -f $SECURITY_YAML.tmp.yaml --ignore-not-found
    rm -f $SECURITY_YAML.tmp.yaml
    rm -f tmp-exec.tar
    echo "Killing data collection..."
    pkill -f "python3 $COLLECT_SCRIPT test_tmp_exec"
    echo "Test finished! Check ../dataset/node_metrics_test_tmp_exec.csv to see the exec_cnt spikes."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo -e "\n================================================="
echo "2. Starting metric collection in the background"
echo "================================================="
python3 $COLLECT_SCRIPT test_tmp_exec &
sleep 5 # Wait for collection to initialize

echo -e "\n================================================="
echo "3. Patching and Deploying fault to sw-wk2..."
echo "================================================="

# Create a temporary yaml that forces the pod onto sw-wk2 and replaces variables
cat $SECURITY_YAML | \
sed 's/{{EXEC_PATH}}/\/tmp/g' | \
sed 's/{{EXEC_INTERVAL}}/3/g' > $SECURITY_YAML.tmp.yaml

# Inject nodeSelector targeting sw-wk2
sed -i 's/      containers:/      nodeSelector:\n        kubernetes.io\/hostname: sw-wk2\n      containers:/g' $SECURITY_YAML.tmp.yaml

k3s kubectl apply -f $SECURITY_YAML.tmp.yaml

echo -e "\nFault running! Waiting 60 seconds to collect 6 windows of metrics..."
echo "You should see Tetragon process_exec events captured by the background script."
for i in {1..6}; do
    echo "Waiting... ($((i*10)) / 60s)"
    sleep 10
done

# The trap cleanup will execute automatically when the loop finishes
