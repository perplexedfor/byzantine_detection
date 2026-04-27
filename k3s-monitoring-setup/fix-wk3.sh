#!/bin/bash
# =============================================================================
# fix-wk3.sh — Apply infra-level stall fixes on sw-wk3 (192.168.56.12)
# Run as: sudo bash fix-wk3.sh   (must be executed ON sw-wk3 via SSH)
# =============================================================================
set -euo pipefail

echo "====== [fix-wk3] Starting fixes on $(hostname) ======"

# ---------------------------------------------------------------------------
# Fix 2 — Kubelet: increase log-stream idle timeout to 30m
# NOTE: node-ip and flannel-iface are NOT set here because they are
#       already in the systemd ExecStart line. Adding them again causes
#       K3s to merge duplicates and crash.
# ---------------------------------------------------------------------------
echo ""
echo "--- Fix 2: Updating /etc/rancher/k3s/config.yaml ---"
sudo mkdir -p /etc/rancher/k3s/

# Remove any node-ip / flannel-iface lines to avoid duplication with ExecStart
sudo sed -i '/^node-ip:/d' /etc/rancher/k3s/config.yaml 2>/dev/null || true
sudo sed -i '/^flannel-iface:/d' /etc/rancher/k3s/config.yaml 2>/dev/null || true

# Remove old kubelet-arg block if present (to avoid duplicating those too)
sudo sed -i '/^kubelet-arg:/,/^[^ ]/{ /^kubelet-arg:/d; /^  - /d; }' /etc/rancher/k3s/config.yaml 2>/dev/null || true

# Append the kubelet args
cat <<'EOF' | sudo tee -a /etc/rancher/k3s/config.yaml > /dev/null
kubelet-arg:
  - "streaming-connection-idle-timeout=30m"
  - "node-status-update-frequency=10s"
  - "eviction-hard=memory.available<100Mi"
  - "kube-reserved=cpu=100m,memory=128Mi"
EOF

# Ensure snapshotter: native is present
if ! grep -q '^snapshotter:' /etc/rancher/k3s/config.yaml 2>/dev/null; then
    echo 'snapshotter: native' | sudo tee -a /etc/rancher/k3s/config.yaml > /dev/null
fi

echo "Updated config.yaml:"
cat /etc/rancher/k3s/config.yaml

echo ""
echo "Restarting k3s-agent..."
sudo systemctl restart k3s-agent
echo "Waiting 15s for agent to settle..."
sleep 15
sudo systemctl status k3s-agent --no-pager || true

# ---------------------------------------------------------------------------
# Fix 4 — Install tetragon-nurse.sh watchdog
# ---------------------------------------------------------------------------
echo ""
echo "--- Fix 4: Installing tetragon-nurse.sh ---"
sudo tee /usr/local/bin/tetragon-nurse.sh > /dev/null <<'SCRIPT'
#!/bin/bash
STALL_THRESHOLD=50
LOGFILE="/var/log/tetragon-nurse.log"
NAMESPACE="kube-system"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }
log "tetragon-nurse started on $(hostname)"

get_pod() {
    k3s kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=tetragon \
        --field-selector spec.nodeName="$(hostname)" \
        --no-headers -o custom-columns=NAME:.metadata.name 2>/dev/null | head -1
}

LAST_EVENT_TIME=$(date +%s)

while true; do
    sleep 10
    POD=$(get_pod)
    if [ -z "$POD" ]; then continue; fi
    LINES=$(k3s kubectl logs -n "$NAMESPACE" "$POD" -c export-stdout --since=10s 2>/dev/null | wc -l)
    if [ "$LINES" -gt 0 ]; then LAST_EVENT_TIME=$(date +%s); fi
    SILENCE=$(( $(date +%s) - LAST_EVENT_TIME ))
    log "Pod=$POD lines_last_10s=$LINES silence=${SILENCE}s"
    if [ "$SILENCE" -gt "$STALL_THRESHOLD" ]; then
        log "!!! STALL DETECTED (${SILENCE}s silence) — deleting $POD to force restart"
        k3s kubectl delete pod -n "$NAMESPACE" "$POD" --grace-period=0 --force 2>&1 | tee -a "$LOGFILE"
        LAST_EVENT_TIME=$(date +%s)
        sleep 20
    fi
done
SCRIPT
sudo chmod +x /usr/local/bin/tetragon-nurse.sh

echo "--- Fix 4: Registering tetragon-nurse systemd service ---"
sudo tee /etc/systemd/system/tetragon-nurse.service > /dev/null <<'EOF'
[Unit]
Description=Tetragon local health watchdog
After=k3s-agent.service
Wants=k3s-agent.service

[Service]
Type=simple
ExecStart=/usr/local/bin/tetragon-nurse.sh
Restart=always
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now tetragon-nurse
sudo systemctl status tetragon-nurse --no-pager || true

echo ""
echo "====== [fix-wk3] Done ======"
