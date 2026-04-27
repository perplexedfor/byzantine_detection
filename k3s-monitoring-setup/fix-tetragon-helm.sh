#!/bin/bash
# =============================================================================
# fix-tetragon-helm.sh — Fix 3: Raise Tetragon CPU limit via Helm upgrade
# Run from: the control-plane node (sw-master) or any machine with kubeconfig
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="$SCRIPT_DIR/tetragon-values.yaml"

echo "====== [fix-tetragon-helm] Upgrading Tetragon resources ======"

if [ ! -f "$VALUES_FILE" ]; then
    echo "ERROR: $VALUES_FILE not found. Run from the k3s-monitoring-setup directory."
    exit 1
fi

helm upgrade tetragon cilium/tetragon \
    --namespace kube-system \
    -f "$VALUES_FILE"

echo "Waiting for DaemonSet rollout..."
kubectl rollout status daemonset/tetragon -n kube-system

echo ""
echo "====== [fix-tetragon-helm] Done ======"
echo "Verify resource limits:"
echo "  kubectl get daemonset tetragon -n kube-system -o jsonpath='{.spec.template.spec.containers[*].resources}'"
