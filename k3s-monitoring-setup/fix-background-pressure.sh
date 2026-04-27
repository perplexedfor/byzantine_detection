#!/bin/bash
# =============================================================================
# fix-background-pressure.sh — Fix 5: Remove background-pressure from wk-2
# Run from: the control-plane node (sw-master) or any machine with kubeconfig
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Adjust path if your workloads dir is elsewhere
BP_YAML="$SCRIPT_DIR/../workloads/background-pressure.yaml"

echo "====== [fix-background-pressure] Patching nodeAffinity for background-pressure ======"

if [ ! -f "$BP_YAML" ]; then
    echo "ERROR: $BP_YAML not found. Adjust BP_YAML path at the top of this script."
    exit 1
fi

echo "Current nodeAffinity in $BP_YAML:"
grep -A10 "nodeAffinity" "$BP_YAML" || echo "(none found — check yaml structure)"

echo ""
echo "Applying updated background-pressure.yaml..."
kubectl apply -f "$BP_YAML"

echo ""
echo "Verifying pod placement (should only show sensor-gateway / wk-3):"
kubectl get pods -l app=background-pressure -o wide

echo ""
echo "====== [fix-background-pressure] Done ======"
echo "NOTE: You must manually edit $BP_YAML first to remove the 'compute' value"
echo "      from .spec.affinity.nodeAffinity before running this script."
echo "      The yaml change is documented in the fix plan (Fix 5 section)."
