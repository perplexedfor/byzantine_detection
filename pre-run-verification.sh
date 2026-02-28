#!/bin/bash

# PRE-RUN VERIFICATION SCRIPT
# Run this BEFORE running scenario_runner.py to ensure all components are properly configured

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}K3s Cluster Configuration Verification${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. Check Kubernetes Cluster
echo -e "${BLUE}1. Checking Kubernetes Cluster...${NC}"
if kubectl cluster-info >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Cluster is accessible${NC}"
    kubectl version --short
else
    echo -e "${RED}✗ Cluster is NOT accessible${NC}"
    exit 1
fi

echo ""

# 2. Check Tetragon DaemonSet
echo -e "${BLUE}2. Checking Tetragon Deployment...${NC}"
TETRAGON_PODS=$(kubectl get pods -n kube-system --no-headers 2>/dev/null | grep tetragon | wc -l)
EXPECTED_PODS=3

if [ $TETRAGON_PODS -eq $EXPECTED_PODS ]; then
    echo -e "${GREEN}✓ Tetragon running on all $EXPECTED_PODS nodes${NC}"
    kubectl get pods -n kube-system -o wide -l app=tetragon
else
    echo -e "${RED}✗ Tetragon pods mismatch: found $TETRAGON_PODS, expected $EXPECTED_PODS${NC}"
    echo "   Available Tetragon pods:"
    kubectl get pods -n kube-system -o wide --selector=app=tetragon 2>/dev/null || echo "   None found"
fi

echo ""

# 3. Check Node-Exporter (if using v2 setup)
echo -e "${BLUE}3. Checking Node-Exporter Deployment...${NC}"
NE_PODS=$(kubectl get pods -n default --no-headers 2>/dev/null | grep node-exporter | wc -l)

if [ $NE_PODS -ge 1 ]; then
    echo -e "${GREEN}✓ Node-Exporter running on $NE_PODS nodes${NC}"
    kubectl get pods -n default -o wide -l app=node-exporter
else
    echo -e "${YELLOW}⚠ Node-Exporter not found in default namespace${NC}"
    echo "  This is OK if using kube-prometheus-stack (Helm) instead"
fi

echo ""

# 4. Check Prometheus
echo -e "${BLUE}4. Checking Prometheus Service...${NC}"

# Check if using kube-prometheus-stack
PROM_NAMESPACE="monitoring"
PROM_SERVICE="prometheus-kube-prometheus-prometheus"

if kubectl get svc -n $PROM_NAMESPACE $PROM_SERVICE >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus service found (kube-prometheus-stack${NC}"
    kubectl get svc -n $PROM_NAMESPACE $PROM_SERVICE
    PROM_URL="http://localhost:9090"
elif kubectl get svc -n default prometheus-service >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus service found (custom setup)${NC}"
    kubectl get svc -n default prometheus-service
    PROM_URL="http://localhost:9090"
else
    echo -e "${RED}✗ Prometheus service NOT found${NC}"
    echo "  Expected: $PROM_SERVICE in namespace $PROM_NAMESPACE"
    PROM_URL=""
fi

echo ""

# 5. Check Port-Forward
if [ ! -z "$PROM_URL" ]; then
    echo -e "${BLUE}5. Checking Prometheus Port-Forward...${NC}"
    
    if curl -s $PROM_URL/api/v1/query?query=up >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Prometheus accessible at $PROM_URL${NC}"
        
        # Check how many targets are up
        TARGETS=$(curl -s $PROM_URL/api/v1/targets | grep -o '"health":"up"' | wc -l)
        echo "  Active targets: $TARGETS"
        
        # Check node-exporter targets specifically
        NE_TARGETS=$(curl -s $PROM_URL/api/v1/targets | grep -o '"job":"node-exporter"' | wc -l)
        echo "  Node-Exporter targets: $NE_TARGETS"
        
        if [ $NE_TARGETS -ge 3 ]; then
            echo -e "${GREEN}  ✓ All 3 nodes reporting metrics${NC}"
        elif [ $NE_TARGETS -gt 0 ]; then
            echo -e "${YELLOW}  ⚠ Only $NE_TARGETS nodes reporting metrics (expected 3)${NC}"
        else
            echo -e "${RED}  ✗ No node-exporter metrics found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Prometheus NOT accessible at $PROM_URL${NC}"
        echo "  To fix: Run './pf-prometheus.sh' in another terminal"
        echo "  This script should be in: k3s-monitoring-setup/pf-prometheus.sh"
    fi
fi

echo ""

# 6. Check TCP-Connect Policy
echo -e "${BLUE}6. Checking TCP-Connect Tracing Policy...${NC}"
if kubectl get tracingpolicy tcp-connect -n kube-system >/dev/null 2>&1; then
    echo -e "${GREEN}✓ TCP-Connect TracingPolicy is deployed${NC}"
else
    echo -e "${YELLOW}⚠ TCP-Connect TracingPolicy NOT found${NC}"
    echo "  This policy enables outbound connection tracking"
    echo "  Deploy with: kubectl apply -f k3s-monitoring-setup/tcp-connect-policy.yaml"
fi

echo ""

# 7. Check All Nodes are Ready
echo -e "${BLUE}7. Checking Node Status...${NC}"
READY_NODES=$(kubectl get nodes --no-headers | grep -c "Ready")
TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l)

if [ $READY_NODES -eq $TOTAL_NODES ]; then
    echo -e "${GREEN}✓ All $TOTAL_NODES nodes are Ready${NC}"
    kubectl get nodes
else
    echo -e "${RED}✗ Only $READY_NODES/$TOTAL_NODES nodes are Ready${NC}"
    kubectl get nodes
fi

echo ""

# 8. Check Baseline Workloads
echo -e "${BLUE}8. Checking Baseline Workloads...${NC}"
BASELINE_PODS=$(kubectl get pods -n default --no-headers 2>/dev/null | grep -E "nginx|redis|traffic" | wc -l)

if [ $BASELINE_PODS -gt 0 ]; then
    echo -e "${GREEN}✓ Baseline workloads running ($BASELINE_PODS pods)${NC}"
    kubectl get pods -n default | grep -E "nginx|redis|traffic|api" || echo "  None currently"
else
    echo -e "${YELLOW}⚠ No baseline workloads detected${NC}"
    echo "  Expected baseline (before running scenario_runner):"
    echo "  - nginx-deployment"
    echo "  - redis-deployment"
    echo "  - traffic-generator"
    echo "  - api-deployment"
    echo ""
    echo "  To start baseline: bash k3s-monitoring-setup.sh or run_normal_baseline.sh"
fi

echo ""

# 9. Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VERIFICATION SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Before running scenario_runner.py, ensure:"
echo "  ✓ Kubernetes cluster is accessible"
echo "  ✓ Tetragon is running on all 3 nodes"
echo "  ✓ Node-Exporter is running (or kube-prometheus-stack is installed)"
echo "  ✓ Prometheus is accessible (port-forward running)"
echo "  ✓ All nodes are in 'Ready' state"
echo "  ✓ Baseline workloads are running"
echo ""
echo "If any checks are YELLOW (⚠) or RED (✗), fix them before proceeding!"
echo ""

if [ $READY_NODES -eq $TOTAL_NODES ] && [ ${NE_PODS:-0} -ge 1 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}System appears ready for scenario_runner!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Some checks need attention before running${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
