
#!/bin/bash

SERVICE="svc/prometheus-service"
LOCAL_PORT=9090
REMOTE_PORT=9090

while true
do
    echo "$(date) → Checking cluster..."

    kubectl cluster-info >/dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "Cluster unreachable. Waiting..."
        sleep 5
        continue
    fi

    echo "$(date) → Starting port-forward..."

    kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring

    echo "$(date) → Port-forward crashed. Restarting..."
    sleep 3
done