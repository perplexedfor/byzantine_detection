#!/bin/bash
# Port-forward watchdog for Prometheus
# Restarts the tunnel immediately if it crashes, so no data is lost.

NAMESPACE="monitoring"
SERVICE="svc/prometheus-kube-prometheus-prometheus"
LOCAL_PORT=9090
REMOTE_PORT=9090
LOGFILE="/tmp/pf-watchdog.log"
PIDFILE="/tmp/pf-watchdog.pid"

echo $$ > "$PIDFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started (PID $$)" | tee -a "$LOGFILE"

cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog stopping — killing port-forward..." | tee -a "$LOGFILE"
    kill "$PF_PID" 2>/dev/null
    rm -f "$PIDFILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting port-forward..." | tee -a "$LOGFILE"

    kubectl port-forward "$SERVICE" "$LOCAL_PORT:$REMOTE_PORT" -n "$NAMESPACE" \
        >> "$LOGFILE" 2>&1 &
    PF_PID=$!

    # Wait for the tunnel to be ready
    sleep 3

    # Confirm it's actually listening
    if ! ss -tlnp | grep -q ":$LOCAL_PORT"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: port-forward failed to bind — retrying in 5s" | tee -a "$LOGFILE"
        kill "$PF_PID" 2>/dev/null
        sleep 5
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tunnel UP on localhost:$LOCAL_PORT (PID $PF_PID)" | tee -a "$LOGFILE"

    # Block until port-forward dies
    wait "$PF_PID"
    EXIT_CODE=$?

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port-forward exited (code $EXIT_CODE) — restarting in 2s..." | tee -a "$LOGFILE"
    sleep 2
done