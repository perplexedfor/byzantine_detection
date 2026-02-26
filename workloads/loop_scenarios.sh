#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "====================================================="
echo "   Automated Kubernetes ML Dataset Generator Loop"
echo "====================================================="
echo "This script will infinitely loop the scenario_runner.py"
echo "to generate a massive dataset for Deep Learning."
echo "Press Ctrl+C to stop the loop at any time."
echo "====================================================="

# Wait a few seconds for user to read warning
sleep 5

LOOP_COUNT=1

while true; do
    echo ""
    echo "====================================================="
    echo "   STARTING GENERATION LOOP #$LOOP_COUNT"
    echo "====================================================="
    
    # Run the Python script (it now auto-appends to the CSV)
    python3 scenario_runner.py
    
    echo "Finished Loop #$LOOP_COUNT."
    echo "Waiting 60 seconds before starting the next loop to ensure total cluster normalization..."
    sleep 60
    
    LOOP_COUNT=$((LOOP_COUNT+1))
done
