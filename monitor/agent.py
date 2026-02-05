import time
import psutil
import json
import os
import sys
import socket
import hashlib

# Configuration
NODE_NAME = os.getenv("NODE_NAME", socket.gethostname())
INTERVAL = int(os.getenv("INTERVAL", "5"))
SECRET_KEY = "byzantine-secret-key-123" # Shared secret

class MetricsAgent:
    def compute_sig(self, data_string):
        # HMAC-style Signature: SHA256(Data + Secret)
        # Stateless to allowing verification without history sync
        payload = f"{data_string}{SECRET_KEY}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_metrics(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        bytes_sent = net_io.bytes_sent
        bytes_recv = net_io.bytes_recv
        
        # Latency
        try:
            start_time = time.time()
            socket.gethostbyname('kubernetes.default.svc.cluster.local')
            latency = (time.time() - start_time) * 1000
        except:
            latency = -1
            
        data = {
            "node": NODE_NAME,
            "timestamp": time.time(),
            "cpu": cpu_percent,
            "mem": mem_percent,
            "net_out": bytes_sent,
            "net_in": bytes_recv,
            "latency": latency
        }
        
        # Sign the Data
        data_string = json.dumps(data, sort_keys=True)
        signature = self.compute_sig(data_string)
        
        data["signature"] = signature
        
        return data

def main():
    print(f"Starting STATLESS SECURE monitoring agent on {NODE_NAME}...")
    psutil.cpu_percent(interval=None) 
    
    agent = MetricsAgent()
    
    while True:
        try:
            metrics = agent.get_metrics()
            print(json.dumps(metrics), flush=True)
            time.sleep(INTERVAL)
        except Exception as e:
            print(f"Error collecting metrics: {e}", file=sys.stderr)
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
