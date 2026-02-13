import kopf
import kubernetes
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import requests
import time
from kubernetes import client, config

# --- MONITORING CONFIG ---
# In-Cluster: http://prometheus-service.default.svc:9090
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-service.default.svc:9090")
MODEL_PATH = os.getenv("MODEL_PATH", "v2/ml/lstm_model.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "v2/ml/scaler.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "v2/ml/threshold.txt")

# --- MODEL DEFINITION ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = self.hidden2latent(hidden[-1])
        hidden_decoded = self.latent2hidden(latent)
        hidden_repeated = hidden_decoded.unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(hidden_repeated)
        output = self.output_layer(decoded)
        return output

# --- TRUST TRACKER ---
class TrustTracker:
    def __init__(self):
        self.scores = {}  # {node_name: float}
        self.status = {}  # {node_name: str}

    def get_score(self, node):
        return self.scores.get(node, 100.0)

    def update(self, node, is_anomaly):
        current = self.get_score(node)
        if is_anomaly:
            new_score = max(0.0, current - 5.0) # Decay
        else:
            new_score = min(100.0, current + 1.0) # Reward
        
        self.scores[node] = new_score
        return new_score

# --- GLOBAL STATE ---
MODEL = None
SCALER = None
THRESHOLD = 0.5
TRUST_SYSTEM = TrustTracker()
SEQUENCE_LENGTH = 10

def fetch_prometheus_data(node_name):
    """
    Fetch last 10 data points (50s) for a specific node.
    Returns: np.array of shape (10, 4) or None if fetch fails.
    Feature Order: [cpu, mem, net_in, net_out]
    """
    end_time = time.time()
    start_time = end_time - 60 
    
    queries = {
        "cpu": f'100 - (avg by (instance) (irate(node_cpu_seconds_total{{mode="idle"}}[1m])) * 100)',
        "mem": f'100 * (1 - ((node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes) / node_memory_MemTotal_bytes))',
        "net_in": f'sum by (instance) (rate(node_network_receive_bytes_total[1m]))',
        "net_out": f'sum by (instance) (rate(node_network_transmit_bytes_total[1m]))'
    }
    
    try:
        # Helper to process one query
        def get_series(query_str):
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params={
                'query': query_str,
                'start': start_time,
                'end': end_time,
                'step': '5s'
            }, timeout=3.0) 
            return resp.json()['data']['result']

        cpu_series = get_series(queries['cpu'])
        mem_series = get_series(queries['mem'])
        in_series = get_series(queries['net_in'])
        out_series = get_series(queries['net_out'])
        
        # Heuristic: Match series where instance IP resolves to Node IP?
        # For Demo/MVP: Just take the first series found (Assuming single node cluster or specific target)
        # In a real multi-node env, we MUST match regex.
        
        # Simplification: If series is empty, return None
        if not cpu_series: return None
        
        # Taking the first available series for MVP demo (Kind Worker)
        c_vals = cpu_series[0]['values']
        m_vals = mem_series[0]['values'] if mem_series else []
        i_vals = in_series[0]['values'] if in_series else []
        o_vals = out_series[0]['values'] if out_series else []
        
        # Align data
        length = min(len(c_vals), len(m_vals), len(i_vals), len(o_vals))
        if length < SEQUENCE_LENGTH:
            return None # Not enough data yet
            
        # Take last 10
        data = []
        for i in range(1, SEQUENCE_LENGTH + 1):
            idx = -i
            row = [
                float(c_vals[idx][1]),
                float(m_vals[idx][1]),
                float(i_vals[idx][1]),
                float(o_vals[idx][1])
            ]
            data.insert(0, row)
            
        return np.array(data)

    except Exception as e:
        print(f"Error fetching Prometheus data: {e}", flush=True)
        return None


# --- KOPF HANDLERS ---
@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    global MODEL, SCALER, THRESHOLD
    print("--- 🛡️ Starting Byzantine Defense Operator v2 ---", flush=True)
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
        
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = LSTMAutoencoder(input_dim=4)
            MODEL.load_state_dict(torch.load(MODEL_PATH))
            MODEL.eval()
        
        if os.path.exists(SCALER_PATH):
            SCALER = joblib.load(SCALER_PATH)

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, 'r') as f:
                THRESHOLD = float(f.read().strip())
        
        print(f"✅ Loaded Brain (Threshold: {THRESHOLD:.4f})", flush=True)
    except Exception as e:
        print(f"❌ Artifact Config Error: {e}", flush=True)

@kopf.timer('security.example.com', 'v1', 'byzantinepolicies', interval=10.0)
def reconcile(spec, name, logger, **kwargs):
    # print(f"--- Reconcile Loop {time.time()} ---", flush=True)
    if MODEL is None or SCALER is None:
        logger.warning("Waiting for model/scaler...")
        return

    # Policy Settings
    limit_score = 40.0 # Banned if < 40
    
    # Scan Nodes
    v1 = client.CoreV1Api()
    for node in v1.list_node().items:
        n_name = node.metadata.name
        if "control-plane" in n_name: continue
        
        # 1. Fetch
        # print(f"Fetching data for {n_name}...", flush=True)
        raw_seq = fetch_prometheus_data(n_name)
        if raw_seq is None:
            # logger.info(f"Skipping {n_name} (Insufficient Data)")
            continue
            
        # 2. Preprocess
        seq_scaled = SCALER.transform(raw_seq)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0) # (1, 10, 4)
        
        # 3. Inference
        with torch.no_grad():
            reconstruction = MODEL(seq_tensor)
            loss = torch.mean(torch.abs(reconstruction - seq_tensor)).item()
            
        is_anomaly = loss > THRESHOLD
        
        # 4. Trust Update
        score = TRUST_SYSTEM.update(n_name, is_anomaly)
        logger.info(f"Node: {n_name} | Loss: {loss:.4f} | Anomaly: {is_anomaly} | Trust: {score}")
        print(f"Node: {n_name} | Loss: {loss:.4f} | Trust: {score}", flush=True)
        
        # 5. Enforcement
        if score < limit_score:
            cordon_node(n_name, logger)
        elif spec.get('autoRemediate', True) and score > 60:
             uncordon_node(n_name, logger)

def cordon_node(node_name, logger):
    v1 = client.CoreV1Api()
    node = v1.read_node(node_name)
    if not node.spec.unschedulable:
        v1.patch_node(node_name, {"spec": {"unschedulable": True}})
        logger.error(f"🚫 Cordoned {node_name} due to Low Trust")
        print(f"🚫 Cordoned {node_name}", flush=True)

def uncordon_node(node_name, logger):
    v1 = client.CoreV1Api()
    node = v1.read_node(node_name)
    if node.spec.unschedulable:
        v1.patch_node(node_name, {"spec": {"unschedulable": False}})
        logger.info(f"✅ Uncordoned {node_name} (Trust Recovered)")
        print(f"✅ Uncordoned {node_name}", flush=True)
