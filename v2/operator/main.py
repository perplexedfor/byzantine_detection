import kopf
import kubernetes
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import requests
import time
import threading
import json
import subprocess
from kubernetes import client, config
from collections import deque

# --- PATHS ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.join(_HERE, "v2", "ml", "ml")
if not os.path.exists(_ML_DIR):
    # Local dev fallback
    _ML_DIR = os.path.abspath(os.path.join(_HERE, "..", "ml", "ml"))

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-service.default.svc:9090")
MODEL_PATH     = os.getenv("MODEL_PATH",     os.path.join(_ML_DIR, "lstm_model.pth"))
SCALER_PATH    = os.getenv("SCALER_PATH",    os.path.join(_ML_DIR, "scaler.pkl"))
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", os.path.join(_HERE, "v2", "ml", "threshold.txt"))
if not os.path.exists(THRESHOLD_PATH):
    THRESHOLD_PATH = os.path.abspath(os.path.join(_HERE, "..", "ml", "threshold.txt"))
RF_MODEL_PATH  = os.getenv("RF_MODEL_PATH",  os.path.join(_ML_DIR, "rf_binary.pkl"))

SEQUENCE_LENGTH = 30
N_FEATURES      = 12

# Feature order (must match preprocess.py / feature_order.json)
FEATURES = [
    "avg_cpu", "avg_mem", "net_bytes_in", "net_bytes_out",
    "net_internal_bytes_in", "net_internal_bytes_out", "net_drop_rate",
    "exec_count", "unique_process_count", "tmp_exec_count",
    "outbound_connect_count", "mining_port_count",
]

# Network byte columns that need log1p (must match preprocess.py)
BYTE_COLS_IDX = [2, 3, 4, 5]  # indices of net_bytes_in/out, net_internal_bytes_in/out

# Feature weights (Synced with winning train_lstm.py config)
AE_FEATURE_WEIGHTS = torch.tensor([
    5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    5.0, 5.0, 10.0, 10.0, 20.0
])

# Sparse branch indices (eBPF features starting from drop_rate + ebpf)
SPARSE_INDICES = [6, 7, 8, 9, 10, 11]


# ── Trust Score Configuration ────────────────────────────────────────────────
# Hybrid trust scoring: RF is primary, AE catches novel attacks
TRUST_DECAY_RF_ONLY   = -2.0    # RF flags anomaly, AE says normal (low-confidence single-model)
TRUST_DECAY_AE_ONLY   = -15.0   # AE flags anomaly (High severity Security/Novelty)
TRUST_DECAY_BOTH      = -25.0   # BOTH flag anomaly (Catastrophic)
TRUST_REWARD_NORMAL   = +4.0    # Both agree normal — fast recovery after transient stress
TRUST_INITIAL         = 100.0
TRUST_CORDON_BELOW    = 40.0    # Cordon node if trust drops below this
TRUST_UNCORDON_ABOVE  = 60.0    # Uncordon node if trust recovers above this

# Cold-start grace period: skip trust penalties for the first N inferences
# after a node's buffer fills, to let transient startup data flush out.
GRACE_PERIOD_TICKS    = 5       # ~50s at 10s interval


# ── Model Definitions ────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, sparse_indices=None):
        super().__init__()
        self.sparse_indices = sparse_indices
        self.encoder_lstm  = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer  = nn.Linear(hidden_dim, input_dim)

        if sparse_indices:
            n_sparse = len(sparse_indices)
            self.sparse_branch = nn.Sequential(
                nn.Linear(n_sparse, 4),
                nn.ReLU(),
                nn.Linear(4, n_sparse)
            )

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent  = self.hidden2latent(h_n[-1])
        h_dec   = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(h_dec)
        recon = self.output_layer(dec_out)

        if self.sparse_indices:
            sparse_in = x[:, :, self.sparse_indices]
            recon[:, :, self.sparse_indices] += self.sparse_branch(sparse_in)
        return recon


# ── Trust Tracker (Hybrid) ───────────────────────────────────────────────────

class HybridTrustTracker:
    """Trust score system using both LSTM AE and RF signals.

    Trust decays faster when both models agree on an anomaly,
    and recovers slowly when both agree the node is healthy.
    """
    def __init__(self):
        self.scores = {}   # {node_name: float}
        self.history = {}  # {node_name: list of dicts} — for logging

    def get_score(self, node):
        return self.scores.get(node, TRUST_INITIAL)

    def update(self, node, ae_flags, rf_flags, rf_label="normal"):
        current = self.get_score(node)

        if ae_flags and rf_flags:
            delta = TRUST_DECAY_BOTH         # -25: catastrophic
            reason = f"BOTH flag (RF={rf_label})"
        elif rf_flags:
            delta = TRUST_DECAY_RF_ONLY      # -5: RF detected known infra fault
            reason = f"RF flags (RF={rf_label})"
        elif ae_flags:
            delta = TRUST_DECAY_AE_ONLY      # -15: AE detected Zero-Day/Security attack
            reason = "AE flags (SECURITY NOVELTY)"
        else:
            delta = TRUST_REWARD_NORMAL      # +2: both agree normal
            reason = "both normal"

        new_score = max(0.0, min(TRUST_INITIAL, current + delta))
        self.scores[node] = new_score
        return new_score, delta, reason

    def get_status(self, node):
        score = self.get_score(node)
        if score < TRUST_CORDON_BELOW:
            return "BANNED"
        elif score < TRUST_UNCORDON_ABOVE:
            return "PROBATION"
        return "HEALTHY"


# ── Global State ─────────────────────────────────────────────────────────────
LSTM_MODEL  = None
RF_MODEL    = None
SCALER      = None
THRESHOLD   = 0.0   # Loaded from threshold.txt at startup
TRUST       = HybridTrustTracker()
V1_API      = None
HTTP_SESSION = requests.Session()
NODE_BUFFERS: dict = {}      # {node_name: deque of feature vectors}
NODE_INFERENCE_COUNT: dict = {}  # {node_name: int} — tracks how many inferences have run
TETRAGON_CACHE: dict = {}    # {node_name: {metric_name: count}}
TETRAGON_LOCK = threading.Lock()


# ── Prometheus Data Fetch ────────────────────────────────────────────────────

def get_node_ip(node_name):
    try:
        if V1_API is None:
            return None
        node = V1_API.read_node(node_name)
        for addr in node.status.addresses:
            if addr.type == "InternalIP":
                return addr.address
    except Exception:
        pass
    return None


def fetch_prometheus_data(node_name: str):
    """Fetch 11-feature snapshot for a node, append to rolling buffer.
    Returns (SEQUENCE_LENGTH, 11) array when buffer is full, else None.
    """
    node_ip = get_node_ip(node_name)
    if node_ip is None:
        return None

    queries = {
        "avg_cpu":  f'1 - avg(rate(node_cpu_seconds_total{{instance=~"{node_ip}:.*",mode="idle"}}[1m]))',
        "avg_mem":  f'1 - (node_memory_MemAvailable_bytes{{instance=~"{node_ip}:.*"}} / node_memory_MemTotal_bytes{{instance=~"{node_ip}:.*"}})',
        "net_bytes_in":  f'rate(node_network_receive_bytes_total{{instance=~"{node_ip}:.*",device="enp0s3"}}[1m])',
        "net_bytes_out": f'rate(node_network_transmit_bytes_total{{instance=~"{node_ip}:.*",device="enp0s3"}}[1m])',
        "net_internal_bytes_in":  f'rate(node_network_receive_bytes_total{{instance=~"{node_ip}:.*",device="enp0s8"}}[1m])',
        "net_internal_bytes_out": f'rate(node_network_transmit_bytes_total{{instance=~"{node_ip}:.*",device="enp0s8"}}[1m])',
        "net_drop_rate": f'rate(node_network_receive_drop_total{{instance=~"{node_ip}:.*",device="enp0s3"}}[1m]) + rate(node_network_transmit_drop_total{{instance=~"{node_ip}:.*",device="enp0s3"}}[1m])',
    }

    # Tetragon eBPF metrics are populated by the event_processor thread
    with TETRAGON_LOCK:
        node_ebpf = TETRAGON_CACHE.get(node_name, {}).copy()
        # Reset specific "burst/count" metrics after reading so they act as "events per interval"
        # Just like we did in the cleaning/labeling scripts
        if node_name in TETRAGON_CACHE:
            TETRAGON_CACHE[node_name]["exec_count"] = 0.0
            TETRAGON_CACHE[node_name]["tmp_exec_count"] = 0.0
            TETRAGON_CACHE[node_name]["outbound_connect_count"] = 0.0
            TETRAGON_CACHE[node_name]["mining_port_count"] = 0.0
            # unique_process_count is usually a set in collection, but here we'll just track the count per window
            TETRAGON_CACHE[node_name]["unique_process_count"] = 0.0

    ebpf_data = {
        "exec_count": float(node_ebpf.get("exec_count", 0)),
        "unique_process_count": float(node_ebpf.get("unique_process_count", 0)),
        "tmp_exec_count": float(node_ebpf.get("tmp_exec_count", 0)),
        "outbound_connect_count": float(node_ebpf.get("outbound_connect_count", 0)),
        "mining_port_count": float(node_ebpf.get("mining_port_count", 0)),
    }

    def instant_query(q):
        try:
            resp = HTTP_SESSION.get(f"{PROMETHEUS_URL}/api/v1/query",
                                    params={"query": q}, timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data or "result" not in data["data"]:
                print(f"[{node_name}] Unexpected Prom response: {data}", flush=True)
                return None
            results = data["data"]["result"]
            if len(results) == 0:
                print(f"[{node_name}] Empty Prom result for: {q[:60]}...", flush=True)
                return None
            # Since we filter by instance in the query, just take the first result
            return float(results[0]["value"][1])
        except Exception as e:
            print(f"[{node_name}] Prometheus Request Failed: {e}", flush=True)
            return None

    try:
        values = []
        for feat_name in FEATURES:
            if feat_name in queries:
                val = instant_query(queries[feat_name])
                if val is None:
                    return None
                values.append(val)
            else:
                values.append(ebpf_data.get(feat_name, 0.0))

        if node_name not in NODE_BUFFERS:
            NODE_BUFFERS[node_name] = deque(maxlen=SEQUENCE_LENGTH)
        NODE_BUFFERS[node_name].append(values)

        if len(NODE_BUFFERS[node_name]) < SEQUENCE_LENGTH:
            print(f"[{node_name}] Buffering data... ({len(NODE_BUFFERS[node_name])}/{SEQUENCE_LENGTH})", flush=True)
            return None

        # On first buffer fill, flush stale Tetragon startup counts so the
        # first inference window isn't poisoned by accumulated boot events.
        if node_name not in NODE_INFERENCE_COUNT:
            NODE_INFERENCE_COUNT[node_name] = 0
            with TETRAGON_LOCK:
                if node_name in TETRAGON_CACHE:
                    for k in TETRAGON_CACHE[node_name]:
                        TETRAGON_CACHE[node_name][k] = 0.0
            print(f"[{node_name}] Buffer full — flushed startup eBPF counts", flush=True)

        return np.array(list(NODE_BUFFERS[node_name]))

    except Exception as e:
        print(f"[{node_name}] Master Exception in fetch_prometheus_data: {e}", flush=True)
        return None


def preprocess_sequence(raw_seq):
    """Apply log1p on byte cols, then scale. Returns (1, SEQ_LEN, N_FEATURES) tensor."""
    seq = raw_seq.copy()
    # log1p transform on network byte columns (match preprocess.py)
    seq[:, BYTE_COLS_IDX] = np.log1p(seq[:, BYTE_COLS_IDX])
    # Scale
    seq_scaled = SCALER.transform(seq)
    return seq_scaled


def run_inference(seq_scaled):
    """Run both LSTM AE and RF on a preprocessed sequence.
    Returns (ae_flags, rf_flags, rf_label, ae_loss).
    """
    seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)  # (1, 30, 11)

    # --- LSTM Autoencoder ---
    ae_flags = False
    ae_loss = 0.0
    if LSTM_MODEL is not None:
        with torch.no_grad():
            recon = LSTM_MODEL(seq_tensor)
            w = AE_FEATURE_WEIGHTS.to(recon.device)
            # Use winning MSE logic (Mean Squared Error) for inference
            sq_err = ((recon - seq_tensor) ** 2) * w
            ae_loss = torch.mean(sq_err, dim=(1,2))[0].item()
            ae_flags = ae_loss > THRESHOLD

    # --- Random Forest ---
    rf_flags = False
    rf_label = "normal"
    if RF_MODEL is not None:
        # Clip scaled values to [0,1] — the MinMaxScaler was fit on normal
        # training data, so live values can exceed this range during/after
        # stress transitions, producing std/min/max the RF never trained on.
        seq_clipped = np.clip(seq_scaled, 0.0, 1.0)

        # Flatten window to tabular features (mean/std/min/max per feature)
        stats = []
        for i in range(seq_clipped.shape[1]):
            col = seq_clipped[:, i]
            stats.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
        flat = np.array(stats, dtype=np.float32).reshape(1, -1)
        flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)

        rf_label = RF_MODEL.predict(flat)[0]
        rf_flags = (rf_label != "normal")


    return ae_flags, rf_flags, rf_label, ae_loss


# ── Tetragon Log Streaming ──────────────────────────────────────────────────

def process_tetragon_event(event_line):
    try:
        event = json.loads(event_line)
        node_name = event.get('node_name', 'unknown')
        
        with TETRAGON_LOCK:
            if node_name not in TETRAGON_CACHE:
                TETRAGON_CACHE[node_name] = {
                    "exec_count": 0, "unique_process_count": 0, "tmp_exec_count": 0,
                    "outbound_connect_count": 0, "mining_port_count": 0
                }
            
            cache = TETRAGON_CACHE[node_name]
            
            if 'process_exec' in event:
                cache["exec_count"] += 1
                binary = event['process_exec']['process'].get('binary', '')
                # Minimal unique process tracking (just increment if non-empty for simplicity)
                if binary: cache["unique_process_count"] += 1
                
                if any(binary.startswith(p) for p in ['/tmp/', '/dev/shm/', '/var/tmp/']):
                    cache["tmp_exec_count"] += 1
            
            if 'process_kprobe' in event:
                kprobe = event['process_kprobe']
                if kprobe.get('function_name') == 'tcp_connect':
                    cache["outbound_connect_count"] += 1
                    # Port check logic
                    args = kprobe.get('args', [])
                    for arg in args:
                        sock = arg.get('sock_arg') or arg.get('sock') or {}
                        dport = sock.get('dport') or arg.get('dport')
                        if dport in [3333, 4444, 5555, 6666, 7777, 8332, 8333]:
                            cache["mining_port_count"] += 1
    except:
        pass

def tetragon_streamer_thread():
    """Background thread to discover and stream Tetragon logs per node pod."""
    print("  🚀 Starting Tetragon eBPF Streamer...", flush=True)
    while True:
        try:
            v1 = client.CoreV1Api()
            pods = v1.list_namespaced_pod("kube-system", label_selector="app.kubernetes.io/name=tetragon")

            for pod in pods.items:
                pod_name  = pod.metadata.name
                node_name = pod.spec.node_name

                def stream_pod(p_name=pod_name, n_name=node_name):
                    """Stream logs using the Kubernetes Python API (no kubectl needed)."""
                    print(f"  📡 Tetragon stream started: pod={p_name} node={n_name}", flush=True)
                    while True:
                        try:
                            v1_log = client.CoreV1Api()
                            resp = v1_log.read_namespaced_pod_log(
                                name=p_name,
                                namespace="kube-system",
                                container="export-stdout",
                                follow=True,
                                tail_lines=0,
                                _preload_content=False  # Stream raw bytes
                            )
                            for line in resp.stream():
                                decoded = line.decode("utf-8", errors="ignore").strip()
                                if decoded:
                                    process_tetragon_event(decoded)
                        except Exception as e:
                            print(f"  ⚠️ Tetragon stream for {n_name} broke: {e}. Reconnecting...", flush=True)
                            time.sleep(5)

                t = threading.Thread(target=stream_pod, daemon=True)
                t.start()

            # Stay alive — the pod threads are daemon threads
            while True:
                time.sleep(30)
        except Exception as e:
            print(f"  ❌ Tetragon Streamer Error: {e}. Retrying in 30s...", flush=True)
            time.sleep(30)


# ── Kopf Handlers ────────────────────────────────────────────────────────────

@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    global LSTM_MODEL, RF_MODEL, SCALER, THRESHOLD, V1_API
    print("--- 🛡️ Starting Byzantine Defense Operator v2 (Hybrid) ---", flush=True)

    # Start Tetragon Streaming in background — must happen AFTER incluster config is loaded
    # We load kube config first here so the streamer thread can use it
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    V1_API = client.CoreV1Api()

    t = threading.Thread(target=tetragon_streamer_thread, daemon=True)
    t.start()

    try:
        # Load LSTM Autoencoder
        if os.path.exists(MODEL_PATH):
            LSTM_MODEL = LSTMAutoencoder(
                input_dim=N_FEATURES, hidden_dim=64, latent_dim=8,
                sparse_indices=SPARSE_INDICES
            )
            LSTM_MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            LSTM_MODEL.eval()
            print(f"  ✅ LSTM Autoencoder loaded ({MODEL_PATH})", flush=True)
        else:
            print(f"  ⚠️  LSTM model not found at {MODEL_PATH}", flush=True)

        # Load Random Forest
        if os.path.exists(RF_MODEL_PATH):
            RF_MODEL = joblib.load(RF_MODEL_PATH)
            print(f"  ✅ Random Forest loaded ({RF_MODEL_PATH})", flush=True)
        else:
            print(f"  ⚠️  RF model not found at {RF_MODEL_PATH}", flush=True)

        # Load Scaler
        if os.path.exists(SCALER_PATH):
            SCALER = joblib.load(SCALER_PATH)
            print(f"  ✅ Scaler loaded ({SCALER_PATH})", flush=True)

        # Load Threshold
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, 'r') as f:
                THRESHOLD = float(f.read().strip())
            print(f"  ✅ Threshold loaded ({THRESHOLD:.6f})", flush=True)

        mode = "HYBRID" if (LSTM_MODEL and RF_MODEL) else "AE-only" if LSTM_MODEL else "RF-only"
        print(f"\n  🛡️ Detection Mode: {mode}", flush=True)
        print(f"  Trust: decay RF={TRUST_DECAY_RF_ONLY}, AE={TRUST_DECAY_AE_ONLY}, "
              f"BOTH={TRUST_DECAY_BOTH}, reward={TRUST_REWARD_NORMAL}", flush=True)
        print(f"  Cordon < {TRUST_CORDON_BELOW}, Uncordon > {TRUST_UNCORDON_ABOVE}", flush=True)

    except Exception as e:
        print(f"  ❌ Error loading models: {e}", flush=True)


@kopf.timer('security.example.com', 'v1', 'byzantinepolicies', interval=10.0)
def reconcile(spec, name, logger, **kwargs):
    if SCALER is None or V1_API is None:
        return
    if LSTM_MODEL is None and RF_MODEL is None:
        return

    for node in V1_API.list_node().items:
        n_name = node.metadata.name
        # Skip the control-plane node (check labels, not name)
        labels = node.metadata.labels or {}
        if "node-role.kubernetes.io/control-plane" in labels:
            continue

        raw_seq = fetch_prometheus_data(n_name)
        if raw_seq is None:
            continue

        # Preprocess and run hybrid inference
        seq_scaled = preprocess_sequence(raw_seq)
        ae_flags, rf_flags, rf_label, ae_loss = run_inference(seq_scaled)

        # Track per-node inference count for grace period
        NODE_INFERENCE_COUNT[n_name] = NODE_INFERENCE_COUNT.get(n_name, 0) + 1
        in_grace = NODE_INFERENCE_COUNT[n_name] <= GRACE_PERIOD_TICKS

        if in_grace:
            # During grace period: log but DON'T penalize trust
            score = TRUST.get_score(n_name)
            delta = 0.0
            reason = f"GRACE ({NODE_INFERENCE_COUNT[n_name]}/{GRACE_PERIOD_TICKS})"
            status = TRUST.get_status(n_name)
        else:
            # Update trust score
            score, delta, reason = TRUST.update(n_name, ae_flags, rf_flags, rf_label)
            status = TRUST.get_status(n_name)

        # Log
        ae_str = "⚠️ AE" if ae_flags else "  AE"
        rf_str = f"⚠️ RF({rf_label})" if rf_flags else "  RF(normal)"
        grace_tag = " [GRACE]" if in_grace else ""
        log_line = (f"[{n_name}] AE_loss={ae_loss:.4f} | {ae_str} | {rf_str} | "
                    f"Trust={score:5.1f} ({delta:+.0f}) | {status} | {reason}{grace_tag}")
        print(log_line, flush=True)

        # Log to file
        log_path = os.path.join(_HERE, "operator_metrics.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {log_line}\n")

        # Enforce cordon/uncordon (never during grace period)
        if not in_grace:
            if score < TRUST_CORDON_BELOW:
                cordon_node(n_name, logger)
            elif spec.get('autoRemediate', True) and score > TRUST_UNCORDON_ABOVE:
                uncordon_node(n_name, logger)


def cordon_node(node_name, logger):
    try:
        node = V1_API.read_node(node_name)
        if not node.spec.unschedulable:
            V1_API.patch_node(node_name, {"spec": {"unschedulable": True}})
            print(f"🚫 CORDONED  {node_name}  (trust={TRUST.get_score(node_name):.1f})", flush=True)
    except Exception as e:
        print(f"❌ Failed to cordon {node_name}: {e}", flush=True)


def uncordon_node(node_name, logger):
    try:
        node = V1_API.read_node(node_name)
        if node.spec.unschedulable:
            V1_API.patch_node(node_name, {"spec": {"unschedulable": False}})
            print(f"✅ UNCORDONED {node_name}  (trust={TRUST.get_score(node_name):.1f})", flush=True)

            # Flush stale data so RF doesn't immediately re-flag from
            # old anomalous samples still sitting in the sliding window.
            if node_name in NODE_BUFFERS:
                NODE_BUFFERS[node_name].clear()
            if node_name in NODE_INFERENCE_COUNT:
                del NODE_INFERENCE_COUNT[node_name]  # triggers fresh grace period
            with TETRAGON_LOCK:
                if node_name in TETRAGON_CACHE:
                    for k in TETRAGON_CACHE[node_name]:
                        TETRAGON_CACHE[node_name][k] = 0.0
            print(f"  🔄 Flushed buffers for {node_name} — fresh grace period", flush=True)
    except Exception as e:
        print(f"❌ Failed to uncordon {node_name}: {e}", flush=True)

