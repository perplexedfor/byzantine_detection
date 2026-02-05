import subprocess
import json
import time
import sys
import os
from ml.anomaly_detector import AnomalyDetector

# Configuration
import hashlib

# Configuration
THRESHOLD = -0.045 
ISOLATED_NODES = set()
DASHBOARD_STATE_FILE = "dashboard/state.json"
SECRET_KEY = "byzantine-secret-key-123"

class SignatureVerifier:
    def __init__(self):
        pass

    def verify(self, data):
        node = data.get('node')
        signature = data.get('signature')
        
        # Remove signature from data to reconstruct the payload
        payload_data = data.copy()
        if 'signature' in payload_data:
            del payload_data['signature']
        
        data_string = json.dumps(payload_data, sort_keys=True)
        
        # Recompute hash (Stateless)
        payload = f"{data_string}{SECRET_KEY}"
        expected_hash = hashlib.sha256(payload.encode()).hexdigest()
        
        if signature == expected_hash:
            return True
        else:
            print(f"[SECURITY] Signature Mismatch for {node}!")
            # print(f"Expected: {expected_hash}")
            # print(f"Received: {signature}")
            return False

class TrustTracker:
    def __init__(self):
        self.scores = {}  # {node: score}
        self.status = {}  # {node: status_str} ('Healthy', 'Probation', 'Banned', 'Compromised')

    def get_score(self, node):
        return self.scores.get(node, 100.0)

    def set_compromised(self, node):
        self.scores[node] = 0.0
        self.status[node] = "Compromised"
        print(f"[SECURITY] Critical Integrity Violation on {node}. Banning immediately.")
        self._apply_cordon_drain(node)

    def update(self, node, is_anomaly):
        current = self.get_score(node)
        
        if is_anomaly:
            # Decay fast
            new_score = max(0.0, current - 15.0)
        else:
            # Recover slowly
            new_score = min(100.0, current + 2.0)
        
        self.scores[node] = new_score
        self._enforce_policy(node, new_score)
        return new_score

    def _enforce_policy(self, node, score):
        # Policy Definition
        # 75-100: Healthy (Green)
        # 30-75:  Probation (Yellow) -> Taint
        # 0-30:   Banned (Red) -> Cordon + Drain

        current_status = self.status.get(node, "Healthy")
        if current_status == "Compromised":
            return # Do not unban a compromised node automatically

        if score < 30:
            new_status = "Banned"
            if current_status != "Banned":
                print(f"[Policy] Banning {node} (Score: {score:.1f})")
                self._apply_cordon_drain(node)
        
        elif score < 75:
            new_status = "Probation"
            if current_status != "Probation" and current_status != "Banned":
                print(f"[Policy] Placing {node} on Probation (Score: {score:.1f})")
                self._apply_taint(node)
            elif current_status == "Banned":
                 # Recovery from Ban not implemented automatically for safety, but logic allows it
                 pass 
        
        else:
            new_status = "Healthy"
            if current_status == "Probation":
                print(f"[Policy] Restoring {node} to Healthy (Score: {score:.1f})")
                self._remove_taint(node)
            elif current_status == "Banned":
                # Manual intervention usually required to unban, but let's allow Uncordon if score recovers
                print(f"[Policy] Restoring {node} to Healthy from Ban (Score: {score:.1f})")
                self._remove_cordon(node)
                self._remove_taint(node)

        self.status[node] = new_status

    def _apply_taint(self, node):
        try:
            # taint key=suspicious:NoSchedule
            subprocess.run(["wsl", "kubectl", "taint", "nodes", node, "suspicious=true:NoSchedule", "--overwrite"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _remove_taint(self, node):
        try:
            subprocess.run(["wsl", "kubectl", "taint", "nodes", node, "suspicious:NoSchedule-"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _apply_cordon_drain(self, node):
        try:
            subprocess.run(["wsl", "kubectl", "cordon", node], check=True)
            # Async drain
            subprocess.Popen(["wsl", "kubectl", "drain", node, "--ignore-daemonsets", "--delete-emptydir-data", "--force"], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to ban {node}: {e}")

    def _remove_cordon(self, node):
        try:
            subprocess.run(["wsl", "kubectl", "uncordon", node], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def save_dashboard_state(node_states):
    max_retries = 3
    for i in range(max_retries):
        try:
            os.makedirs("dashboard", exist_ok=True)
            tmp_file = DASHBOARD_STATE_FILE + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(node_states, f)
            
            if os.path.exists(DASHBOARD_STATE_FILE):
                os.remove(DASHBOARD_STATE_FILE) # Windows replace workaround
            os.replace(tmp_file, DASHBOARD_STATE_FILE)
            break
        except Exception as e:
            if i == max_retries - 1:
                print(f"Error saving state: {e}")
            time.sleep(0.1)

def run_controller():
    print("Loading ML Model...")
    try:
        detector = AnomalyDetector()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    tracker = TrustTracker()
    verifier = SignatureVerifier()
    
    print(f"Starting Trust-Aware Controller (Threshold={THRESHOLD})...")
    print("Policies: <75=Probation, <30=Banned, InvalidHash=Compromised")

    cmd = ["wsl", "kubectl", "logs", "-l", "app=monitor-agent", "-f"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

    node_states = {} 

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            try:
                line = line.strip()
                if not line: continue
                
                data = json.loads(line)
                
                # Ignore stale logs (older than 5 seconds) to avoid false positives during rollouts
                if time.time() - data.get('timestamp', 0) > 5:
                    continue

                node = data['node']
                
                # 1. SECURITY CHECK (Hash Chain)
                if not verifier.verify(data):
                    tracker.set_compromised(node)
                    # Update state to reflect compromise
                    node_states[node] = {
                        "last_seen": time.time(),
                        "cpu": data.get('cpu'),
                        "score": 0.0,
                        "status": "Compromised",
                        "raw_status": "Tampered"
                    }
                    save_dashboard_state(node_states)
                    continue # Skip ML check if tampered
                
                # 2. ANOMALY CHECK (ML)
                status, raw_score = detector.predict(data, threshold=THRESHOLD)
                is_anomaly = (status == "Anomaly")
                
                # 3. TRUST UPDATE
                trust_score = tracker.update(node, is_anomaly)
                
                if is_anomaly:
                    print(f"ANOMALY: {node} (Raw ML: {raw_score:.3f}, Trust: {trust_score:.1f})")
                
                # Update State for Dashboard
                node_states[node] = {
                    "last_seen": time.time(),
                    "cpu": data.get('cpu'),
                    "score": trust_score, 
                    "status": tracker.status.get(node, "Healthy"),
                    "raw_status": status
                }
                save_dashboard_state(node_states)

            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"Error processing loop: {e}")

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        process.terminate()

if __name__ == "__main__":
    run_controller()

