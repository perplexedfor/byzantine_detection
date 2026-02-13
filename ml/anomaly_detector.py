import joblib
import pandas as pd
import os
import numpy as np

class AnomalyDetector:
    def __init__(self, model_path="ml/model.pkl", scaler_path="ml/scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = ['cpu', 'mem', 'net_in', 'net_out', 'latency']

    def predict(self, metrics_dict, threshold=-0.085):
        """
        metrics_dict: dict with keys matching features
        threshold: score below this is Anomaly (calibrated to -0.085)
        Returns: Normal/Anomaly, Score
        """
        # Create DataFrame from single record
        df = pd.DataFrame([metrics_dict])
        
        # Ensure correct column order
        X = df[self.features]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Score samples. Lower is more anomalous. 
        # API returns negative for anomaly, positive for normal based on offset.
        # We override with manual threshold to tune sensitivity.
        score = self.model.decision_function(X_scaled)[0]
        
        status = "Normal" if score > threshold else "Anomaly"
        return status, score

if __name__ == "__main__":
    # Test run
    print("Loading detector...")
    detector = AnomalyDetector()
    
    # Test normal looking data
    normal_data = {'cpu': 1.5, 'mem': 20.0, 'net_in': 350000, 'net_out': 200000, 'latency': 3.5}
    status, score = detector.predict(normal_data)
    print(f"NORMAL_TEST: {status} ({score:.3f})")
    
    # Test anomaly (Extreme CPU)
    anomaly_data = {'cpu': 99.0, 'mem': 20.0, 'net_in': 350000, 'net_out': 200000, 'latency': 3.5}
    status, score = detector.predict(anomaly_data)
    print(f"CPU_SPIKE_TEST: {status} ({score:.3f})")

    # Test Anomaly (Network Spike)
    anomaly_net = {'cpu': 1.5, 'mem': 20.0, 'net_in': 90000000, 'net_out': 200000, 'latency': 3.5}
    status, score = detector.predict(anomaly_net)
    print(f"NET_SPIKE_TEST: {status} ({score:.3f})")

