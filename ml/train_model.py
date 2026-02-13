import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "production_baseline.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml", "scaler.pkl")

def train():
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Select features for training
    features = ['cpu', 'mem', 'net_in', 'net_out', 'latency']
    X = df[features]

    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")
    # contamination=0.01 because we know our baseline data is CLEAN/NORMAL
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_scaled)
    
    # Check scores to calibrate threshold
    # Higher score = More Normal. Lower (negative) = Anomaly.
    scores = model.decision_function(X_scaled)
    print(f"Training Data Scores -> Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}")
    print(f"Suggested Threshold (Min - 0.05): {scores.min() - 0.05:.3f}")

    print("Saving model and scaler...")
    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
