import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from ml.anomaly_detector import AnomalyDetector

# Configuration
DATA_FILE = "dataset/fault_test.csv"
OUTPUT_DIR = "evaluation/plots"
MODEL_PATH = "ml/model.pkl"
SCALER_PATH = "ml/scaler.pkl"

def generate_graphs():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Detector to get scores
    print("Loading model for scoring...")
    try:
        detector = AnomalyDetector()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Calculate scores for all point
    print("Scoring data points...")
    scores = []
    for index, row in df.iterrows():
        metrics = row.to_dict()
        # We need the raw score, not the threshold decision
        # accessing internal model directly for batch scoring would be faster, but let's stick to the class
        # The class predict returns (status, score)
        _, score = detector.predict(metrics)
        scores.append(score)
    
    df['anomaly_score'] = scores
    
    # Add a pseudo-time axis (Index as Time for MVP)
    df['time_step'] = df.index

    # --- Plot 1: CPU Usage Over Time (The Attack) ---
    print("Generating CPU plot...")
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    # Plot CPU for each node
    sns.lineplot(data=df, x='time_step', y='cpu', hue='node', marker='o')
    
    plt.title("CPU Usage During Byzantine Attack Simulation", fontsize=15)
    plt.xlabel("Time Step (Data Point Index)", fontsize=12)
    plt.ylabel("CPU Usage (%)", fontsize=12)
    plt.axhline(y=5.0, color='r', linestyle='--', label='Normal CPU Max')
    plt.legend()
    
    cpu_plot_path = os.path.join(OUTPUT_DIR, "cpu_attack_visualization.png")
    plt.savefig(cpu_plot_path)
    print(f"Saved {cpu_plot_path}")
    plt.close()

    # --- Plot 2: Anomaly Score Over Time (The Detection) ---
    print("Generating Anomaly Score plot...")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x='time_step', y='anomaly_score', hue='node', marker='x')
    
    plt.title("Anomaly Scores During Attack", fontsize=15)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Anomaly Score (Lower is Abnormal)", fontsize=12)
    
    # Threshold line
    THRESHOLD = 0.08
    plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
    
    # Shade the anomaly region
    plt.fill_between(df['time_step'], -0.2, THRESHOLD, color='red', alpha=0.1, label='Anomaly Zone')
    
    plt.legend(loc='upper right')
    plt.ylim(bottom=-0.15, top=0.2) # Zoom in on relevant range
    
    score_plot_path = os.path.join(OUTPUT_DIR, "anomaly_score_visualization.png")
    plt.savefig(score_plot_path)
    print(f"Saved {score_plot_path}")
    plt.close()

if __name__ == "__main__":
    generate_graphs()
