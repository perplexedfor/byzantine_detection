import pandas as pd
from ml.anomaly_detector import AnomalyDetector
import sys

def test_on_file(filepath):
    print(f"Testing anomaly detector on {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    detector = AnomalyDetector()
    
    anomalies_found = 0
    total = 0
    
    with open("test_results.txt", "w") as f:
        f.write(f"{'Node':<15} {'CPU':<6} {'Score':<8} {'Status'}\n")
        f.write("-" * 40 + "\n")
        
        for index, row in df.iterrows():
            metrics = row.to_dict()
            status, score = detector.predict(metrics)
            total += 1
            
            if status == "Anomaly":
                anomalies_found += 1
                f.write(f"{row['node']:<15} {row['cpu']:<6.1f} {score:<8.3f} {status} <--- DETECTED\n")
            else:
                f.write(f"{row['node']:<15} {row['cpu']:<6.1f} {score:<8.3f} {status}\n")

        f.write("-" * 40 + "\n")
        f.write(f"Total Rows: {total}\n")
        f.write(f"Anomalies Detected: {anomalies_found}\n")
    
    print(f"Done. Found {anomalies_found} anomalies.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml/test_file.py <csv_file>")
    else:
        test_on_file(sys.argv[1])
