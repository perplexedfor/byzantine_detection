import subprocess
import json
import csv
import time
import os
import argparse
import sys

def collect_data(duration_seconds=60, output_file="dataset/normal_behavior.csv"):
    print(f"Collecting data for {duration_seconds} seconds to {output_file}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['node', 'timestamp', 'cpu', 'mem', 'net_out', 'net_in', 'latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        # Command to stream logs from all monitor-agent pods
        # We use wsl if on Windows calling into WSL, otherwise just kubectl if in linux context
        # Assuming running from Windows host based on previous interactions
        cmd = ["wsl", "kubectl", "logs", "-l", "app=monitor-agent", "-f"]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        except FileNotFoundError:
             print("Error: 'wsl' command not found. Are you on Windows?")
             return

        start_time = time.time()
        count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        print("Subprocess exited.")
                        break
                    continue
                
                try:
                    line = line.strip()
                    if not line: continue
                    
                    data = json.loads(line)
                    # Filter to ensure we only have expected keys
                    row = {k: data.get(k, 0) for k in fieldnames}
                    writer.writerow(row)
                    count += 1
                    if count % 10 == 0:
                        print(f"Collected {count} data points...")
                        csvfile.flush()
                except json.JSONDecodeError:
                    # Ignore non-JSON logs (pod startup logs etc)
                    pass
                except Exception as e:
                    print(f"Error processing line: {e}")
                    
        except KeyboardInterrupt:
            print("Stopping collection...")
        finally:
            process.terminate()
            print(f"Data collection finished. Total records: {count}. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--output", type=str, default="dataset/normal_behavior.csv", help="Output CSV file path")
    args = parser.parse_args()
    collect_data(args.duration, args.output)
