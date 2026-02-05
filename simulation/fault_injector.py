import subprocess
import argparse
import time
import sys

def inject_cpu_fault(node_name, duration):
    print(f"Injecting CPU spike on {node_name} for {duration} seconds...")
    # Run a CPU intensive process: yes > /dev/null
    cmd = ["wsl", "docker", "exec", "-d", node_name, "sh", "-c", f"timeout {duration} yes > /dev/null"]
    try:
        subprocess.run(cmd, check=True)
        print(f"CPU fault started on {node_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to inject CPU fault: {e}")

def inject_memory_fault(node_name, duration):
    print(f"Injecting Memory leak on {node_name} for {duration} seconds...")
    # allocate 500MB (approx)
    python_mem_load = "python3 -c \"a=' '*500*1024*1024; import time; time.sleep(" + str(duration) + ")\""
    cmd = ["wsl", "docker", "exec", "-d", node_name, "sh", "-c", python_mem_load]
    try:
        subprocess.run(cmd, check=True)
        print(f"Memory fault started on {node_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to inject Memory fault: {e}")

def inject_network_fault(node_name, duration, delay_ms=300):
    # This requires 'tc' (traffic control) which might not be installed in kind nodes by default
    # But let's try. Kind nodes are usually Ubuntu/Debian based.
    print(f"Injecting Network latency ({delay_ms}ms) on {node_name} for {duration} seconds...")
    
    # Add delay
    add_cmd = ["wsl", "docker", "exec", node_name, "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", f"{delay_ms}ms"]
    
    # Remove delay after duration
    del_cmd = ["wsl", "docker", "exec", node_name, "tc", "qdisc", "del", "dev", "eth0", "root"]
    
    try:
        print("Applying network delay...")
        subprocess.run(add_cmd, check=True)
        
        print(f"Waiting {duration} seconds...")
        time.sleep(duration)
        
        print("Removing network delay...")
        subprocess.run(del_cmd, check=True)
        print("Network fault cleaned up.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed during network fault injection (tc tool might be missing): {e}")
        # Try to cleanup just in case
        try:
            subprocess.run(del_cmd, stderr=subprocess.DEVNULL)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Byzantine Fault Injector")
    parser.add_argument("--node", type=str, required=True, help="Target Kind node (e.g., kind-worker)")
    parser.add_argument("--fault", type=str, required=True, choices=["cpu", "memory", "network"], help="Type of fault")
    parser.add_argument("--duration", type=int, default=30, help="Duration of fault in seconds")
    
    args = parser.parse_args()
    
    if args.fault == "cpu":
        inject_cpu_fault(args.node, args.duration)
    elif args.fault == "memory":
        inject_memory_fault(args.node, args.duration)
    elif args.fault == "network":
        inject_network_fault(args.node, args.duration)

if __name__ == "__main__":
    main()
