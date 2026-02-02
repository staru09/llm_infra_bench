"""
llama.cpp Server Runner and Benchmark Script
Starts llama.cpp server, waits for health check, runs benchmark, then cleans up.
"""
import subprocess
import time
import requests
import sys
import os

CONFIG = {
    "binary_path": "./llama-server",
    "model_path": "./Meta-Llama-3-8B-Instruct-F16.gguf",
    "port": 8080,
    "ctx_size": 8192,
    "n_gpu_layers": 999,
    "n_parallel": 32,
    "extra_args": [
        "-cb",  # Continuous batching
        "-fa"   # Flash attention
    ]
}


def wait_for_server(url, timeout=300):
    """Poll server until ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health")
            print("[OK] llama.cpp Server is ready")
            return True
        except requests.exceptions.ConnectionError:
            elapsed = int(time.time() - start)
            print(f"[WAIT] llama.cpp starting... ({elapsed}s)")
            time.sleep(1)
    return False


def run_benchmark(concurrency=20, dataset="benchmark_dataset.json"):
    """Start llama.cpp server and run benchmark."""
    if not os.path.exists(CONFIG["binary_path"]):
        print(f"[ERROR] Binary not found at {CONFIG['binary_path']}")
        return

    cmd = [
        CONFIG["binary_path"],
        "-m", CONFIG["model_path"],
        "--port", str(CONFIG["port"]),
        "-c", str(CONFIG["ctx_size"]),
        "-ngl", str(CONFIG["n_gpu_layers"]),
        "-np", str(CONFIG["n_parallel"]),
        "--host", "0.0.0.0"
    ] + CONFIG["extra_args"]

    print(f"[START] Launching llama.cpp: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        server_url = f"http://localhost:{CONFIG['port']}"
        if not wait_for_server(server_url):
            raise TimeoutError("Server failed to start within timeout")

        print("[RUN] Starting Benchmark...")
        subprocess.run([
            sys.executable, "main.py",
            "--url", server_url,
            "--backend", "llamacpp",
            "--concurrency", str(concurrency),
            "--dataset", dataset
        ], check=True)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[STOP] Terminating server...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    run_benchmark()
