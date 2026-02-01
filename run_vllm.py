"""
vLLM Server Runner and Benchmark Script
Starts vLLM server, waits for health check, runs benchmark, then cleans up.
"""
import subprocess
import time
import requests
import sys

CONFIG = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "port": 8000,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "dtype": "float16",
    "max_model_len": 4096,
    "extra_args": [
        "--disable-log-requests",
        "--enforce-eager"
    ]
}


def wait_for_server(url, timeout=300):
    """Poll server until ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/v1/models")
            print("[OK] vLLM Server is ready")
            return True
        except requests.exceptions.ConnectionError:
            elapsed = int(time.time() - start)
            print(f"[WAIT] vLLM starting... ({elapsed}s)")
            time.sleep(2)
    return False


def run_benchmark(concurrency=20, dataset="sharegpt_data.json", model=None):
    """Start vLLM server and run benchmark."""
    
    # Use provided model or fallback to CONFIG
    model_path = model if model else CONFIG["model"]
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(CONFIG["port"]),
        "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
        "--dtype", CONFIG["dtype"],
        "--max-model-len", str(CONFIG["max_model_len"])
    ] + CONFIG["extra_args"]

    print(f"[START] Launching vLLM: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    try:
        server_url = f"http://localhost:{CONFIG['port']}"
        if not wait_for_server(server_url):
            raise TimeoutError("Server failed to start within timeout")

        print("[RUN] Starting Benchmark...")
        subprocess.run([
            sys.executable, "main.py",
            "--url", server_url,
            "--backend", "vllm",
            "--concurrency", str(concurrency),
            "--dataset", dataset,
            "--model", model_path
        ], check=True)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[STOP] Terminating server...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vLLM Benchmark Runner")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="sharegpt_data.json")
    parser.add_argument("--model", type=str, default=None, help="Model path (default: from CONFIG)")
    args = parser.parse_args()
    
    run_benchmark(concurrency=args.concurrency, dataset=args.dataset, model=args.model)