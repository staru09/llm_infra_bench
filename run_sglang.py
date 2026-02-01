"""
SGLang Server Runner and Benchmark Script
Starts SGLang server, waits for health check, runs benchmark, then cleans up.
"""
import subprocess
import time
import requests
import sys

CONFIG = {
    "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
    "port": 30000,
    "tp_size": 1,
    "mem_fraction_static": 0.85,
    "extra_args": [
        "--trust-remote-code",
        "--disable-log-requests"
    ]
}


def wait_for_server(url, timeout=300):
    """Poll server until ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/v1/models")
            print("[OK] SGLang Server is ready")
            return True
        except requests.exceptions.ConnectionError:
            elapsed = int(time.time() - start)
            print(f"[WAIT] SGLang starting... ({elapsed}s)")
            time.sleep(2)
    return False


def run_benchmark(concurrency=20, dataset="benchmark_dataset.json"):
    """Start SGLang server and run benchmark."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", CONFIG["model_path"],
        "--port", str(CONFIG["port"]),
        "--tp", str(CONFIG["tp_size"]),
        "--mem-fraction-static", str(CONFIG["mem_fraction_static"])
    ] + CONFIG["extra_args"]

    print(f"[START] Launching SGLang: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        server_url = f"http://localhost:{CONFIG['port']}"
        if not wait_for_server(server_url):
            raise TimeoutError("Server failed to start within timeout")

        print("[RUN] Starting Benchmark...")
        subprocess.run([
            sys.executable, "main.py",
            "--url", server_url,
            "--backend", "sglang",
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