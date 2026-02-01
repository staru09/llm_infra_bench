import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime

import aiohttp

from metrics import RequestMetrics, MetricsCalculator, BenchmarkResults
from gpu_monitor import GPUMonitor, GPUMetrics, NVML_AVAILABLE

MODEL_NAME = "Qwen/Qwen3-8B"
TIMEOUT = 300


def get_model_short_name(model_name: str) -> str:
    """Extract short model name for directory naming."""
    # Remove org prefix (e.g., 'meta-llama/')
    short_name = model_name.split('/')[-1]
    return short_name


def save_results_json(
    results: BenchmarkResults, 
    raw_metrics: list, 
    gpu_metrics: GPUMetrics,
    gpu_samples: list,
    args, 
    duration: float, 
    output_dir: str = "results"
):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{args.backend}_{timestamp}.json"
    
    output = {
        "metadata": {
            "backend": args.backend,
            "timestamp": timestamp,
            "url": args.url,
            "concurrency": args.concurrency,
            "dataset": args.dataset,
            "total_duration_sec": round(duration, 3),
            "total_requests": len(raw_metrics),
        },
        "summary": {
            "ttft_mean_ms": round(results.ttft_mean_ms, 3),
            "ttft_p99_ms": round(results.ttft_p99_ms, 3),
            "e2e_mean_ms": round(results.e2e_mean_ms, 3),
            "e2e_p99_ms": round(results.e2e_p99_ms, 3),
            "itl_mean_ms": round(results.itl_mean_ms, 3),
            "itl_p99_ms": round(results.itl_p99_ms, 3),
            "tpot_mean_ms": round(results.tpot_mean_ms, 3),
            "system_tps": round(results.system_tps, 3),
            "system_rps": round(results.system_rps, 3),
            "mean_prefill_speed_tokens_per_sec": round(results.mean_prefill_speed_tokens_per_sec, 3),
        },
        "gpu": {
            "gpu_name": gpu_metrics.gpu_name,
            "samples_count": gpu_metrics.samples_count,
            "gpu_util_mean": round(gpu_metrics.gpu_util_mean, 2),
            "gpu_util_max": round(gpu_metrics.gpu_util_max, 2),
            "gpu_util_min": round(gpu_metrics.gpu_util_min, 2),
            "memory_used_mean_mb": round(gpu_metrics.memory_used_mean_mb, 1),
            "memory_used_max_mb": round(gpu_metrics.memory_used_max_mb, 1),
            "memory_total_mb": round(gpu_metrics.memory_total_mb, 1),
            "power_mean_watts": round(gpu_metrics.power_mean_watts, 1),
            "power_max_watts": round(gpu_metrics.power_max_watts, 1),
            "temp_mean_c": round(gpu_metrics.temp_mean_c, 1),
            "temp_max_c": round(gpu_metrics.temp_max_c, 1),
        },
        "per_request": [
            {
                "request_id": m.request_id,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "ttft_ms": round(m.ttft * 1000, 3),
                "e2e_ms": round(m.e2e_latency * 1000, 3),
                "tpot_ms": round(m.tpot * 1000, 3),
            }
            for m in raw_metrics
        ],
        "gpu_samples": gpu_samples  # Time-series GPU data for plotting
    }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"[SAVED] JSON: {filename}")
    return filename


def save_results_csv(
    results: BenchmarkResults, 
    raw_metrics: list,
    gpu_metrics: GPUMetrics,
    args, 
    duration: float, 
    output_dir: str = "results"
):
    """Save benchmark results to CSV files (summary + per-request)."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary CSV (append mode for comparing runs)
    summary_file = f"{output_dir}/benchmark_summary.csv"
    file_exists = os.path.exists(summary_file)
    
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "backend", "concurrency", "total_requests", "duration_sec",
                "ttft_mean_ms", "ttft_p99_ms", "e2e_mean_ms", "e2e_p99_ms",
                "itl_mean_ms", "itl_p99_ms", "tpot_mean_ms", 
                "system_tps", "system_rps", "prefill_speed",
                "gpu_name", "gpu_util_mean", "gpu_util_max",
                "mem_used_mean_mb", "mem_used_max_mb", "mem_total_mb",
                "power_mean_w", "power_max_w", "temp_mean_c", "temp_max_c"
            ])
        writer.writerow([
            timestamp, args.backend, args.concurrency, len(raw_metrics), round(duration, 3),
            round(results.ttft_mean_ms, 3), round(results.ttft_p99_ms, 3),
            round(results.e2e_mean_ms, 3), round(results.e2e_p99_ms, 3),
            round(results.itl_mean_ms, 3), round(results.itl_p99_ms, 3),
            round(results.tpot_mean_ms, 3),
            round(results.system_tps, 3), round(results.system_rps, 3),
            round(results.mean_prefill_speed_tokens_per_sec, 3),
            gpu_metrics.gpu_name, round(gpu_metrics.gpu_util_mean, 2), round(gpu_metrics.gpu_util_max, 2),
            round(gpu_metrics.memory_used_mean_mb, 1), round(gpu_metrics.memory_used_max_mb, 1),
            round(gpu_metrics.memory_total_mb, 1),
            round(gpu_metrics.power_mean_watts, 1), round(gpu_metrics.power_max_watts, 1),
            round(gpu_metrics.temp_mean_c, 1), round(gpu_metrics.temp_max_c, 1)
        ])
    
    # Per-request CSV
    per_request_file = f"{output_dir}/{args.backend}_{timestamp}_requests.csv"
    with open(per_request_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "input_tokens", "output_tokens", "ttft_ms", "e2e_ms", "tpot_ms"])
        for m in raw_metrics:
            writer.writerow([
                m.request_id, m.input_tokens, m.output_tokens,
                round(m.ttft * 1000, 3), round(m.e2e_latency * 1000, 3), round(m.tpot * 1000, 3)
            ])
    
    print(f"[SAVED] CSV Summary: {summary_file}")
    print(f"[SAVED] CSV Requests: {per_request_file}")
    return summary_file, per_request_file


async def send_request(session, url, prompt_data, sem):
    """Sends a single request and captures raw timestamps."""
    async with sem:
        req_start = time.perf_counter()
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_data["prompt"]}],
            "stream": True,
            "max_tokens": prompt_data["max_output_tokens"],
            "ignore_eos": prompt_data.get("ignore_eos", False),
            "temperature": 0.0
        }

        first_token_time = None
        last_token_time = None
        chunk_times = []
        token_count = 0

        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"Error {resp.status}: {text}")
                    return None

                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]": 
                        continue
                    if line.startswith("data: "):
                        current_time = time.perf_counter()
                        chunk_times.append(current_time)
                        
                        if first_token_time is None:
                            first_token_time = current_time
                        last_token_time = current_time
                        token_count += 1

            if first_token_time is None: 
                return None

            return RequestMetrics(
                request_id=prompt_data["id"],
                timestamp_request_sent=req_start,
                timestamp_first_token=first_token_time,
                timestamp_last_token=last_token_time,
                chunk_timestamps=chunk_times,
                input_tokens=len(prompt_data["prompt"]) // 4,
                output_tokens=token_count
            )

        except Exception as e:
            print(f"Request failed: {e}")
            return None


async def main(args):
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor(sample_interval=0.1)
    gpu_monitor.start()
    
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
        tasks = []
        start_time = time.perf_counter()
        
        print(f"Starting Benchmark on {args.url} with {args.concurrency} concurrent requests...")
        
        for item in dataset:
            task = asyncio.create_task(send_request(session, args.url, item, sem))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    # Stop GPU monitoring
    gpu_monitor.stop()
    gpu_metrics = gpu_monitor.get_metrics()
    gpu_samples = gpu_monitor.get_samples_as_list()

    valid_results = [r for r in results if r is not None]
    final_stats = MetricsCalculator.aggregate(valid_results, total_duration)

    # Console Output
    print(f"\n=== RESULTS FOR {args.backend} ===")
    print(f"Total Requests: {len(valid_results)}")
    print(f"Duration:       {total_duration:.2f}s")
    print(f"System TPS:     {final_stats.system_tps:.2f} tok/s")
    print(f"TTFT (ms):      Mean: {final_stats.ttft_mean_ms:.2f} | P99: {final_stats.ttft_p99_ms:.2f}")
    print(f"E2E (ms):       Mean: {final_stats.e2e_mean_ms:.2f} | P99: {final_stats.e2e_p99_ms:.2f}")
    print(f"TPOT (ms):      {final_stats.tpot_mean_ms:.2f}")
    print(f"ITL P99 (ms):   {final_stats.itl_p99_ms:.2f}")
    
    # GPU Stats
    if gpu_metrics.samples_count > 0:
        print(f"\n=== GPU METRICS ({gpu_metrics.gpu_name}) ===")
        print(f"GPU Util:       Mean: {gpu_metrics.gpu_util_mean:.1f}% | Max: {gpu_metrics.gpu_util_max:.1f}%")
        print(f"Memory:         Mean: {gpu_metrics.memory_used_mean_mb:.0f}MB | Max: {gpu_metrics.memory_used_max_mb:.0f}MB / {gpu_metrics.memory_total_mb:.0f}MB")
        print(f"Power:          Mean: {gpu_metrics.power_mean_watts:.1f}W | Max: {gpu_metrics.power_max_watts:.1f}W")
        print(f"Temperature:    Mean: {gpu_metrics.temp_mean_c:.1f}C | Max: {gpu_metrics.temp_max_c:.1f}C")

    # Save Results - organized by backend/model-concurrency
    model_name = getattr(args, 'model', MODEL_NAME) or MODEL_NAME
    model_short = get_model_short_name(model_name)
    output_dir = f"results/{args.backend}/{model_short}-concurrency-{args.concurrency}"
    save_results_json(final_stats, valid_results, gpu_metrics, gpu_samples, args, total_duration, output_dir)
    save_results_csv(final_stats, valid_results, gpu_metrics, args, total_duration, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--dataset", type=str, default="sharegpt_data.json")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--backend", type=str, default="unknown")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name for result organization")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results (JSON + CSV)")
    args = parser.parse_args()
    asyncio.run(main(args))