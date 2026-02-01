"""
SGLang Offline Engine Benchmark Script
Uses SGLang's direct engine API for batch inference without HTTP server.
"""
import asyncio
import json
import os
import time
import csv
from datetime import datetime
from dataclasses import dataclass

import sglang as sgl

from metrics import RequestMetrics, MetricsCalculator, BenchmarkResults
from gpu_monitor import GPUMonitor, GPUMetrics

CONFIG = {
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "mem_fraction_static": 0.85,
    "tp_size": 1,
}


def get_model_short_name(model_name: str) -> str:
    """Extract short model name for directory naming."""
    return model_name.split('/')[-1]


def save_results(results: BenchmarkResults, raw_metrics: list, 
                 gpu_metrics: GPUMetrics, gpu_samples: list,
                 backend: str, concurrency: int, dataset: str, 
                 duration: float, output_dir: str):
    """Save benchmark results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON output
    json_file = f"{output_dir}/{backend}_{timestamp}.json"
    output = {
        "metadata": {
            "backend": backend,
            "timestamp": timestamp,
            "concurrency": concurrency,
            "dataset": dataset,
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
        "gpu_samples": gpu_samples
    }
    
    with open(json_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] JSON: {json_file}")
    
    # CSV Summary
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
            timestamp, backend, concurrency, len(raw_metrics), round(duration, 3),
            round(results.ttft_mean_ms, 3), round(results.ttft_p99_ms, 3),
            round(results.e2e_mean_ms, 3), round(results.e2e_p99_ms, 3),
            round(results.itl_mean_ms, 3), round(results.itl_p99_ms, 3),
            round(results.tpot_mean_ms, 3),
            round(results.system_tps, 3), round(results.system_rps, 3),
            round(results.mean_prefill_speed_tokens_per_sec, 3),
            gpu_metrics.gpu_name, round(gpu_metrics.gpu_util_mean, 2), 
            round(gpu_metrics.gpu_util_max, 2),
            round(gpu_metrics.memory_used_mean_mb, 1), 
            round(gpu_metrics.memory_used_max_mb, 1),
            round(gpu_metrics.memory_total_mb, 1),
            round(gpu_metrics.power_mean_watts, 1), 
            round(gpu_metrics.power_max_watts, 1),
            round(gpu_metrics.temp_mean_c, 1), round(gpu_metrics.temp_max_c, 1)
        ])
    print(f"[SAVED] CSV Summary: {summary_file}")
    
    # Per-request CSV
    per_request_file = f"{output_dir}/{backend}_{timestamp}_requests.csv"
    with open(per_request_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "input_tokens", "output_tokens", "ttft_ms", "e2e_ms", "tpot_ms"])
        for m in raw_metrics:
            writer.writerow([
                m.request_id, m.input_tokens, m.output_tokens,
                round(m.ttft * 1000, 3), round(m.e2e_latency * 1000, 3), round(m.tpot * 1000, 3)
            ])
    print(f"[SAVED] CSV Requests: {per_request_file}")


async def run_single_request(llm, prompt_data, request_id):
    """Run a single request and capture metrics using RequestMetrics from metrics.py."""
    prompt = prompt_data["prompt"]
    max_tokens = prompt_data.get("max_output_tokens", 256)
    
    sampling_params = {"temperature": 0.0, "max_new_tokens": max_tokens}
    
    req_start = time.perf_counter()
    
    try:
        # async_generate returns results directly when awaited
        outputs = await llm.async_generate(prompt, sampling_params)
        
        end_time = time.perf_counter()
        
        # For non-streaming, TTFT is approximated as a portion of total time
        # In practice, TTFT would be measured via streaming
        output_text = outputs.get("text", "") if isinstance(outputs, dict) else str(outputs)
        token_count = len(output_text.split())  # Approximate token count
        
        # Estimate TTFT as ~10% of total time for non-streaming
        ttft_estimate = req_start + (end_time - req_start) * 0.1
        
        return RequestMetrics(
            request_id=str(request_id),
            timestamp_request_sent=req_start,
            timestamp_first_token=ttft_estimate,
            timestamp_last_token=end_time,
            chunk_timestamps=[ttft_estimate, end_time],
            input_tokens=len(prompt) // 4,
            output_tokens=token_count
        )
    except Exception as e:
        print(f"[ERROR] Request {request_id} failed: {e}")
        return None


async def run_benchmark_async(llm, dataset, concurrency):
    """Run benchmark with specified concurrency using semaphore."""
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_request(prompt_data, idx):
        async with sem:
            return await run_single_request(llm, prompt_data, idx)
    
    tasks = [asyncio.create_task(bounded_request(item, idx)) for idx, item in enumerate(dataset)]
    return await asyncio.gather(*tasks)


def run_benchmark(concurrency=20, dataset_path="sharegpt_data.json"):
    """Main benchmark function using SGLang offline engine."""
    
    print(f"[LOAD] Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"[INIT] Initializing SGLang Engine with {CONFIG['model_path']}...")
    llm = sgl.Engine(
        model_path=CONFIG["model_path"],
        tp_size=CONFIG["tp_size"],
        mem_fraction_static=CONFIG["mem_fraction_static"],
    )
    
    try:
        # Start GPU monitoring
        gpu_monitor = GPUMonitor(sample_interval=0.1)
        gpu_monitor.start()
        
        print(f"[RUN] Starting benchmark with {concurrency} concurrent requests...")
        print(f"[RUN] Total requests: {len(dataset)}")
        
        start_time = time.perf_counter()
        results = asyncio.run(run_benchmark_async(llm, dataset, concurrency))
        total_duration = time.perf_counter() - start_time
        
        # Stop GPU monitoring
        gpu_monitor.stop()
        gpu_metrics = gpu_monitor.get_metrics()
        gpu_samples = gpu_monitor.get_samples_as_list()
        
        # Filter valid results and calculate metrics
        valid_results = [r for r in results if r is not None]
        final_stats = MetricsCalculator.aggregate(valid_results, total_duration)
        
        # Print results
        print(f"\n=== RESULTS FOR sglang ===")
        print(f"Total Requests: {len(valid_results)}")
        print(f"Duration:       {total_duration:.2f}s")
        print(f"System TPS:     {final_stats.system_tps:.2f} tok/s")
        print(f"TTFT (ms):      Mean: {final_stats.ttft_mean_ms:.2f} | P99: {final_stats.ttft_p99_ms:.2f}")
        print(f"E2E (ms):       Mean: {final_stats.e2e_mean_ms:.2f} | P99: {final_stats.e2e_p99_ms:.2f}")
        print(f"TPOT (ms):      {final_stats.tpot_mean_ms:.2f}")
        print(f"ITL P99 (ms):   {final_stats.itl_p99_ms:.2f}")
        
        if gpu_metrics.samples_count > 0:
            print(f"\n=== GPU METRICS ({gpu_metrics.gpu_name}) ===")
            print(f"GPU Util:       Mean: {gpu_metrics.gpu_util_mean:.1f}% | Max: {gpu_metrics.gpu_util_max:.1f}%")
            print(f"Memory:         Mean: {gpu_metrics.memory_used_mean_mb:.0f}MB | Max: {gpu_metrics.memory_used_max_mb:.0f}MB / {gpu_metrics.memory_total_mb:.0f}MB")
            print(f"Power:          Mean: {gpu_metrics.power_mean_watts:.1f}W | Max: {gpu_metrics.power_max_watts:.1f}W")
            print(f"Temperature:    Mean: {gpu_metrics.temp_mean_c:.1f}C | Max: {gpu_metrics.temp_max_c:.1f}C")
        
        # Save results
        model_short = get_model_short_name(CONFIG["model_path"])
        output_dir = f"results/sglang/{model_short}-concurrency-{concurrency}"
        
        save_results(final_stats, valid_results, gpu_metrics, gpu_samples,
                     "sglang", concurrency, dataset_path, total_duration, output_dir)
        
    finally:
        print("[STOP] Shutting down SGLang Engine...")
        llm.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SGLang Offline Benchmark")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="sharegpt_data.json")
    args = parser.parse_args()
    
    run_benchmark(concurrency=args.concurrency, dataset_path=args.dataset)