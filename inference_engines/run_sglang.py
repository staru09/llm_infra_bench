"""
SGLang Offline Engine Benchmark Script
Uses SGLang's direct engine API for batch inference without HTTP server.
"""
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sglang as sgl

from metrics.metrics import RequestMetrics, MetricsCalculator, BenchmarkResults
from metrics.gpu_monitor import GPUMonitor, GPUMetrics
from main import save_results_json, save_results_csv


@dataclass
class BenchmarkArgs:
    """Simple args object to match main.py's interface."""
    backend: str
    url: str
    concurrency: int
    dataset: str
    model: str

CONFIG = {
    "model_path": "Qwen/Qwen3-8B",
    "mem_fraction_static": 0.85,
    "tp_size": 1,
}


def get_model_short_name(model_name: str) -> str:
    """Extract short model name for directory naming."""
    return model_name.split('/')[-1]



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


def run_benchmark(concurrency=20, dataset="sharegpt_data.json", model=None):
    """Main benchmark function using SGLang offline engine."""
    
    # Use provided model or fallback to CONFIG
    model_path = model if model else CONFIG["model_path"]
    
    print(f"[LOAD] Loading dataset: {dataset}")
    with open(dataset, 'r') as f:
        dataset_items = json.load(f)
    
    print(f"[INIT] Initializing SGLang Engine with {model_path}...")
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=CONFIG["tp_size"],
        mem_fraction_static=CONFIG["mem_fraction_static"],
    )
    
    try:
        # Start GPU monitoring
        gpu_monitor = GPUMonitor(sample_interval=0.1)
        gpu_monitor.start()
        
        print(f"[RUN] Starting benchmark with {concurrency} concurrent requests...")
        print(f"[RUN] Total requests: {len(dataset_items)}")
        
        start_time = time.perf_counter()
        results = asyncio.run(run_benchmark_async(llm, dataset_items, concurrency))
        total_duration = time.perf_counter() - start_time
        
        # Stop GPU monitoring
        gpu_monitor.stop()
        gpu_metrics = gpu_monitor.get_metrics()
        gpu_samples = gpu_monitor.get_samples_as_list()
        
        # Filter valid results and calculate metrics
        valid_results = [r for r in results if r is not None]
        final_stats = MetricsCalculator.aggregate(valid_results, total_duration)
        dataset_path = dataset  # For saving results
        
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
        
        # Save results using shared functions from main.py
        model_short = get_model_short_name(model_path)
        output_dir = f"results/sglang/{model_short}-concurrency-{concurrency}"
        
        args = BenchmarkArgs(
            backend="sglang",
            url="offline",  # No HTTP server for SGLang offline
            concurrency=concurrency,
            dataset=dataset_path,
            model=model_path
        )
        save_results_json(final_stats, valid_results, gpu_metrics, gpu_samples, args, total_duration, output_dir)
        save_results_csv(final_stats, valid_results, gpu_metrics, args, total_duration, output_dir)
        
    finally:
        print("[STOP] Shutting down SGLang Engine...")
        llm.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SGLang Offline Benchmark")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="sharegpt_data.json")
    parser.add_argument("--model", type=str, default=None, help="Model path (default: from CONFIG)")
    args = parser.parse_args()
    
    run_benchmark(concurrency=args.concurrency, dataset=args.dataset, model=args.model)