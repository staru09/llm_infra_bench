import argparse
import asyncio
import json
import time
import aiohttp
import numpy as np
from metrics import RequestMetrics, MetricsCalculator

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Must match server
TIMEOUT = 300  # 5 minutes timeout per request

async def send_request(session, url, prompt_data, sem):
    """Sends a single request and captures raw timestamps."""
    async with sem:  # Concurrency control
        req_start = time.perf_counter()
        
        # Construct payload compatible with OpenAI API
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_data["prompt"]}],
            "stream": True,
            "max_tokens": prompt_data["max_output_tokens"],
            "ignore_eos": prompt_data.get("ignore_eos", False),
            "temperature": 0.0  # Deterministic
        }

        # Metrics capture containers
        first_token_time = None
        last_token_time = None
        chunk_times = []
        token_count = 0
        input_tokens = 0 # In real usage, you'd calculate this via tokenizer locally

        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"Error {resp.status}: {text}")
                    return None

                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]": continue
                    if line.startswith("data: "):
                        current_time = time.perf_counter()
                        chunk_times.append(current_time)
                        
                        if first_token_time is None:
                            first_token_time = current_time
                        last_token_time = current_time
                        token_count += 1 # Approximation: 1 chunk = 1 token (usually true for streaming)

            # Create Metric Object
            if first_token_time is None: return None # Request failed or empty

            return RequestMetrics(
                request_id=prompt_data["id"],
                timestamp_request_sent=req_start,
                timestamp_first_token=first_token_time,
                timestamp_last_token=last_token_time,
                chunk_timestamps=chunk_times,
                input_tokens=len(prompt_data["prompt"]) // 4, # Rough approx if no tokenizer
                output_tokens=token_count
            )

        except Exception as e:
            print(f"Request failed: {e}")
            return None

async def main(args):
    # 1. Load Dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # 2. Prepare Load
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
        tasks = []
        start_time = time.perf_counter()
        
        print(f"Starting Benchmark on {args.url} with {args.concurrency} concurrent requests...")
        
        for item in dataset:
            task = asyncio.create_task(send_request(session, args.url, item, sem))
            tasks.append(task)
        
        # 3. Wait for completion
        results = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    # 4. Process Metrics
    valid_results = [r for r in results if r is not None]
    final_stats = MetricsCalculator.aggregate(valid_results, total_duration)

    # 5. Output
    print(f"\n=== RESULTS FOR {args.backend} ===")
    print(f"Total Requests: {len(valid_results)}")
    print(f"Duration:       {total_duration:.2f}s")
    print(f"System TPS:     {final_stats.system_tps:.2f} tok/s")
    print(f"TTFT (ms):      Mean: {final_stats.ttft_mean_ms:.2f} | P99: {final_stats.ttft_p99_ms:.2f}")
    print(f"E2E (ms):       Mean: {final_stats.e2e_mean_ms:.2f}")
    print(f"TPOT (ms):      {final_stats.tpot_mean_ms:.2f}")
    print(f"ITL P99 (ms):   {final_stats.itl_p99_ms:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--dataset", type=str, default="benchmark_dataset.json")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--backend", type=str, default="unknown")
    args = parser.parse_args()
    asyncio.run(main(args))