"""
Unified Benchmark Runner
Run benchmarks against different LLM inference backends from a single entry point.

Usage:
    python benchmark.py --backend vllm
    python benchmark.py --backend sglang
    python benchmark.py --backend all
    python benchmark.py --backend vllm --model meta-llama/Llama-3.1-70B-Instruct
"""
import argparse
import sys

DEFAULT_MODEL = "Qwen/Qwen3-8B"
CONCURRENCY_LEVELS = [20, 50, 80, 100]


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run vLLM benchmark at all concurrency levels (20, 50, 80, 100)
    python benchmark.py --backend vllm

    # Run single concurrency level
    python benchmark.py --backend sglang --concurrency 50

    # Use custom model
    python benchmark.py --backend vllm --model meta-llama/Llama-3.1-70B-Instruct

    # Run all backends
    python benchmark.py --backend all
        """
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        required=True,
        choices=["vllm", "sglang", "llamacpp", "all"],
        help="Backend to benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for inference (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=None,
        help="Number of concurrent requests. If not specified, runs all levels: 20, 50, 80, 100"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="sharegpt_data.json",
        help="Path to dataset JSON file"
    )
    
    args = parser.parse_args()
    
    # Determine backends to run
    if args.backend == "all":
        backends_to_run = ["vllm", "sglang"]
    else:
        backends_to_run = [args.backend]
    
    # Determine concurrency levels
    if args.concurrency is not None:
        concurrency_levels = [args.concurrency]
    else:
        concurrency_levels = CONCURRENCY_LEVELS
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Backends:    {', '.join(backends_to_run)}")
    print(f"Concurrency: {concurrency_levels}")
    print(f"Dataset:     {args.dataset}")
    print(f"{'='*60}\n")
    
    for backend in backends_to_run:
        print(f"\n{'='*60}")
        print(f"BACKEND: {backend.upper()}")
        print(f"{'='*60}\n")
        
        # Import the appropriate backend module
        if backend == "vllm":
            from inference_engines.run_vllm import run_benchmark
        elif backend == "sglang":
            from inference_engines.run_sglang import run_benchmark
        elif backend == "llamacpp":
            from inference_engines.run_llamacpp import run_benchmark
        else:
            print(f"[WARN] Unknown backend: {backend}")
            continue
        
        # Run benchmark for each concurrency level
        for concurrency in concurrency_levels:
            print(f"\n{'-'*40}")
            print(f"[{backend.upper()}] Concurrency: {concurrency}")
            print(f"{'-'*40}\n")
            
            try:
                run_benchmark(
                    concurrency=concurrency,
                    dataset=args.dataset,
                    model=args.model
                )
            except Exception as e:
                print(f"[ERROR] {backend} @ {concurrency} failed: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
