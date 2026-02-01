"""
Unified Benchmark Runner
Run benchmarks against different LLM inference backends from a single entry point.

Usage:
    python benchmark.py --backend vllm
    python benchmark.py --backend sglang
    python benchmark.py --backend llamacpp
    python benchmark.py --backend all
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py --backend vllm --concurrency 20
    python benchmark.py --backend sglang --dataset custom_prompts.json
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
        "--concurrency", 
        type=int, 
        default=20,
        help="Number of concurrent requests (default: 20)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="benchmark_dataset.json",
        help="Path to dataset JSON file"
    )
    
    args = parser.parse_args()
    
    backends_to_run = []
    
    if args.backend == "all":
        backends_to_run = ["vllm", "sglang", "llamacpp"]
    else:
        backends_to_run = [args.backend]
    
    for backend in backends_to_run:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {backend.upper()}")
        print(f"{'='*60}\n")
        
        if backend == "vllm":
            from run_vllm import run_benchmark
        elif backend == "sglang":
            from run_sglang import run_benchmark
        elif backend == "llamacpp":
            from run_llamacpp import run_benchmark
        
        run_benchmark(
            concurrency=args.concurrency,
            dataset=args.dataset
        )


if __name__ == "__main__":
    main()
