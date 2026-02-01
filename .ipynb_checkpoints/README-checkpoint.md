# LLM Inference Benchmark

A comprehensive benchmarking suite for comparing LLM inference engines (vLLM, SGLang, llama.cpp) with detailed latency, throughput, and GPU utilization metrics.

## Features

- **Multi-backend support**: vLLM, SGLang, llama.cpp
- **Async load testing**: Simulates concurrent users with configurable concurrency
- **Comprehensive metrics**: TTFT, E2E latency, ITL, TPOT, throughput (TPS/RPS)
- **GPU monitoring**: Real-time GPU utilization, memory, power, and temperature
- **Export formats**: JSON and CSV for analysis and plotting

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd llm_infra_bench

# Install dependencies
pip install aiohttp numpy requests pynvml python-dotenv

# For dataset generation (optional)
pip install openai  # or use OpenRouter
```

## Quick Start

### 1. Generate Dataset

Create a benchmark dataset using Claude Opus via OpenRouter:

```bash
# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Generate 18 diverse prompts
python dataset.py
```

This creates `benchmark_dataset.json` with prompts across categories: coding, reasoning, creative writing, summarization, etc.

### 2. Run Benchmarks

**Option A: Unified Runner (Recommended)**

```bash
# Run single backend
python benchmark.py --backend vllm --concurrency 20

# Run all backends sequentially
python benchmark.py --backend all --concurrency 10

# Custom dataset
python benchmark.py --backend sglang --dataset my_prompts.json
```

**Option B: Individual Backend Scripts**

```bash
# These scripts start the server, run benchmark, then cleanup
python run_vllm.py
python run_sglang.py
python run_llamacpp.py
```

**Option C: Direct Client (server already running)**

```bash
python main.py --url http://localhost:8000 --backend vllm --concurrency 20
```

## Configuration Parameters

### main.py (Benchmark Client)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--url` | `http://localhost:8000` | Server endpoint |
| `--backend` | `unknown` | Backend name (for labeling results) |
| `--concurrency` | `10` | Number of concurrent requests |
| `--dataset` | `benchmark_dataset.json` | Path to prompts JSON file |
| `--output` | `results` | Directory to save JSON/CSV results |

### benchmark.py (Unified Runner)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | (required) | `vllm`, `sglang`, `llamacpp`, or `all` |
| `--concurrency` | `20` | Concurrent requests |
| `--dataset` | `benchmark_dataset.json` | Prompts file |

### Backend Configurations

Edit the `CONFIG` dict in each runner script:

**run_vllm.py**
```python
CONFIG = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "port": 8000,
    "tensor_parallel_size": 1,      # GPUs for tensor parallelism
    "gpu_memory_utilization": 0.90,
    "dtype": "float16",             # or "bfloat16" for H100
    "max_model_len": 4096,
}
```

**run_sglang.py**
```python
CONFIG = {
    "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
    "port": 30000,
    "tp_size": 1,
    "mem_fraction_static": 0.85,
}
```

**run_llamacpp.py**
```python
CONFIG = {
    "binary_path": "./llama-server",
    "model_path": "./model.gguf",
    "port": 8080,
    "ctx_size": 8192,
    "n_gpu_layers": 999,  # Full GPU offload
    "n_parallel": 32,     # Concurrent slots
}
```

## Output Metrics

### Console Output

```
=== RESULTS FOR vllm ===
Total Requests: 18
Duration:       12.45s
System TPS:     342.50 tok/s
TTFT (ms):      Mean: 45.23 | P99: 112.50
E2E (ms):       Mean: 234.12 | P99: 567.89
TPOT (ms):      12.34
ITL P99 (ms):   15.67

=== GPU METRICS (NVIDIA RTX 4090) ===
GPU Util:       Mean: 87.3% | Max: 99.0%
Memory:         Mean: 18432MB | Max: 19200MB / 24576MB
Power:          Mean: 312.5W | Max: 450.0W
Temperature:    Mean: 68.2C | Max: 74.0C
```

### Saved Files

```
results/
├── benchmark_summary.csv         # All runs (append mode)
├── vllm_20260201_143000.json     # Full export with GPU time-series
├── vllm_20260201_143000_requests.csv
└── ...
```

### Metrics Definitions

| Metric | Description |
|--------|-------------|
| **TTFT** | Time To First Token - latency until first token arrives |
| **E2E** | End-to-End latency - total request duration |
| **ITL** | Inter-Token Latency - time between consecutive tokens |
| **TPOT** | Time Per Output Token - (E2E - TTFT) / (tokens - 1) |
| **TPS** | Tokens Per Second - system throughput |
| **RPS** | Requests Per Second - system throughput |

## Project Structure

```
llm_infra_bench/
├── main.py           # Async benchmark client
├── metrics.py        # Metrics dataclasses and calculator
├── gpu_monitor.py    # NVIDIA GPU monitoring (pynvml)
├── dataset.py        # Synthetic dataset generator (OpenRouter)
├── benchmark.py      # Unified runner for all backends
├── run_vllm.py       # vLLM server launcher
├── run_sglang.py     # SGLang server launcher
├── run_llamacpp.py   # llama.cpp server launcher
└── results/          # Output directory
```

## Example Workflow

```bash
# 1. Generate dataset
python dataset.py

# 2. Run benchmarks on all backends
python benchmark.py --backend all --concurrency 20

# 3. Results saved to results/benchmark_summary.csv
# 4. Use your favorite tool to plot and compare!
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- `pynvml` for GPU monitoring
- Backend-specific: vLLM, SGLang, or llama.cpp installed

## License

MIT
