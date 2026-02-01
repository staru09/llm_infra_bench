# LLM Inference Benchmark

A comprehensive benchmarking suite for comparing LLM inference engines (vLLM, SGLang) with detailed latency, throughput, and GPU utilization metrics.

## 🔬 Experiment Results & Conclusions

We benchmarked **vLLM** and **SGLang** on an **NVIDIA A100-SXM4-80GB** with two models:

- **Llama-3.1-8B-Instruct**
- **Qwen3-8B**

### Key Findings

| Metric                     | vLLM         | SGLang       | Winner                      |
| -------------------------- | ------------ | ------------ | --------------------------- |
| **Low Concurrency (20)**   | 64s duration | 80s duration | ✅ vLLM                     |
| **High Concurrency (100)** | 24s duration | 23s duration | ✅ SGLang                   |
| **Peak TPS**               | 2,311 tok/s  | 2,950 tok/s  | ✅ SGLang (+28%)            |
| **GPU Utilization**        | 73%          | 96%          | ✅ SGLang (Full Saturation) |
| **Power Efficiency**       | 289W         | 343W         | ✅ vLLM (-19%)              |

### Recommendations

| Use Case                              | Recommended Backend |
| ------------------------------------- | ------------------- |
| **Chatbots, APIs, Interactive Apps**  | vLLM                |
| **Batch Processing, Data Generation** | SGLang              |
| **Power-Constrained Environments**    | vLLM                |
| **Maximum Throughput**                | SGLang              |

### Visualization

Run the visualization script to generate comparison plots:

```bash
python visualize.py
```

Generated plots in `plots/`:

- `throughput_all_combined.png` - All models & backends comparison
- `cross_model_comparison.png` - Peak performance by model
- `latency_heatmap.png` - E2E latency across configurations
- `gpu_comparison.png` - GPU utilization & power consumption
- `scaling_efficiency.png` - TPS per concurrent request
- `summary_table.png` - Full results summary

---

## Features

- **Multi-backend support**: vLLM, SGLang
- **Multi-model support**: Dynamic model selection via `--model` flag
- **Async load testing**: Configurable concurrency levels (default: 20, 50, 80, 100)
- **Comprehensive metrics**: TTFT, E2E latency, ITL, TPOT, throughput (TPS/RPS)
- **GPU monitoring**: Real-time GPU utilization, memory, power, and temperature
- **Visualization**: Auto-generated comparison charts
- **Export formats**: JSON and CSV for analysis

## Installation

```bash
git clone https://github.com/staru09/llm_infra_bench.git
cd llm_infra_bench

# Install dependencies
pip install aiohttp numpy requests pynvml pandas matplotlib
```

## Quick Start

### Run Full Benchmark Suite

```bash
# Run vLLM at all concurrency levels (20, 50, 80, 100)
python benchmark.py --backend vllm

# Run SGLang at all concurrency levels
python benchmark.py --backend sglang

# Run both backends
python benchmark.py --backend all

# Custom model
python benchmark.py --backend vllm --model Qwen/Qwen3-8B

# Single concurrency level
python benchmark.py --backend sglang --concurrency 50
```

### Generate Visualizations

```bash
python visualize.py
# Plots saved to plots/
```

## Configuration

### benchmark.py

| Parameter       | Default                            | Description                |
| --------------- | ---------------------------------- | -------------------------- |
| `--backend`     | (required)                         | `vllm`, `sglang`, or `all` |
| `--model`       | `meta-llama/Llama-3.1-8B-Instruct` | Model to benchmark         |
| `--concurrency` | All (20,50,80,100)                 | Single concurrency level   |
| `--dataset`     | `sharegpt_data.json`               | Dataset file               |

### Backend Configurations

Edit the `CONFIG` dict in each runner script:

**run_vllm.py**

```python
CONFIG = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "port": 8000,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "dtype": "float16",
    "max_model_len": 4096,
}
```

**run_sglang.py** (Offline Engine)

```python
CONFIG = {
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "tp_size": 1,
    "mem_fraction_static": 0.85,
}
```

## Output

### Results Structure

```
results/
├── vllm/
│   ├── Llama-3.1-8B-Instruct-concurrency-20/
│   │   ├── benchmark_summary.csv
│   │   ├── vllm_20260201_143000.json
│   │   └── vllm_20260201_143000_requests.csv
│   └── Qwen3-8B-concurrency-100/
│       └── ...
└── sglang/
    └── ...
```

### Metrics Definitions

| Metric   | Description                                             |
| -------- | ------------------------------------------------------- |
| **TTFT** | Time To First Token - latency until first token arrives |
| **E2E**  | End-to-End latency - total request duration             |
| **ITL**  | Inter-Token Latency - time between consecutive tokens   |
| **TPOT** | Time Per Output Token - (E2E - TTFT) / (tokens - 1)     |
| **TPS**  | Tokens Per Second - system throughput                   |
| **RPS**  | Requests Per Second - completed requests throughput     |
