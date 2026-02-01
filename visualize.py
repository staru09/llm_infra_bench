"""
Benchmark Results Visualization Script
Generates charts and plots from benchmark results.

Usage:
    python visualize.py                    # Generate all plots
    python visualize.py --output plots/    # Custom output directory
"""
import os
import json
import glob
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'vllm': '#1f77b4',      # Blue
    'sglang': '#ff7f0e',    # Orange
    'llamacpp': '#2ca02c',  # Green
}


def load_all_results(results_dir="results"):
    """Load all benchmark results from the results directory."""
    data = []
    
    for backend in ['vllm', 'sglang', 'llamacpp']:
        backend_dir = os.path.join(results_dir, backend)
        if not os.path.exists(backend_dir):
            continue
            
        # Find all concurrency directories
        for conc_dir in glob.glob(os.path.join(backend_dir, "*-concurrency-*")):
            # Extract concurrency from directory name
            try:
                concurrency = int(conc_dir.split("-concurrency-")[-1])
            except ValueError:
                continue
            
            # Load summary CSV
            summary_files = glob.glob(os.path.join(conc_dir, "benchmark_summary.csv"))
            if summary_files:
                df = pd.read_csv(summary_files[0])
                if len(df) > 0:
                    row = df.iloc[-1].to_dict()  # Get last row (most recent)
                    row['backend'] = backend
                    row['concurrency'] = concurrency
                    row['dir'] = conc_dir
                    data.append(row)
            
            # Load JSON for detailed data
            json_files = glob.glob(os.path.join(conc_dir, "*.json"))
            if json_files and data:
                with open(json_files[-1], 'r') as f:
                    json_data = json.load(f)
                    data[-1]['json_data'] = json_data
    
    return pd.DataFrame(data)


def plot_throughput_comparison(df, output_dir):
    """Line chart: System TPS vs Concurrency by backend."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        ax.plot(
            backend_df['concurrency'], 
            backend_df['system_tps'],
            marker='o', 
            linewidth=2,
            markersize=8,
            label=backend.upper(),
            color=COLORS.get(backend, '#333333')
        )
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Throughput (Tokens/sec)', fontsize=12)
    ax.set_title('System Throughput vs Concurrency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] throughput_comparison.png")


def plot_latency_comparison(df, output_dir):
    """Line chart: TTFT and E2E latency vs Concurrency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TTFT Plot
    ax1 = axes[0]
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        ax1.plot(
            backend_df['concurrency'], 
            backend_df['ttft_mean_ms'],
            marker='o', 
            linewidth=2,
            markersize=8,
            label=backend.upper(),
            color=COLORS.get(backend, '#333333')
        )
    
    ax1.set_xlabel('Concurrency', fontsize=12)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)
    ax1.set_title('Time to First Token (Mean)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # E2E Plot
    ax2 = axes[1]
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        ax2.plot(
            backend_df['concurrency'], 
            backend_df['e2e_mean_ms'] / 1000,  # Convert to seconds
            marker='s', 
            linewidth=2,
            markersize=8,
            label=backend.upper(),
            color=COLORS.get(backend, '#333333')
        )
    
    ax2.set_xlabel('Concurrency', fontsize=12)
    ax2.set_ylabel('E2E Latency (seconds)', fontsize=12)
    ax2.set_title('End-to-End Latency (Mean)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] latency_comparison.png")


def plot_duration_comparison(df, output_dir):
    """Bar chart: Total benchmark duration by backend and concurrency."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    backends = df['backend'].unique()
    concurrencies = sorted(df['concurrency'].unique())
    x = np.arange(len(concurrencies))
    width = 0.35 / len(backends) * 2
    
    for i, backend in enumerate(backends):
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        durations = []
        for conc in concurrencies:
            row = backend_df[backend_df['concurrency'] == conc]
            durations.append(row['duration_sec'].values[0] if len(row) > 0 else 0)
        
        offset = (i - len(backends)/2 + 0.5) * width
        bars = ax.bar(x + offset, durations, width, label=backend.upper(), 
                      color=COLORS.get(backend, '#333333'), alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, durations):
            if val > 0:
                ax.annotate(f'{val:.1f}s',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Duration (seconds)', fontsize=12)
    ax.set_title('Benchmark Duration by Concurrency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(concurrencies)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] duration_comparison.png")


def plot_gpu_utilization(df, output_dir):
    """Bar chart: GPU utilization comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    backends = df['backend'].unique()
    concurrencies = sorted(df['concurrency'].unique())
    x = np.arange(len(concurrencies))
    width = 0.35 / len(backends) * 2
    
    # GPU Utilization
    ax1 = axes[0]
    for i, backend in enumerate(backends):
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        values = []
        for conc in concurrencies:
            row = backend_df[backend_df['concurrency'] == conc]
            values.append(row['gpu_util_mean'].values[0] if len(row) > 0 else 0)
        
        offset = (i - len(backends)/2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=backend.upper(), 
                color=COLORS.get(backend, '#333333'), alpha=0.8)
    
    ax1.set_xlabel('Concurrency', fontsize=12)
    ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax1.set_title('Mean GPU Utilization', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(concurrencies)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Power Draw
    ax2 = axes[1]
    for i, backend in enumerate(backends):
        backend_df = df[df['backend'] == backend].sort_values('concurrency')
        values = []
        for conc in concurrencies:
            row = backend_df[backend_df['concurrency'] == conc]
            values.append(row['power_mean_w'].values[0] if len(row) > 0 else 0)
        
        offset = (i - len(backends)/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=backend.upper(), 
                color=COLORS.get(backend, '#333333'), alpha=0.8)
    
    ax2.set_xlabel('Concurrency', fontsize=12)
    ax2.set_ylabel('Power (Watts)', fontsize=12)
    ax2.set_title('Mean Power Consumption', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(concurrencies)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_utilization.png'), dpi=150)
    plt.close()
    print(f"[SAVED] gpu_utilization.png")


def plot_per_request_histogram(df, output_dir):
    """Histogram: Per-request latency distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find the highest concurrency for each backend
    for ax, metric, title in [
        (axes[0], 'ttft_ms', 'TTFT Distribution (Highest Concurrency)'),
        (axes[1], 'e2e_ms', 'E2E Latency Distribution (Highest Concurrency)')
    ]:
        for backend in df['backend'].unique():
            backend_df = df[df['backend'] == backend]
            if len(backend_df) == 0:
                continue
            
            # Get highest concurrency row
            max_conc_row = backend_df.loc[backend_df['concurrency'].idxmax()]
            
            if 'json_data' in max_conc_row and max_conc_row['json_data']:
                per_request = max_conc_row['json_data'].get('per_request', [])
                if per_request:
                    values = [r[metric] for r in per_request if metric in r]
                    if values:
                        ax.hist(values, bins=30, alpha=0.6, 
                               label=f"{backend.upper()} (n={len(values)})",
                               color=COLORS.get(backend, '#333333'))
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_histogram.png'), dpi=150)
    plt.close()
    print(f"[SAVED] latency_histogram.png")


def plot_gpu_timeseries(df, output_dir):
    """Line chart: GPU metrics over time (from gpu_samples)."""
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend]
        if len(backend_df) == 0:
            continue
        
        # Get highest concurrency row with json_data
        max_conc_row = backend_df.loc[backend_df['concurrency'].idxmax()]
        
        if 'json_data' not in max_conc_row or not max_conc_row['json_data']:
            continue
            
        gpu_samples = max_conc_row['json_data'].get('gpu_samples', [])
        if not gpu_samples:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        timestamps = [s.get('elapsed_sec', i) for i, s in enumerate(gpu_samples)]
        
        # GPU Utilization
        ax1 = axes[0, 0]
        values = [s.get('gpu_util', 0) for s in gpu_samples]
        ax1.plot(timestamps, values, color=COLORS.get(backend, '#333333'), linewidth=1)
        ax1.fill_between(timestamps, values, alpha=0.3, color=COLORS.get(backend, '#333333'))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('GPU Util (%)')
        ax1.set_title('GPU Utilization Over Time')
        ax1.set_ylim(0, 105)
        
        # Memory
        ax2 = axes[0, 1]
        values = [s.get('memory_used_mb', 0) / 1024 for s in gpu_samples]  # Convert to GB
        ax2.plot(timestamps, values, color=COLORS.get(backend, '#333333'), linewidth=1)
        ax2.fill_between(timestamps, values, alpha=0.3, color=COLORS.get(backend, '#333333'))
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Memory (GB)')
        ax2.set_title('GPU Memory Usage Over Time')
        
        # Power
        ax3 = axes[1, 0]
        values = [s.get('power_watts', 0) for s in gpu_samples]
        ax3.plot(timestamps, values, color=COLORS.get(backend, '#333333'), linewidth=1)
        ax3.fill_between(timestamps, values, alpha=0.3, color=COLORS.get(backend, '#333333'))
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power Consumption Over Time')
        
        # Temperature
        ax4 = axes[1, 1]
        values = [s.get('temp_c', 0) for s in gpu_samples]
        ax4.plot(timestamps, values, color=COLORS.get(backend, '#333333'), linewidth=1)
        ax4.fill_between(timestamps, values, alpha=0.3, color=COLORS.get(backend, '#333333'))
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('GPU Temperature Over Time')
        
        fig.suptitle(f'{backend.upper()} - GPU Metrics (Concurrency {int(max_conc_row["concurrency"])})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gpu_timeseries_{backend}.png'), dpi=150)
        plt.close()
        print(f"[SAVED] gpu_timeseries_{backend}.png")


def plot_summary_table(df, output_dir):
    """Generate a summary comparison table as an image."""
    # Prepare data for table
    backends = df['backend'].unique()
    concurrencies = sorted(df['concurrency'].unique())
    
    metrics = ['system_tps', 'duration_sec', 'ttft_mean_ms', 'e2e_mean_ms', 'gpu_util_mean']
    metric_labels = ['TPS', 'Duration (s)', 'TTFT (ms)', 'E2E (ms)', 'GPU Util (%)']
    
    fig, ax = plt.subplots(figsize=(14, len(backends) * len(concurrencies) * 0.5 + 2))
    ax.axis('off')
    
    # Build table data
    table_data = []
    row_labels = []
    
    for backend in backends:
        for conc in concurrencies:
            row = df[(df['backend'] == backend) & (df['concurrency'] == conc)]
            if len(row) > 0:
                row = row.iloc[0]
                row_labels.append(f"{backend.upper()} @ {conc}")
                table_data.append([
                    f"{row['system_tps']:.0f}",
                    f"{row['duration_sec']:.1f}",
                    f"{row['ttft_mean_ms']:.0f}",
                    f"{row['e2e_mean_ms']:.0f}",
                    f"{row['gpu_util_mean']:.1f}%"
                ])
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=metric_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for j in range(len(metric_labels)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Style row labels
        for i in range(len(row_labels)):
            table[(i+1, -1)].set_facecolor('#D9E2F3')
    
    plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BENCHMARK VISUALIZATION")
    print(f"{'='*60}")
    print(f"Results dir: {args.results}")
    print(f"Output dir:  {args.output}")
    print(f"{'='*60}\n")
    
    # Load all results
    print("[LOAD] Loading benchmark results...")
    df = load_all_results(args.results)
    
    if len(df) == 0:
        print("[ERROR] No results found!")
        return
    
    print(f"[OK] Found {len(df)} benchmark runs")
    print(f"     Backends: {df['backend'].unique().tolist()}")
    print(f"     Concurrency levels: {sorted(df['concurrency'].unique().tolist())}")
    print()
    
    # Generate plots
    print("[PLOT] Generating visualizations...")
    
    plot_throughput_comparison(df, args.output)
    plot_latency_comparison(df, args.output)
    plot_duration_comparison(df, args.output)
    plot_gpu_utilization(df, args.output)
    plot_per_request_histogram(df, args.output)
    plot_gpu_timeseries(df, args.output)
    plot_summary_table(df, args.output)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE - All plots saved to: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
