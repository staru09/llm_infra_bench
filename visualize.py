"""
Benchmark Results Visualization Script
Generates charts and plots from benchmark results.

Supports multiple models and backends comparison.

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
BACKEND_COLORS = {
    'vllm': '#1f77b4',      # Blue
    'sglang': '#ff7f0e',    # Orange
}
MODEL_MARKERS = {
    'Llama-3.1-8B-Instruct': 'o',
    'Qwen3-8B': 's',
}
LINESTYLES = {
    'vllm': '-',
    'sglang': '--',
}


def load_all_results(results_dir="results"):
    """Load all benchmark results from the results directory."""
    data = []
    
    for backend in ['vllm', 'sglang', 'llamacpp']:
        backend_dir = os.path.join(results_dir, backend)
        if not os.path.exists(backend_dir):
            continue
            
        # Find all model-concurrency directories
        for conc_dir in glob.glob(os.path.join(backend_dir, "*-concurrency-*")):
            dir_name = os.path.basename(conc_dir)
            
            # Extract model name and concurrency
            try:
                parts = dir_name.rsplit("-concurrency-", 1)
                model_name = parts[0]
                concurrency = int(parts[1])
            except (ValueError, IndexError):
                continue
            
            # Load summary CSV
            summary_files = glob.glob(os.path.join(conc_dir, "benchmark_summary.csv"))
            if summary_files:
                df = pd.read_csv(summary_files[0])
                if len(df) > 0:
                    row = df.iloc[-1].to_dict()
                    row['backend'] = backend
                    row['model'] = model_name
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


def plot_throughput_by_model(df, output_dir):
    """Line chart: TPS vs Concurrency, grouped by model with backend comparison."""
    models = df['model'].unique()
    
    for model in models:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_df = df[df['model'] == model]
        
        for backend in model_df['backend'].unique():
            backend_df = model_df[model_df['backend'] == backend].sort_values('concurrency')
            ax.plot(
                backend_df['concurrency'], 
                backend_df['system_tps'],
                marker=MODEL_MARKERS.get(model, 'o'),
                linewidth=2,
                markersize=8,
                linestyle=LINESTYLES.get(backend, '-'),
                label=backend.upper(),
                color=BACKEND_COLORS.get(backend, '#333')
            )
        
        ax.set_xlabel('Concurrency', fontsize=12)
        ax.set_ylabel('Throughput (Tokens/sec)', fontsize=12)
        ax.set_title(f'{model} - Throughput vs Concurrency', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = model.replace('/', '_').replace('.', '_')
        plt.savefig(os.path.join(output_dir, f'throughput_{safe_name}.png'), dpi=150)
        plt.close()
        print(f"[SAVED] throughput_{safe_name}.png")


def plot_cross_model_comparison(df, output_dir):
    """Bar chart: Compare models side-by-side for each backend at max concurrency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get max concurrency for each model-backend combo
    max_conc = df.groupby(['backend', 'model'])['concurrency'].max().reset_index()
    summary = df.merge(max_conc, on=['backend', 'model', 'concurrency'])
    
    models = sorted(df['model'].unique())
    backends = sorted(df['backend'].unique())
    x = np.arange(len(backends))
    width = 0.35
    
    # TPS Comparison
    ax1 = axes[0]
    for i, model in enumerate(models):
        values = []
        for backend in backends:
            row = summary[(summary['backend'] == backend) & (summary['model'] == model)]
            values.append(row['system_tps'].values[0] if len(row) > 0 else 0)
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=model, alpha=0.8)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.annotate(f'{val:.0f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Backend', fontsize=12)
    ax1.set_ylabel('Throughput (TPS)', fontsize=12)
    ax1.set_title('Peak Throughput by Model & Backend', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([b.upper() for b in backends])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Duration Comparison
    ax2 = axes[1]
    for i, model in enumerate(models):
        values = []
        for backend in backends:
            row = summary[(summary['backend'] == backend) & (summary['model'] == model)]
            values.append(row['duration_sec'].values[0] if len(row) > 0 else 0)
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=model, alpha=0.8)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.annotate(f'{val:.1f}s',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Backend', fontsize=12)
    ax2.set_ylabel('Duration (seconds)', fontsize=12)
    ax2.set_title('Peak Concurrency Duration by Model & Backend', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.upper() for b in backends])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_model_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] cross_model_comparison.png")


def plot_all_throughput_combined(df, output_dir):
    """Line chart: All models and backends on one chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for backend in df['backend'].unique():
        for model in df['model'].unique():
            subset = df[(df['backend'] == backend) & (df['model'] == model)]
            if len(subset) == 0:
                continue
            subset = subset.sort_values('concurrency')
            
            label = f"{backend.upper()} - {model}"
            ax.plot(
                subset['concurrency'],
                subset['system_tps'],
                marker=MODEL_MARKERS.get(model, 'o'),
                linewidth=2,
                markersize=8,
                linestyle=LINESTYLES.get(backend, '-'),
                label=label,
                color=BACKEND_COLORS.get(backend, '#333')
            )
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Throughput (Tokens/sec)', fontsize=12)
    ax.set_title('All Models & Backends - Throughput Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_all_combined.png'), dpi=150)
    plt.close()
    print(f"[SAVED] throughput_all_combined.png")


def plot_latency_heatmap(df, output_dir):
    """Heatmap: E2E latency across models, backends, and concurrency."""
    models = sorted(df['model'].unique())
    backends = sorted(df['backend'].unique())
    concurrencies = sorted(df['concurrency'].unique())
    
    fig, axes = plt.subplots(1, len(backends), figsize=(7*len(backends), 6))
    if len(backends) == 1:
        axes = [axes]
    
    for ax, backend in zip(axes, backends):
        # Create matrix: rows=models, cols=concurrency
        matrix = np.zeros((len(models), len(concurrencies)))
        
        for i, model in enumerate(models):
            for j, conc in enumerate(concurrencies):
                row = df[(df['backend'] == backend) & 
                        (df['model'] == model) & 
                        (df['concurrency'] == conc)]
                if len(row) > 0:
                    matrix[i, j] = row['e2e_mean_ms'].values[0] / 1000  # Convert to seconds
        
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(concurrencies)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(concurrencies)
        ax.set_yticklabels(models)
        ax.set_xlabel('Concurrency')
        ax.set_title(f'{backend.upper()} - E2E Latency (seconds)', fontweight='bold')
        
        # Add value annotations
        for i in range(len(models)):
            for j in range(len(concurrencies)):
                if matrix[i, j] > 0:
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', 
                           color='white' if matrix[i, j] > matrix.max()/2 else 'black', fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Latency (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_heatmap.png'), dpi=150)
    plt.close()
    print(f"[SAVED] latency_heatmap.png")


def plot_gpu_comparison(df, output_dir):
    """Grouped bar chart: GPU utilization and power by model and backend."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use max concurrency data
    max_conc = df.groupby(['backend', 'model'])['concurrency'].max().reset_index()
    summary = df.merge(max_conc, on=['backend', 'model', 'concurrency'])
    
    models = sorted(df['model'].unique())
    backends = sorted(df['backend'].unique())
    
    # Create labels: "Backend - Model"
    labels = []
    gpu_utils = []
    powers = []
    colors = []
    
    for backend in backends:
        for model in models:
            row = summary[(summary['backend'] == backend) & (summary['model'] == model)]
            if len(row) > 0:
                labels.append(f"{backend.upper()}\n{model[:10]}...")
                gpu_utils.append(row['gpu_util_mean'].values[0])
                powers.append(row['power_mean_w'].values[0])
                colors.append(BACKEND_COLORS.get(backend, '#333'))
    
    x = np.arange(len(labels))
    
    # GPU Utilization
    ax1 = axes[0]
    bars1 = ax1.bar(x, gpu_utils, color=colors, alpha=0.8)
    ax1.set_xlabel('Backend - Model', fontsize=12)
    ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax1.set_title('Peak GPU Utilization', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, gpu_utils):
        ax1.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Power Draw
    ax2 = axes[1]
    bars2 = ax2.bar(x, powers, color=colors, alpha=0.8)
    ax2.set_xlabel('Backend - Model', fontsize=12)
    ax2.set_ylabel('Power (Watts)', fontsize=12)
    ax2.set_title('Peak Power Consumption', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, powers):
        ax2.annotate(f'{val:.0f}W', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] gpu_comparison.png")


def plot_scaling_efficiency(df, output_dir):
    """Line chart: TPS per unit concurrency (efficiency) across scaling."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for backend in df['backend'].unique():
        for model in df['model'].unique():
            subset = df[(df['backend'] == backend) & (df['model'] == model)]
            if len(subset) == 0:
                continue
            subset = subset.sort_values('concurrency')
            
            # Calculate efficiency: TPS / Concurrency
            efficiency = subset['system_tps'] / subset['concurrency']
            
            label = f"{backend.upper()} - {model}"
            ax.plot(
                subset['concurrency'],
                efficiency,
                marker=MODEL_MARKERS.get(model, 'o'),
                linewidth=2,
                markersize=8,
                linestyle=LINESTYLES.get(backend, '-'),
                label=label,
                color=BACKEND_COLORS.get(backend, '#333')
            )
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Efficiency (TPS / Concurrency)', fontsize=12)
    ax.set_title('Scaling Efficiency - TPS per Concurrent Request', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_efficiency.png'), dpi=150)
    plt.close()
    print(f"[SAVED] scaling_efficiency.png")


def plot_summary_table(df, output_dir):
    """Generate a comprehensive summary table as an image."""
    # Get max concurrency for each combo
    max_conc = df.groupby(['backend', 'model'])['concurrency'].max().reset_index()
    summary = df.merge(max_conc, on=['backend', 'model', 'concurrency'])
    summary = summary.sort_values(['model', 'backend'])
    
    fig, ax = plt.subplots(figsize=(16, len(summary) * 0.6 + 2))
    ax.axis('off')
    
    # Build table
    headers = ['Model', 'Backend', 'Conc', 'TPS', 'Duration', 'E2E (s)', 'GPU %', 'Power (W)']
    table_data = []
    
    for _, row in summary.iterrows():
        table_data.append([
            row['model'][:20],
            row['backend'].upper(),
            int(row['concurrency']),
            f"{row['system_tps']:.0f}",
            f"{row['duration_sec']:.1f}s",
            f"{row['e2e_mean_ms']/1000:.1f}",
            f"{row['gpu_util_mean']:.0f}%",
            f"{row['power_mean_w']:.0f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title('Benchmark Results Summary (Peak Concurrency)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BENCHMARK VISUALIZATION")
    print(f"{'='*60}")
    print(f"Results dir: {args.results}")
    print(f"Output dir:  {args.output}")
    print(f"{'='*60}\n")
    
    print("[LOAD] Loading benchmark results...")
    df = load_all_results(args.results)
    
    if len(df) == 0:
        print("[ERROR] No results found!")
        return
    
    print(f"[OK] Found {len(df)} benchmark runs")
    print(f"     Backends: {df['backend'].unique().tolist()}")
    print(f"     Models: {df['model'].unique().tolist()}")
    print(f"     Concurrency: {sorted(df['concurrency'].unique().tolist())}")
    print()
    
    print("[PLOT] Generating visualizations...\n")
    
    # Per-model plots
    plot_throughput_by_model(df, args.output)
    
    # Combined plots
    plot_all_throughput_combined(df, args.output)
    plot_cross_model_comparison(df, args.output)
    plot_latency_heatmap(df, args.output)
    plot_gpu_comparison(df, args.output)
    plot_scaling_efficiency(df, args.output)
    plot_summary_table(df, args.output)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE - All plots saved to: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
