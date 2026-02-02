"""
GPU Monitoring Utilities
Samples GPU utilization, memory, and power during benchmark runs.
Requires: pynvml (pip install pynvml)
"""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[WARN] pynvml not installed. GPU metrics disabled. Install with: pip install pynvml")


@dataclass
class GPUSample:
    """Single GPU measurement snapshot."""
    timestamp: float
    gpu_utilization: float  # Percentage (0-100)
    memory_used_mb: float
    memory_total_mb: float
    power_watts: float
    temperature_c: float


@dataclass 
class GPUMetrics:
    """Aggregated GPU metrics from a benchmark run."""
    gpu_name: str = ""
    samples_count: int = 0
    
    gpu_util_mean: float = 0.0
    gpu_util_max: float = 0.0
    gpu_util_min: float = 0.0
    
    memory_used_mean_mb: float = 0.0
    memory_used_max_mb: float = 0.0
    memory_total_mb: float = 0.0
    
    power_mean_watts: float = 0.0
    power_max_watts: float = 0.0
    
    temp_mean_c: float = 0.0
    temp_max_c: float = 0.0


class GPUMonitor:
    """
    Background GPU monitor that samples metrics at regular intervals.
    
    Usage:
        monitor = GPUMonitor(sample_interval=0.1)
        monitor.start()
        # ... run benchmark ...
        monitor.stop()
        metrics = monitor.get_metrics()
    """
    
    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.1):
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.samples: List[GPUSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._gpu_name = "Unknown"
        
    def start(self):
        """Start background GPU monitoring."""
        if not NVML_AVAILABLE:
            print("[WARN] GPU monitoring disabled (pynvml not available)")
            return False
            
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._gpu_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode('utf-8')
        except Exception as e:
            print(f"[WARN] Failed to initialize NVML: {e}")
            return False
        
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        print(f"[GPU] Monitoring started: {self._gpu_name}")
        return True
    
    def stop(self):
        """Stop GPU monitoring and cleanup."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        print(f"[GPU] Monitoring stopped. Collected {len(self.samples)} samples.")
    
    def _sample_loop(self):
        """Background sampling loop."""
        while self._running:
            try:
                sample = self._take_sample()
                if sample:
                    self.samples.append(sample)
            except Exception as e:
                pass  # Silently continue on sampling errors
            
            time.sleep(self.sample_interval)
    
    def _take_sample(self) -> Optional[GPUSample]:
        """Take a single GPU measurement."""
        if not self._handle:
            return None
            
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW to W
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return GPUSample(
                timestamp=time.perf_counter(),
                gpu_utilization=util.gpu,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
                power_watts=power,
                temperature_c=temp
            )
        except Exception:
            return None
    
    def get_metrics(self) -> GPUMetrics:
        """Calculate aggregated metrics from samples."""
        if not self.samples:
            return GPUMetrics(gpu_name=self._gpu_name)
        
        utils = [s.gpu_utilization for s in self.samples]
        mems = [s.memory_used_mb for s in self.samples]
        powers = [s.power_watts for s in self.samples]
        temps = [s.temperature_c for s in self.samples]
        
        return GPUMetrics(
            gpu_name=self._gpu_name,
            samples_count=len(self.samples),
            gpu_util_mean=sum(utils) / len(utils),
            gpu_util_max=max(utils),
            gpu_util_min=min(utils),
            memory_used_mean_mb=sum(mems) / len(mems),
            memory_used_max_mb=max(mems),
            memory_total_mb=self.samples[0].memory_total_mb,
            power_mean_watts=sum(powers) / len(powers),
            power_max_watts=max(powers),
            temp_mean_c=sum(temps) / len(temps),
            temp_max_c=max(temps)
        )
    
    def get_samples_as_list(self) -> List[dict]:
        """Return raw samples as list of dicts (for JSON export)."""
        return [
            {
                "timestamp": s.timestamp,
                "gpu_util": s.gpu_utilization,
                "memory_mb": round(s.memory_used_mb, 1),
                "power_w": round(s.power_watts, 1),
                "temp_c": s.temperature_c
            }
            for s in self.samples
        ]
