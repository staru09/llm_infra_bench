import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RequestMetrics:
    """
    Holds the raw timestamps and token counts for a SINGLE request.
    Populate this in your async client loop.
    """
    request_id: str
    timestamp_request_sent: float
    timestamp_first_token: float
    timestamp_last_token: float
    
    # List of timestamps for every chunk received (used for ITL)
    chunk_timestamps: List[float] = field(default_factory=list)
    
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def ttft(self) -> float:
        """Time To First Token (seconds)"""
        return self.timestamp_first_token - self.timestamp_request_sent

    @property
    def e2e_latency(self) -> float:
        """End-to-End Latency (seconds)"""
        return self.timestamp_last_token - self.timestamp_request_sent

    @property
    def prefill_speed(self) -> float:
        """
        Speed of processing input tokens (Tokens/sec).
        Formula: Input_Tokens / TTFT
        """
        if self.ttft <= 0: return 0.0
        return self.input_tokens / self.ttft

    @property
    def inter_token_latencies(self) -> List[float]:
        """
        Calculates the time difference between consecutive chunks.
        Returns a list of latency values (seconds).
        """
        if len(self.chunk_timestamps) < 2:
            return []
        
        itls = [
            t2 - t1 
            for t1, t2 in zip(self.chunk_timestamps[:-1], self.chunk_timestamps[1:])
        ]
        return itls

    @property
    def tpot(self) -> float:
        """
        Time Per Output Token (seconds).
        Formula: (E2E Latency - TTFT) / (Output_Tokens - 1)
        """
        if self.output_tokens <= 1:
            return 0.0
        
        return (self.e2e_latency - self.ttft) / (self.output_tokens - 1)


@dataclass
class BenchmarkResults:
    """
    Stores the final aggregated metrics for the entire experiment.
    """
    ttft_mean_ms: float
    ttft_p99_ms: float
    
    e2e_mean_ms: float
    e2e_p99_ms: float
    
    itl_mean_ms: float
    itl_p99_ms: float
    tpot_mean_ms: float
    
    system_tps: float  # Total Tokens / Duration
    system_rps: float  # Total Requests / Duration
    
    mean_prefill_speed_tokens_per_sec: float


class MetricsCalculator:
    """
    Static logic to process a list of RequestMetrics into BenchmarkResults.
    """
    
    @staticmethod
    def aggregate(
        requests: List[RequestMetrics], 
        total_benchmark_duration_sec: float
    ) -> BenchmarkResults:
        
        if not requests:
            return BenchmarkResults(*([0.0] * 10))

        ttfts = [r.ttft for r in requests]
        e2es = [r.e2e_latency for r in requests]
        tpots = [r.tpot for r in requests]
        prefills = [r.prefill_speed for r in requests]
        

        all_itls = []
        for r in requests:
            all_itls.extend(r.inter_token_latencies)
            
        total_output_tokens = sum(r.output_tokens for r in requests)
        total_requests = len(requests)
        
        system_tps = total_output_tokens / total_benchmark_duration_sec
        system_rps = total_requests / total_benchmark_duration_sec

        def get_stats(data_seconds: List[float]):
            if not data_seconds: return 0.0, 0.0
            data_ms = [x * 1000 for x in data_seconds] # Convert to ms
            return np.mean(data_ms), np.percentile(data_ms, 99)

        ttft_mean, ttft_p99 = get_stats(ttfts)
        e2e_mean, e2e_p99 = get_stats(e2es)
        itl_mean, itl_p99 = get_stats(all_itls)
        tpot_mean, _ = get_stats(tpots)

        return BenchmarkResults(
            ttft_mean_ms=ttft_mean,
            ttft_p99_ms=ttft_p99,
            e2e_mean_ms=e2e_mean,
            e2e_p99_ms=e2e_p99,
            itl_mean_ms=itl_mean,
            itl_p99_ms=itl_p99,
            tpot_mean_ms=tpot_mean,
            system_tps=system_tps,
            system_rps=system_rps,
            mean_prefill_speed_tokens_per_sec=np.mean(prefills) if prefills else 0.0
        )

# Usage 
# if __name__ == "__main__":
#     req1 = RequestMetrics(
#         request_id="1",
#         timestamp_request_sent=100.0,
#         timestamp_first_token=100.5,    # TTFT = 0.5s
#         timestamp_last_token=102.0,     # E2E = 2.0s
#         chunk_timestamps=[100.5, 101.0, 101.5, 102.0], # 4 chunks, ITLs: 0.5, 0.5, 0.5
#         input_tokens=50,
#         output_tokens=4 # 4 tokens generated
#     )
    
#     results = MetricsCalculator.aggregate([req1], total_benchmark_duration_sec=2.0)
    
#     print("--- Benchmark Results ---")
#     print(f"TTFT (ms): {results.ttft_mean_ms:.2f}")
#     print(f"System TPS: {results.system_tps:.2f}")
#     print(f"Prefill Speed (Tok/s): {results.mean_prefill_speed_tokens_per_sec:.2f}")
#     print(f"ITL P99 (ms): {results.itl_p99_ms:.2f}")