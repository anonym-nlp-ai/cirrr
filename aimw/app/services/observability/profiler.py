"""
Performance profiling and benchmarking system for vLLM optimization.

This module provides:
- CPU and memory profiling
- GPU utilization monitoring
- Request latency analysis
- Throughput benchmarking
- Performance bottleneck identification
- Optimization recommendations
- Load testing capabilities
"""

import asyncio
import time
import statistics
import cProfile
import pstats
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading
from collections import defaultdict, deque
from loguru import logger
from pydantic import BaseModel

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not available - GPU monitoring disabled")

from aimw.app.services.observability.telemetry import get_telemetry_manager, trace_function


class ProfilerType(str, Enum):
    """Types of profiling"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    FULL = "full"


class BenchmarkType(str, Enum):
    """Types of benchmarks"""
    SINGLE_REQUEST = "single_request"
    BATCH_PROCESSING = "batch_processing"
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    ENDURANCE_TEST = "endurance_test"


@dataclass
class ProfileResult:
    """Result of a profiling session"""
    profiler_type: ProfilerType
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_stats: Optional[Dict[str, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None
    gpu_stats: Optional[Dict[str, Any]] = None
    latency_stats: Optional[Dict[str, Any]] = None
    throughput_stats: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # File paths to artifacts


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    benchmark_type: BenchmarkType
    duration_seconds: int = 60
    concurrent_requests: int = 1
    warmup_seconds: int = 10
    cooldown_seconds: int = 5
    sample_requests: List[Dict[str, Any]] = field(default_factory=list)
    target_qps: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_success_rate: float = 0.95


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.cpu_samples = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=1000)
        self.gpu_samples = deque(maxlen=1000) if GPU_AVAILABLE else None
        
    async def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_freq = psutil.cpu_freq()
                cpu_count = psutil.cpu_count()
                
                cpu_sample = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'cpu_freq_mhz': cpu_freq.current if cpu_freq else None,
                    'cpu_count': cpu_count
                }
                self.cpu_samples.append(cpu_sample)
                
                # Memory monitoring
                memory = psutil.virtual_memory()
                memory_sample = {
                    'timestamp': datetime.now(),
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_total_gb': memory.total / (1024**3)
                }
                self.memory_samples.append(memory_sample)
                
                # GPU monitoring (if available)
                if GPU_AVAILABLE and self.gpu_samples is not None:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Monitor first GPU
                            gpu_sample = {
                                'timestamp': datetime.now(),
                                'gpu_load': gpu.load * 100,
                                'gpu_memory_used': gpu.memoryUsed,
                                'gpu_memory_total': gpu.memoryTotal,
                                'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                'gpu_temperature': gpu.temperature
                            }
                            self.gpu_samples.append(gpu_sample)
                    except Exception as e:
                        logger.warning(f"GPU monitoring error: {str(e)}")
                
                await asyncio.sleep(self.interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = {}
        
        # CPU stats
        if self.cpu_samples:
            cpu_values = [sample['cpu_percent'] for sample in self.cpu_samples]
            stats['cpu'] = {
                'avg_percent': statistics.mean(cpu_values),
                'max_percent': max(cpu_values),
                'min_percent': min(cpu_values),
                'p95_percent': statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) > 20 else max(cpu_values),
                'sample_count': len(cpu_values)
            }
        
        # Memory stats
        if self.memory_samples:
            memory_values = [sample['memory_percent'] for sample in self.memory_samples]
            memory_used_gb = [sample['memory_used_gb'] for sample in self.memory_samples]
            stats['memory'] = {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'min_percent': min(memory_values),
                'p95_percent': statistics.quantiles(memory_values, n=20)[18] if len(memory_values) > 20 else max(memory_values),
                'avg_used_gb': statistics.mean(memory_used_gb),
                'max_used_gb': max(memory_used_gb),
                'sample_count': len(memory_values)
            }
        
        # GPU stats
        if self.gpu_samples:
            gpu_load_values = [sample['gpu_load'] for sample in self.gpu_samples]
            gpu_memory_values = [sample['gpu_memory_percent'] for sample in self.gpu_samples]
            gpu_temp_values = [sample['gpu_temperature'] for sample in self.gpu_samples]
            
            stats['gpu'] = {
                'avg_load_percent': statistics.mean(gpu_load_values),
                'max_load_percent': max(gpu_load_values),
                'min_load_percent': min(gpu_load_values),
                'avg_memory_percent': statistics.mean(gpu_memory_values),
                'max_memory_percent': max(gpu_memory_values),
                'avg_temperature': statistics.mean(gpu_temp_values),
                'max_temperature': max(gpu_temp_values),
                'sample_count': len(gpu_load_values)
            }
        
        return stats


class LatencyProfiler:
    """Latency profiling for requests"""
    
    def __init__(self):
        self.latency_samples = deque(maxlen=10000)
        self.request_times = {}
        
    def start_request(self, request_id: str):
        """Start timing a request"""
        self.request_times[request_id] = time.time()
    
    def end_request(self, request_id: str, success: bool = True) -> Optional[float]:
        """End timing a request and return latency"""
        if request_id not in self.request_times:
            return None
            
        start_time = self.request_times.pop(request_id)
        latency = time.time() - start_time
        
        self.latency_samples.append({
            'timestamp': datetime.now(),
            'latency_seconds': latency,
            'success': success,
            'request_id': request_id
        })
        
        return latency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get latency statistics"""
        if not self.latency_samples:
            return {}
        
        successful_samples = [s for s in self.latency_samples if s['success']]
        failed_samples = [s for s in self.latency_samples if not s['success']]
        
        if not successful_samples:
            return {'total_requests': len(self.latency_samples), 'success_rate': 0.0}
        
        latencies = [sample['latency_seconds'] for sample in successful_samples]
        
        return {
            'total_requests': len(self.latency_samples),
            'successful_requests': len(successful_samples),
            'failed_requests': len(failed_samples),
            'success_rate': len(successful_samples) / len(self.latency_samples),
            'avg_latency_ms': statistics.mean(latencies) * 1000,
            'median_latency_ms': statistics.median(latencies) * 1000,
            'min_latency_ms': min(latencies) * 1000,
            'max_latency_ms': max(latencies) * 1000,
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) > 20 else max(latencies) * 1000,
            'p99_latency_ms': statistics.quantiles(latencies, n=100)[98] * 1000 if len(latencies) > 100 else max(latencies) * 1000
        }


class ThroughputProfiler:
    """Throughput profiling"""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.request_timestamps = deque()
        
    def record_request(self, timestamp: Optional[datetime] = None):
        """Record a request"""
        timestamp = timestamp or datetime.now()
        self.request_timestamps.append(timestamp)
        
        # Clean old timestamps outside the window
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
    
    def get_current_qps(self) -> float:
        """Get current queries per second"""
        if len(self.request_timestamps) < 2:
            return 0.0
        
        time_span = (self.request_timestamps[-1] - self.request_timestamps[0]).total_seconds()
        if time_span == 0:
            return 0.0
        
        return len(self.request_timestamps) / time_span
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throughput statistics"""
        return {
            'current_qps': self.get_current_qps(),
            'total_requests_in_window': len(self.request_timestamps),
            'window_seconds': self.window_seconds
        }


class PerformanceProfiler:
    """Main performance profiler"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./profiling_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_monitor = SystemMonitor()
        self.latency_profiler = LatencyProfiler()
        self.throughput_profiler = ThroughputProfiler()
        self.telemetry = get_telemetry_manager()
        
        self.active_profiles: Dict[str, Any] = {}
        
    async def start_profiling(
        self,
        profile_id: str,
        profiler_type: ProfilerType = ProfilerType.FULL,
        enable_cpu_profiling: bool = True,
        enable_memory_profiling: bool = True
    ) -> str:
        """Start a profiling session"""
        if profile_id in self.active_profiles:
            raise ValueError(f"Profile {profile_id} is already active")
        
        profile_data = {
            'type': profiler_type,
            'start_time': datetime.now(),
            'cpu_profiler': None,
            'memory_snapshot': None
        }
        
        # Start CPU profiling
        if enable_cpu_profiling and profiler_type in [ProfilerType.CPU, ProfilerType.FULL]:
            profile_data['cpu_profiler'] = cProfile.Profile()
            profile_data['cpu_profiler'].enable()
        
        # Start memory profiling
        if enable_memory_profiling and profiler_type in [ProfilerType.MEMORY, ProfilerType.FULL]:
            tracemalloc.start()
            profile_data['memory_snapshot'] = tracemalloc.take_snapshot()
        
        # Start system monitoring
        if profiler_type in [ProfilerType.FULL, ProfilerType.GPU]:
            await self.system_monitor.start_monitoring()
        
        self.active_profiles[profile_id] = profile_data
        logger.info(f"Started profiling session: {profile_id} ({profiler_type.value})")
        
        return profile_id
    
    async def stop_profiling(self, profile_id: str) -> ProfileResult:
        """Stop a profiling session and return results"""
        if profile_id not in self.active_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile_data = self.active_profiles.pop(profile_id)
        end_time = datetime.now()
        duration = (end_time - profile_data['start_time']).total_seconds()
        
        result = ProfileResult(
            profiler_type=profile_data['type'],
            start_time=profile_data['start_time'],
            end_time=end_time,
            duration_seconds=duration
        )
        
        # Stop CPU profiling
        if profile_data['cpu_profiler']:
            profile_data['cpu_profiler'].disable()
            
            # Save CPU profile
            cpu_profile_path = self.output_dir / f"{profile_id}_cpu_profile.prof"
            profile_data['cpu_profiler'].dump_stats(str(cpu_profile_path))
            result.artifacts['cpu_profile'] = str(cpu_profile_path)
            
            # Generate CPU stats
            stats = pstats.Stats(profile_data['cpu_profiler'])
            result.cpu_stats = self._analyze_cpu_profile(stats)
        
        # Stop memory profiling
        if profile_data['memory_snapshot'] and tracemalloc.is_tracing():
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(profile_data['memory_snapshot'], 'lineno')
            
            result.memory_stats = self._analyze_memory_profile(top_stats)
            tracemalloc.stop()
        
        # Stop system monitoring and get stats
        if profile_data['type'] in [ProfilerType.FULL, ProfilerType.GPU]:
            await self.system_monitor.stop_monitoring()
            system_stats = self.system_monitor.get_stats()
            
            if 'cpu' in system_stats:
                result.cpu_stats = {**(result.cpu_stats or {}), **system_stats['cpu']}
            if 'memory' in system_stats:
                result.memory_stats = {**(result.memory_stats or {}), **system_stats['memory']}
            if 'gpu' in system_stats:
                result.gpu_stats = system_stats['gpu']
        
        # Get latency and throughput stats
        result.latency_stats = self.latency_profiler.get_stats()
        result.throughput_stats = self.throughput_profiler.get_stats()
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        # Save full report
        report_path = self.output_dir / f"{profile_id}_report.json"
        self._save_report(result, report_path)
        result.artifacts['report'] = str(report_path)
        
        logger.info(f"Completed profiling session: {profile_id}")
        return result
    
    def record_request_start(self, request_id: str):
        """Record the start of a request"""
        self.latency_profiler.start_request(request_id)
        self.throughput_profiler.record_request()
    
    def record_request_end(self, request_id: str, success: bool = True):
        """Record the end of a request"""
        return self.latency_profiler.end_request(request_id, success)
    
    def _analyze_cpu_profile(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Analyze CPU profiling results"""
        # Get top functions by cumulative time
        stats.sort_stats('cumulative')
        
        # Capture stats output
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            stats.print_stats(20)  # Top 20 functions
            output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Parse basic stats
        total_calls = stats.total_calls
        total_time = stats.total_tt
        
        return {
            'total_calls': total_calls,
            'total_time_seconds': total_time,
            'calls_per_second': total_calls / total_time if total_time > 0 else 0,
            'top_functions': output
        }
    
    def _analyze_memory_profile(self, top_stats) -> Dict[str, Any]:
        """Analyze memory profiling results"""
        if not top_stats:
            return {}
        
        total_size_diff = sum(stat.size_diff for stat in top_stats)
        total_count_diff = sum(stat.count_diff for stat in top_stats)
        
        # Top memory consumers
        top_consumers = []
        for stat in top_stats[:10]:
            top_consumers.append({
                'file': stat.traceback.filename,
                'line': stat.traceback.lineno,
                'size_diff_mb': stat.size_diff / (1024 * 1024),
                'count_diff': stat.count_diff
            })
        
        return {
            'total_size_diff_mb': total_size_diff / (1024 * 1024),
            'total_count_diff': total_count_diff,
            'top_consumers': top_consumers
        }
    
    def _generate_recommendations(self, result: ProfileResult) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if result.cpu_stats:
            avg_cpu = result.cpu_stats.get('avg_percent', 0)
            if avg_cpu > 80:
                recommendations.append(
                    f"High CPU usage detected ({avg_cpu:.1f}%). Consider optimizing hot code paths or scaling horizontally."
                )
            elif avg_cpu < 20:
                recommendations.append(
                    f"Low CPU usage detected ({avg_cpu:.1f}%). System may be I/O bound or underutilized."
                )
        
        # Memory recommendations
        if result.memory_stats:
            avg_memory = result.memory_stats.get('avg_percent', 0)
            if avg_memory > 85:
                recommendations.append(
                    f"High memory usage detected ({avg_memory:.1f}%). Consider optimizing memory usage or increasing available memory."
                )
        
        # GPU recommendations
        if result.gpu_stats:
            avg_gpu_load = result.gpu_stats.get('avg_load_percent', 0)
            avg_gpu_memory = result.gpu_stats.get('avg_memory_percent', 0)
            
            if avg_gpu_load > 90:
                recommendations.append(
                    f"High GPU load detected ({avg_gpu_load:.1f}%). Consider optimizing model or increasing batch size."
                )
            elif avg_gpu_load < 30:
                recommendations.append(
                    f"Low GPU utilization detected ({avg_gpu_load:.1f}%). Consider increasing batch size or concurrent requests."
                )
            
            if avg_gpu_memory > 90:
                recommendations.append(
                    f"High GPU memory usage detected ({avg_gpu_memory:.1f}%). Consider reducing model size or batch size."
                )
        
        # Latency recommendations
        if result.latency_stats:
            p95_latency = result.latency_stats.get('p95_latency_ms', 0)
            success_rate = result.latency_stats.get('success_rate', 1.0)
            
            if p95_latency > 5000:  # 5 seconds
                recommendations.append(
                    f"High latency detected (P95: {p95_latency:.0f}ms). Consider optimizing inference pipeline or adding caching."
                )
            
            if success_rate < 0.95:
                recommendations.append(
                    f"Low success rate detected ({success_rate:.1%}). Investigate error causes and improve error handling."
                )
        
        # Throughput recommendations
        if result.throughput_stats:
            current_qps = result.throughput_stats.get('current_qps', 0)
            if current_qps < 1:
                recommendations.append(
                    f"Low throughput detected ({current_qps:.2f} QPS). Consider optimizing request processing or adding parallelization."
                )
        
        return recommendations
    
    def _save_report(self, result: ProfileResult, path: Path):
        """Save profiling report to file"""
        report_data = {
            'profiler_type': result.profiler_type.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration_seconds': result.duration_seconds,
            'cpu_stats': result.cpu_stats,
            'memory_stats': result.memory_stats,
            'gpu_stats': result.gpu_stats,
            'latency_stats': result.latency_stats,
            'throughput_stats': result.throughput_stats,
            'recommendations': result.recommendations,
            'artifacts': result.artifacts
        }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_profiler(output_dir: Optional[Path] = None) -> PerformanceProfiler:
    """Get or create the global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler(output_dir)
    return _profiler


# Decorators for automatic profiling

def profile_function(
    profiler_type: ProfilerType = ProfilerType.CPU,
    profile_id: Optional[str] = None
):
    """Decorator to profile a function"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            profiler = get_profiler()
            pid = profile_id or f"{func.__name__}_{int(time.time())}"
            
            await profiler.start_profiling(pid, profiler_type)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                await profiler.stop_profiling(pid)
        
        def sync_wrapper(*args, **kwargs):
            profiler = get_profiler()
            pid = profile_id or f"{func.__name__}_{int(time.time())}"
            
            asyncio.run(profiler.start_profiling(pid, profiler_type))
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                asyncio.run(profiler.stop_profiling(pid))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator 