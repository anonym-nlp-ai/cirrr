"""
Health check system for vLLM inference server.

This module provides comprehensive health monitoring including:
- Basic connectivity checks
- Model readiness verification
- Performance health metrics
- Resource utilization monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import aiohttp
import psutil
from loguru import logger
from pydantic import BaseModel

from aimw.app.core.vllm_client_config import LLMConfig, get_vllm_client_settings


class HealthStatus(str, Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult(BaseModel):
    """Result of a health check operation"""
    status: HealthStatus
    message: str
    timestamp: datetime
    latency_ms: Optional[float] = None
    details: Optional[Dict] = None


class HealthMetrics(BaseModel):
    """Comprehensive health metrics"""
    status: HealthStatus
    uptime_seconds: float
    last_successful_request: Optional[datetime] = None
    total_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    error_rate_percent: float = 0.0


class VLLMHealthChecker:
    """Comprehensive health checker for vLLM inference server"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_vllm_client_settings()
        self.start_time = time.time()
        self.request_history: List[Tuple[datetime, bool, float]] = []
        self.max_history_size = 1000
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
        return self._session

    async def check_connectivity(self) -> HealthCheckResult:
        """Check basic connectivity to vLLM server"""
        start_time = time.time()
        
        try:
            session = await self._get_session()
            health_url = f"{self.config.api_base}/health"
            
            async with session.get(health_url) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        message="Server is reachable",
                        timestamp=datetime.now(),
                        latency_ms=latency_ms
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Server returned status {response.status}",
                        timestamp=datetime.now(),
                        latency_ms=latency_ms
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Connection timeout",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {str(e)}",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def check_model_readiness(self) -> HealthCheckResult:
        """Check if the model is ready to serve requests"""
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Test with a simple prompt
            test_payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            async with session.post(
                f"{self.config.api_base}/v1/chat/completions",
                json=test_payload
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return HealthCheckResult(
                            status=HealthStatus.HEALTHY,
                            message="Model is ready and responding",
                            timestamp=datetime.now(),
                            latency_ms=latency_ms,
                            details={"model": self.config.model}
                        )
                    else:
                        return HealthCheckResult(
                            status=HealthStatus.DEGRADED,
                            message="Model responded but with unexpected format",
                            timestamp=datetime.now(),
                            latency_ms=latency_ms
                        )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Model request failed with status {response.status}",
                        timestamp=datetime.now(),
                        latency_ms=latency_ms
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model readiness check failed: {str(e)}",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def check_performance_health(self) -> HealthCheckResult:
        """Check performance-related health metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Performance thresholds
            cpu_threshold = 80.0
            memory_threshold = 85.0
            
            issues = []
            if cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Check recent error rate
            error_rate = self._calculate_error_rate()
            if error_rate > 10.0:  # 10% error rate threshold
                issues.append(f"High error rate: {error_rate:.1f}%")
                
            if not issues:
                status = HealthStatus.HEALTHY
                message = "Performance metrics are healthy"
            elif len(issues) == 1:
                status = HealthStatus.DEGRADED
                message = f"Performance issue detected: {issues[0]}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple performance issues: {', '.join(issues)}"
            
            return HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "error_rate_percent": error_rate
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Performance check failed: {str(e)}",
                timestamp=datetime.now()
            )

    async def comprehensive_health_check(self) -> HealthMetrics:
        """Perform comprehensive health check and return metrics"""
        try:
            # Run all health checks
            connectivity_result = await self.check_connectivity()
            readiness_result = await self.check_model_readiness()
            performance_result = await self.check_performance_health()
            
            # Determine overall status
            statuses = [connectivity_result.status, readiness_result.status, performance_result.status]
            
            if HealthStatus.UNHEALTHY in statuses:
                overall_status = HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                overall_status = HealthStatus.DEGRADED
            elif HealthStatus.UNKNOWN in statuses:
                overall_status = HealthStatus.UNKNOWN
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Calculate metrics
            uptime = time.time() - self.start_time
            total_requests = len(self.request_history)
            failed_requests = sum(1 for _, success, _ in self.request_history if not success)
            error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0.0
            
            avg_latency = 0.0
            if total_requests > 0:
                avg_latency = sum(latency for _, _, latency in self.request_history) / total_requests
            
            last_successful = None
            for timestamp, success, _ in reversed(self.request_history):
                if success:
                    last_successful = timestamp
                    break
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            return HealthMetrics(
                status=overall_status,
                uptime_seconds=uptime,
                last_successful_request=last_successful,
                total_requests=total_requests,
                failed_requests=failed_requests,
                average_latency_ms=avg_latency,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                error_rate_percent=error_rate
            )
            
        except Exception as e:
            logger.error(f"Comprehensive health check failed: {str(e)}")
            return HealthMetrics(
                status=HealthStatus.UNKNOWN,
                uptime_seconds=time.time() - self.start_time,
                error_rate_percent=100.0
            )

    def record_request(self, success: bool, latency_ms: float):
        """Record a request for health metrics"""
        timestamp = datetime.now()
        self.request_history.append((timestamp, success, latency_ms))
        
        # Trim history if too large
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]

    def _calculate_error_rate(self, window_minutes: int = 5) -> float:
        """Calculate error rate for the specified time window"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_requests = [
            (timestamp, success) for timestamp, success, _ in self.request_history
            if timestamp >= cutoff
        ]
        
        if not recent_requests:
            return 0.0
            
        failed = sum(1 for _, success in recent_requests if not success)
        return (failed / len(recent_requests)) * 100

    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()


# Global health checker instance
_health_checker: Optional[VLLMHealthChecker] = None


def get_health_checker() -> VLLMHealthChecker:
    """Get the global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = VLLMHealthChecker()
    return _health_checker


async def quick_health_check() -> bool:
    """Quick health check that returns True if service is healthy"""
    checker = get_health_checker()
    try:
        result = await checker.check_connectivity()
        return result.status == HealthStatus.HEALTHY
    except Exception as e:
        logger.error(f"Quick health check failed: {str(e)}")
        return False 