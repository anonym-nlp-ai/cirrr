"""
Resilient vLLM client with circuit breaker pattern and retry mechanisms.

This module provides a robust wrapper around the vLLM client that includes:
- Circuit breaker pattern for fault tolerance
- Exponential backoff retry mechanisms
- Health monitoring integration
- Request/response metrics collection
- Connection pooling and resource management
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import random
from functools import wraps
from loguru import logger
from pydantic import BaseModel

from aimw.app.services.vllm.vllm_client import BaseLLMClient, create_client
from aimw.app.services.vllm.health_check import get_health_checker, HealthStatus
from aimw.app.core.vllm_client_config import LLMConfig, Message


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service is back


class RetryConfig(BaseModel):
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout_seconds: int = 60  # Time before attempting recovery
    success_threshold: int = 3  # Successful requests needed to close circuit
    request_timeout_seconds: int = 30  # Individual request timeout


class RequestMetrics(BaseModel):
    """Metrics for a single request"""
    timestamp: datetime
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    circuit_breaker_state: CircuitBreakerState


class CircuitBreaker:
    """Circuit breaker implementation for vLLM requests"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.request_history: List[RequestMetrics] = []
        
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit breaker state"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
            
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > 
                timedelta(seconds=self.config.recovery_timeout_seconds)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
            
        # HALF_OPEN state - allow limited requests
        return True
    
    def record_success(self):
        """Record a successful request"""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker recovered to CLOSED state")
        
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker returned to OPEN state")


class ResilientVLLMClient:
    """Resilient wrapper for vLLM client with circuit breaker and retry logic"""
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.client: Optional[BaseLLMClient] = None
        self.health_checker = get_health_checker()
        self.metrics_history: List[RequestMetrics] = []
        self.max_metrics_history = 10000
        
    async def _get_client(self) -> BaseLLMClient:
        """Get or create the underlying client"""
        if self.client is None:
            self.client = create_client(self.config)
        return self.client
    
    async def _execute_with_timeout(self, coro, timeout_seconds: int):
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout_seconds} seconds")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter"""
        delay = min(
            self.retry_config.base_delay_seconds * 
            (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay_seconds
        )
        
        if self.retry_config.jitter:
            # Add jitter to avoid thundering herd
            delay *= (0.5 + random.random() * 0.5)
            
        return delay
    
    async def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception type and attempt count"""
        if attempt >= self.retry_config.max_attempts:
            return False
            
        # Don't retry on certain types of errors
        non_retryable_errors = (
            ValueError,  # Invalid input
            KeyError,    # Missing required fields
        )
        
        if isinstance(exception, non_retryable_errors):
            return False
            
        # Check if service might be recovering
        if isinstance(exception, (ConnectionError, TimeoutError)):
            # Quick health check to see if service is back
            is_healthy = await self.health_checker.quick_health_check()
            return not is_healthy  # Retry if not healthy
            
        return True
    
    async def chat_completion(
        self, 
        messages: List[Message], 
        stream: Optional[bool] = None,
        request_id: Optional[str] = None
    ) -> Union[str, dict]:
        """Execute chat completion with resilience patterns"""
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker is OPEN - service unavailable")
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                # Add delay for retries
                if attempt > 1:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying request (attempt {attempt}) after {delay:.2f}s delay")
                    await asyncio.sleep(delay)
                
                # Execute request with timeout
                client = await self._get_client()
                
                if asyncio.iscoroutinefunction(client.chat_completion):
                    coro = client.chat_completion(messages, stream)
                    result = await self._execute_with_timeout(
                        coro, 
                        self.circuit_breaker.config.request_timeout_seconds
                    )
                else:
                    # Wrap synchronous call in executor
                    loop = asyncio.get_event_loop()
                    result = await self._execute_with_timeout(
                        loop.run_in_executor(
                            None, 
                            lambda: client.chat_completion(messages, stream)
                        ),
                        self.circuit_breaker.config.request_timeout_seconds
                    )
                
                # Record success
                latency_ms = (time.time() - start_time) * 1000
                self.circuit_breaker.record_success()
                self.health_checker.record_request(True, latency_ms)
                
                # Record metrics
                metrics = RequestMetrics(
                    timestamp=datetime.now(),
                    latency_ms=latency_ms,
                    success=True,
                    retry_count=attempt - 1,
                    circuit_breaker_state=self.circuit_breaker.state
                )
                self._record_metrics(metrics)
                
                logger.debug(f"Request successful on attempt {attempt} ({latency_ms:.1f}ms)")
                return result
                
            except Exception as e:
                last_exception = e
                latency_ms = (time.time() - start_time) * 1000
                
                logger.warning(f"Request failed on attempt {attempt}: {str(e)}")
                
                # Record failure
                self.circuit_breaker.record_failure()
                self.health_checker.record_request(False, latency_ms)
                
                # Check if we should retry
                should_retry = await self._should_retry(e, attempt)
                
                if not should_retry:
                    # Record final failure metrics
                    metrics = RequestMetrics(
                        timestamp=datetime.now(),
                        latency_ms=latency_ms,
                        success=False,
                        error_message=str(e),
                        retry_count=attempt - 1,
                        circuit_breaker_state=self.circuit_breaker.state
                    )
                    self._record_metrics(metrics)
                    break
        
        # All retries exhausted
        logger.error(f"Request failed after {self.retry_config.max_attempts} attempts")
        raise last_exception
    
    def _record_metrics(self, metrics: RequestMetrics):
        """Record request metrics"""
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history:]
    
    def get_metrics_summary(self, window_minutes: int = 15) -> Dict[str, Any]:
        """Get summary metrics for the specified time window"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff
        ]
        
        if not recent_metrics:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "average_latency_ms": 0.0,
                "circuit_breaker_state": self.circuit_breaker.state.value
            }
        
        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.success)
        success_rate = (successful_requests / total_requests) * 100
        
        latencies = [m.latency_ms for m in recent_metrics]
        avg_latency = sum(latencies) / len(latencies)
        
        retry_counts = [m.retry_count for m in recent_metrics]
        avg_retries = sum(retry_counts) / len(retry_counts)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "average_latency_ms": avg_latency,
            "average_retry_count": avg_retries,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failure_count": self.circuit_breaker.failure_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        health_metrics = await self.health_checker.comprehensive_health_check()
        client_metrics = self.get_metrics_summary()
        
        return {
            "health_status": health_metrics.status.value,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "uptime_seconds": health_metrics.uptime_seconds,
            "recent_success_rate": client_metrics["success_rate"],
            "recent_average_latency_ms": client_metrics["average_latency_ms"],
            "total_requests": health_metrics.total_requests,
            "failed_requests": health_metrics.failed_requests,
            "error_rate_percent": health_metrics.error_rate_percent
        }


# Convenience functions for creating resilient clients

def create_resilient_client(
    config: Optional[LLMConfig] = None,
    **kwargs
) -> ResilientVLLMClient:
    """Create a resilient vLLM client with default settings"""
    return ResilientVLLMClient(config=config, **kwargs)


def resilient_chat_completion(
    messages: List[Message],
    config: Optional[LLMConfig] = None,
    stream: Optional[bool] = None,
    **kwargs
) -> Union[str, dict]:
    """Convenience function for one-off resilient chat completions"""
    async def _execute():
        client = create_resilient_client(config)
        return await client.chat_completion(messages, stream, **kwargs)
    
    return asyncio.run(_execute())


# Decorator for adding resilience to existing functions
def with_resilience(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
):
    """Decorator to add resilience patterns to any async function"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_cfg = retry_config or RetryConfig()
            cb_cfg = circuit_breaker_config or CircuitBreakerConfig()
            cb = CircuitBreaker(cb_cfg)
            
            if not cb.can_execute():
                raise RuntimeError("Circuit breaker is OPEN")
            
            last_exception = None
            
            for attempt in range(1, retry_cfg.max_attempts + 1):
                try:
                    if attempt > 1:
                        delay = min(
                            retry_cfg.base_delay_seconds * 
                            (retry_cfg.exponential_base ** (attempt - 1)),
                            retry_cfg.max_delay_seconds
                        )
                        if retry_cfg.jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        await asyncio.sleep(delay)
                    
                    result = await func(*args, **kwargs)
                    cb.record_success()
                    return result
                    
                except Exception as e:
                    last_exception = e
                    cb.record_failure()
                    
                    if attempt >= retry_cfg.max_attempts:
                        break
                        
            raise last_exception
            
        return wrapper
    return decorator 