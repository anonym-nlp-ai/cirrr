"""
Batch processing system for vLLM requests.

This module provides:
- Efficient batching of multiple requests
- Request queuing and prioritization
- Adaptive batch sizing based on load
- Request deduplication and caching
- Load balancing across multiple vLLM instances
- Performance optimization for throughput
"""

import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
from loguru import logger
from pydantic import BaseModel

from aimw.app.services.vllm.resilient_client import ResilientVLLMClient, create_resilient_client
from aimw.app.services.observability.telemetry import get_telemetry_manager, trace_function
from aimw.app.core.vllm_client_config import LLMConfig, Message


class RequestPriority(str, Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BatchingStrategy(str, Enum):
    """Batching strategies"""
    SIZE_BASED = "size_based"          # Batch when size threshold reached
    TIME_BASED = "time_based"          # Batch after timeout
    ADAPTIVE = "adaptive"              # Adaptive based on load
    HYBRID = "hybrid"                  # Size + time based


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    id: str
    messages: List[Message]
    stream: bool = False
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 8
    min_batch_size: int = 1
    max_wait_time_ms: int = 100
    strategy: BatchingStrategy = BatchingStrategy.HYBRID
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_deduplication: bool = True
    priority_weights: Dict[RequestPriority, float] = field(default_factory=lambda: {
        RequestPriority.LOW: 0.25,
        RequestPriority.NORMAL: 1.0,
        RequestPriority.HIGH: 2.0,
        RequestPriority.CRITICAL: 4.0
    })
    adaptive_sizing: bool = True
    load_balancing: bool = True


class RequestCache:
    """Cache for request responses"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_request(self, messages: List[Message]) -> str:
        """Generate hash for request caching"""
        content = json.dumps([msg.dict() for msg in messages], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, messages: List[Message]) -> Optional[Any]:
        """Get cached response if available and not expired"""
        key = self._hash_request(messages)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                self.hit_count += 1
                return response
            else:
                # Expired, remove from cache
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, messages: List[Message], response: Any):
        """Cache response"""
        key = self._hash_request(messages)
        self.cache[key] = (response, datetime.now())
    
    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


class LoadBalancer:
    """Load balancer for multiple vLLM instances"""
    
    def __init__(self, clients: List[ResilientVLLMClient]):
        self.clients = clients
        self.request_counts = [0] * len(clients)
        self.response_times = [deque(maxlen=100) for _ in clients]
        self.error_counts = [0] * len(clients)
    
    def get_best_client(self) -> Tuple[ResilientVLLMClient, int]:
        """Get the best client based on load and performance"""
        if not self.clients:
            raise RuntimeError("No clients available")
        
        # Calculate scores for each client
        scores = []
        for i, client in enumerate(self.clients):
            # Lower is better
            load_score = self.request_counts[i]
            
            # Average response time (lower is better)
            avg_response_time = (
                sum(self.response_times[i]) / len(self.response_times[i])
                if self.response_times[i] else 0
            )
            
            # Error rate (lower is better)
            error_rate = self.error_counts[i] / max(self.request_counts[i], 1)
            
            # Combined score (weighted)
            score = load_score * 0.4 + avg_response_time * 0.4 + error_rate * 100 * 0.2
            scores.append(score)
        
        # Select client with lowest score
        best_index = scores.index(min(scores))
        return self.clients[best_index], best_index
    
    def record_request(self, client_index: int, response_time: float, success: bool):
        """Record request metrics for load balancing"""
        self.request_counts[client_index] += 1
        self.response_times[client_index].append(response_time)
        if not success:
            self.error_counts[client_index] += 1


class VLLMBatchProcessor:
    """Main batch processor for vLLM requests"""
    
    def __init__(self, config: BatchConfig, clients: Optional[List[ResilientVLLMClient]] = None):
        self.config = config
        self.clients = clients or [create_resilient_client()]
        self.load_balancer = LoadBalancer(self.clients) if config.load_balancing else None
        
        # Request queues by priority
        self.request_queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        
        # Cache and deduplication
        self.cache = RequestCache(config.cache_ttl_seconds) if config.enable_caching else None
        self.dedup_map: Dict[str, List[BatchRequest]] = defaultdict(list)
        
        # Processing state
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        self.batch_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.processing_times = deque(maxlen=1000)
        self.batch_sizes = deque(maxlen=1000)
        
        # Telemetry
        self.telemetry = get_telemetry_manager()
        
        # Start batch processing task
        self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start the background batch processing task"""
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._batch_processing_loop())
    
    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        while True:
            try:
                await self._process_batch()
                
                # Dynamic wait time based on strategy
                wait_time = self._calculate_wait_time()
                await asyncio.sleep(wait_time / 1000.0)  # Convert ms to seconds
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    def _calculate_wait_time(self) -> int:
        """Calculate dynamic wait time based on load and strategy"""
        if self.config.strategy == BatchingStrategy.TIME_BASED:
            return self.config.max_wait_time_ms
        
        # For adaptive and hybrid strategies, adjust based on queue size
        total_queued = sum(len(queue) for queue in self.request_queues.values())
        
        if total_queued == 0:
            return self.config.max_wait_time_ms
        elif total_queued >= self.config.max_batch_size:
            return 1  # Process immediately
        else:
            # Adaptive wait time based on queue fill ratio
            fill_ratio = total_queued / self.config.max_batch_size
            return max(1, int(self.config.max_wait_time_ms * (1 - fill_ratio)))
    
    async def submit_request(
        self,
        messages: List[Message],
        stream: bool = False,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Submit a request for batch processing"""
        # Check cache first
        if self.cache and not stream:
            cached_response = self.cache.get(messages)
            if cached_response:
                return cached_response
        
        # Create request
        request_id = f"{int(time.time() * 1000)}_{hash(str(messages))}"
        request = BatchRequest(
            id=request_id,
            messages=messages,
            stream=stream,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Handle deduplication
        if self.config.enable_deduplication:
            request_hash = self._hash_request(messages)
            self.dedup_map[request_hash].append(request)
        
        # Add to appropriate priority queue
        self.request_queues[priority].append(request)
        self.total_requests += 1
        
        # Record metrics
        self.telemetry.record_vllm_request(
            model="queued",
            latency_seconds=0,
            tokens=0,
            success=True
        )
        
        # Return future result
        return await request.future
    
    def _hash_request(self, messages: List[Message]) -> str:
        """Generate hash for request deduplication"""
        content = json.dumps([msg.dict() for msg in messages], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _process_batch(self):
        """Process a batch of requests"""
        if self.is_processing:
            return
        
        async with self.processing_lock:
            if self.is_processing:
                return
            
            self.is_processing = True
            
            try:
                # Collect requests for batch
                batch_requests = self._collect_batch_requests()
                
                if not batch_requests:
                    return
                
                # Process the batch
                await self._execute_batch(batch_requests)
                
            finally:
                self.is_processing = False
    
    def _collect_batch_requests(self) -> List[BatchRequest]:
        """Collect requests from queues to form a batch"""
        batch_requests = []
        remaining_capacity = self.config.max_batch_size
        
        # Process by priority (highest first)
        for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH, 
                        RequestPriority.NORMAL, RequestPriority.LOW]:
            queue = self.request_queues[priority]
            
            while queue and remaining_capacity > 0:
                request = queue.popleft()
                batch_requests.append(request)
                remaining_capacity -= 1
        
        # Check if we have minimum batch size or timeout
        if len(batch_requests) < self.config.min_batch_size:
            # Check if oldest request has exceeded wait time
            if batch_requests:
                oldest_request = min(batch_requests, key=lambda r: r.timestamp)
                wait_time = (datetime.now() - oldest_request.timestamp).total_seconds() * 1000
                
                if wait_time < self.config.max_wait_time_ms:
                    # Put requests back and wait
                    for request in batch_requests:
                        self.request_queues[request.priority].appendleft(request)
                    return []
        
        return batch_requests
    
    @trace_function("vllm_batch_execution")
    async def _execute_batch(self, batch_requests: List[BatchRequest]):
        """Execute a batch of requests"""
        if not batch_requests:
            return
        
        start_time = time.time()
        
        async with self.telemetry.async_trace_span(
            "vllm.batch_processing",
            attributes={
                "batch.size": len(batch_requests),
                "batch.strategy": self.config.strategy.value
            }
        ) as span:
            try:
                # Group requests by deduplication
                if self.config.enable_deduplication:
                    deduplicated_groups = self._group_deduplicated_requests(batch_requests)
                    
                    # Process each unique request
                    for unique_requests in deduplicated_groups.values():
                        await self._process_request_group(unique_requests)
                else:
                    # Process each request individually
                    await asyncio.gather(*[
                        self._process_single_request(request) 
                        for request in batch_requests
                    ], return_exceptions=True)
                
                # Record batch metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.batch_sizes.append(len(batch_requests))
                self.total_batches += 1
                
                if span:
                    span.set_attribute("batch.processing_time_ms", processing_time * 1000)
                    span.set_attribute("batch.total_processed", self.total_batches)
                
                logger.debug(f"Processed batch of {len(batch_requests)} requests in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Set error for all requests in batch
                for request in batch_requests:
                    if not request.future.done():
                        request.future.set_exception(e)
    
    def _group_deduplicated_requests(self, batch_requests: List[BatchRequest]) -> Dict[str, List[BatchRequest]]:
        """Group requests by deduplication hash"""
        groups = defaultdict(list)
        
        for request in batch_requests:
            request_hash = self._hash_request(request.messages)
            groups[request_hash].append(request)
        
        return groups
    
    async def _process_request_group(self, requests: List[BatchRequest]):
        """Process a group of deduplicated requests"""
        if not requests:
            return
        
        # Use the first request as representative
        primary_request = requests[0]
        
        try:
            response = await self._process_single_request(primary_request)
            
            # Set result for all deduplicated requests
            for request in requests:
                if not request.future.done():
                    request.future.set_result(response)
            
            # Cache the response
            if self.cache and not primary_request.stream:
                self.cache.set(primary_request.messages, response)
                
        except Exception as e:
            # Set error for all requests in group
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_single_request(self, request: BatchRequest) -> Any:
        """Process a single request"""
        start_time = time.time()
        client_index = 0
        
        try:
            # Get client (load balanced if enabled)
            if self.load_balancer:
                client, client_index = self.load_balancer.get_best_client()
            else:
                client = self.clients[0]
            
            # Execute request
            response = await client.chat_completion(
                messages=request.messages,
                stream=request.stream,
                request_id=request.id
            )
            
            # Record success metrics
            processing_time = time.time() - start_time
            if self.load_balancer:
                self.load_balancer.record_request(client_index, processing_time, True)
            
            # Set result
            if not request.future.done():
                request.future.set_result(response)
            
            return response
            
        except Exception as e:
            # Record failure metrics
            processing_time = time.time() - start_time
            if self.load_balancer:
                self.load_balancer.record_request(client_index, processing_time, False)
            
            # Retry logic
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                # Re-queue with higher priority
                higher_priority = self._get_higher_priority(request.priority)
                request.priority = higher_priority
                self.request_queues[higher_priority].appendleft(request)
                logger.warning(f"Retrying request {request.id} (attempt {request.retry_count})")
                return  # Don't set exception yet
            
            # Set exception after max retries
            if not request.future.done():
                request.future.set_exception(e)
            
            raise e
    
    def _get_higher_priority(self, current_priority: RequestPriority) -> RequestPriority:
        """Get higher priority for retry"""
        priority_order = [RequestPriority.LOW, RequestPriority.NORMAL, 
                         RequestPriority.HIGH, RequestPriority.CRITICAL]
        
        current_index = priority_order.index(current_priority)
        if current_index < len(priority_order) - 1:
            return priority_order[current_index + 1]
        return current_priority
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics"""
        queue_sizes = {
            priority.value: len(queue) 
            for priority, queue in self.request_queues.items()
        }
        
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        avg_batch_size = (
            sum(self.batch_sizes) / len(self.batch_sizes)
            if self.batch_sizes else 0
        )
        
        stats = {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "queue_sizes": queue_sizes,
            "avg_processing_time": avg_processing_time,
            "avg_batch_size": avg_batch_size,
            "is_processing": self.is_processing
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    async def shutdown(self):
        """Gracefully shutdown the batch processor"""
        logger.info("Shutting down batch processor...")
        
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining requests
        remaining_requests = []
        for queue in self.request_queues.values():
            remaining_requests.extend(queue)
            queue.clear()
        
        if remaining_requests:
            logger.info(f"Processing {len(remaining_requests)} remaining requests...")
            await self._execute_batch(remaining_requests)
        
        logger.info("Batch processor shutdown complete")


# Global batch processor instance
_batch_processor: Optional[VLLMBatchProcessor] = None


def get_batch_processor(
    config: Optional[BatchConfig] = None,
    clients: Optional[List[ResilientVLLMClient]] = None
) -> VLLMBatchProcessor:
    """Get or create the global batch processor"""
    global _batch_processor
    
    if _batch_processor is None:
        _batch_processor = VLLMBatchProcessor(
            config=config or BatchConfig(),
            clients=clients
        )
    
    return _batch_processor


# Convenience functions

async def batch_chat_completion(
    messages: List[Message],
    stream: bool = False,
    priority: RequestPriority = RequestPriority.NORMAL,
    config: Optional[BatchConfig] = None
) -> Any:
    """Submit a chat completion request for batch processing"""
    processor = get_batch_processor(config)
    return await processor.submit_request(messages, stream, priority) 