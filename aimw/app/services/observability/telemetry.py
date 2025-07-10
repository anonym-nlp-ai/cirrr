"""
OpenTelemetry integration for comprehensive observability.

This module provides:
- Distributed tracing for request flows
- Custom metrics collection for business logic
- Performance profiling and monitoring
- Integration with Prometheus and Jaeger
- Automatic instrumentation for HTTP and database calls
"""

import asyncio
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Any, Optional, List, Union
from functools import wraps
import os
from datetime import datetime

from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes
from loguru import logger
from pydantic import BaseModel


class TelemetryConfig(BaseModel):
    """Configuration for telemetry setup"""
    service_name: str = "cir3-aimw"
    service_version: str = "0.1.15"
    environment: str = "development"
    jaeger_endpoint: Optional[str] = "http://localhost:14268/api/traces"
    prometheus_port: int = 8000
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_auto_instrumentation: bool = True
    trace_sample_rate: float = 1.0


class CustomMetrics:
    """Custom application metrics for CIR3"""
    
    def __init__(self, meter):
        self.meter = meter
        
        # Request metrics
        self.request_counter = meter.create_counter(
            name="cir3_requests_total",
            description="Total number of requests processed",
            unit="1"
        )
        
        self.request_duration = meter.create_histogram(
            name="cir3_request_duration_seconds",
            description="Request processing duration",
            unit="s"
        )
        
        # vLLM specific metrics
        self.vllm_requests = meter.create_counter(
            name="cir3_vllm_requests_total",
            description="Total vLLM requests",
            unit="1"
        )
        
        self.vllm_latency = meter.create_histogram(
            name="cir3_vllm_latency_seconds",
            description="vLLM request latency",
            unit="s"
        )
        
        self.vllm_tokens = meter.create_counter(
            name="cir3_vllm_tokens_total",
            description="Total tokens processed by vLLM",
            unit="1"
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = meter.create_up_down_counter(
            name="cir3_circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=half-open, 2=open)",
            unit="1"
        )
        
        self.circuit_breaker_failures = meter.create_counter(
            name="cir3_circuit_breaker_failures_total",
            description="Circuit breaker failures",
            unit="1"
        )
        
        # QA generation metrics
        self.qa_generation_requests = meter.create_counter(
            name="cir3_qa_generation_total",
            description="Total QA generation requests",
            unit="1"
        )
        
        self.qa_pairs_generated = meter.create_counter(
            name="cir3_qa_pairs_total",
            description="Total QA pairs generated",
            unit="1"
        )
        
        self.document_length = meter.create_histogram(
            name="cir3_document_length_chars",
            description="Input document length in characters",
            unit="chars"
        )
        
        # Agent interaction metrics
        self.agent_interactions = meter.create_counter(
            name="cir3_agent_interactions_total",
            description="Agent interactions by type",
            unit="1"
        )
        
        self.consensus_rounds = meter.create_histogram(
            name="cir3_consensus_rounds",
            description="Number of consensus rounds",
            unit="1"
        )
        
        # Health metrics
        self.health_checks = meter.create_counter(
            name="cir3_health_checks_total",
            description="Health check results",
            unit="1"
        )


class TelemetryManager:
    """Central telemetry management"""
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None
        self.custom_metrics: Optional[CustomMetrics] = None
        self._initialized = False
        
    def initialize(self):
        """Initialize OpenTelemetry components"""
        if self._initialized:
            return
            
        try:
            # Configure resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            })
            
            # Setup tracing
            if self.config.enable_tracing:
                self._setup_tracing(resource)
                
            # Setup metrics
            if self.config.enable_metrics:
                self._setup_metrics(resource)
                
            # Setup auto-instrumentation
            if self.config.enable_auto_instrumentation:
                self._setup_auto_instrumentation()
                
            self._initialized = True
            logger.info("OpenTelemetry telemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {str(e)}")
            
    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing"""
        # Configure tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Add Jaeger exporter if endpoint is configured
        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                endpoint=self.config.jaeger_endpoint,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection"""
        # Configure Prometheus metric reader
        prometheus_reader = PrometheusMetricReader(port=self.config.prometheus_port)
        
        # Configure meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        )
        
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
        self.custom_metrics = CustomMetrics(self.meter)
        
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation"""
        try:
            # Instrument HTTP clients
            RequestsInstrumentor().instrument()
            AioHttpClientInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
            logger.info("Auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Auto-instrumentation setup failed: {str(e)}")
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        status: Optional[StatusCode] = None
    ):
        """Context manager for creating trace spans"""
        if not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                    
            try:
                yield span
                if status:
                    span.set_status(Status(status))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    @asynccontextmanager
    async def async_trace_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        status: Optional[StatusCode] = None
    ):
        """Async context manager for creating trace spans"""
        if not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                    
            try:
                yield span
                if status:
                    span.set_status(Status(status))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def record_vllm_request(
        self,
        model: str,
        latency_seconds: float,
        tokens: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Record vLLM request metrics"""
        if not self.custom_metrics:
            return
            
        labels = {
            "model": model,
            "status": "success" if success else "error"
        }
        
        if error:
            labels["error_type"] = error
            
        self.custom_metrics.vllm_requests.add(1, labels)
        self.custom_metrics.vllm_latency.record(latency_seconds, labels)
        
        if success:
            self.custom_metrics.vllm_tokens.add(tokens, {"model": model})
    
    def record_qa_generation(
        self,
        document_length: int,
        qa_pairs_count: int,
        processing_time: float,
        success: bool
    ):
        """Record QA generation metrics"""
        if not self.custom_metrics:
            return
            
        labels = {"status": "success" if success else "error"}
        
        self.custom_metrics.qa_generation_requests.add(1, labels)
        self.custom_metrics.document_length.record(document_length)
        
        if success:
            self.custom_metrics.qa_pairs_generated.add(qa_pairs_count)
            
        self.custom_metrics.request_duration.record(processing_time, labels)
    
    def record_agent_interaction(
        self,
        agent_type: str,
        interaction_type: str,
        success: bool
    ):
        """Record agent interaction metrics"""
        if not self.custom_metrics:
            return
            
        labels = {
            "agent_type": agent_type,
            "interaction_type": interaction_type,
            "status": "success" if success else "error"
        }
        
        self.custom_metrics.agent_interactions.add(1, labels)
    
    def record_consensus_round(self, round_count: int):
        """Record consensus round metrics"""
        if not self.custom_metrics:
            return
            
        self.custom_metrics.consensus_rounds.record(round_count)
    
    def record_circuit_breaker_state(self, state: str, failures: int = 0):
        """Record circuit breaker state"""
        if not self.custom_metrics:
            return
            
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
        self.custom_metrics.circuit_breaker_state.add(state_value)
        
        if failures > 0:
            self.custom_metrics.circuit_breaker_failures.add(failures)
    
    def record_health_check(self, component: str, status: str):
        """Record health check results"""
        if not self.custom_metrics:
            return
            
        labels = {
            "component": component,
            "status": status
        }
        
        self.custom_metrics.health_checks.add(1, labels)


# Global telemetry manager instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager(config: Optional[TelemetryConfig] = None) -> TelemetryManager:
    """Get or create the global telemetry manager"""
    global _telemetry_manager
    
    if _telemetry_manager is None:
        if config is None:
            # Load from environment
            config = TelemetryConfig(
                service_name=os.getenv("OTEL_SERVICE_NAME", "cir3-aimw"),
                service_version=os.getenv("OTEL_SERVICE_VERSION", "0.1.15"),
                environment=os.getenv("OTEL_ENVIRONMENT", "development"),
                jaeger_endpoint=os.getenv("OTEL_JAEGER_ENDPOINT"),
                prometheus_port=int(os.getenv("OTEL_PROMETHEUS_PORT", "8000")),
                enable_tracing=os.getenv("OTEL_ENABLE_TRACING", "true").lower() == "true",
                enable_metrics=os.getenv("OTEL_ENABLE_METRICS", "true").lower() == "true",
            )
        
        _telemetry_manager = TelemetryManager(config)
        _telemetry_manager.initialize()
    
    return _telemetry_manager


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Decorator for tracing function calls"""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                telemetry = get_telemetry_manager()
                async with telemetry.async_trace_span(
                    func_name,
                    attributes=attributes
                ) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                telemetry = get_telemetry_manager()
                with telemetry.trace_span(
                    func_name,
                    attributes=attributes
                ) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def trace_vllm_request(model: str):
    """Decorator for tracing vLLM requests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            telemetry = get_telemetry_manager()
            start_time = time.time()
            
            async with telemetry.async_trace_span(
                "vllm.request",
                attributes={
                    "vllm.model": model,
                    "vllm.operation": "chat_completion"
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success metrics
                    latency = time.time() - start_time
                    tokens = getattr(result, 'usage', {}).get('total_tokens', 0)
                    
                    telemetry.record_vllm_request(
                        model=model,
                        latency_seconds=latency,
                        tokens=tokens,
                        success=True
                    )
                    
                    if span:
                        span.set_attribute("vllm.tokens", tokens)
                        span.set_attribute("vllm.latency_ms", latency * 1000)
                    
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    latency = time.time() - start_time
                    
                    telemetry.record_vllm_request(
                        model=model,
                        latency_seconds=latency,
                        tokens=0,
                        success=False,
                        error=type(e).__name__
                    )
                    
                    raise
        
        return wrapper
    return decorator


# Convenience functions for common operations

def initialize_telemetry(config: Optional[TelemetryConfig] = None):
    """Initialize telemetry system"""
    manager = get_telemetry_manager(config)
    logger.info(f"Telemetry initialized for service: {manager.config.service_name}")


def get_current_span():
    """Get the current active span"""
    return trace.get_current_span()


def add_span_attribute(key: str, value: Any):
    """Add attribute to current span"""
    span = get_current_span()
    if span:
        span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add event to current span"""
    span = get_current_span()
    if span:
        span.add_event(name, attributes or {})


def record_exception(exception: Exception):
    """Record exception in current span"""
    span = get_current_span()
    if span:
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception))) 