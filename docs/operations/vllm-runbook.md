# vLLM Operations Runbook

## Table of Contents
1. [Service Overview](#service-overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Health Checks](#health-checks)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Incident Response](#incident-response)
9. [Maintenance](#maintenance)

## Service Overview

### What is vLLM?
vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It provides:
- Fast inference with continuous batching
- GPU memory optimization
- OpenAI-compatible API
- Distributed inference support

### Service Dependencies
- **GPU Hardware**: NVIDIA GPUs with CUDA support
- **Container Runtime**: Docker with NVIDIA container toolkit
- **Storage**: Local SSD for model caching
- **Network**: High-bandwidth network for model loading
- **Monitoring**: Prometheus, Grafana, Jaeger

### SLA Targets
- **Availability**: 99.9% uptime
- **Latency**: P95 < 10 seconds, P99 < 30 seconds
- **Throughput**: > 10 requests/second
- **Error Rate**: < 5%

## Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  vLLM Service   │────│  GPU Cluster    │
│   (Nginx)       │    │  (Docker)       │    │  (CUDA)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │────│  Health Checks  │────│  Telemetry      │
│   (Prometheus)  │    │  (Custom)       │    │  (OpenTelemetry)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Port Allocation
- **7654**: vLLM HTTPS API endpoint
- **8000**: Internal vLLM HTTP endpoint
- **9054**: Prometheus metrics endpoint
- **3000**: Grafana dashboard
- **16686**: Jaeger tracing UI

## Deployment

### Prerequisites
1. **Hardware Requirements**:
   - NVIDIA GPU with 24GB+ VRAM (A100, H100 recommended)
   - 64GB+ system RAM
   - 500GB+ SSD storage
   - 100Mbps+ network bandwidth

2. **Software Requirements**:
   ```bash
   # NVIDIA drivers
   nvidia-smi  # Should show GPU information
   
   # Docker with NVIDIA support
   docker run --rm --gpus all nvidia/cuda:11.0-runtime-ubuntu18.04 nvidia-smi
   
   # Docker Compose
   docker-compose --version
   ```

### Deployment Steps

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd cirrr
   ```

2. **Configure Environment**:
   ```bash
   # Copy and edit vLLM configuration
   cp vllm-serve-engine-api/.env.example vllm-serve-engine-api/.env
   
   # Edit configuration
   nano vllm-serve-engine-api/.env
   ```

3. **Set Required Variables**:
   ```env
   # Model configuration
   MODEL_NAME=google/gemma-3-27b-it
   TENSOR_PARALLEL_SIZE=4
   GPU_MEMORY_UTILIZATION=0.9
   
   # Security
   API_KEY=your-secure-api-key
   
   # SSL certificates
   SSL_CERT_DIR=/path/to/ssl/certs
   ```

4. **Deploy Services**:
   ```bash
   # Start vLLM service
   cd vllm-serve-engine-api
   docker-compose up -d
   
   # Start monitoring
   cd ../instrumentation
   docker-compose up -d
   ```

5. **Verify Deployment**:
   ```bash
   # Check service health
   curl -k -H "Authorization: Bearer your-api-key" \
        https://localhost:7654/health
   
   # Check metrics
   curl https://localhost:9054/metrics
   ```

### Configuration Management

#### vLLM Server Configuration
Key configuration files:
- `vllm-serve-engine-api/.env`: Environment variables
- `vllm-serve-engine-api/vllm-conf/vllm-server-config.yaml`: Server config
- `conf/vllm-client-conf.env`: Client configuration

#### Critical Configuration Parameters
```yaml
# Model settings
model: google/gemma-3-27b-it
dtype: auto
tensor_parallel_size: 4

# Performance settings
gpu_memory_utilization: 0.9
max_num_seqs: 256
max_model_len: 8192

# API settings
port: 8000
ssl_keyfile: /path/to/key.pem
ssl_certfile: /path/to/cert.pem
```

## Monitoring

### Key Metrics to Monitor

#### Service Health Metrics
- `up{job="vllm"}`: Service availability (0/1)
- `cir3_health_checks_total`: Health check results
- `cir3_circuit_breaker_state`: Circuit breaker state

#### Performance Metrics
- `vllm:e2e_request_latency_seconds`: End-to-end latency
- `cir3_vllm_requests_total`: Request count by status
- `cir3_vllm_tokens_total`: Token processing count
- `cir3_vllm_latency_seconds`: Request latency distribution

#### Resource Metrics
- `container_cpu_usage_seconds_total`: CPU usage
- `container_memory_usage_bytes`: Memory usage
- `nvidia_ml_py_memory_used_bytes`: GPU memory usage
- `nvidia_ml_py_gpu_utilization`: GPU utilization

### Monitoring Dashboards

#### Grafana Dashboards
1. **vLLM Overview**: Service health and performance
2. **Resource Utilization**: CPU, Memory, GPU usage
3. **Request Analytics**: Latency, throughput, errors
4. **Business Metrics**: QA generation stats

#### Alert Rules
Critical alerts are configured for:
- Service downtime (> 1 minute)
- High error rate (> 10%)
- High latency (P95 > 10s)
- Resource exhaustion (Memory > 90%)

## Health Checks

### Health Check Endpoints

#### Basic Health Check
```bash
curl -k -H "Authorization: Bearer $API_KEY" \
     https://localhost:7654/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Detailed Health Check
```bash
curl -k -H "Authorization: Bearer $API_KEY" \
     https://localhost:7654/v1/models
```

#### Application Health Check
```python
from aimw.app.services.vllm.health_check import get_health_checker

async def check_health():
    checker = get_health_checker()
    result = await checker.comprehensive_health_check()
    print(f"Status: {result.status}")
    print(f"Uptime: {result.uptime_seconds}s")
    print(f"Success rate: {result.error_rate_percent}%")
```

### Health Check Automation
Health checks run automatically:
- **Kubernetes liveness probe**: Every 30 seconds
- **Kubernetes readiness probe**: Every 10 seconds
- **Prometheus health checks**: Every 15 seconds
- **Application health monitor**: Continuous

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
**Symptoms**: Container exits immediately
**Causes**:
- GPU not available
- Insufficient memory
- Model download failure
- Configuration errors

**Resolution**:
```bash
# Check GPU availability
nvidia-smi

# Check container logs
docker-compose logs vllm

# Verify configuration
docker-compose config

# Check disk space
df -h

# Check memory
free -h
```

#### 2. High Latency
**Symptoms**: P95 latency > 10 seconds
**Causes**:
- GPU memory pressure
- Large batch sizes
- Model quantization issues
- Network bottlenecks

**Resolution**:
```bash
# Check GPU memory
nvidia-smi

# Review performance metrics
curl https://localhost:9054/metrics | grep vllm

# Adjust configuration
# Reduce gpu_memory_utilization
# Decrease max_num_seqs
# Enable model quantization
```

#### 3. Out of Memory Errors
**Symptoms**: CUDA OOM errors in logs
**Causes**:
- Model too large for GPU
- Excessive concurrent requests
- Memory leaks

**Resolution**:
```bash
# Check GPU memory
nvidia-smi

# Reduce memory usage
export GPU_MEMORY_UTILIZATION=0.8
export MAX_NUM_SEQS=128

# Restart service
docker-compose restart vllm
```

#### 4. Circuit Breaker Open
**Symptoms**: All requests failing with "Circuit breaker is OPEN"
**Causes**:
- Service health issues
- High error rates
- Network connectivity problems

**Resolution**:
```bash
# Check service health
curl -k https://localhost:7654/health

# Review error logs
docker-compose logs vllm | grep -i error

# Check network connectivity
telnet localhost 7654

# Reset circuit breaker (in application)
```

#### 5. SSL Certificate Issues
**Symptoms**: SSL handshake failures
**Causes**:
- Expired certificates
- Incorrect certificate paths
- Permissions issues

**Resolution**:
```bash
# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Verify certificate permissions
ls -la /path/to/ssl/

# Update certificates
cp new-cert.pem /path/to/ssl/
docker-compose restart vllm
```

### Debugging Tools

#### Container Inspection
```bash
# Container status
docker-compose ps

# Container logs
docker-compose logs -f vllm

# Execute into container
docker-compose exec vllm bash

# Container resource usage
docker stats
```

#### Performance Analysis
```bash
# Profile CPU usage
python -m cProfile -o profile.prof your_script.py

# Memory profiling
python -m memory_profiler your_script.py

# GPU profiling
nsys profile python your_script.py
```

#### Network Debugging
```bash
# Check port connectivity
telnet localhost 7654

# Network traffic analysis
tcpdump -i any port 7654

# SSL handshake testing
openssl s_client -connect localhost:7654
```

## Performance Tuning

### GPU Optimization

#### Memory Optimization
```yaml
# Increase memory utilization
gpu_memory_utilization: 0.95

# Enable memory pooling
block_size: 16

# Use appropriate data type
dtype: "float16"  # or "bfloat16"
```

#### Parallelization
```yaml
# Multi-GPU setup
tensor_parallel_size: 4  # Number of GPUs

# Pipeline parallelism (for large models)
pipeline_parallel_size: 2
```

### Request Optimization

#### Batching Configuration
```yaml
# Increase batch size
max_num_seqs: 512

# Continuous batching
max_num_batched_tokens: 8192

# Prefill optimization
max_paddings: 512
```

#### Caching
```yaml
# Enable prefix caching
enable_prefix_caching: true

# Set cache size
max_num_sequences: 1024
```

### Model Optimization

#### Quantization
```yaml
# Enable quantization
quantization: "awq"  # or "gptq", "squeezellm"

# Load in specific format
load_format: "auto"
```

#### Model Selection
- **For latency**: Use smaller models (7B-13B parameters)
- **For throughput**: Use larger models with quantization
- **For memory**: Use quantized models or model sharding

### System Optimization

#### Container Resources
```yaml
deploy:
  resources:
    limits:
      memory: 64G
      cpus: '16'
    reservations:
      memory: 32G
      cpus: '8'
```

#### Storage Optimization
- Use SSD for model storage
- Enable model caching
- Use tmpfs for temporary files

## Incident Response

### Severity Levels

#### P0 - Critical (Service Down)
- **Response Time**: 15 minutes
- **Escalation**: Immediate page to on-call
- **Examples**: Complete service outage, data corruption

#### P1 - High (Degraded Performance)
- **Response Time**: 1 hour
- **Escalation**: Page during business hours
- **Examples**: High latency, elevated error rates

#### P2 - Medium (Minor Issues)
- **Response Time**: 4 hours
- **Escalation**: Email notification
- **Examples**: Non-critical feature issues

#### P3 - Low (Cosmetic Issues)
- **Response Time**: Next business day
- **Escalation**: Ticket creation
- **Examples**: Documentation issues, minor UI problems

### Incident Response Procedures

#### 1. Detection and Alerting
- Monitor alerting channels (PagerDuty, Slack)
- Check monitoring dashboards
- Verify incident scope and impact

#### 2. Initial Response
```bash
# Quick health check
curl -k https://localhost:7654/health

# Check recent logs
docker-compose logs --tail=100 vllm

# Review metrics dashboard
# Check GPU status
nvidia-smi
```

#### 3. Mitigation Steps
```bash
# For service outage
docker-compose restart vllm

# For memory issues
docker-compose down
docker system prune -f
docker-compose up -d

# For configuration issues
# Rollback to last known good configuration
git checkout HEAD~1 -- vllm-serve-engine-api/.env
docker-compose restart vllm
```

#### 4. Communication
- Update incident status page
- Notify stakeholders
- Document timeline and actions

#### 5. Resolution and Post-Mortem
- Verify service restoration
- Document root cause
- Implement preventive measures
- Schedule post-mortem review

### Emergency Contacts
- **On-call Engineer**: +1-xxx-xxx-xxxx
- **Team Lead**: team-lead@company.com
- **DevOps Team**: devops@company.com
- **Escalation Manager**: manager@company.com

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Review monitoring dashboards
- [ ] Check alert status
- [ ] Monitor resource usage
- [ ] Review error logs

#### Weekly
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Check certificate expiration
- [ ] Backup configuration

#### Monthly
- [ ] Model updates and testing
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Performance tuning

#### Quarterly
- [ ] Infrastructure review
- [ ] Disaster recovery testing
- [ ] Documentation updates
- [ ] Team training

### Backup and Recovery

#### Configuration Backup
```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz \
    vllm-serve-engine-api/.env \
    vllm-serve-engine-api/vllm-conf/ \
    conf/

# Store in secure location
aws s3 cp config-backup-*.tar.gz s3://backup-bucket/
```

#### Model Backup
```bash
# Backup model cache
tar -czf model-cache-$(date +%Y%m%d).tar.gz ~/.cache/huggingface/

# Store model backup
aws s3 cp model-cache-*.tar.gz s3://model-backup-bucket/
```

#### Recovery Procedures
```bash
# Restore configuration
tar -xzf config-backup-YYYYMMDD.tar.gz

# Restore model cache
tar -xzf model-cache-YYYYMMDD.tar.gz -C ~/

# Restart services
docker-compose down && docker-compose up -d
```

### Security Maintenance

#### Certificate Rotation
```bash
# Generate new certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Update configuration
cp cert.pem /path/to/ssl/
cp key.pem /path/to/ssl/

# Restart service
docker-compose restart vllm
```

#### API Key Rotation
```bash
# Generate new API key
export NEW_API_KEY=$(openssl rand -hex 32)

# Update configuration
sed -i "s/API_KEY=.*/API_KEY=$NEW_API_KEY/" .env

# Restart service
docker-compose restart vllm

# Update client configurations
```

### Performance Monitoring

#### Set Up Continuous Monitoring
```bash
# Enable performance profiling
python -m aimw.app.services.observability.profiler \
    --profile-type full \
    --duration 3600 \
    --output-dir /var/log/performance/

# Schedule regular benchmarks
crontab -e
# Add: 0 2 * * * /path/to/benchmark-script.sh
```

#### Capacity Planning
- Monitor growth trends
- Forecast resource needs
- Plan infrastructure scaling
- Review cost optimization

This runbook should be updated regularly and kept accessible to all team members responsible for vLLM operations. 