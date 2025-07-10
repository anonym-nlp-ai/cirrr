# Monitoring Setup Guide for CIR3 and vLLM

## Overview
This guide provides step-by-step instructions for setting up comprehensive monitoring for CIR3 and vLLM in production environments.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Component Setup](#component-setup)
4. [Configuration](#configuration)
5. [Dashboard Installation](#dashboard-installation)
6. [Alerting Setup](#alerting-setup)
7. [Troubleshooting](#troubleshooting)
8. [Production Considerations](#production-considerations)

## Architecture Overview

### Monitoring Stack
```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│    CIR3 App     │   vLLM Service  │      Health Checks          │
│  (FastAPI)      │   (Docker)      │    (Custom Probes)          │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                 │                       │
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Telemetry Collection                           │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  OpenTelemetry  │   Prometheus    │       Custom Metrics        │
│   (Tracing)     │   (Metrics)     │      (Application)          │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                 │                       │
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Storage & Analysis                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│     Jaeger      │   Prometheus    │        Grafana              │
│   (Traces)      │   (TSDB)        │     (Visualization)         │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                 │                       │
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Alerting                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Alertmanager   │    PagerDuty    │        Slack/Email          │
│   (Routing)     │  (Escalation)   │     (Notifications)         │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Data Flow
1. **Application Metrics**: Applications expose metrics via `/metrics` endpoints
2. **Traces**: OpenTelemetry collects and exports traces to Jaeger
3. **Collection**: Prometheus scrapes metrics from all services
4. **Storage**: Time-series data stored in Prometheus TSDB
5. **Visualization**: Grafana queries Prometheus and displays dashboards
6. **Alerting**: Prometheus evaluates rules and sends alerts to Alertmanager

## Prerequisites

### System Requirements
- **RAM**: 8GB+ for monitoring stack
- **CPU**: 4+ cores
- **Storage**: 100GB+ SSD for metrics storage
- **Network**: Reliable connectivity to all monitored services

### Software Dependencies
```bash
# Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
pip install docker-compose

# SSL certificates (for production)
openssl req -x509 -newkey rsa:4096 -keyout monitoring.key -out monitoring.crt -days 365 -nodes
```

### Environment Preparation
```bash
# Create monitoring directories
mkdir -p /opt/monitoring/{prometheus,grafana,jaeger,alertmanager}
mkdir -p /opt/monitoring/ssl
mkdir -p /var/log/monitoring

# Set permissions
chown -R 472:472 /opt/monitoring/grafana  # Grafana user
chown -R 65534:65534 /opt/monitoring/prometheus  # Nobody user
```

## Component Setup

### 1. Prometheus Setup

#### Create Configuration Directory
```bash
mkdir -p /opt/monitoring/prometheus/{data,config,rules}
```

#### Prometheus Configuration (`/opt/monitoring/prometheus/config/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'cir3-production'
    environment: 'production'

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # vLLM Service
  - job_name: 'vllm'
    scheme: https
    static_configs:
      - targets: ['vllm:7654']
    metrics_path: /metrics
    tls_config:
      insecure_skip_verify: true
    scrape_interval: 10s
    bearer_token: 'your-vllm-api-key'

  # CIR3 Application
  - job_name: 'cir3-app'
    static_configs:
      - targets: ['cir3-app:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor (container metrics)
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # GPU metrics (if available)
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-exporter:9445']
```

#### Alert Rules (`/opt/monitoring/prometheus/rules/alerts.yml`)
```bash
# Copy the comprehensive alert rules we created earlier
cp instrumentation/prometheus/alerts.yml /opt/monitoring/prometheus/rules/
```

### 2. Grafana Setup

#### Create Configuration
```bash
mkdir -p /opt/monitoring/grafana/{data,provisioning/{dashboards,datasources,plugins}}
```

#### Datasource Configuration (`/opt/monitoring/grafana/provisioning/datasources/prometheus.yml`)
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
```

#### Dashboard Provisioning (`/opt/monitoring/grafana/provisioning/dashboards/dashboard.yml`)
```yaml
apiVersion: 1

providers:
  - name: 'CIR3 Dashboards'
    orgId: 1
    folder: 'CIR3'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

### 3. Jaeger Setup

#### Jaeger Configuration (`/opt/monitoring/jaeger/jaeger.yml`)
```yaml
collector:
  zipkin:
    host-port: ":9411"

storage:
  type: elasticsearch
  elasticsearch:
    server-urls: http://elasticsearch:9200
    num-shards: 1
    num-replicas: 0
```

### 4. Alertmanager Setup

#### Alertmanager Configuration (`/opt/monitoring/alertmanager/config.yml`)
```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourcompany.com'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'

  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'CIR3 Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-warning'
        title: 'CIR3 Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Configuration

### Docker Compose Setup

#### Main Monitoring Stack (`/opt/monitoring/docker-compose.yml`)
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=90d'
      - '--storage.tsdb.retention.size=50GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--log.level=info'
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/config:/etc/prometheus
      - ./prometheus/rules:/etc/prometheus/rules
      - ./prometheus/data:/prometheus
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  grafana:
    image: grafana/grafana:10.0.0
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:1.47
    container_name: jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector HTTP
      - "14250:14250"  # Jaeger collector gRPC
      - "9411:9411"    # Zipkin
    networks:
      - monitoring
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: alertmanager
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager:/etc/alertmanager
    networks:
      - monitoring
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: node-exporter
    command:
      - '--path.rootfs=/host'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    volumes:
      - '/:/host:ro,rslave'
    networks:
      - monitoring
    restart: unless-stopped

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - monitoring
    restart: unless-stopped

networks:
  monitoring:
    driver: bridge
    external: true

volumes:
  prometheus_data:
  grafana_data:
```

### Application Integration

#### CIR3 Application Configuration
Add to your FastAPI application:

```python
# main.py
from aimw.app.services.observability.telemetry import initialize_telemetry, TelemetryConfig
from prometheus_client import make_asgi_app, generate_latest

# Initialize telemetry
telemetry_config = TelemetryConfig(
    service_name="cir3-app",
    service_version="0.1.15",
    environment="production",
    jaeger_endpoint="http://jaeger:14268/api/traces",
    prometheus_port=8000,
    enable_tracing=True,
    enable_metrics=True
)
initialize_telemetry(telemetry_config)

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

#### vLLM Integration
Update vLLM client usage:

```python
from aimw.app.services.vllm.resilient_client import create_resilient_client
from aimw.app.services.observability.telemetry import trace_vllm_request

@trace_vllm_request("google/gemma-3-27b-it")
async def generate_response(messages):
    client = create_resilient_client()
    return await client.chat_completion(messages)
```

## Dashboard Installation

### Import Pre-built Dashboards

#### 1. Copy Dashboard Files
```bash
# Copy the vLLM Grafana dashboard we created
cp instrumentation/grafana/vllm-grafana.json /opt/monitoring/grafana/dashboards/

# Download additional dashboards
curl -o /opt/monitoring/grafana/dashboards/node-exporter.json \
     https://grafana.com/api/dashboards/1860/revisions/27/download

curl -o /opt/monitoring/grafana/dashboards/docker-containers.json \
     https://grafana.com/api/dashboards/193/revisions/4/download
```

#### 2. Custom CIR3 Dashboard
Create `/opt/monitoring/grafana/dashboards/cir3-overview.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "CIR3 Overview",
    "tags": ["cir3", "application"],
    "panels": [
      {
        "title": "QA Generation Rate",
        "type": "stat",
        "targets": [{
          "expr": "rate(cir3_qa_generation_total[5m])",
          "legendFormat": "QA/sec"
        }]
      },
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(cir3_request_duration_seconds_bucket[5m]))",
          "legendFormat": "P95"
        }]
      }
    ]
  }
}
```

## Alerting Setup

### 1. Configure Notification Channels

#### Slack Integration
```bash
# Create Slack webhook
# Go to Slack > Apps > Incoming Webhooks
# Copy webhook URL to alertmanager config
```

#### PagerDuty Integration
```bash
# Create PagerDuty integration
# Go to PagerDuty > Integrations > Prometheus
# Copy integration key to alertmanager config
```

### 2. Test Alerting
```bash
# Test alert rule
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert for monitoring setup"
    }
  }]'
```

## Deployment

### 1. Start Monitoring Stack
```bash
# Create network
docker network create monitoring

# Start services
cd /opt/monitoring
docker-compose up -d

# Verify services
docker-compose ps
```

### 2. Verify Setup
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana
curl http://localhost:3000/api/health

# Check Jaeger
curl http://localhost:16686/api/services

# Check Alertmanager
curl http://localhost:9093/api/v1/status
```

### 3. Configure Dashboards
1. Open Grafana: http://localhost:3000
2. Login (admin/admin123)
3. Import dashboards from `/var/lib/grafana/dashboards`
4. Configure data sources if needed

## Troubleshooting

### Common Issues

#### 1. Prometheus Not Scraping Targets
```bash
# Check target status
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "down")'

# Check connectivity
docker exec prometheus wget -qO- http://target:port/metrics

# Check configuration
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

#### 2. Grafana Dashboards Not Loading
```bash
# Check provisioning logs
docker logs grafana

# Verify dashboard files
ls -la /opt/monitoring/grafana/dashboards/

# Test datasource
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up
```

#### 3. Alerts Not Firing
```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Check alertmanager config
docker exec alertmanager amtool config show

# Test alert routing
docker exec alertmanager amtool config routes test
```

### Logs and Debugging
```bash
# View logs
docker-compose logs -f prometheus
docker-compose logs -f grafana
docker-compose logs -f alertmanager

# Check metrics exposure
curl http://localhost:8000/metrics  # CIR3 app
curl -k https://localhost:7654/metrics  # vLLM

# Validate configurations
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
docker exec alertmanager amtool check-config /etc/alertmanager/config.yml
```

## Production Considerations

### Security
- Use SSL/TLS for all communications
- Configure authentication for Grafana
- Restrict network access with firewalls
- Use secrets management for API keys

### High Availability
- Deploy Prometheus in HA mode
- Use external storage for long-term retention
- Set up Grafana clustering
- Configure Alertmanager clustering

### Performance
- Tune retention policies
- Configure recording rules for complex queries
- Use remote storage for scale
- Monitor monitoring system resource usage

### Backup and Recovery
```bash
# Backup Prometheus data
tar -czf prometheus-backup-$(date +%Y%m%d).tar.gz /opt/monitoring/prometheus/data

# Backup Grafana
tar -czf grafana-backup-$(date +%Y%m%d).tar.gz /opt/monitoring/grafana

# Automated backup script
cat > /opt/monitoring/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/monitoring"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Stop services briefly for consistent backup
docker-compose stop prometheus grafana

# Backup data
tar -czf $BACKUP_DIR/prometheus-$DATE.tar.gz /opt/monitoring/prometheus/data
tar -czf $BACKUP_DIR/grafana-$DATE.tar.gz /opt/monitoring/grafana/data

# Restart services
docker-compose start prometheus grafana

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x /opt/monitoring/backup.sh

# Schedule backup
echo "0 2 * * * /opt/monitoring/backup.sh" | crontab -
```

### Monitoring the Monitoring
- Set up alerts for monitoring system health
- Monitor disk usage for metrics storage
- Track query performance in Grafana
- Monitor Prometheus memory usage

This completes the comprehensive monitoring setup for CIR3 and vLLM. Regular maintenance and monitoring of the monitoring system itself is crucial for production reliability. 