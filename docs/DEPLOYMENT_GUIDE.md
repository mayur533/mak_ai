# AI Assistant System - Deployment Guide

## Overview

This guide covers deploying the AI Assistant System in various environments, from local development to production deployments.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 1GB free space
- **Network**: Internet connection for API calls

### Required Dependencies

- Google Gemini API key(s)
- Virtual environment support
- Git (for version control)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-assistant-system.git
cd ai-assistant-system
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies (Linux)
sudo apt update
sudo apt install -y xdotool scrot

# Install system dependencies (macOS)
brew install xdotool

# Install system dependencies (Windows)
# Download and install required tools manually
```

### 4. Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_primary_api_key_here
GEMINI_API_KEY_SECONDARY=your_secondary_api_key_here
GEMINI_MODEL=gemini-2.5-flash
VOICE_ENABLED=false
DEBUG_MODE=false
LOG_LEVEL=INFO
MAX_RETRIES=3
DB_PATH=db/agent_memory.db
```

### 5. Run the System

```bash
# Basic run
python main.py

# With voice mode
python main.py --voice

# With debug mode
python main.py --debug

# With custom config
python main.py --config custom.env
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    xdotool \
    scrot \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY .env .

# Create non-root user
RUN useradd -m -u 1000 ai_user && \
    chown -R ai_user:ai_user /app
USER ai_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "main.py"]
```

### 2. Build and Run

```bash
# Build Docker image
docker build -t ai-assistant-system .

# Run container
docker run -d \
    --name ai-assistant \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -e GEMINI_API_KEY=your_api_key \
    ai-assistant-system

# View logs
docker logs -f ai-assistant
```

### 3. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ai-assistant:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_API_KEY_SECONDARY=${GEMINI_API_KEY_SECONDARY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  grafana-storage:
```

Run with Docker Compose:

```bash
docker-compose up -d
```

## Kubernetes Deployment

### 1. Namespace

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-assistant
```

### 2. ConfigMap

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-assistant-config
  namespace: ai-assistant
data:
  GEMINI_MODEL: "gemini-2.5-flash"
  VOICE_ENABLED: "false"
  DEBUG_MODE: "false"
  LOG_LEVEL: "INFO"
  MAX_RETRIES: "3"
  DB_PATH: "db/agent_memory.db"
```

### 3. Secret

Create `k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-assistant-secrets
  namespace: ai-assistant
type: Opaque
data:
  GEMINI_API_KEY: <base64-encoded-primary-key>
  GEMINI_API_KEY_SECONDARY: <base64-encoded-secondary-key>
```

### 4. Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-assistant
  namespace: ai-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-assistant
  template:
    metadata:
      labels:
        app: ai-assistant
    spec:
      containers:
      - name: ai-assistant
        image: ai-assistant-system:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ai-assistant-config
        - secretRef:
            name: ai-assistant-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: ai-assistant-data
```

### 5. Service

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-assistant-service
  namespace: ai-assistant
spec:
  selector:
    app: ai-assistant
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 6. Persistent Volume Claim

Create `k8s/pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ai-assistant-data
  namespace: ai-assistant
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### 7. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ai-assistant

# View logs
kubectl logs -f deployment/ai-assistant -n ai-assistant

# Get service URL
kubectl get service ai-assistant-service -n ai-assistant
```

## Production Deployment

### 1. Environment Setup

```bash
# Create production environment
python3 -m venv prod_env
source prod_env/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Set production environment variables
export ENV=production
export DEBUG_MODE=false
export LOG_LEVEL=WARNING
```

### 2. Process Management with Systemd

Create `/etc/systemd/system/ai-assistant.service`:

```ini
[Unit]
Description=AI Assistant System
After=network.target

[Service]
Type=simple
User=ai-assistant
Group=ai-assistant
WorkingDirectory=/opt/ai-assistant-system
Environment=PATH=/opt/ai-assistant-system/prod_env/bin
ExecStart=/opt/ai-assistant-system/prod_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable ai-assistant
sudo systemctl start ai-assistant
sudo systemctl status ai-assistant
```

### 3. Reverse Proxy with Nginx

Create `/etc/nginx/sites-available/ai-assistant`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/ai-assistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Observability

### 1. Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-assistant'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 5s
```

### 2. Grafana Dashboard

Import the provided dashboard configuration from `monitoring/grafana-dashboard.json`.

### 3. Health Checks

The system provides health check endpoints:

- `/health` - Basic health status
- `/ready` - Readiness check
- `/metrics` - Prometheus metrics

### 4. Log Management

Configure log rotation in `/etc/logrotate.d/ai-assistant`:

```
/opt/ai-assistant-system/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ai-assistant ai-assistant
    postrotate
        systemctl reload ai-assistant
    endscript
}
```

## Security Considerations

### 1. API Key Management

- Store API keys in environment variables or secrets
- Use different keys for different environments
- Rotate keys regularly
- Monitor API key usage

### 2. Network Security

- Use HTTPS in production
- Implement rate limiting
- Use firewall rules to restrict access
- Consider VPN for internal deployments

### 3. Access Control

- Run services with minimal privileges
- Use non-root users
- Implement authentication if needed
- Monitor access logs

### 4. Data Protection

- Encrypt sensitive data at rest
- Use secure database connections
- Implement backup strategies
- Follow data retention policies

## Backup and Recovery

### 1. Database Backup

```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/ai-assistant"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
cp /opt/ai-assistant-system/db/agent_memory.db $BACKUP_DIR/agent_memory_$DATE.db

# Backup configuration
cp /opt/ai-assistant-system/.env $BACKUP_DIR/env_$DATE

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "env_*" -mtime +30 -delete
```

### 2. Automated Backups

```bash
# Add to crontab
crontab -e
# Add: 0 2 * * * /opt/ai-assistant-system/scripts/backup.sh
```

### 3. Recovery Procedure

```bash
# Stop the service
sudo systemctl stop ai-assistant

# Restore database
cp /opt/backups/ai-assistant/agent_memory_YYYYMMDD_HHMMSS.db /opt/ai-assistant-system/db/agent_memory.db

# Restore configuration
cp /opt/backups/ai-assistant/env_YYYYMMDD_HHMMSS /opt/ai-assistant-system/.env

# Start the service
sudo systemctl start ai-assistant
```

## Performance Tuning

### 1. System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "vm.max_map_count=262144" >> /etc/sysctl.conf
echo "fs.file-max=65536" >> /etc/sysctl.conf
sysctl -p
```

### 2. Application Tuning

- Adjust thread pool sizes
- Configure connection pooling
- Optimize database queries
- Implement caching strategies

### 3. Monitoring Performance

- Set up performance alerts
- Monitor resource usage
- Track response times
- Analyze error rates

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check logs: `journalctl -u ai-assistant -f`
   - Verify configuration
   - Check dependencies

2. **High memory usage**
   - Monitor context size
   - Check for memory leaks
   - Adjust limits

3. **API rate limiting**
   - Check API key usage
   - Implement backoff strategies
   - Use secondary keys

4. **Database issues**
   - Check disk space
   - Verify permissions
   - Check database integrity

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG_MODE=true
export LOG_LEVEL=DEBUG
python main.py
```

### Log Analysis

```bash
# View recent errors
grep "ERROR" /opt/ai-assistant-system/logs/app.log | tail -20

# Monitor real-time logs
tail -f /opt/ai-assistant-system/logs/app.log

# Analyze performance
grep "duration" /opt/ai-assistant-system/logs/app.log | awk '{print $NF}' | sort -n
```

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review logs for errors
   - Check disk space
   - Update dependencies

2. **Monthly**
   - Rotate logs
   - Backup data
   - Review performance metrics

3. **Quarterly**
   - Update system packages
   - Review security patches
   - Plan capacity upgrades

### Updates

```bash
# Update application
cd /opt/ai-assistant-system
git pull origin main
pip install -r requirements.txt --upgrade
sudo systemctl restart ai-assistant

# Update system packages
sudo apt update && sudo apt upgrade
```

## Support

For deployment issues:

1. Check the logs first
2. Verify configuration
3. Test in development environment
4. Review documentation
5. Contact support team

## Additional Resources

- [API Reference](API_REFERENCE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Security Best Practices](SECURITY.md)


