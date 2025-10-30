# GMCS Deployment Guide

Complete guide for deploying GMCS to production.

## Quick Start

```bash
# Clone repository
git clone https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate
cd Chaotic-Neuro-Computational-Substrate

# Install dependencies
pip install -r requirements.txt

# Run backend
python src/main.py

# Run frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Production Deployment

### Docker Deployment

**Build**:
```bash
docker-compose build
```

**Run**:
```bash
docker-compose up -d
```

**Stop**:
```bash
docker-compose down
```

### Manual Deployment

#### Backend
```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Run with Gunicorn
gunicorn src.api.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

#### Frontend
```bash
cd frontend
npm install
npm run build
npm start
```

## Environment Configuration

### Backend (.env)
```bash
# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# JAX
JAX_PLATFORM_NAME=gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false

# THRML
THRML_MODE=accuracy
THRML_TEMPERATURE=1.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/gmcs.log

# Security
API_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- GPU: NVIDIA GTX 1060 or equivalent
- Storage: 10 GB
- OS: Linux, macOS, Windows

### Recommended
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA RTX 3080 or better
- Storage: 50+ GB SSD
- OS: Linux (Ubuntu 20.04+)

## GPU Setup

### NVIDIA CUDA
```bash
# Install CUDA Toolkit 11.8+
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install cuDNN
sudo apt-get install libcudnn8
```

### AMD ROCm
```bash
# Install ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
```

### Apple Silicon
JAX automatically uses Metal backend on M1/M2/M3 Macs.

## Performance Tuning

### JAX Configuration
```python
# In src/config/defaults.py
JAX_CONFIG = {
    "jax_enable_x64": False,  # Use float32 for speed
    "jax_platform_name": "gpu",
    "xla_python_client_preallocate": False,
    "xla_python_client_mem_fraction": 0.75
}
```

### Backend Optimization
```python
# Increase worker count
WORKERS = os.cpu_count()

# Enable caching
CACHE_ENABLED = True
CACHE_SIZE = 1000

# Batch size
BATCH_SIZE = 32
```

### Frontend Optimization
```javascript
// next.config.js
module.exports = {
  reactStrictMode: true,
  swcMinify: true,
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production'
  },
  images: {
    unoptimized: false
  }
}
```

## Monitoring

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Expected response
{"status": "healthy", "timestamp": "..."}
```

### Logging
```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gmcs.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

request_count = Counter('gmcs_requests_total', 'Total requests')
request_duration = Histogram('gmcs_request_duration_seconds', 'Request duration')
```

## Security

### API Authentication
```python
# Add API key middleware
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

### CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### HTTPS/TLS
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Run with HTTPS
uvicorn src.api.server:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
version: '3.8'
services:
  gmcs-backend:
    image: gmcs:latest
    deploy:
      replicas: 4
    ports:
      - "8000-8003:8000"
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancing
```nginx
# nginx.conf
upstream gmcs_backend {
    least_conn;
    server gmcs-backend-1:8000;
    server gmcs-backend-2:8000;
    server gmcs-backend-3:8000;
    server gmcs-backend-4:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://gmcs_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws {
        proxy_pass http://gmcs_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Backup & Recovery

### Database Backup
```bash
# Backup configurations
tar -czf backup_$(date +%Y%m%d).tar.gz \
  saved_configs/ \
  saved_sessions/ \
  model_registry/

# Restore
tar -xzf backup_20250101.tar.gz
```

### State Persistence
```python
# Auto-save every 5 minutes
import schedule

def backup_state():
    # Save current state
    with open('backup/state.pkl', 'wb') as f:
        pickle.dump(sim_state, f)

schedule.every(5).minutes.do(backup_state)
```

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
nvidia-smi

# Check JAX
python -c "import jax; print(jax.devices())"

# Set environment
export CUDA_VISIBLE_DEVICES=0
```

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16

# Enable memory growth
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
```

### Slow Performance
```bash
# Profile
python -m cProfile -o profile.stats src/main.py

# Analyze
python -m pstats profile.stats
```

## CI/CD

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy GMCS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t gmcs:latest .
      
      - name: Run tests
        run: docker run gmcs:latest pytest
      
      - name: Deploy
        run: |
          docker-compose down
          docker-compose up -d
```

## Monitoring & Alerts

### Grafana Dashboard
```yaml
# docker-compose.yml
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - ./grafana:/etc/grafana/provisioning
```

### Alert Rules
```yaml
# alerts.yml
groups:
  - name: gmcs
    rules:
      - alert: HighErrorRate
        expr: rate(gmcs_errors_total[5m]) > 0.05
        annotations:
          summary: "High error rate detected"
```

## See Also

- [Architecture Documentation](architecture.md)
- [API Reference](API_REFERENCE.md)
- [Security Policy](../SECURITY.md)

