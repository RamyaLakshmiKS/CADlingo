# CADlingo Production Deployment Guide

## Overview

This guide covers production deployment, API integration, quality monitoring, and scaling considerations for CADlingo.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [API Integration](#api-integration)
3. [Quality Monitoring](#quality-monitoring)
4. [Performance Optimization](#performance-optimization)
5. [Scaling Considerations](#scaling-considerations)
6. [Security Best Practices](#security-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Docker Deployment

### Quick Start

```bash
# Clone and navigate
git clone https://github.com/RamyaLakshmiKS/CADlingo.git
cd CADlingo

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cadlingo-ui
docker-compose logs -f cadlingo-api
```

### Services

| Service | Container | Port | URL |
|---------|-----------|------|-----|
| **Streamlit UI** | cadlingo-ui | 8501 | http://localhost:8501 |
| **FastAPI Backend** | cadlingo-api | 8000 | http://localhost:8000/docs |

### Environment Variables

Create `.env` file for configuration:

```bash
# .env
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
API_HOST=0.0.0.0
API_PORT=8000
MAX_BATCH_SIZE=50
MODEL_PATH=/app/results/models/best_model
```

### Volume Mounting

Persistent data storage:

```yaml
volumes:
  - ./data:/app/data              # Training data
  - ./results:/app/results        # Models and outputs
```

### Health Checks

Both services include automated health checks:

```bash
# Check UI health
curl http://localhost:8501

# Check API health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Updating Deployment

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose down
docker-compose up -d --build

# Verify services
docker-compose ps
```

---

## API Integration

### Authentication (Future)

Currently open, but can add API key authentication:

```python
# Future implementation
headers = {
    "X-API-Key": "your-api-key-here"
}
```

### Endpoint Examples

#### 1. Basic Generation

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "description": "A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room",
        "num_beams": 4,
        "temperature": 1.0
    }
)

result = response.json()
print(result['code'])
```

**Response:**
```json
{
  "success": true,
  "code": "LAYER \"walls\" continuous\nRECT 0.0 0.0 12.0 10.0 \"bedroom\"...",
  "description": "A 1200 sq ft floor plan...",
  "generation_time": 1.45,
  "metadata": {
    "room_count": 4,
    "code_length": 523,
    "num_beams": 4,
    "temperature": 1.0
  }
}
```

#### 2. Generation with Validation

```python
response = requests.post(
    "http://localhost:8000/generate-with-validation",
    json={
        "description": "A 1500 sq ft home with 3 bedrooms, kitchen, and 2 bathrooms",
        "num_beams": 6
    }
)

result = response.json()
print(f"Quality Score: {result['quality_score']}%")
print(f"Validation: {result['validation_report']}")
```

**Response:**
```json
{
  "success": true,
  "code": "...",
  "validation_report": {
    "is_valid": true,
    "overall_score": 92.5,
    "overlap_count": 0,
    "valid_rooms": 6,
    "total_rooms": 6,
    "issues": []
  },
  "quality_score": 92.5,
  "generation_time": 2.1
}
```

#### 3. Download DXF File

```python
response = requests.post(
    "http://localhost:8000/generate-dxf",
    json={"description": "Small apartment with bedroom, kitchen, bathroom"}
)

# Save DXF file
with open("floor_plan.dxf", "wb") as f:
    f.write(response.content)
```

#### 4. Batch Processing

```python
descriptions = [
    "Studio apartment with kitchen and bathroom",
    "2-bedroom family home with living room",
    "3-bedroom house with 2 bathrooms and kitchen"
]

response = requests.post(
    "http://localhost:8000/batch-generate",
    json=descriptions,
    params={"num_beams": 4}
)

results = response.json()
print(f"Processed {results['total']} floor plans")

for result in results['results']:
    if result['success']:
        print(f"✓ {result['description'][:50]}...")
    else:
        print(f"✗ Error: {result['error']}")
```

---

## Quality Monitoring

### Automated Evaluation

CADlingo includes comprehensive quality metrics beyond BLEU:

```python
from src.improvements.improvement_modules import AutomatedEvaluator

evaluator = AutomatedEvaluator()

# Evaluate single prediction
evaluation = evaluator.evaluate_prediction(
    predicted_code=generated_code,
    reference_code=ground_truth_code,
    predicted_rooms=parsed_predicted_rooms,
    reference_rooms=parsed_reference_rooms
)

print(evaluation['scores']['combined_score'])
```

**Metrics Tracked:**

| Metric | Description | Target |
|--------|-------------|--------|
| IoU | Layout overlap accuracy | >0.90 |
| Room Count Accuracy | Predicted vs actual rooms | >0.95 |
| Room Type Accuracy | Correct room classifications | >0.92 |
| Adjacency Accuracy | Room relationship preservation | >0.85 |
| Plausibility Score | Architectural feasibility | >90% |
| Validation Score | Geometric validity | >80% |

### Logging and Monitoring

```python
# Enable detailed logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cadlingo.log'),
        logging.StreamHandler()
    ]
)
```

### Quality Dashboard

Monitor metrics in real-time using the Streamlit UI:

- **Overall Quality Score**: Combined geometric + validation score
- **Room Count**: Predicted vs expected
- **Layout Check**: Overlap detection status
- **Issue List**: Specific validation problems

---

## Performance Optimization

### Model Loading

Models are cached after first load:

```python
# Automatic caching in API
@app.on_event("startup")
async def startup_event():
    global generator
    generator = CADGenerator()  # Loaded once
```

### Batch Processing

For high-throughput scenarios:

```python
# Process multiple requests efficiently
results = []
for batch in chunks(descriptions, batch_size=10):
    batch_results = generator.batch_generate(batch)
    results.extend(batch_results)
```

### GPU Acceleration

If GPU is available:

```python
# Automatically uses GPU if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Performance Benchmarks:**

| Hardware | Generation Time | Batch Throughput |
|----------|----------------|------------------|
| CPU (Intel i7) | 2-3 sec/plan | ~20 plans/min |
| GPU (NVIDIA T4) | 0.5-1 sec/plan | ~60 plans/min |
| GPU (NVIDIA A100) | 0.2-0.5 sec/plan | ~120 plans/min |

---

## Scaling Considerations

### Horizontal Scaling

Use load balancer with multiple API instances:

```yaml
# docker-compose.scale.yml
services:
  cadlingo-api:
    deploy:
      replicas: 3
```

```bash
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### Kubernetes Deployment

For production-grade orchestration:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cadlingo-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cadlingo-api
  template:
    metadata:
      labels:
        app: cadlingo-api
    spec:
      containers:
      - name: api
        image: cadlingo:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Database Integration (Future)

Store generated plans for analytics:

```python
# Pseudo-code for future implementation
from sqlalchemy import create_engine
from models import FloorPlan, Generation

# Log each generation
generation = Generation(
    description=description,
    code=generated_code,
    quality_score=validation_score,
    timestamp=datetime.now()
)
db.session.add(generation)
db.session.commit()
```

---

## Security Best Practices

### API Rate Limiting

```python
# Future: Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_floor_plan(request: GenerationRequest):
    ...
```

### Input Validation

Already implemented:

```python
class GenerationRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=500)
    num_beams: int = Field(default=4, ge=1, le=8)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
```

### HTTPS in Production

Use reverse proxy (Nginx) for HTTPS:

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name cadlingo.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Troubleshooting

### Common Issues

#### Container won't start

```bash
# Check logs
docker-compose logs cadlingo-api

# Common causes:
# - Port already in use
# - Insufficient memory
# - Model files missing

# Solutions:
docker-compose down
docker system prune -a
docker-compose up -d --build
```

#### Model loading fails

```bash
# Verify model exists
ls -lh results/models/best_model/

# Re-train if missing
python src/models/train.py
```

#### High memory usage

```bash
# Monitor resource usage
docker stats

# Reduce batch size if needed
# Edit ui/api.py:
MAX_BATCH_SIZE = 25  # Instead of 50
```

#### Slow generation

- Check GPU availability: `nvidia-smi`
- Reduce `num_beams` parameter
- Use batch processing for multiple plans
- Consider model optimization (quantization)

### Health Check Failures

```bash
# API health check
curl -f http://localhost:8000/health || echo "API unhealthy"

# If fails:
docker-compose restart cadlingo-api
docker-compose logs -f cadlingo-api
```

---

## Support and Contributions

For production deployment assistance:
- **Issues**: https://github.com/RamyaLakshmiKS/CADlingo/issues
- **Email**: ramyalakshmi.ks@gmail.com
- **Documentation**: `/docs` directory

### Contributing

Contributions welcome! Focus areas:
- Performance optimization
- Additional validation rules
- Enhanced UI features
- Dataset augmentation
- Multi-language support

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Status**: Production-Ready Prototype
