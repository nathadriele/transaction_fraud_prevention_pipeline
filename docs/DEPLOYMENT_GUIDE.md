# Guia de Deploy - Sistema de Prevenção de Fraudes

## Visão Geral

Este guia documenta o processo completo de deploy do Sistema de Prevenção de Fraudes em diferentes ambientes, desde desenvolvimento local até produção em cloud.

## Arquiteturas de Deploy

### 1. **Deploy Local (Desenvolvimento)**

#### Pré-requisitos
- Python 3.8+
- pip ou conda
- Git

#### Instalação Rápida
```bash
# Clone o repositório
git clone <repository-url>
cd transacional_fraud_prevention_pipeline

# Instale dependências
pip install -r requirements.txt

# Execute o dashboard
streamlit run src/dashboard/app.py
```

#### Scripts de Inicialização
```bash
# Windows
start_dashboard.bat

# Linux/Mac
./start_dashboard.sh

# Python (multiplataforma)
python start_dashboard.py
```

### 2. **Deploy com Docker**

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte
COPY . .

# Expõe porta do Streamlit
EXPOSE 8501

# Comando de inicialização
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  fraud-dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app/src
    restart: unless-stopped
    
  # Opcional: Banco de dados
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: fraud_detection
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

#### Comandos Docker
```bash
# Build da imagem
docker build -t fraud-dashboard .

# Executar container
docker run -p 8501:8501 fraud-dashboard

# Com Docker Compose
docker-compose up -d
```

### 3. **Deploy em Cloud**

#### AWS (Amazon Web Services)

##### EC2 + Docker
```bash
# 1. Criar instância EC2 (Ubuntu 20.04)
# 2. Instalar Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

# 3. Deploy da aplicação
git clone <repository-url>
cd transacional_fraud_prevention_pipeline
docker-compose up -d

# 4. Configurar Security Group (porta 8501)
```

##### ECS (Elastic Container Service)
```json
{
  "family": "fraud-dashboard",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fraud-dashboard",
      "image": "your-account.dkr.ecr.region.amazonaws.com/fraud-dashboard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fraud-dashboard",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Platform (GCP)

##### Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/fraud-dashboard', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/fraud-dashboard']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'fraud-dashboard',
      '--image', 'gcr.io/$PROJECT_ID/fraud-dashboard',
      '--platform', 'managed',
      '--region', 'us-central1',
      '--allow-unauthenticated'
    ]
```

##### Comandos GCP
```bash
# Build e deploy
gcloud builds submit --config cloudbuild.yaml

# Deploy direto
gcloud run deploy fraud-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure

##### Container Instances
```bash
# Criar resource group
az group create --name fraud-detection-rg --location eastus

# Deploy container
az container create \
  --resource-group fraud-detection-rg \
  --name fraud-dashboard \
  --image your-registry/fraud-dashboard:latest \
  --dns-name-label fraud-dashboard-unique \
  --ports 8501
```

### 4. **Deploy com Streamlit Cloud**

#### Configuração
1. Fork o repositório no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione o repositório
5. Configure:
   - **Main file path**: `src/dashboard/app.py`
   - **Python version**: 3.9
   - **Requirements file**: `requirements.txt`

#### secrets.toml (para configurações sensíveis)
```toml
[database]
host = "your-db-host"
username = "your-username"
password = "your-password"

[api]
fraud_detection_key = "your-api-key"
```

## Configurações de Produção

### 1. **Variáveis de Ambiente**

```bash
# .env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@host:5432/fraud_db
REDIS_URL=redis://localhost:6379
API_KEY=your-secure-api-key
```

### 2. **Configuração de Logging**

```python
# config/logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: default
    filename: logs/fraud_detection.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
loggers:
  fraud_detection:
    level: INFO
    handlers: [console, file]
    propagate: no
root:
  level: INFO
  handlers: [console]
```

### 3. **Configuração de Segurança**

```python
# config/security.py
SECURITY_CONFIG = {
    'authentication': {
        'enabled': True,
        'method': 'oauth2',  # ou 'basic', 'ldap'
        'providers': ['google', 'github']
    },
    'authorization': {
        'roles': ['admin', 'analyst', 'viewer'],
        'permissions': {
            'admin': ['read', 'write', 'delete', 'configure'],
            'analyst': ['read', 'write'],
            'viewer': ['read']
        }
    },
    'data_encryption': {
        'enabled': True,
        'algorithm': 'AES-256',
        'key_rotation': '30d'
    }
}
```

### 4. **Configuração de Performance**

```python
# config/performance.py
PERFORMANCE_CONFIG = {
    'caching': {
        'enabled': True,
        'backend': 'redis',
        'ttl': 3600,  # 1 hora
        'max_size': '1GB'
    },
    'database': {
        'connection_pool_size': 20,
        'connection_timeout': 30,
        'query_timeout': 60
    },
    'api': {
        'rate_limiting': {
            'enabled': True,
            'requests_per_minute': 100
        },
        'timeout': 30
    }
}
```

## Scripts de Deploy

### 1. **Script de Deploy Automatizado**

```bash
#!/bin/bash
# deploy.sh

set -e

echo "Iniciando deploy do Sistema de Prevenção de Fraudes..."

# Configurações
ENVIRONMENT=${1:-production}
VERSION=$(git describe --tags --always)

echo "Configurações:"
echo "   Ambiente: $ENVIRONMENT"
echo "   Versão: $VERSION"

# Build da aplicação
echo "Building aplicação..."
docker build -t fraud-dashboard:$VERSION .
docker tag fraud-dashboard:$VERSION fraud-dashboard:latest

# Testes
echo "Executando testes..."
python -m pytest tests/ -v

# Deploy
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Deploy em produção..."
    docker push your-registry/fraud-dashboard:$VERSION
    docker push your-registry/fraud-dashboard:latest

    # Atualizar serviço
    kubectl set image deployment/fraud-dashboard fraud-dashboard=your-registry/fraud-dashboard:$VERSION
    kubectl rollout status deployment/fraud-dashboard
else
    echo "Deploy em desenvolvimento..."
    docker-compose up -d
fi

echo "Deploy concluído com sucesso!"
```

### 2. **Script de Rollback**

```bash
#!/bin/bash
# rollback.sh

PREVIOUS_VERSION=${1:-latest}

echo "Executando rollback para versão: $PREVIOUS_VERSION"

kubectl rollout undo deployment/fraud-dashboard
kubectl rollout status deployment/fraud-dashboard

echo "Rollback concluído!"
```

## Monitoramento e Observabilidade

### 1. **Health Checks**

```python
# src/health.py
from fastapi import FastAPI
import psutil
import pandas as pd

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
def detailed_health():
    return {
        "status": "healthy",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "database": check_database_connection(),
        "models": check_models_status()
    }
```

### 2. **Métricas com Prometheus**

```python
# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Métricas de negócio
fraud_detections = Counter('fraud_detections_total', 'Total fraud detections')
transaction_processing_time = Histogram('transaction_processing_seconds', 'Transaction processing time')
active_alerts = Gauge('active_alerts', 'Number of active alerts')

# Métricas de sistema
model_accuracy = Gauge('model_accuracy', 'Model accuracy score', ['model_name'])
```

### 3. **Alertas**

```yaml
# alerts.yaml
groups:
  - name: fraud-detection
    rules:
      - alert: HighFraudRate
        expr: fraud_rate > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Taxa de fraude muito alta"
          
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Queda na acurácia do modelo"
```

## Segurança em Produção

### 1. **Checklist de Segurança**

- [ ] Autenticação habilitada
- [ ] HTTPS configurado
- [ ] Dados sensíveis criptografados
- [ ] Rate limiting implementado
- [ ] Logs de auditoria ativos
- [ ] Backup automatizado
- [ ] Monitoramento de segurança
- [ ] Testes de penetração realizados

### 2. **Configuração HTTPS**

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name fraud-dashboard.company.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Escalabilidade

### 1. **Load Balancing**

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-dashboard
  template:
    metadata:
      labels:
        app: fraud-dashboard
    spec:
      containers:
      - name: fraud-dashboard
        image: fraud-dashboard:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### 2. **Auto Scaling**

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-dashboard-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-dashboard
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Problemas Comuns

1. **Dashboard não carrega**
   - Verificar logs: `docker logs fraud-dashboard`
   - Verificar porta: `netstat -tulpn | grep 8501`
   - Verificar firewall

2. **Erro de memória**
   - Aumentar limites de container
   - Otimizar cache do Streamlit
   - Implementar paginação

3. **Performance lenta**
   - Verificar cache Redis
   - Otimizar queries de banco
   - Implementar CDN

### Comandos Úteis

```bash
# Logs em tempo real
kubectl logs -f deployment/fraud-dashboard

# Status dos pods
kubectl get pods -l app=fraud-dashboard

# Métricas de recursos
kubectl top pods

# Reiniciar deployment
kubectl rollout restart deployment/fraud-dashboard
```

---

**Deploy realizado com sucesso! Sistema pronto para produção.**
