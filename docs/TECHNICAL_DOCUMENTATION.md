# Documentação Técnica - Sistema de Prevenção de Fraudes

## Visão Geral

O Sistema de Prevenção de Fraudes é uma solução completa para detecção e prevenção de fraudes em transações financeiras, combinando técnicas de Machine Learning, regras de negócio e análises estatísticas.

## Arquitetura do Sistema

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE PREVENÇÃO DE FRAUDES          │
├─────────────────────────────────────────────────────────────┤
│  Dashboard Streamlit                                        │
│  ├── Monitoramento em Tempo Real                           │
│  ├── Análise de Dados                                      │
│  ├── Gestão de Regras                                      │
│  └── Relatórios Executivos                                 │
├─────────────────────────────────────────────────────────────┤
│  Engine de Detecção                                        │
│  ├── Modelos Supervisionados (RF, XGBoost, LR)            │
│  ├── Modelos Não Supervisionados (Isolation Forest, LOF)  │
│  ├── Regras de Negócio Customizáveis                      │
│  └── Sistema de Scoring Integrado                         │
├─────────────────────────────────────────────────────────────┤
│  Pipeline de Dados                                         │
│  ├── Carregamento de Dados                                │
│  ├── Preprocessamento e Feature Engineering               │
│  ├── Validação de Qualidade                               │
│  └── Geração de Dados Sintéticos                          │
├─────────────────────────────────────────────────────────────┤
│  Infraestrutura                                            │
│  ├── Configuração Centralizada                            │
│  ├── Sistema de Logging                                   │
│  ├── Utilitários e Helpers                                │
│  └── Testes Automatizados                                 │
└─────────────────────────────────────────────────────────────┘
```

## Estrutura do Projeto

```
transacional_fraud_prevention_pipeline/
├── src/                          # Código fonte principal
│   ├── data/                     # Módulos de dados
│   │   ├── data_loader.py        # Carregamento de dados
│   │   ├── preprocessor.py       # Preprocessamento
│   │   ├── synthetic_data_generator.py  # Geração de dados sintéticos
│   │   └── exploratory_analysis.py      # Análise exploratória
│   ├── models/                   # Modelos de ML e regras
│   │   ├── supervised_models.py  # Modelos supervisionados
│   │   ├── unsupervised_models.py # Modelos não supervisionados
│   │   └── business_rules.py     # Regras de negócio
│   ├── features/                 # Engenharia de features
│   ├── evaluation/               # Métricas e validação
│   ├── dashboard/                # Interface Streamlit
│   │   └── app.py               # Aplicação principal
│   └── utils/                    # Utilitários
│       ├── config.py            # Configuração
│       ├── logger.py            # Sistema de logging
│       └── helpers.py           # Funções auxiliares
├── notebooks/                    # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_fraud_detection_modeling.ipynb
│   └── 03_fraud_prevention_demo.ipynb
├── data/                        # Dados
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   └── synthetic/               # Dados sintéticos
├── models/                      # Modelos treinados
├── config/                      # Arquivos de configuração
├── tests/                       # Testes automatizados
├── docs/                        # Documentação
└── logs/                        # Arquivos de log
```

## Configuração e Instalação

### Pré-requisitos

- Python 3.9+
- pip ou conda
- Git

### Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd transacional_fraud_prevention_pipeline

# Instale as dependências
pip install -r requirements.txt

# Configure o ambiente
cp config/config.example.yaml config/config.yaml

# Execute testes (opcional)
pytest tests/ -v

# Inicie o dashboard
streamlit run src/dashboard/app.py
```

## Módulos de Dados

### DataLoader (`src/data/data_loader.py`)

Responsável pelo carregamento e validação de dados de diferentes fontes.

**Principais funcionalidades:**
- Carregamento de dados CSV, Parquet, Excel
- Validação de esquemas de dados
- Conversão automática de tipos
- Geração de resumos estatísticos

**Exemplo de uso:**
```python
from src.data.data_loader import DataLoader

loader = DataLoader()
users_df, transactions_df = loader.load_synthetic_data()
summary = loader.get_data_summary(transactions_df)
```

### SyntheticDataGenerator (`src/data/synthetic_data_generator.py`)

Gera dados transacionais sintéticos realistas para desenvolvimento e testes.

**Características dos dados gerados:**
- Usuários com perfis demográficos variados
- Transações com padrões temporais realistas
- Fraudes com características específicas
- Features derivadas automaticamente

**Exemplo de uso:**
```python
from src.data.synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()
users_df, transactions_df = generator.generate_complete_dataset()
generator.save_dataset(users_df, transactions_df)
```

### DataPreprocessor (`src/data/preprocessor.py`)

Preprocessa dados para modelagem de ML.

**Funcionalidades:**
- Engenharia de features temporais
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização de features numéricas
- Balanceamento de classes

**Exemplo de uso:**
```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
processed_df = preprocessor.fit_transform(transactions_df)
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(processed_df)
```

## Modelos de Machine Learning

### Modelos Supervisionados (`src/models/supervised_models.py`)

Implementa modelos de classificação para detecção de fraudes.

**Modelos disponíveis:**
- Random Forest
- XGBoost
- Logistic Regression
- Ensemble (Voting/Stacking)

**Funcionalidades:**
- Treinamento com balanceamento de classes (SMOTE)
- Validação cruzada
- Avaliação com múltiplas métricas
- Análise de importância de features

**Exemplo de uso:**
```python
from src.models.supervised_models import SupervisedModels

models = SupervisedModels()
training_results = models.train_models(X_train, y_train, balance_method='smote')
evaluation_results = models.evaluate_models(X_test, y_test)
best_model = models.get_best_model()
```

### Modelos Não Supervisionados (`src/models/unsupervised_models.py`)

Implementa modelos de detecção de anomalias.

**Modelos disponíveis:**
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN Clustering
- K-Means Clustering
- Autoencoder (opcional, requer TensorFlow)

**Exemplo de uso:**
```python
from src.models.unsupervised_models import UnsupervisedModels

models = UnsupervisedModels()
results = models.train_all_models(X_train)
anomalies = models.predict_anomalies(X_test, 'isolation_forest')
```

## Sistema de Regras de Negócio

### BusinessRulesEngine (`src/models/business_rules.py`)

Engine flexível para criação e gestão de regras de negócio.

**Tipos de regras implementadas:**
- Regras de valor (transações de alto valor)
- Regras de velocidade (muitas transações em pouco tempo)
- Regras de localização (países/regiões suspeitas)
- Regras temporais (horários incomuns)
- Regras customizadas (definidas pelo usuário)

**Exemplo de uso:**
```python
from src.models.business_rules import BusinessRulesEngine

engine = BusinessRulesEngine()

# Avalia uma transação
transaction = {
    'amount': 15000,
    'country': 'XX',
    'transaction_hour': 3
}

result = engine.evaluate_transaction(transaction)
print(f"Ação: {result['final_action']}, Score: {result['risk_score']}")
```

## Dashboard e Monitoramento

### Streamlit Dashboard (`src/dashboard/app.py`)

Interface web interativa para monitoramento e gestão do sistema.

**Páginas disponíveis:**
- **Visão Geral**: Métricas principais e estatísticas
- **Análise de Dados**: Visualizações exploratórias
- **Modelos ML**: Performance e comparação de modelos
- **Regras de Negócio**: Gestão e estatísticas de regras
- **Alertas**: Sistema de alertas em tempo real
- **Configurações**: Ajustes do sistema

**Executar dashboard:**
```bash
streamlit run src/dashboard/app.py
```

## Configuração

### Arquivo de Configuração (`config/config.yaml`)

Centraliza todas as configurações do sistema:

```yaml
# Configurações de dados
data:
  synthetic:
    n_samples: 10000
    fraud_rate: 0.02
  preprocessing:
    test_size: 0.2
    stratify: true

# Configurações de modelos
models:
  supervised:
    random_forest:
      n_estimators: 100
      max_depth: 10
  unsupervised:
    isolation_forest:
      contamination: 0.02

# Regras de negócio
business_rules:
  amount_rules:
    max_single_transaction: 10000
  velocity_rules:
    max_transactions_per_hour: 50
```

## Testes

### Estrutura de Testes

```
tests/
├── test_data_loader.py          # Testes do carregador de dados
├── test_business_rules.py       # Testes das regras de negócio
├── test_supervised_models.py    # Testes dos modelos supervisionados
└── test_integration.py          # Testes de integração
```

### Executar Testes

```bash
# Todos os testes
pytest tests/ -v

# Com coverage
pytest tests/ --cov=src --cov-report=html

# Teste específico
pytest tests/test_data_loader.py -v
```

## Métricas e Avaliação

### Métricas Implementadas

**Para modelos supervisionados:**
- Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- Confusion Matrix
- Classification Report

**Para modelos não supervisionados:**
- Taxa de anomalias detectadas
- Silhouette Score (clustering)
- Reconstruction Error (autoencoder)

**Para regras de negócio:**
- Taxa de acionamento por regra
- Distribuição de ações (flag/review/block)
- Score de risco médio

## Deploy e Produção

### Considerações para Produção

1. **Escalabilidade**: Use processamento em lote para grandes volumes
2. **Latência**: Otimize modelos para resposta < 100ms
3. **Monitoramento**: Implemente alertas para drift de modelo
4. **Segurança**: Criptografe dados sensíveis
5. **Backup**: Mantenha backups de modelos e configurações

### API REST (Exemplo)

```python
from fastapi import FastAPI
from src.models.supervised_models import SupervisedModels

app = FastAPI()
models = SupervisedModels()

@app.post("/predict")
async def predict_fraud(transaction: dict):
    prediction = models.predict(transaction)
    return {"is_fraud": bool(prediction), "confidence": float(prediction)}
```

## Troubleshooting

### Problemas Comuns

1. **Erro de importação**: Verifique se o PYTHONPATH inclui o diretório raiz
2. **Dados não encontrados**: Execute o gerador de dados sintéticos
3. **Modelos não treinados**: Execute o pipeline de treinamento
4. **Dashboard não carrega**: Verifique se o Streamlit está instalado

### Logs

Os logs são salvos em `logs/fraud_detection.log` e incluem:
- Informações de treinamento de modelos
- Acionamento de regras de negócio
- Erros e warnings do sistema
- Métricas de performance

## Suporte

Para suporte técnico:
- Consulte a documentação em `docs/`
- Execute os notebooks em `notebooks/` para exemplos
- Verifique os testes em `tests/` para casos de uso
- Abra issues no repositório para bugs ou melhorias

## Roadmap

### Próximas Funcionalidades

1. **API REST completa** para integração
2. **Retreinamento automático** de modelos
3. **Integração com bancos de dados** externos
4. **Sistema de feedback** para melhoria contínua
5. **Deployment containerizado** com Docker
6. **Monitoramento avançado** com Prometheus/Grafana
