## Transactional Fraud Prevention System v.1.0.1

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![PyTest](https://img.shields.io/badge/PyTest-Testing-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple.svg)
![Data Pipeline](https://img.shields.io/badge/Data%20Pipeline-Real--Time-lightgrey.svg)
![Fraud Detection](https://img.shields.io/badge/Fraud%20Detection-Active-critical.svg)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20development-yellow.svg)

The project, in addition to its primary goal of learning and experimentation, also aims to provide a comprehensive solution for the detection and prevention of fraud in financial transactions, integrating Machine Learning, business rules, and advanced statistical analyses. The solution includes an interactive dashboard for real-time monitoring, exploratory analysis, and alert management.

## About the Project
This system is being developed to detect and prevent fraud in financial transactions with high accuracy, reducing false positives while maintaining a seamless experience for legitimate users. The solution integrates multiple technologies and approaches:

- Supervised and unsupervised Machine Learning models
- Customizable and dynamic business rules engine
- Advanced statistical and exploratory analyses
- Interactive web dashboard for monitoring and management
- Automated data pipeline for real-time processing
- Intelligent alert system with different priority levels

## Visual Demonstration of the System
The system provides a complete and intuitive web interface through a Streamlit dashboard. Below are the main screens and functionalities:

### 1. Execution

![Tela Principal](images/image_1.png)

- Dependency check completed: All required libraries (Streamlit, Pandas, NumPy, Plotly, Scikit-learn) are installed.
- Data files successfully located: The required .csv and .json files for the dashboard were found in the data/ directory.
- Dashboard successfully launched via Streamlit: The command streamlit run src/dashboard/app.py was executed correctly.
- Local access to the dashboard: The dashboard was automatically opened in the browser at http://localhost:8501.
- Documentation available: The technical documentation is located at docs/TECHNICAL_DOCUMENTATION.md.

### 2. Real-Time Monitoring Dashboard

![Dashboard de Monitoramento](images/image_2.png)

The monitoring dashboard displays:
- Model performance metrics for fraud detection
- Real-time transaction charts
- Indicators of active and pending alerts
- Geographic distribution of transactions
- Timeline of detected fraud events

### 3. Exploratory Data Analysis

![Análise de Dados](images/image_3.png)

The data analysis section offers:
- Interactive charts showing the distribution of transaction values
- Temporal analysis of fraud patterns
- Correlations between different variables
- Detailed descriptive statistics
- Visualizations of outliers and anomalies

### 4. Analysis by Merchant Category

![Análise por Categoria](images/image_4.png)

This screen presents:
- Distribution of fraud cases by merchant category
- Fraud rate specific to each type of merchant
- Comparative bar charts
- Insights into higher-risk categories
- Recommendations based on the identified patterns

### 5. Temporal Analysis and Patterns

![Análise Temporal](images/image_5.png)

The temporal analysis shows:
- Fraud patterns over time
- Seasonality and trends
- Peak times for fraudulent activity
- Comparisons between periods
- Forecasts based on historical data

### 6. Machine Learning Models

![Modelos ML](images/image_6.png)

The models section presents:
- Comparative performance across different algorithms
- Precision, recall, and F1-score metrics
- ROC and Precision–Recall curves
- Feature importance for each model
- Configurations and parameters of the trained models

### 7. Business Rules System

![Regras de Negócio](images/image_7.png)

The rules system offers:
- An interface for creating and editing custom rules
- Conditional logic for different scenarios
- Configurable thresholds for alerts
- History of rule application
- Validation and testing of new rules

### 8. Alert Management

![Sistema de Alertas](images/image_8.png)

The alert system includes:
- A list of active alerts with different priority levels
- Filters by status, type, and time period
- Detailed view of each alert with transaction context
- Available actions for each alert
- Resolution history and feedback

### 9. Geographic Analysis

![Análise Geográfica](images/image_9.png)

The geographic analysis shows:
- Map of transaction distribution by country/region
- Identification of suspicious geographic patterns
- Fraud rate by location
- Alerts for transactions in unusual locations
- Visualization of suspicious transaction routes

### 10. System Settings

![Configurações](images/image_10.png)

The settings screen allows:
- Uploading new datasets for analysis
- Configuration of model parameters
- Adjustment of detection thresholds
- Configuration of notifications and alerts
- Backup and restore of settings

### 11. Reports and Detailed Metrics

![Relatórios](images/image_11.png)

The reports include:
- Detailed model performance metrics
- In-depth analysis of false positives and false negatives
- Identification of long-term trends
- Comparisons across different observation periods
- Export of results for external analysis

### 12. Individual Transaction Analysis

![Detalhes de Transação](images/image_12.png)

The individual analysis shows:
- Complete details of each transaction
- Risk score calculated by the models
- Factors that contributed to the classification
- User history and behavioral patterns
- Recommended actions based on the analysis

### 13. Executive Dashboard

![Dashboard Executivo](images/image_13.png)

The executive dashboard presents:
- Key KPIs to support management
- Consolidated summary of system performance
- Estimated financial impact achieved through fraud prevention
- Identified trends and future projections
- Strategic recommendations for decision-making

### 14. Navigation Interface and Menu

![Interface de Navegação](images/image_14.png)

The navigation interface offers:
- An intuitive sidebar menu for quick access
- Real-time system status
- Connectivity and performance information
- Links to documentation and support
- Visual indicators of system status and alerts

## Installation and Execution

### Method 1: Automated Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/nathadriele/transaction_fraud_prevention_pipeline
cd transacional_fraud_prevention_pipeline

# Run the automated initialization script
python start_dashboard.py
```

The automated script will:
- Check the Python version
- Install missing dependencies
- Configure the environment
- Automatically start the dashboard
- Open the browser at the correct URL

### Method 2: Docker

```bash
# Build e execução com Docker Compose
docker-compose up -d

# Acesse: http://localhost:8501
```

### Method 3: Manual Installation

```bash
# Instale as dependências
pip install -r requirements.txt

# Configure o ambiente
cp config/config.example.yaml config/config.yaml

# Execute o dashboard
streamlit run src/dashboard/app.py
```

**Dashboard available at: http://localhost:8501**

## Principais Funcionalidades

### Detecção de Fraudes
- **Modelos Supervisionados**: Random Forest, XGBoost, Logistic Regression
- **Detecção de Anomalias**: Isolation Forest, Local Outlier Factor
- **Análise Comportamental**: Padrões de usuário e transações
- **Scoring em Tempo Real**: Classificação instantânea de risco
- **Ensemble de Modelos**: Combinação inteligente de múltiplos algoritmos

### Sistema de Regras de Negócio
- **Engine Customizável**: Criação de regras específicas do negócio
- **Alertas Inteligentes**: Baseados em thresholds dinâmicos
- **Lógica Condicional**: Regras complexas com múltiplas condições
- **Gestão de Exceções**: Tratamento de casos especiais
- **Validação Automática**: Teste de eficácia das regras

### Dashboard e Monitoramento
- **Interface Web Interativa**: Dashboard Streamlit responsivo
- **Métricas em Tempo Real**: KPIs atualizados automaticamente
- **Visualizações Avançadas**: Gráficos interativos com Plotly
- **Análise Exploratória**: Ferramentas de investigação de dados
- **Relatórios Executivos**: Resumos para tomada de decisão

### Pipeline de Dados
- **Geração de Dados Sintéticos**: Criação de datasets realistas
- **Preprocessamento Automático**: Limpeza e transformação de dados
- **Feature Engineering**: Criação de variáveis preditivas
- **Validação de Qualidade**: Verificação de integridade dos dados
- **ETL Automatizado**: Extração, transformação e carregamento

### Sistema de Alertas
- **Classificação por Prioridade**: Crítico, Alto, Médio, Baixo
- **Notificações em Tempo Real**: Alertas instantâneos
- **Gestão de Workflow**: Acompanhamento de resolução
- **Histórico Completo**: Rastreabilidade de todas as ações
- **Integração Externa**: APIs para sistemas terceiros

## Tecnologias Utilizadas

### Core
- **Python 3.9+** - Linguagem principal
- **Streamlit** - Framework para dashboard web
- **Docker** - Containerização e deployment

### Machine Learning
- **scikit-learn** - Algoritmos de ML clássicos
- **XGBoost** - Gradient boosting otimizado
- **TensorFlow** - Deep learning e redes neurais
- **imbalanced-learn** - Tratamento de dados desbalanceados

### Análise de Dados
- **pandas** - Manipulação de dados estruturados
- **numpy** - Computação numérica
- **scipy** - Análises estatísticas avançadas
- **matplotlib/seaborn** - Visualizações estáticas

### Visualização
- **Plotly** - Gráficos interativos
- **Streamlit** - Interface web responsiva
- **Jupyter** - Notebooks para análise exploratória

### Desenvolvimento
- **pytest** - Framework de testes
- **black** - Formatação de código
- **flake8** - Linting e qualidade de código
- **pre-commit** - Hooks de validação

## Como Usar o Sistema

### 1. Acesso ao Dashboard
Após a instalação, acesse `http://localhost:8501` para abrir o dashboard principal.

### 2. Navegação
Use o menu lateral para navegar entre as diferentes seções:
- **Visão Geral**: Métricas principais e status do sistema
- **Análise de Dados**: Exploração e visualização de dados
- **Modelos ML**: Performance e configuração dos modelos
- **Regras**: Gestão de regras de negócio
- **Alertas**: Monitoramento e gestão de alertas
- **Configurações**: Ajustes e upload de dados

### 3. Análise de Transações
- Visualize transações em tempo real
- Analise padrões e tendências
- Identifique anomalias e fraudes
- Configure alertas personalizados

### 4. Gestão de Modelos
- Compare performance entre modelos
- Ajuste parâmetros e thresholds
- Retreine modelos com novos dados
- Monitore drift e degradação

### 5. Configuração de Regras
- Crie regras customizadas de negócio
- Defina condições e ações
- Teste regras antes da implementação
- Monitore eficácia das regras

## Notebooks de Análise

O projeto inclui notebooks Jupyter para análise exploratória:

### 1. Análise Exploratória de Dados
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
- Estatísticas descritivas
- Visualizações de distribuições
- Análise de correlações
- Identificação de padrões

### 2. Modelagem de Fraudes
```bash
jupyter notebook notebooks/02_fraud_detection_modeling.ipynb
```
- Treinamento de modelos
- Validação cruzada
- Otimização de hiperparâmetros
- Avaliação de performance

### 3. Demonstração do Sistema
```bash
jupyter notebook notebooks/03_fraud_prevention_demo.ipynb
```
- Casos de uso práticos
- Simulações de cenários
- Exemplos de implementação
- Testes de integração

## Testes e Qualidade

### Execução de Testes
```bash
# Execute todos os testes
pytest tests/

# Execute com coverage
pytest tests/ --cov=src --cov-report=html

# Testes específicos
pytest tests/test_business_rules.py
pytest tests/test_data_loader.py
```

### Métricas de Performance
- **Precision**: > 95%
- **Recall**: > 90%
- **F1-Score**: > 92%
- **False Positive Rate**: < 2%
- **Latência**: < 100ms
- **Throughput**: > 1000 transações/segundo

## Documentação Técnica

### Guias Disponíveis
- [Guia de Deploy](docs/DEPLOYMENT_GUIDE.md) - Instruções completas de deployment
- [Documentação Técnica](docs/TECHNICAL_DOCUMENTATION.md) - Arquitetura e implementação
- [Configurações](config/) - Arquivos de configuração do sistema
- [Notebooks](notebooks/) - Análises exploratórias e modelagem

### Estrutura de Dados
O sistema trabalha com os seguintes tipos de dados:
- **Transações**: Dados de transações financeiras
- **Usuários**: Informações de perfil e comportamento
- **Alertas**: Registros de eventos suspeitos
- **Modelos**: Métricas e configurações de ML
- **Regras**: Definições de regras de negócio

### APIs e Integrações
- **REST API**: Endpoints para integração externa
- **Webhooks**: Notificações em tempo real
- **Batch Processing**: Processamento em lote
- **Stream Processing**: Análise de dados em tempo real

## Casos de Uso

### 1. Detecção em Tempo Real
- Análise imediata de transações em execução
- Bloqueio automático de atividades potencialmente fraudulentas
- Notificações instantâneas para as equipes responsáveis

### 2. Análise Investigativa
- Investigação de padrões suspeitos
- Análise forense de fraudes
- Relatórios detalhados para auditoria

### 3. Gestão de Risco
- Monitoramento de KPIs de risco
- Ajuste de políticas de segurança
- Otimização de regras de negócio

### 4. Compliance e Auditoria
- Rastreabilidade completa de decisões
- Relatórios para órgãos reguladores
- Documentação de processos

## Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

### Padrões de Desenvolvimento
- Siga as convenções PEP 8 para Python
- Adicione testes para novas funcionalidades
- Documente mudanças no README
- Use commits semânticos

### Reportar Issues
- Use templates de issue disponíveis
- Inclua logs e screenshots quando relevante
- Descreva passos para reproduzir problemas
- Sugira soluções quando possível

### Status do Projeto
- **Versão Atual**: 1.0.1
- **Status**: Em atualização / manutenção
- **Última Atualização**: 21/11/2025
- **Próxima Release**: A definir

---

**Sistema de Prevenção de Fraudes** - Protegendo transações financeiras com inteligência artificial e análise avançada de dados. **Desenvolvido principalmente com foco no aprendizado**.
