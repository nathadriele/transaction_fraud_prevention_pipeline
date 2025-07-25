## Sistema de Prevenção de Fraudes Transacionais v.1.0.0

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

Uma solução de detecção e prevenção de fraudes em transações financeiras, combinando Machine Learning, regras de negócio e análises estatísticas avançadas. O sistema oferece um dashboard interativo para monitoramento em tempo real, análise de dados e gestão de alertas de fraude.

## Sobre o Projeto

Este sistema está sendo desenvolvido para detectar e prevenir fraudes em transações financeiras com alta precisão, reduzindo falsos positivos e mantendo uma experiência fluida para usuários legítimos. A solução integra múltiplas tecnologias e abordagens:

- **Modelos de Machine Learning** supervisionados e não supervisionados
- **Engine de regras de negócio** customizáveis e dinâmicas
- **Análises estatísticas** e exploratórias avançadas
- **Dashboard web interativo** para monitoramento e gestão
- **Pipeline de dados** automatizado para processamento em tempo real
- **Sistema de alertas** inteligente com diferentes níveis de prioridade

## Demonstração Visual do Sistema

O sistema oferece uma interface web completa e intuitiva através de um dashboard Streamlit. Abaixo estão as principais telas e funcionalidades:

### 1. Tela Principal - Visão Geral do Sistema

![Tela Principal](images/image_1.png)

A tela principal apresenta uma visão geral completa do sistema com métricas em tempo real. Observa-se:
- Painel de indicadores principais com total de transações processadas
- Taxa de fraude atual e tendências
- Gráficos de distribuição de transações por categoria
- Status do sistema e última atualização dos dados
- Navegação lateral para acesso às diferentes funcionalidades

### 2. Dashboard de Monitoramento em Tempo Real

![Dashboard de Monitoramento](images/image_2.png)

O dashboard de monitoramento exibe:
- Métricas de performance dos modelos de detecção
- Gráficos de transações em tempo real
- Indicadores de alertas ativos e pendentes
- Distribuição geográfica das transações
- Timeline de eventos de fraude detectados

### 3. Análise Exploratória de Dados

![Análise de Dados](images/image_3.png)

A seção de análise de dados oferece:
- Gráficos interativos de distribuição de valores de transação
- Análise temporal de padrões de fraude
- Correlações entre diferentes variáveis
- Estatísticas descritivas detalhadas
- Visualizações de outliers e anomalias

### 4. Análise por Categoria de Comerciante

![Análise por Categoria](images/image_4.png)

Esta tela apresenta:
- Distribuição de fraudes por categoria de estabelecimento
- Taxa de fraude específica para cada tipo de comerciante
- Gráficos de barras comparativos
- Insights sobre categorias de maior risco
- Recomendações baseadas nos padrões identificados

### 5. Análise Temporal e Padrões

![Análise Temporal](images/image_5.png)

A análise temporal mostra:
- Padrões de fraude ao longo do tempo
- Sazonalidade e tendências
- Horários de pico para atividades fraudulentas
- Comparação entre períodos
- Previsões baseadas em dados históricos

### 6. Modelos de Machine Learning

![Modelos ML](images/image_6.png)

A seção de modelos apresenta:
- Performance comparativa entre diferentes algoritmos
- Métricas de precisão, recall e F1-score
- Curvas ROC e Precision-Recall
- Importância das features para cada modelo
- Configurações e parâmetros dos modelos treinados

### 7. Sistema de Regras de Negócio

![Regras de Negócio](images/image_7.png)

O sistema de regras oferece:
- Interface para criação e edição de regras customizadas
- Lógica condicional para diferentes cenários
- Thresholds configuráveis para alertas
- Histórico de aplicação das regras
- Validação e teste de novas regras

### 8. Gestão de Alertas

![Sistema de Alertas](images/image_8.png)

O sistema de alertas inclui:
- Lista de alertas ativos com diferentes prioridades
- Filtros por status, tipo e período
- Detalhes de cada alerta com contexto da transação
- Ações disponíveis para cada alerta
- Histórico de resoluções e feedback

### 9. Análise Geográfica

![Análise Geográfica](images/image_9.png)

A análise geográfica mostra:
- Mapa de distribuição de transações por país/região
- Identificação de padrões geográficos suspeitos
- Taxa de fraude por localização
- Alertas para transações em locais incomuns
- Visualização de rotas de transações suspeitas

### 10. Configurações do Sistema

![Configurações](images/image_10.png)

A tela de configurações permite:
- Upload de novos datasets para análise
- Configuração de parâmetros dos modelos
- Ajuste de thresholds de detecção
- Configuração de notificações e alertas
- Backup e restore de configurações

### 11. Relatórios e Métricas Detalhadas

![Relatórios](images/image_11.png)

Os relatórios incluem:
- Métricas de performance detalhadas
- Análise de falsos positivos e negativos
- Tendências de longo prazo
- Comparação entre períodos
- Exportação de dados para análise externa

### 12. Análise de Transações Individuais

![Detalhes de Transação](images/image_12.png)

A análise individual mostra:
- Detalhes completos de cada transação
- Score de risco calculado pelos modelos
- Fatores que contribuíram para a classificação
- Histórico do usuário e padrões comportamentais
- Ações recomendadas baseadas na análise

### 13. Dashboard Executivo

![Dashboard Executivo](images/image_13.png)

O dashboard executivo apresenta:
- KPIs principais para gestão
- Resumo de performance do sistema
- Impacto financeiro da prevenção de fraudes
- Tendências e projeções
- Recomendações estratégicas

### 14. Interface de Navegação e Menu

![Interface de Navegação](images/image_14.png)

A interface de navegação oferece:
- Menu lateral intuitivo para acesso rápido
- Status em tempo real do sistema
- Informações de conectividade e performance
- Links para documentação e suporte
- Indicadores visuais de status e alertas

## Instalação e Execução

### Método 1: Script Automático (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/nathadriele/transaction_fraud_prevention_pipeline
cd transacional_fraud_prevention_pipeline

# Execute o script de inicialização automática
python start_dashboard.py
```

O script automático irá:
- Verificar a versão do Python
- Instalar dependências ausentes
- Configurar o ambiente
- Iniciar o dashboard automaticamente
- Abrir o navegador na URL correta

### Método 2: Docker

```bash
# Build e execução com Docker Compose
docker-compose up -d

# Acesse: http://localhost:8501
```

### Método 3: Instalação Manual

```bash
# Instale as dependências
pip install -r requirements.txt

# Configure o ambiente
cp config/config.example.yaml config/config.yaml

# Execute o dashboard
streamlit run src/dashboard/app.py
```

**Dashboard disponível em: http://localhost:8501**

## Arquitetura do Sistema

```
├── src/                    # Código fonte principal
│   ├── data/              # Módulos de dados e ETL
│   ├── models/            # Modelos de ML e regras
│   ├── features/          # Engenharia de features
│   ├── evaluation/        # Métricas e validação
│   └── dashboard/         # Interface Streamlit
├── notebooks/             # Análises exploratórias
├── data/                  # Dados (raw, processed, synthetic)
├── models/                # Modelos treinados
├── config/                # Configurações
├── tests/                 # Testes unitários
├── images/                # Screenshots do sistema
└── docs/                  # Documentação
```

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
- Análise instantânea de transações
- Bloqueio automático de fraudes
- Notificações imediatas para equipes

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
- **Versão Atual**: 1.0.0
- **Status**: Produção
- **Última Atualização**: 24/07/2025
- **Próxima Release**: A definir

---

**Sistema de Prevenção de Fraudes** - Protegendo transações financeiras com inteligência artificial e análise avançada de dados. **Desenvolvido principalmente com foco no aprendizado**.
