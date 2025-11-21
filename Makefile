# Makefile para Sistema de Prevenção de Fraudes

.PHONY: help install install-dev setup clean test test-quick lint format \
        dashboard train predict generate-data eda jupyter docs \
        docker-build docker-run quality-check deploy-staging deploy-production \
        backup-models monitor install-all

# Variáveis
PYTHON     := python
PIP        := pip
STREAMLIT  := streamlit
DATE       := $(shell date +%Y%m%d)

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  install         - Instala dependências básicas"
	@echo "  install-dev     - Instala dependências de desenvolvimento"
	@echo "  install-all     - Instala tudo e executa setup"
	@echo "  setup           - Configuração inicial do projeto"
	@echo "  clean           - Remove arquivos temporários"
	@echo "  test            - Executa testes com coverage"
	@echo "  test-quick      - Executa testes rápidos (sem coverage)"
	@echo "  lint            - Executa linting (flake8, mypy)"
	@echo "  format          - Formata código (black, isort)"
	@echo "  dashboard       - Inicia dashboard Streamlit"
	@echo "  train           - Treina modelos"
	@echo "  predict         - Executa predições"
	@echo "  generate-data   - Gera dados sintéticos"
	@echo "  eda             - Executa análise exploratória"
	@echo "  jupyter         - Inicia Jupyter Lab"
	@echo "  docs            - Gera documentação com Sphinx"
	@echo "  docker-build    - Build da imagem Docker"
	@echo "  docker-run      - Executa container Docker"
	@echo "  quality-check   - Lint + testes"
	@echo "  backup-models   - Faz backup dos modelos"
	@echo "  deploy-staging  - Deploy para ambiente de staging (exemplo)"
	@echo "  deploy-production - Deploy para produção (exemplo)"
	@echo "  monitor         - Inicia módulo de monitoramento"

# Instalação
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

# Configuração inicial
setup: install
	@echo "Configurando projeto..."
	@mkdir -p config logs
	@test -f config/config.yaml || cp config/config.example.yaml config/config.yaml
	@echo "Projeto configurado com sucesso!"

# Limpeza
clean:
	@echo "Limpando arquivos temporários..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "Limpeza concluída!"

# Testes
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-quick:
	$(PYTHON) -m pytest tests/ -v

# Linting e formatação
lint:
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

# Dashboard
dashboard:
	$(STREAMLIT) run src/dashboard/app.py

# Treinamento e predição
train:
	$(PYTHON) -m src.models.train

predict:
	$(PYTHON) -m src.models.predict

# Geração de dados sintéticos
generate-data:
	$(PYTHON) -m src.data.synthetic_data_generator

# Análise exploratória
eda:
	$(PYTHON) -m src.data.exploratory_analysis

# Jupyter
jupyter:
	jupyter lab

# Documentação
docs:
	@echo "Gerando documentação..."
	@mkdir -p docs/_build/html
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentação gerada em docs/_build/html/"

# Docker
docker-build:
	docker build -t fraud-prevention .

docker-run:
	docker run -p 8501:8501 fraud-prevention

# Verificação de qualidade
quality-check: lint test
	@echo "✅ Verificação de qualidade concluída!"

# Deploy
deploy-staging:
	@echo "Fazendo deploy para staging..."
	# Adicionar comandos reais de deploy aqui

deploy-production:
	@echo "Fazendo deploy para produção..."
	# Adicionar comandos reais de deploy aqui

# Backup de modelos
backup-models:
	@echo "Fazendo backup dos modelos..."
	@mkdir -p backups/models_$(DATE)
	@if [ -d "models" ]; then cp -r models/* backups/models_$(DATE)/; fi
	@echo "Backup concluído!"

# Monitoramento
monitor:
	@echo "Iniciando monitoramento..."
	$(PYTHON) -m src.monitoring.monitor

# Instalação completa
install-all: install-dev setup
	@echo "Instalação completa finalizada!"
