# Makefile – Sistema de Prevenção de Fraudes

.PHONY: help install install-dev setup clean test test-quick lint format \
        dashboard train predict generate-data eda jupyter docs \
        docker-build docker-run quality-check deploy-staging deploy-production \
        backup-models monitor install-all

PYTHON     := python
PIP        := pip
STREAMLIT  := streamlit
DATE       := $(shell date +%Y%m%d)

help:
	@echo "Comandos disponíveis:"
	@echo "  install          - Instala dependências básicas"
	@echo "  install-dev      - Instala dependências de desenvolvimento"
	@echo "  install-all      - Instala tudo e executa setup"
	@echo "  setup            - Configuração inicial do projeto"
	@echo "  clean            - Remove arquivos temporários"
	@echo "  test             - Executa testes com coverage"
	@echo "  test-quick       - Executa testes rápidos"
	@echo "  lint             - Linting (flake8, mypy)"
	@echo "  format           - Formata código"
	@echo "  dashboard        - Inicia dashboard Streamlit"
	@echo "  train            - Treina modelos"
	@echo "  predict          - Executa predições"
	@echo "  generate-data    - Gera dados sintéticos"
	@echo "  eda              - Análise exploratória"
	@echo "  jupyter          - Inicia Jupyter Lab"
	@echo "  docs             - Gera documentação"
	@echo "  docker-build     - Build Docker"
	@echo "  docker-run       - Executa container Docker"
	@echo "  quality-check    - Lint + testes"
	@echo "  backup-models    - Backup dos modelos"
	@echo "  deploy-staging   - Deploy para staging"
	@echo "  deploy-production - Deploy para produção"
	@echo "  monitor          - Inicia módulo de monitoramento"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

setup: install
	@mkdir -p config logs
	@test -f config/config.yaml || cp config/config.example.yaml config/config.yaml
	@echo "Projeto configurado com sucesso!"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "Limpeza concluída!"

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-quick:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

dashboard:
	$(STREAMLIT) run src/dashboard/app.py

train:
	$(PYTHON) -m src.models.train

predict:
	$(PYTHON) -m src.models.predict

generate-data:
	$(PYTHON) -m src.data.synthetic_data_generator

eda:
	$(PYTHON) -m src.data.exploratory_analysis

jupyter:
	jupyter lab

docs:
	@mkdir -p docs/_build/html
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentação gerada em docs/_build/html/"

docker-build:
	docker build -t fraud-prevention .

docker-run:
	docker run -p 8501:8501 fraud-prevention

quality-check: lint test
	@echo "Verificação de qualidade concluída!"

deploy-staging:
	@echo "Deploy para staging (placeholder)"

deploy-production:
	@echo "Deploy para produção (placeholder)"

backup-models:
	@mkdir -p backups/models_$(DATE)
	@if [ -d "models" ]; then cp -r models/* backups/models_$(DATE)/; fi
	@echo "Backup concluído!"

monitor:
	$(PYTHON) -m src.monitoring.monitor

install-all: install-dev setup
	@echo "Instalação completa finalizada!"
