# Makefile para Sistema de Prevenção de Fraudes

.PHONY: help install install-dev setup clean test lint format dashboard train predict docs

# Variáveis
PYTHON := python
PIP := pip
STREAMLIT := streamlit

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  install      - Instala dependências básicas"
	@echo "  install-dev  - Instala dependências de desenvolvimento"
	@echo "  setup        - Configuração inicial do projeto"
	@echo "  clean        - Remove arquivos temporários"
	@echo "  test         - Executa testes"
	@echo "  lint         - Executa linting"
	@echo "  format       - Formata código"
	@echo "  dashboard    - Inicia dashboard Streamlit"
	@echo "  train        - Treina modelos"
	@echo "  predict      - Executa predições"
	@echo "  docs         - Gera documentação"
	@echo "  jupyter      - Inicia Jupyter Lab"

# Instalação
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# Configuração inicial
setup: install
	@echo "Configurando projeto..."
	@if not exist config\config.yaml copy config\config.example.yaml config\config.yaml
	@if not exist logs mkdir logs
	@echo "Projeto configurado com sucesso!"

# Limpeza
clean:
	@echo "Limpando arquivos temporários..."
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del "%%f"
	@for /r . %%f in (*.pyo) do @if exist "%%f" del "%%f"
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist .coverage del .coverage
	@if exist htmlcov rd /s /q htmlcov
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
	@if not exist docs\_build mkdir docs\_build
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentação gerada em docs/_build/html/"

# Docker (se aplicável)
docker-build:
	docker build -t fraud-prevention .

docker-run:
	docker run -p 8501:8501 fraud-prevention

# Verificação de qualidade
quality-check: lint test
	@echo "✅ Verificação de qualidade concluída!"

# Deploy (exemplo)
deploy-staging:
	@echo "Fazendo deploy para staging..."
	# Adicionar comandos de deploy aqui

deploy-production:
	@echo "Fazendo deploy para produção..."
	# Adicionar comandos de deploy aqui

# Backup de modelos
backup-models:
	@echo "Fazendo backup dos modelos..."
	@if not exist backups mkdir backups
	@if exist models xcopy models backups\models_%date:~-4,4%%date:~-10,2%%date:~-7,2%\ /E /I /Y
	@echo "✅ Backup concluído!"

# Monitoramento
monitor:
	@echo "Iniciando monitoramento..."
	$(PYTHON) -m src.monitoring.monitor

# Instalação completa
install-all: install-dev setup
	@echo "✅ Instalação completa finalizada!"
