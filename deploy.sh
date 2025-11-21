#!/bin/bash

# Script de Deploy Automatizado - Sistema de Prevenção de Fraudes
# Suporta deploy local, Docker e cloud (AWS, GCP, Azure)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ENVIRONMENT=${1:-local}
CLOUD_PROVIDER=${2:-}
VERSION=$(git describe --tags --always 2>/dev/null || echo "v1.0.1")
PROJECT_NAME="fraud-prevention-system"
PYTHON_BIN=${PYTHON_BIN:-python3}
PIP_BIN=${PIP_BIN:-pip3}

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           DEPLOY - SISTEMA DE PREVENÇÃO DE FRAUDES           ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configurações do Deploy:${NC}"
echo "   Ambiente: $ENVIRONMENT"
echo "   Versão:   $VERSION"
echo "   Projeto:  $PROJECT_NAME"
echo

check_prerequisites() {
    echo -e "${BLUE}Verificando pré-requisitos...${NC}"

    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        echo -e "${RED}ERRO: $PYTHON_BIN não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: $PYTHON_BIN encontrado${NC}"

    if ! command -v "$PIP_BIN" &> /dev/null; then
        echo -e "${RED}ERRO: $PIP_BIN não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: $PIP_BIN encontrado${NC}"

    if ! command -v git &> /dev/null; then
        echo -e "${YELLOW}AVISO: Git não encontrado (opcional)${NC}"
    else
        echo -e "${GREEN}OK: Git encontrado${NC}"
    fi
}

install_dependencies() {
    echo -e "${BLUE}Instalando dependências...${NC}"

    if [ -f "requirements.txt" ]; then
        "$PIP_BIN" install -r requirements.txt
        echo -e "${GREEN}OK: Dependências instaladas${NC}"
    else
        echo -e "${RED}ERRO: Arquivo requirements.txt não encontrado${NC}"
        exit 1
    fi
}

run_tests() {
    echo -e "${BLUE}Executando testes automatizados...${NC}"

    if [ -d "tests" ]; then
        "$PYTHON_BIN" -m pytest tests/ -v || {
            echo -e "${YELLOW}AVISO: Alguns testes falharam, mas o deploy continuará.${NC}"
        }
    else
        echo -e "${YELLOW}AVISO: Diretório de testes não encontrado${NC}"
    fi

    echo -e "${BLUE}Verificando importações básicas do dashboard...${NC}"
    "$PYTHON_BIN" - << 'EOF'
import sys
sys.path.append('src')
try:
    from src.dashboard.app import load_final_results  # noqa: F401
    print('OK: Importações do dashboard OK')
except Exception as e:
    print(f'ERRO: Erro nas importações do dashboard: {e}')
    sys.exit(1)
EOF
}

deploy_local() {
    echo -e "${BLUE}Deploy Local...${NC}"

    check_prerequisites
    install_dependencies
    run_tests

    echo -e "${BLUE}Iniciando dashboard...${NC}"
    echo -e "${GREEN}Dashboard disponível em: http://localhost:8501${NC}"
    echo -e "${YELLOW}Para encerrar, use Ctrl+C${NC}"
    echo

    "$PYTHON_BIN" start_dashboard.py
}

deploy_docker() {
    echo -e "${BLUE}Deploy com Docker...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}ERRO: Docker não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: Docker encontrado${NC}"

    echo -e "${BLUE}Build da imagem Docker...${NC}"
    docker build -t $PROJECT_NAME:$VERSION .
    docker tag $PROJECT_NAME:$VERSION $PROJECT_NAME:latest
    echo -e "${GREEN}OK: Imagem criada: $PROJECT_NAME:$VERSION${NC}"

    if [ -f "docker-compose.yml" ]; then
        echo -e "${BLUE}Parando containers existentes (docker-compose)...${NC}"
        docker-compose down 2>/dev/null || true

        echo -e "${BLUE}Iniciando containers com docker-compose...${NC}"
        docker-compose up -d

        echo -e "${BLUE}Aguardando inicialização...${NC}"
        sleep 10

        if docker-compose ps | grep -q "Up"; then
            echo -e "${GREEN}OK: Deploy Docker concluído com sucesso!${NC}"
            echo -e "${GREEN}Dashboard disponível em: http://localhost:8501${NC}"
            echo -e "${BLUE}Para ver logs: docker-compose logs -f${NC}"
            echo -e "${BLUE}Para parar: docker-compose down${NC}"
        else
            echo -e "${RED}ERRO: Erro no deploy via docker-compose${NC}"
            docker-compose logs || true
            exit 1
        fi
    else
        echo -e "${YELLOW}AVISO: docker-compose.yml não encontrado. Executando container simples.${NC}"
        docker run -d -p 8501:8501 --name $PROJECT_NAME $PROJECT_NAME:latest
        echo -e "${GREEN}Dashboard disponível em: http://localhost:8501${NC}"
    fi
}

deploy_cloud() {
    echo -e "${BLUE}Deploy em Cloud...${NC}"

    case "$CLOUD_PROVIDER" in
        "aws")
            deploy_aws
            ;;
        "gcp")
            deploy_gcp
            ;;
        "azure")
            deploy_azure
            ;;
        *)
            echo -e "${RED}ERRO: Provedor de cloud não especificado ou inválido${NC}"
            echo "Uso: $0 cloud [aws|gcp|azure]"
            exit 1
            ;;
    esac
}

deploy_aws() {
    echo -e "${BLUE}Deploy AWS...${NC}"

    if ! command -v aws &> /dev/null; then
        echo -e "${RED}ERRO: AWS CLI não encontrado${NC}"
        exit 1
    fi

    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${RED}ERRO: Variável de ambiente AWS_ACCOUNT_ID não definida${NC}"
        echo "Defina: export AWS_ACCOUNT_ID=SEU_ID_DE_CONTA_AWS"
        exit 1
    fi

    local ECR_REPO="$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/$PROJECT_NAME"

    echo -e "${BLUE}Realizando login no ECR...${NC}"
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"

    echo -e "${BLUE}Build e push da imagem para ECR...${NC}"
    docker build -t $PROJECT_NAME:$VERSION .
    docker tag $PROJECT_NAME:$VERSION "$ECR_REPO:latest"
    docker push "$ECR_REPO:latest"

    echo -e "${GREEN}OK: Imagem publicada no ECR: $ECR_REPO:latest${NC}"
    echo -e "${YELLOW}⚠ Deploy em ECS/EC2/EKS deve ser configurado separadamente.${NC}"
}

deploy_gcp() {
    echo -e "${BLUE}Deploy GCP (Cloud Run)...${NC}"

    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}ERRO: gcloud CLI não encontrado${NC}"
        exit 1
    fi

    gcloud run deploy "$PROJECT_NAME" \
        --source . \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated

    echo -e "${GREEN}OK: Deploy GCP concluído${NC}"
    echo -e "${YELLOW}Consulte o Console GCP para obter a URL pública do serviço.${NC}"
}

deploy_azure() {
    echo -e "${BLUE}Deploy Azure (Container Instances)...${NC}"

    if ! command -v az &> /dev/null; then
        echo -e "${RED}ERRO: Azure CLI não encontrado${NC}"
        exit 1
    fi

    echo -e "${YELLOW}⚠ Certifique-se de que a imagem $PROJECT_NAME:latest está publicada em um registry acessível pelo Azure.${NC}"

    az container create \
        --resource-group fraud-detection-rg \
        --name "$PROJECT_NAME" \
        --image "$PROJECT_NAME:latest" \
        --dns-name-label "$PROJECT_NAME-$(date +%s)" \
        --ports 8501

    echo -e "${GREEN}OK: Deploy Azure concluído${NC}"
}

show_help() {
    echo "Uso: $0 [AMBIENTE] [OPÇÕES]"
    echo
    echo "Ambientes:"
    echo "  local           Deploy local (padrão)"
    echo "  docker          Deploy com Docker/Docker Compose"
    echo "  cloud [prov]    Deploy em cloud (aws|gcp|azure)"
    echo
    echo "Exemplos:"
    echo "  $0                     # Deploy local"
    echo "  $0 local               # Deploy local"
    echo "  $0 docker              # Deploy com Docker"
    echo "  $0 cloud aws           # Deploy na AWS"
    echo "  $0 cloud gcp           # Deploy no GCP"
    echo "  $0 cloud azure         # Deploy no Azure"
    echo
    echo "Opções:"
    echo "  -h, --help             Mostra esta ajuda"
    echo "  -v, --version          Mostra versão do script"
}

main() {
    case "$ENVIRONMENT" in
        "local")
            deploy_local
            ;;
        "docker")
            deploy_docker
            ;;
        "cloud")
            deploy_cloud
            ;;
        "-h"|"--help")
            show_help
            ;;
        "-v"|"--version")
            echo "Deploy Script - versão ${VERSION}"
            ;;
        *)
            echo -e "${RED}ERRO: Ambiente inválido: $ENVIRONMENT${NC}"
            show_help
            exit 1
            ;;
    esac
}

trap 'echo -e "\n${YELLOW}Deploy interrompido pelo usuário${NC}"; exit 130' INT TERM

main "$@"
