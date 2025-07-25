#!/bin/bash

# Script de Deploy Automatizado - Sistema de Prevenção de Fraudes
# Suporta deploy local, Docker e cloud

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
ENVIRONMENT=${1:-local}
VERSION=$(git describe --tags --always 2>/dev/null || echo "v1.0.0")
PROJECT_NAME="fraud-prevention-system"

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           DEPLOY - SISTEMA DE PREVENÇÃO DE FRAUDES          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configurações do Deploy:${NC}"
echo "   Ambiente: $ENVIRONMENT"
echo "   Versão: $VERSION"
echo "   Projeto: $PROJECT_NAME"
echo

# Função para verificar pré-requisitos
check_prerequisites() {
    echo -e "${BLUE}Verificando pré-requisitos...${NC}"

    # Verifica Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERRO: Python 3 não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: Python 3 encontrado${NC}"

    # Verifica pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        echo -e "${RED}ERRO: pip não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: pip encontrado${NC}"

    # Verifica Git
    if ! command -v git &> /dev/null; then
        echo -e "${YELLOW}AVISO: Git não encontrado (opcional)${NC}"
    else
        echo -e "${GREEN}OK: Git encontrado${NC}"
    fi
}

# Função para instalar dependências
install_dependencies() {
    echo -e "${BLUE}Instalando dependências...${NC}"

    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        echo -e "${GREEN}OK: Dependências instaladas${NC}"
    else
        echo -e "${RED}ERRO: Arquivo requirements.txt não encontrado${NC}"
        exit 1
    fi
}

# Função para executar testes
run_tests() {
    echo -e "${BLUE}Executando testes...${NC}"

    if [ -d "tests" ]; then
        python3 -m pytest tests/ -v || {
            echo -e "${YELLOW}AVISO: Alguns testes falharam, mas continuando deploy...${NC}"
        }
    else
        echo -e "${YELLOW}AVISO: Diretório de testes não encontrado${NC}"
    fi

    # Testa importações básicas
    python3 -c "
import sys
sys.path.append('src')
try:
    from src.dashboard.app import load_final_results
    print('OK: Importações do dashboard OK')
except Exception as e:
    print(f'ERRO: Erro nas importações: {e}')
    sys.exit(1)
    "
}

# Função para deploy local
deploy_local() {
    echo -e "${BLUE}Deploy Local...${NC}"

    check_prerequisites
    install_dependencies
    run_tests

    echo -e "${BLUE}Iniciando dashboard...${NC}"
    echo -e "${GREEN}Dashboard será executado em: http://localhost:8501${NC}"
    echo -e "${YELLOW}Para parar: Ctrl+C${NC}"
    echo

    python3 start_dashboard.py
}

# Função para deploy com Docker
deploy_docker() {
    echo -e "${BLUE}Deploy com Docker...${NC}"

    # Verifica Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}ERRO: Docker não encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK: Docker encontrado${NC}"

    # Build da imagem
    echo -e "${BLUE}Building imagem Docker...${NC}"
    docker build -t $PROJECT_NAME:$VERSION .
    docker tag $PROJECT_NAME:$VERSION $PROJECT_NAME:latest
    echo -e "${GREEN}OK: Imagem criada: $PROJECT_NAME:$VERSION${NC}"

    # Para containers existentes
    echo -e "${BLUE}Parando containers existentes...${NC}"
    docker-compose down 2>/dev/null || true

    # Inicia containers
    echo -e "${BLUE}Iniciando containers...${NC}"
    docker-compose up -d

    # Aguarda inicialização
    echo -e "${BLUE}Aguardando inicialização...${NC}"
    sleep 10

    # Verifica status
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}OK: Deploy Docker concluído com sucesso!${NC}"
        echo -e "${GREEN}Dashboard disponível em: http://localhost:8501${NC}"
        echo -e "${BLUE}Para ver logs: docker-compose logs -f${NC}"
        echo -e "${BLUE}Para parar: docker-compose down${NC}"
    else
        echo -e "${RED}ERRO: Erro no deploy Docker${NC}"
        docker-compose logs
        exit 1
    fi
}

# Função para deploy em cloud
deploy_cloud() {
    echo -e "${BLUE}Deploy em Cloud...${NC}"

    case $2 in
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

# Função para deploy AWS
deploy_aws() {
    echo -e "${BLUE}Deploy AWS...${NC}"

    # Verifica AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}ERRO: AWS CLI não encontrado${NC}"
        exit 1
    fi

    # Build e push para ECR
    echo -e "${BLUE}Push para ECR...${NC}"
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
    docker tag $PROJECT_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/$PROJECT_NAME:latest
    docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/$PROJECT_NAME:latest

    echo -e "${GREEN}OK: Deploy AWS concluído${NC}"
}

# Função para deploy GCP
deploy_gcp() {
    echo -e "${BLUE}Deploy GCP...${NC}"

    # Verifica gcloud
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}ERRO: gcloud CLI não encontrado${NC}"
        exit 1
    fi

    # Deploy para Cloud Run
    gcloud run deploy $PROJECT_NAME \
        --source . \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated

    echo -e "${GREEN}OK: Deploy GCP concluído${NC}"
}

# Função para deploy Azure
deploy_azure() {
    echo -e "${BLUE}Deploy Azure...${NC}"

    # Verifica Azure CLI
    if ! command -v az &> /dev/null; then
        echo -e "${RED}ERRO: Azure CLI não encontrado${NC}"
        exit 1
    fi

    # Deploy para Container Instances
    az container create \
        --resource-group fraud-detection-rg \
        --name $PROJECT_NAME \
        --image $PROJECT_NAME:latest \
        --dns-name-label $PROJECT_NAME-$(date +%s) \
        --ports 8501

    echo -e "${GREEN}OK: Deploy Azure concluído${NC}"
}

# Função para mostrar ajuda
show_help() {
    echo "Uso: $0 [AMBIENTE] [OPÇÕES]"
    echo
    echo "Ambientes:"
    echo "  local     Deploy local (padrão)"
    echo "  docker    Deploy com Docker"
    echo "  cloud     Deploy em cloud (requer provedor)"
    echo
    echo "Exemplos:"
    echo "  $0                    # Deploy local"
    echo "  $0 local              # Deploy local"
    echo "  $0 docker             # Deploy com Docker"
    echo "  $0 cloud aws          # Deploy na AWS"
    echo "  $0 cloud gcp          # Deploy no GCP"
    echo "  $0 cloud azure        # Deploy no Azure"
    echo
    echo "Opções:"
    echo "  -h, --help           Mostra esta ajuda"
    echo "  -v, --version        Mostra versão"
}

# Função principal
main() {
    case $ENVIRONMENT in
        "local")
            deploy_local
            ;;
        "docker")
            deploy_docker
            ;;
        "cloud")
            deploy_cloud $@
            ;;
        "-h"|"--help")
            show_help
            ;;
        "-v"|"--version")
            echo "Deploy Script v1.0.0"
            ;;
        *)
            echo -e "${RED}ERRO: Ambiente inválido: $ENVIRONMENT${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Tratamento de sinais
trap 'echo -e "\n${YELLOW}Deploy interrompido pelo usuário${NC}"; exit 130' INT TERM

# Executa função principal
main $@
