#!/bin/bash

# Script de inicialização do Dashboard de Prevenção de Fraudes
# Para Linux/Mac

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           SISTEMA DE PREVENÇÃO DE FRAUDES                   ║"
echo "║                                                              ║"
echo "║              Dashboard Interativo - Streamlit               ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo

echo -e "${BLUE}Iniciando dashboard...${NC}"
echo

# Verifica se Python está instalado
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}ERRO: Python não encontrado!${NC}"
        echo -e "${YELLOW}Instale Python 3.8+ em: https://python.org${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}OK: Python encontrado${NC}"
echo

# Verifica versão do Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${BLUE}Versão do Python: $PYTHON_VERSION${NC}"

# Executa o script de inicialização
echo -e "${BLUE}Executando script de inicialização...${NC}"
echo

$PYTHON_CMD start_dashboard.py

if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}ERRO: Erro ao executar o dashboard${NC}"
    echo -e "${YELLOW}Verifique os logs acima para mais detalhes${NC}"
    exit 1
fi

echo
echo -e "${GREEN}OK: Dashboard finalizado${NC}"
