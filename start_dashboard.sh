#!/bin/bash

# Script de inicialização do Dashboard (Linux/Mac)

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           SISTEMA DE PREVENÇÃO DE FRAUDES                   ║"
echo "║              Dashboard Interativo - Streamlit               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo -e "${BLUE}Iniciando dashboard...${NC}\n"

# Verificação do Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERRO: Python não encontrado${NC}"
    echo -e "${YELLOW}Instale Python 3.8+: https://python.org${NC}"
    exit 1
fi

echo -e "${GREEN}Python encontrado: $($PYTHON_CMD --version)${NC}\n"

# Execução
echo -e "${BLUE}Executando dashboard...${NC}\n"
$PYTHON_CMD start_dashboard.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao iniciar o dashboard${NC}"
    echo -e "${YELLOW}Verifique os logs exibidos acima${NC}"
    exit 1
fi

echo -e "\n${GREEN}Dashboard finalizado com sucesso${NC}"
