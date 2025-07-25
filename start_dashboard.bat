@echo off
title Sistema de Prevenção de Fraudes - Dashboard

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║           SISTEMA DE PREVENÇÃO DE FRAUDES                   ║
echo ║                                                              ║
echo ║              Dashboard Interativo - Streamlit               ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo Iniciando dashboard...
echo.

REM Verifica se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python não encontrado!
    echo Instale Python 3.8+ em: https://python.org
    pause
    exit /b 1
)

echo OK: Python encontrado
echo.

REM Executa o script de inicialização
echo Executando script de inicialização...
python start_dashboard.py

if errorlevel 1 (
    echo.
    echo ERRO: Erro ao executar o dashboard
    echo Verifique os logs acima para mais detalhes
    pause
    exit /b 1
)

echo.
echo OK: Dashboard finalizado
pause
