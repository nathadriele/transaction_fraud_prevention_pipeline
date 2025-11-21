@echo off
title Sistema de Prevenção de Fraudes - Dashboard

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               SISTEMA DE PREVENÇÃO DE FRAUDES                ║
echo ║              Dashboard Interativo - Streamlit                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo Iniciando dashboard...
echo.

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python não encontrado!
    echo Instale Python 3.8+ em: https://python.org
    pause
    exit /b 1
)

echo Python detectado
echo.

:: Executa dashboard
python start_dashboard.py
if errorlevel 1 (
    echo.
    echo ERRO: Falha ao iniciar o dashboard
    pause
    exit /b 1
)

echo.
echo Dashboard encerrado
pause
