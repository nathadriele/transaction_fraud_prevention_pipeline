#!/usr/bin/env python3
"""
Script de inicialização automática do Dashboard de Prevenção de Fraudes.
Configura o ambiente e inicia o dashboard automaticamente.
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Exibe banner do sistema."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           SISTEMA DE PREVENÇÃO DE FRAUDES                   ║
    ║                                                              ║
    ║              Dashboard Interativo - Streamlit               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Verifica se a versão do Python é adequada."""
    print("Verificando versão do Python...")

    if sys.version_info < (3, 8):
        print("ERRO: Python 3.8+ é necessário")
        print(f"   Versão atual: {sys.version}")
        return False

    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def setup_environment():
    """Configura o ambiente Python."""
    print("\nConfigurando ambiente...")

    # Adiciona paths necessários
    project_root = Path(__file__).parent
    src_path = project_root / 'src'

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"OK: Adicionado ao path: {project_root}")

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"OK: Adicionado ao path: {src_path}")

    return True

def check_dependencies():
    """Verifica se as dependências estão instaladas."""
    print("\nVerificando dependências...")

    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('scikit-learn', 'sklearn')
    ]

    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"OK: {package_name}")
        except ImportError:
            print(f"AUSENTE: {package_name}")
            missing.append(package_name)

    if missing:
        print(f"\nInstalando dependências ausentes...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing)
            print("OK: Dependências instaladas com sucesso!")
        except subprocess.CalledProcessError:
            print("ERRO: Erro ao instalar dependências")
            print("Execute manualmente: pip install " + " ".join(missing))
            return False

    return True

def check_data_files():
    """Verifica se os arquivos de dados existem."""
    print("\nVerificando arquivos de dados...")

    project_root = Path(__file__).parent
    required_files = [
        'data/final_results.csv',
        'data/eda_summary.csv',
        'data/alerts_log.csv',
        'data/model_scores.json'
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"OK: {file_path}")
        else:
            print(f"AUSENTE: {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print("\nAlguns arquivos de dados estão ausentes.")
        print("O dashboard usará dados de exemplo quando necessário.")

    return True

def start_streamlit():
    """Inicia o servidor Streamlit."""
    print("\nIniciando dashboard...")

    project_root = Path(__file__).parent
    app_path = project_root / 'src' / 'dashboard' / 'app.py'

    if not app_path.exists():
        print(f"ERRO: Arquivo do dashboard não encontrado: {app_path}")
        return False

    try:
        print("Executando: streamlit run src/dashboard/app.py")
        print("O dashboard será aberto em: http://localhost:8501")
        print("\nAguarde alguns segundos para o servidor inicializar...")
        
        # Inicia o Streamlit em background
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_path),
            '--server.headless', 'true',
            '--server.port', '8501'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Aguarda alguns segundos para o servidor inicializar
        time.sleep(5)
        
        # Verifica se o processo ainda está rodando
        if process.poll() is None:
            print("OK: Dashboard iniciado com sucesso!")

            # Abre o navegador automaticamente
            try:
                webbrowser.open('http://localhost:8501')
                print("Navegador aberto automaticamente")
            except:
                print("AVISO: Não foi possível abrir o navegador automaticamente")
                print("Acesse manualmente: http://localhost:8501")

            print("\n" + "="*60)
            print("DASHBOARD EXECUTANDO COM SUCESSO!")
            print("="*60)
            print("URL: http://localhost:8501")
            print("Para parar: Ctrl+C")
            print("Documentação: docs/TECHNICAL_DOCUMENTATION.md")
            print("="*60)
            
            # Mantém o processo rodando
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n\nParando dashboard...")
                process.terminate()
                print("OK: Dashboard parado com sucesso!")

            return True
        else:
            print("ERRO: Erro ao iniciar o dashboard")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Erro: {stderr.decode()}")
            return False

    except FileNotFoundError:
        print("ERRO: Streamlit não encontrado")
        print("Instale com: pip install streamlit")
        return False
    except Exception as e:
        print(f"ERRO: Erro inesperado: {e}")
        return False

def main():
    """Função principal."""
    print_banner()
    
    # Executa verificações
    checks = [
        ("Versão do Python", check_python_version),
        ("Configuração do ambiente", setup_environment),
        ("Dependências", check_dependencies),
        ("Arquivos de dados", check_data_files)
    ]
    
    print("Executando verificações preliminares...\n")

    for check_name, check_func in checks:
        print(f"Verificando {check_name}...")
        if not check_func():
            print(f"\nERRO: Falha na verificação: {check_name}")
            print("Corrija os problemas e tente novamente")
            return False
        print()

    print("OK: Todas as verificações passaram!")
    
    # Inicia o dashboard
    if start_streamlit():
        return True
    else:
        print("\nERRO: Falha ao iniciar o dashboard")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nOperação cancelada pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\nERRO crítico: {e}")
        sys.exit(1)
