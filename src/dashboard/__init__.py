"""
Módulo do Dashboard

Contém a aplicação Streamlit para:
- Interface de monitoramento
- Visualizações interativas
- Gestão de regras
- Relatórios executivos
"""

from .app import main as run_dashboard

__all__ = [
    "run_dashboard"
]
