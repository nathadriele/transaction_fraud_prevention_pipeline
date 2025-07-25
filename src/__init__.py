"""
Sistema de Prevenção de Fraudes Transacionais

Este pacote contém todos os módulos necessários para detecção e prevenção
de fraudes em transações financeiras.
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__email__ = "fraud-prevention@company.com"

# Importações principais
from src.utils.config import load_config
from src.utils.logger import get_logger

# Configuração global
config = load_config()
logger = get_logger(__name__)

__all__ = [
    "config",
    "logger",
    "load_config",
    "get_logger"
]
