"""
Módulo de Dados

Contém funcionalidades para:
- Geração de dados sintéticos
- Carregamento e preprocessamento de dados
- Validação de qualidade dos dados
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor

__all__ = [
    "SyntheticDataGenerator",
    "DataLoader", 
    "DataPreprocessor"
]
