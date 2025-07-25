"""
Módulo de Avaliação

Contém funcionalidades para:
- Métricas de avaliação de modelos
- Validação cruzada
- Análise de performance
- Relatórios de avaliação
"""

from .metrics import ModelMetrics
from .validation import ModelValidator
from .reports import EvaluationReports

__all__ = [
    "ModelMetrics",
    "ModelValidator",
    "EvaluationReports"
]
