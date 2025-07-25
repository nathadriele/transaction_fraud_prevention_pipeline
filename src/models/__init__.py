"""
Módulo de Modelos

Contém implementações de:
- Modelos supervisionados de ML
- Modelos não supervisionados para detecção de anomalias
- Regras de negócio
- Ensemble de modelos
"""

from .supervised_models import SupervisedModels
from .unsupervised_models import UnsupervisedModels
from .business_rules import BusinessRulesEngine
from .model_ensemble import ModelEnsemble

__all__ = [
    "SupervisedModels",
    "UnsupervisedModels",
    "BusinessRulesEngine",
    "ModelEnsemble"
]
