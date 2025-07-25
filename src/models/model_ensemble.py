"""
Ensemble de Modelos para Detecção de Fraudes

Combina múltiplos modelos para melhorar a performance de detecção.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import save_model, load_model

logger = get_logger(__name__)


class ModelEnsemble:
    """
    Ensemble de modelos para detecção de fraudes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o ensemble.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        self.models = {}
        self.ensemble_model = None
        self.is_fitted = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Adiciona um modelo ao ensemble.
        
        Args:
            name: Nome do modelo
            model: Modelo treinado
            weight: Peso do modelo no ensemble
        """
        self.models[name] = {
            'model': model,
            'weight': weight
        }
        logger.info(f"Modelo {name} adicionado ao ensemble com peso {weight}")
    
    def create_voting_ensemble(self, voting: str = 'soft'):
        """
        Cria ensemble usando VotingClassifier.
        
        Args:
            voting: Tipo de votação ('hard' ou 'soft')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn é necessário para ensemble")
        
        if len(self.models) < 2:
            raise ValueError("Pelo menos 2 modelos são necessários para ensemble")
        
        # Prepara lista de modelos para VotingClassifier
        estimators = [(name, info['model']) for name, info in self.models.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        logger.info(f"Ensemble criado com {len(self.models)} modelos usando votação {voting}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina o ensemble.
        
        Args:
            X: Features de treino
            y: Target de treino
        """
        if self.ensemble_model is None:
            self.create_voting_ensemble()
        
        logger.info("Treinando ensemble...")
        self.ensemble_model.fit(X, y)
        self.is_fitted = True
        logger.info("Ensemble treinado com sucesso!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições usando o ensemble.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com predições
        """
        if not self.is_fitted:
            raise ValueError("Ensemble não foi treinado")
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades de predição.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com probabilidades
        """
        if not self.is_fitted:
            raise ValueError("Ensemble não foi treinado")
        
        if hasattr(self.ensemble_model, 'predict_proba'):
            return self.ensemble_model.predict_proba(X)
        else:
            raise ValueError("Modelo não suporta predict_proba")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Avalia o ensemble.
        
        Args:
            X: Features de teste
            y: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        if not self.is_fitted:
            raise ValueError("Ensemble não foi treinado")
        
        # Predições
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1] if hasattr(self.ensemble_model, 'predict_proba') else None
        
        # Métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        return metrics
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Retorna pesos dos modelos no ensemble.
        
        Returns:
            Dicionário com pesos
        """
        return {name: info['weight'] for name, info in self.models.items()}
    
    def save_ensemble(self, filepath: str):
        """
        Salva o ensemble.
        
        Args:
            filepath: Caminho para salvar
        """
        if not self.is_fitted:
            raise ValueError("Ensemble não foi treinado")
        
        save_model(self.ensemble_model, filepath)
        logger.info(f"Ensemble salvo em: {filepath}")
    
    def load_ensemble(self, filepath: str):
        """
        Carrega ensemble salvo.
        
        Args:
            filepath: Caminho do arquivo
        """
        self.ensemble_model = load_model(filepath)
        self.is_fitted = True
        logger.info(f"Ensemble carregado de: {filepath}")


def create_simple_ensemble(models_dict: Dict[str, Any]) -> ModelEnsemble:
    """
    Cria ensemble simples a partir de dicionário de modelos.
    
    Args:
        models_dict: Dicionário com modelos {nome: modelo}
        
    Returns:
        Ensemble configurado
    """
    ensemble = ModelEnsemble()
    
    for name, model in models_dict.items():
        ensemble.add_model(name, model)
    
    ensemble.create_voting_ensemble()
    
    return ensemble


def main():
    """Função principal para teste do ensemble."""
    
    print("🤖 Testando Model Ensemble...")
    
    # Exemplo de uso básico
    ensemble = ModelEnsemble()
    
    print(f"✅ Ensemble inicializado")
    print(f"📊 Modelos no ensemble: {len(ensemble.models)}")
    
    # Simula adição de modelos
    class DummyModel:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            return proba / proba.sum(axis=1, keepdims=True)
    
    ensemble.add_model("model1", DummyModel())
    ensemble.add_model("model2", DummyModel())
    
    print(f"✅ Modelos adicionados: {list(ensemble.models.keys())}")
    print(f"📊 Pesos: {ensemble.get_model_weights()}")


if __name__ == "__main__":
    main()
