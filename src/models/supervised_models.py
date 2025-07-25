"""
Modelos Supervisionados para Detecção de Fraudes

Implementa modelos de machine learning supervisionados para classificação
de transações fraudulentas vs legítimas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importações de ML
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import save_model, load_model, ensure_dir

logger = get_logger(__name__)


class SupervisedModels:
    """
    Classe para gerenciar modelos supervisionados de detecção de fraudes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa os modelos supervisionados.
        
        Args:
            config: Configurações do sistema
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn é necessário para modelos supervisionados")
        
        self.config = config or load_config()
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        
        # Inicializa modelos
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa os modelos com configurações."""
        
        # Random Forest
        rf_config = self.config['models']['supervised']['random_forest']
        self.models['random_forest'] = RandomForestClassifier(**rf_config)
        
        # Logistic Regression
        lr_config = self.config['models']['supervised']['logistic_regression']
        self.models['logistic_regression'] = LogisticRegression(**lr_config)
        
        # XGBoost (se disponível)
        if XGBOOST_AVAILABLE:
            xgb_config = self.config['models']['supervised']['xgboost']
            self.models['xgboost'] = xgb.XGBClassifier(**xgb_config)
        
        logger.info(f"Modelos inicializados: {list(self.models.keys())}")
    
    def balance_dataset(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balanceia o dataset para lidar com classes desbalanceadas.
        
        Args:
            X: Features
            y: Target
            method: Método de balanceamento ('smote', 'undersample', 'none')
            
        Returns:
            Tuple com (X_balanced, y_balanced)
        """
        logger.info(f"Balanceando dataset usando método: {method}")
        
        original_fraud_rate = y.mean()
        logger.info(f"Taxa de fraude original: {original_fraud_rate:.2%}")
        
        if method == 'smote':
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
            except Exception as e:
                logger.warning(f"Erro no SMOTE: {e}. Usando dados originais.")
                return X, y
                
        elif method == 'undersample':
            try:
                undersampler = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = undersampler.fit_resample(X, y)
            except Exception as e:
                logger.warning(f"Erro no undersampling: {e}. Usando dados originais.")
                return X, y
                
        else:
            return X, y
        
        new_fraud_rate = y_balanced.mean()
        logger.info(f"Taxa de fraude após balanceamento: {new_fraud_rate:.2%}")
        logger.info(f"Tamanho do dataset: {len(X)} -> {len(X_balanced)}")
        
        return X_balanced, y_balanced
    
    def train_models(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    balance_method: str = 'smote',
                    use_cross_validation: bool = True) -> Dict[str, Any]:
        """
        Treina todos os modelos.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            balance_method: Método de balanceamento
            use_cross_validation: Se deve usar validação cruzada
            
        Returns:
            Dicionário com resultados do treinamento
        """
        logger.info("Iniciando treinamento dos modelos...")
        
        # Balanceia dataset se necessário
        if balance_method != 'none':
            X_train_balanced, y_train_balanced = self.balance_dataset(
                X_train, y_train, balance_method
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Treinando modelo: {model_name}")
            
            try:
                # Treina o modelo
                model.fit(X_train_balanced, y_train_balanced)
                self.trained_models[model_name] = model
                
                # Validação cruzada
                if use_cross_validation:
                    cv_scores = cross_val_score(
                        model, X_train_balanced, y_train_balanced,
                        cv=self.config['evaluation']['cross_validation']['cv_folds'],
                        scoring=self.config['evaluation']['cross_validation']['scoring']
                    )
                    
                    training_results[model_name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        logger.info("Treinamento concluído!")
        return training_results
    
    def evaluate_models(self, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, Dict]:
        """
        Avalia todos os modelos treinados.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info("Avaliando modelos...")
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            logger.info(f"Avaliando modelo: {model_name}")
            
            try:
                # Predições
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Métricas básicas
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred)
                }
                
                # AUC-ROC se probabilidades disponíveis
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                # Matriz de confusão
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = {
                    'tn': int(cm[0, 0]),
                    'fp': int(cm[0, 1]),
                    'fn': int(cm[1, 0]),
                    'tp': int(cm[1, 1])
                }
                
                # Relatório de classificação
                metrics['classification_report'] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, AUC: {metrics.get('roc_auc', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Erro ao avaliar {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        self.model_scores = evaluation_results
        return evaluation_results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict:
        """
        Obtém importância das features para um modelo.
        
        Args:
            model_name: Nome do modelo
            feature_names: Lista com nomes das features
            
        Returns:
            Dicionário com importância das features
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Modelo {model_name} não suporta importância de features")
            return {}
        
        # Cria dicionário ordenado por importância
        feature_importance = dict(zip(feature_names, importances))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def predict(self, 
               X: pd.DataFrame, 
               model_name: str = None,
               return_probabilities: bool = False) -> np.ndarray:
        """
        Faz predições usando um modelo específico ou o melhor modelo.
        
        Args:
            X: Features para predição
            model_name: Nome do modelo (se None, usa o melhor)
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Array com predições
        """
        if model_name is None:
            model_name = self.get_best_model()
        
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
    
    def get_best_model(self) -> str:
        """
        Retorna o nome do melhor modelo baseado no F1-score.
        
        Returns:
            Nome do melhor modelo
        """
        if not self.model_scores:
            raise ValueError("Nenhum modelo foi avaliado ainda")
        
        best_model = max(
            self.model_scores.keys(),
            key=lambda x: self.model_scores[x].get('f1_score', 0)
        )
        
        return best_model
    
    def save_models(self, output_dir: str = None) -> Dict[str, str]:
        """
        Salva todos os modelos treinados.
        
        Args:
            output_dir: Diretório de saída
            
        Returns:
            Dicionário com caminhos dos modelos salvos
        """
        if output_dir is None:
            output_dir = Path(self.config.get('models', {}).get('model_path', 'models'))
        
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        saved_paths = {}
        
        for model_name, model in self.trained_models.items():
            model_path = output_path / f"{model_name}_fraud_detector.pkl"
            save_model(model, model_path)
            saved_paths[model_name] = str(model_path)
            logger.info(f"Modelo {model_name} salvo em: {model_path}")
        
        # Salva também as métricas
        if self.model_scores:
            from src.utils.helpers import save_json
            metrics_path = output_path / "model_metrics.json"
            save_json(self.model_scores, metrics_path)
            saved_paths['metrics'] = str(metrics_path)
        
        return saved_paths
    
    def load_models(self, model_dir: str) -> None:
        """
        Carrega modelos salvos.
        
        Args:
            model_dir: Diretório com modelos salvos
        """
        model_path = Path(model_dir)
        
        for model_file in model_path.glob("*_fraud_detector.pkl"):
            model_name = model_file.stem.replace('_fraud_detector', '')
            model = load_model(model_file)
            self.trained_models[model_name] = model
            logger.info(f"Modelo {model_name} carregado de: {model_file}")
        
        # Carrega métricas se disponível
        metrics_file = model_path / "model_metrics.json"
        if metrics_file.exists():
            from src.utils.helpers import load_json
            self.model_scores = load_json(metrics_file)


def main():
    """Função principal para teste dos modelos supervisionados."""
    
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    print("🤖 Testando Modelos Supervisionados...")
    
    # Carrega e preprocessa dados
    loader = DataLoader()
    users_df, transactions_df = loader.load_synthetic_data()
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(transactions_df)
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(processed_df)
    
    # Treina modelos
    models = SupervisedModels()
    training_results = models.train_models(X_train, y_train)
    
    # Avalia modelos
    evaluation_results = models.evaluate_models(X_test, y_test)
    
    # Salva modelos
    saved_paths = models.save_models()
    
    # Mostra resultados
    print("✅ Modelos treinados e avaliados!")
    print(f"🏆 Melhor modelo: {models.get_best_model()}")
    
    for model_name, metrics in evaluation_results.items():
        if 'error' not in metrics:
            print(f"📊 {model_name}: F1={metrics['f1_score']:.4f}, AUC={metrics.get('roc_auc', 'N/A')}")


if __name__ == "__main__":
    main()
