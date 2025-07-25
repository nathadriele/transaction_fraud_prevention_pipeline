"""
Modelos Não Supervisionados para Detecção de Anomalias

Implementa modelos de detecção de anomalias para identificar
transações suspeitas sem usar labels de fraude.
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

# Importações de ML
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Define Model como None para evitar erros
    Model = None

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import save_model, load_model, ensure_dir

logger = get_logger(__name__)


class UnsupervisedModels:
    """
    Classe para gerenciar modelos não supervisionados de detecção de anomalias.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa os modelos não supervisionados.
        
        Args:
            config: Configurações do sistema
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn é necessário para modelos não supervisionados")
        
        self.config = config or load_config()
        self.models = {}
        self.trained_models = {}
        self.scalers = {}
        
        # Inicializa modelos
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa os modelos com configurações."""
        
        # Isolation Forest
        if_config = self.config['models']['unsupervised']['isolation_forest']
        self.models['isolation_forest'] = IsolationForest(**if_config)
        
        # Local Outlier Factor
        lof_config = self.config['models']['unsupervised']['local_outlier_factor']
        self.models['local_outlier_factor'] = LocalOutlierFactor(**lof_config)
        
        # DBSCAN para clustering
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        
        # K-Means para clustering
        self.models['kmeans'] = KMeans(n_clusters=5, random_state=42)
        
        logger.info(f"Modelos não supervisionados inicializados: {list(self.models.keys())}")
    
    def prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados para modelos não supervisionados.
        
        Args:
            X: Features originais
            
        Returns:
            Features preparadas
        """
        # Remove colunas não numéricas se existirem
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Normaliza os dados
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric),
            columns=X_numeric.columns,
            index=X_numeric.index
        )
        
        self.scalers['standard'] = scaler
        
        return X_scaled
    
    def train_isolation_forest(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina modelo Isolation Forest.
        
        Args:
            X: Features de treino
            
        Returns:
            Resultados do treinamento
        """
        logger.info("Treinando Isolation Forest...")
        
        X_prepared = self.prepare_data(X)
        
        model = self.models['isolation_forest']
        model.fit(X_prepared)
        
        # Predições (1 = normal, -1 = anomalia)
        predictions = model.predict(X_prepared)
        anomaly_scores = model.decision_function(X_prepared)
        
        # Converte para formato binário (1 = anomalia, 0 = normal)
        anomalies = (predictions == -1).astype(int)
        
        self.trained_models['isolation_forest'] = model
        
        results = {
            'model_name': 'isolation_forest',
            'anomaly_rate': anomalies.mean(),
            'total_anomalies': anomalies.sum(),
            'score_mean': anomaly_scores.mean(),
            'score_std': anomaly_scores.std()
        }
        
        logger.info(f"Isolation Forest - Taxa de anomalias: {results['anomaly_rate']:.2%}")
        
        return results
    
    def train_local_outlier_factor(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina modelo Local Outlier Factor.
        
        Args:
            X: Features de treino
            
        Returns:
            Resultados do treinamento
        """
        logger.info("Treinando Local Outlier Factor...")
        
        X_prepared = self.prepare_data(X)
        
        model = self.models['local_outlier_factor']
        
        # LOF não tem método fit separado, usa fit_predict
        predictions = model.fit_predict(X_prepared)
        outlier_scores = model.negative_outlier_factor_
        
        # Converte para formato binário (1 = anomalia, 0 = normal)
        anomalies = (predictions == -1).astype(int)
        
        self.trained_models['local_outlier_factor'] = model
        
        results = {
            'model_name': 'local_outlier_factor',
            'anomaly_rate': anomalies.mean(),
            'total_anomalies': anomalies.sum(),
            'score_mean': outlier_scores.mean(),
            'score_std': outlier_scores.std()
        }
        
        logger.info(f"LOF - Taxa de anomalias: {results['anomaly_rate']:.2%}")
        
        return results
    
    def train_clustering_models(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina modelos de clustering para detecção de anomalias.
        
        Args:
            X: Features de treino
            
        Returns:
            Resultados do treinamento
        """
        logger.info("Treinando modelos de clustering...")
        
        X_prepared = self.prepare_data(X)
        
        results = {}
        
        # DBSCAN
        dbscan = self.models['dbscan']
        dbscan_labels = dbscan.fit_predict(X_prepared)
        
        # Pontos com label -1 são considerados outliers
        dbscan_anomalies = (dbscan_labels == -1).astype(int)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        self.trained_models['dbscan'] = dbscan
        
        results['dbscan'] = {
            'model_name': 'dbscan',
            'anomaly_rate': dbscan_anomalies.mean(),
            'total_anomalies': dbscan_anomalies.sum(),
            'n_clusters': n_clusters_dbscan,
            'noise_points': (dbscan_labels == -1).sum()
        }
        
        # K-Means
        kmeans = self.models['kmeans']
        kmeans_labels = kmeans.fit_predict(X_prepared)
        
        # Calcula distâncias aos centroides
        distances = np.min(kmeans.transform(X_prepared), axis=1)
        
        # Define anomalias como pontos mais distantes (top 5%)
        threshold = np.percentile(distances, 95)
        kmeans_anomalies = (distances > threshold).astype(int)
        
        # Silhouette score para avaliar qualidade do clustering
        try:
            silhouette_avg = silhouette_score(X_prepared, kmeans_labels)
        except:
            silhouette_avg = None
        
        self.trained_models['kmeans'] = kmeans
        
        results['kmeans'] = {
            'model_name': 'kmeans',
            'anomaly_rate': kmeans_anomalies.mean(),
            'total_anomalies': kmeans_anomalies.sum(),
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_
        }
        
        logger.info(f"DBSCAN - Clusters: {n_clusters_dbscan}, Anomalias: {results['dbscan']['anomaly_rate']:.2%}")
        logger.info(f"K-Means - Silhouette: {silhouette_avg:.4f if silhouette_avg else 'N/A'}, Anomalias: {results['kmeans']['anomaly_rate']:.2%}")
        
        return results
    
    def create_autoencoder(self, input_dim: int):
        """
        Cria modelo autoencoder para detecção de anomalias.
        
        Args:
            input_dim: Dimensão de entrada
            
        Returns:
            Modelo autoencoder
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow é necessário para autoencoders")
        
        encoding_dim = self.config['models']['unsupervised']['autoencoder']['encoding_dim']
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        
        # Decoder
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        
        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_autoencoder(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina autoencoder para detecção de anomalias.
        
        Args:
            X: Features de treino
            
        Returns:
            Resultados do treinamento
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow não disponível. Pulando autoencoder.")
            return {}
        
        logger.info("Treinando Autoencoder...")
        
        X_prepared = self.prepare_data(X)
        
        # Cria e treina autoencoder
        autoencoder = self.create_autoencoder(X_prepared.shape[1])
        
        config = self.config['models']['unsupervised']['autoencoder']
        history = autoencoder.fit(
            X_prepared, X_prepared,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.1,
            verbose=0
        )
        
        # Calcula erros de reconstrução
        reconstructed = autoencoder.predict(X_prepared, verbose=0)
        mse = np.mean(np.power(X_prepared - reconstructed, 2), axis=1)
        
        # Define threshold para anomalias (95º percentil)
        threshold = np.percentile(mse, 95)
        anomalies = (mse > threshold).astype(int)
        
        self.trained_models['autoencoder'] = autoencoder
        
        results = {
            'model_name': 'autoencoder',
            'anomaly_rate': anomalies.mean(),
            'total_anomalies': anomalies.sum(),
            'reconstruction_error_mean': mse.mean(),
            'reconstruction_error_std': mse.std(),
            'threshold': threshold,
            'final_loss': history.history['loss'][-1]
        }
        
        logger.info(f"Autoencoder - Taxa de anomalias: {results['anomaly_rate']:.2%}")
        
        return results
    
    def train_all_models(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina todos os modelos não supervisionados.
        
        Args:
            X: Features de treino
            
        Returns:
            Resultados de todos os modelos
        """
        logger.info("Treinando todos os modelos não supervisionados...")
        
        all_results = {}
        
        # Isolation Forest
        try:
            all_results['isolation_forest'] = self.train_isolation_forest(X)
        except Exception as e:
            logger.error(f"Erro no Isolation Forest: {e}")
            all_results['isolation_forest'] = {'error': str(e)}
        
        # Local Outlier Factor
        try:
            all_results['local_outlier_factor'] = self.train_local_outlier_factor(X)
        except Exception as e:
            logger.error(f"Erro no LOF: {e}")
            all_results['local_outlier_factor'] = {'error': str(e)}
        
        # Clustering
        try:
            clustering_results = self.train_clustering_models(X)
            all_results.update(clustering_results)
        except Exception as e:
            logger.error(f"Erro no clustering: {e}")
            all_results['clustering'] = {'error': str(e)}
        
        # Autoencoder
        try:
            autoencoder_results = self.train_autoencoder(X)
            if autoencoder_results:
                all_results['autoencoder'] = autoencoder_results
        except Exception as e:
            logger.error(f"Erro no autoencoder: {e}")
            all_results['autoencoder'] = {'error': str(e)}
        
        logger.info("Treinamento de modelos não supervisionados concluído!")
        
        return all_results
    
    def predict_anomalies(self, 
                         X: pd.DataFrame, 
                         model_name: str) -> np.ndarray:
        """
        Prediz anomalias usando um modelo específico.
        
        Args:
            X: Features para predição
            model_name: Nome do modelo
            
        Returns:
            Array com predições de anomalias (1 = anomalia, 0 = normal)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        # Prepara dados usando o mesmo scaler do treino
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = pd.DataFrame(
            self.scalers['standard'].transform(X_numeric),
            columns=X_numeric.columns,
            index=X_numeric.index
        )
        
        model = self.trained_models[model_name]
        
        if model_name == 'isolation_forest':
            predictions = model.predict(X_scaled)
            return (predictions == -1).astype(int)
        
        elif model_name == 'local_outlier_factor':
            # LOF precisa ser retreinado com novos dados
            logger.warning("LOF não suporta predição em novos dados. Use apenas para dados de treino.")
            return np.zeros(len(X))
        
        elif model_name == 'dbscan':
            # DBSCAN não tem predict, usa fit_predict
            predictions = model.fit_predict(X_scaled)
            return (predictions == -1).astype(int)
        
        elif model_name == 'kmeans':
            # Calcula distâncias aos centroides
            distances = np.min(model.transform(X_scaled), axis=1)
            threshold = np.percentile(distances, 95)  # Mesmo threshold do treino
            return (distances > threshold).astype(int)
        
        elif model_name == 'autoencoder' and TENSORFLOW_AVAILABLE:
            reconstructed = model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
            threshold = np.percentile(mse, 95)  # Mesmo threshold do treino
            return (mse > threshold).astype(int)
        
        else:
            raise ValueError(f"Modelo {model_name} não suportado para predição")
    
    def evaluate_with_labels(self, 
                           X: pd.DataFrame, 
                           y_true: pd.Series) -> Dict[str, Dict]:
        """
        Avalia modelos usando labels verdadeiros (se disponível).
        
        Args:
            X: Features
            y_true: Labels verdadeiros de fraude
            
        Returns:
            Métricas de avaliação para cada modelo
        """
        logger.info("Avaliando modelos não supervisionados com labels...")
        
        evaluation_results = {}
        
        for model_name in self.trained_models.keys():
            try:
                y_pred = self.predict_anomalies(X, model_name)
                
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                metrics = {
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred),
                    'f1_score': f1_score(y_true, y_pred),
                    'anomaly_rate': y_pred.mean()
                }
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao avaliar {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results


def main():
    """Função principal para teste dos modelos não supervisionados."""
    
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    print("🔍 Testando Modelos Não Supervisionados...")
    
    # Carrega e preprocessa dados
    loader = DataLoader()
    users_df, transactions_df = loader.load_synthetic_data()
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(transactions_df)
    
    # Remove target para treino não supervisionado
    X = processed_df.drop(columns=['is_fraud'] if 'is_fraud' in processed_df.columns else [])
    y = processed_df['is_fraud'] if 'is_fraud' in processed_df.columns else None
    
    # Treina modelos
    models = UnsupervisedModels()
    training_results = models.train_all_models(X)
    
    # Avalia com labels se disponível
    if y is not None:
        evaluation_results = models.evaluate_with_labels(X, y)
        
        print("✅ Modelos não supervisionados treinados e avaliados!")
        
        for model_name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                print(f"📊 {model_name}: F1={metrics['f1_score']:.4f}, Anomalias={metrics['anomaly_rate']:.2%}")
    else:
        print("✅ Modelos não supervisionados treinados!")
        
        for model_name, results in training_results.items():
            if 'error' not in results:
                print(f"📊 {model_name}: Anomalias={results.get('anomaly_rate', 0):.2%}")


if __name__ == "__main__":
    main()
