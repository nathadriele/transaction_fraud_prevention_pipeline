"""
Preprocessador de Dados

Módulo responsável por preprocessar e preparar dados para modelagem.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import get_time_features

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocessador de dados para detecção de fraudes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o preprocessador.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        
        # Inicializa transformadores
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Features configuradas
        self.categorical_features = self.config['features']['categorical_features']
        self.numerical_features = self.config['features']['numerical_features']
        
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta transformadores e transforma os dados.
        
        Args:
            df: DataFrame a ser processado
            
        Returns:
            DataFrame transformado
        """
        logger.info("Ajustando transformadores e processando dados...")
        
        # Cria cópia dos dados
        df_processed = df.copy()
        
        # Engenharia de features
        df_processed = self._engineer_features(df_processed)
        
        # Tratamento de valores ausentes
        df_processed = self._handle_missing_values(df_processed, fit=True)
        
        # Codificação de variáveis categóricas
        df_processed = self._encode_categorical_features(df_processed, fit=True)
        
        # Normalização de features numéricas
        df_processed = self._scale_numerical_features(df_processed, fit=True)
        
        # Remove features desnecessárias
        df_processed = self._remove_unnecessary_features(df_processed)
        
        self.is_fitted = True
        logger.info(f"Dados processados: {df_processed.shape}")
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados usando transformadores já ajustados.
        
        Args:
            df: DataFrame a ser transformado
            
        Returns:
            DataFrame transformado
        """
        if not self.is_fitted:
            raise ValueError("Preprocessador não foi ajustado. Use fit_transform primeiro.")
        
        logger.info("Transformando dados...")
        
        # Cria cópia dos dados
        df_processed = df.copy()
        
        # Engenharia de features
        df_processed = self._engineer_features(df_processed)
        
        # Tratamento de valores ausentes
        df_processed = self._handle_missing_values(df_processed, fit=False)
        
        # Codificação de variáveis categóricas
        df_processed = self._encode_categorical_features(df_processed, fit=False)
        
        # Normalização de features numéricas
        df_processed = self._scale_numerical_features(df_processed, fit=False)
        
        # Remove features desnecessárias
        df_processed = self._remove_unnecessary_features(df_processed)
        
        logger.info(f"Dados transformados: {df_processed.shape}")
        
        return df_processed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria novas features."""
        
        logger.debug("Criando features de engenharia...")
        
        # Features temporais
        if 'timestamp' in df.columns:
            time_features = df['timestamp'].apply(get_time_features)
            time_df = pd.DataFrame(time_features.tolist())
            df = pd.concat([df, time_df], axis=1)
        
        # Features de valor
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
            
            # Z-score do valor
            df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Features de velocidade (se disponível)
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            df = self._create_velocity_features(df)
        
        # Features de agregação
        if self.config['features']['engineering']['create_aggregation_features']:
            df = self._create_aggregation_features(df)
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de velocidade de transações."""
        
        # Ordena por usuário e timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Tempo desde última transação
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff()
        df['time_since_last_transaction'] = df['time_since_last_transaction'].dt.total_seconds() / 3600  # em horas
        df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(24)  # 24h para primeira transação
        
        # Contagem de transações por período
        try:
            df['transactions_last_hour'] = df.groupby('user_id').rolling('1h', on='timestamp')['amount'].count().values
            df['transactions_last_day'] = df.groupby('user_id').rolling('1D', on='timestamp')['amount'].count().values
        except Exception as e:
            logger.warning(f"Erro ao criar features de velocidade: {e}")
            df['transactions_last_hour'] = 1
            df['transactions_last_day'] = 1
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de agregação."""
        
        if 'user_id' not in df.columns:
            return df
        
        # Features por usuário
        user_agg = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'merchant_category': lambda x: x.nunique() if 'merchant_category' in df.columns else 0
        }).round(2)
        
        # Achata colunas multi-nível
        user_agg.columns = ['_'.join(col).strip() for col in user_agg.columns]
        user_agg = user_agg.add_prefix('user_')
        
        # Merge com dados originais
        df = df.merge(user_agg, left_on='user_id', right_index=True, how='left')
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Trata valores ausentes."""
        
        # Features numéricas - imputa com mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy='median')
                df[numeric_cols] = self.imputers['numeric'].fit_transform(df[numeric_cols])
            else:
                if 'numeric' in self.imputers:
                    df[numeric_cols] = self.imputers['numeric'].transform(df[numeric_cols])
        
        # Features categóricas - imputa com moda
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.imputers['categorical'].fit_transform(df[categorical_cols])
            else:
                if 'categorical' in self.imputers:
                    df[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Codifica features categóricas."""
        
        categorical_cols = [col for col in self.categorical_features if col in df.columns]
        
        for col in categorical_cols:
            if fit:
                # Usa LabelEncoder para features com muitas categorias
                if df[col].nunique() > 10:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Usa OneHotEncoder para features com poucas categorias
                    self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.encoders[col].fit_transform(df[[col]])
                    
                    # Cria nomes das colunas
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    
                    # Remove coluna original e adiciona codificadas
                    df = df.drop(columns=[col])
                    df = pd.concat([df, encoded_df], axis=1)
            else:
                if col in self.encoders:
                    if isinstance(self.encoders[col], LabelEncoder):
                        # Trata categorias não vistas
                        unique_values = set(self.encoders[col].classes_)
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in unique_values else 'unknown'
                        )
                        # Adiciona 'unknown' se necessário
                        if 'unknown' not in unique_values:
                            self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'unknown')
                        
                        df[col] = self.encoders[col].transform(df[col])
                    else:
                        # OneHotEncoder
                        encoded = self.encoders[col].transform(df[[col]])
                        feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                        
                        df = df.drop(columns=[col])
                        df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Normaliza features numéricas."""
        
        # Identifica colunas numéricas (exceto target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'user_id']]
        
        if len(numeric_cols) > 0:
            if fit:
                self.scalers['standard'] = StandardScaler()
                df[numeric_cols] = self.scalers['standard'].fit_transform(df[numeric_cols])
            else:
                if 'standard' in self.scalers:
                    df[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])
        
        return df
    
    def _remove_unnecessary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features desnecessárias."""
        
        # Features a remover
        features_to_remove = [
            'transaction_id', 'user_id', 'timestamp', 'fraud_pattern'
        ]
        
        # Remove se existirem
        features_to_remove = [col for col in features_to_remove if col in df.columns]
        if features_to_remove:
            df = df.drop(columns=features_to_remove)
        
        return df
    
    def prepare_train_test_split(self, 
                                df: pd.DataFrame, 
                                target_column: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara divisão treino/teste.
        
        Args:
            df: DataFrame processado
            target_column: Nome da coluna target
            
        Returns:
            Tuple com (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Configurações de divisão
        test_size = self.config['data']['preprocessing']['test_size']
        random_state = self.config['data']['preprocessing']['random_state']
        stratify = y if self.config['data']['preprocessing']['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Divisão treino/teste:")
        logger.info(f"  - Treino: {X_train.shape[0]:,} amostras")
        logger.info(f"  - Teste: {X_test.shape[0]:,} amostras")
        logger.info(f"  - Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features após processamento."""
        if not self.is_fitted:
            raise ValueError("Preprocessador não foi ajustado")
        
        # Esta implementação seria mais complexa na prática
        # Retorna lista básica por enquanto
        return ['processed_features']


def main():
    """Função principal para teste do preprocessador."""
    
    from src.data.data_loader import DataLoader
    
    # Carrega dados
    loader = DataLoader()
    users_df, transactions_df = loader.load_synthetic_data()
    
    # Preprocessa dados
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(transactions_df)
    
    print("✅ Dados preprocessados com sucesso!")
    print(f"📊 Shape original: {transactions_df.shape}")
    print(f"📊 Shape processado: {processed_df.shape}")
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(processed_df)
    print(f"📊 Treino: {X_train.shape}, Teste: {X_test.shape}")


if __name__ == "__main__":
    main()
