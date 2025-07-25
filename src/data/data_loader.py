"""
Carregador de Dados

Módulo responsável por carregar e validar dados de diferentes fontes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import validate_dataframe_schema, get_file_hash

logger = get_logger(__name__)


class DataLoader:
    """
    Carregador de dados para o sistema de detecção de fraudes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o carregador de dados.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        
        # Esquemas esperados
        self.transaction_schema = [
            'transaction_id', 'user_id', 'timestamp', 'amount',
            'merchant_category', 'payment_method', 'device_type',
            'country', 'city', 'is_fraud'
        ]
        
        self.user_schema = [
            'user_id', 'age', 'gender', 'income_level',
            'account_age_days', 'country', 'city'
        ]
    
    def load_transactions(self, 
                         filepath: Union[str, Path],
                         validate_schema: bool = True) -> pd.DataFrame:
        """
        Carrega dados de transações.
        
        Args:
            filepath: Caminho para o arquivo de transações
            validate_schema: Se deve validar o esquema dos dados
            
        Returns:
            DataFrame com transações
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o esquema for inválido
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        logger.info(f"Carregando transações de: {filepath}")
        
        # Carrega dados baseado na extensão
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix.lower() == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {filepath.suffix}")
        
        # Validação do esquema
        if validate_schema:
            if not validate_dataframe_schema(df, self.transaction_schema):
                raise ValueError("Esquema de transações inválido")
        
        # Conversões de tipo
        df = self._convert_transaction_types(df)
        
        # Validações básicas
        self._validate_transaction_data(df)
        
        logger.info(f"Transações carregadas: {len(df):,} registros")
        
        return df
    
    def load_users(self, 
                   filepath: Union[str, Path],
                   validate_schema: bool = True) -> pd.DataFrame:
        """
        Carrega dados de usuários.
        
        Args:
            filepath: Caminho para o arquivo de usuários
            validate_schema: Se deve validar o esquema dos dados
            
        Returns:
            DataFrame com usuários
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        logger.info(f"Carregando usuários de: {filepath}")
        
        # Carrega dados
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix.lower() == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {filepath.suffix}")
        
        # Validação do esquema
        if validate_schema:
            if not validate_dataframe_schema(df, self.user_schema):
                raise ValueError("Esquema de usuários inválido")
        
        # Conversões de tipo
        df = self._convert_user_types(df)
        
        logger.info(f"Usuários carregados: {len(df):,} registros")
        
        return df
    
    def load_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega dados sintéticos gerados.
        
        Returns:
            Tuple com (users_df, transactions_df)
        """
        synthetic_path = Path(self.config['data']['synthetic_data_path'])
        
        users_file = synthetic_path / 'synthetic_users.csv'
        transactions_file = synthetic_path / 'synthetic_transactions.csv'
        
        if not users_file.exists() or not transactions_file.exists():
            logger.warning("Dados sintéticos não encontrados. Gerando novos dados...")
            from src.data.synthetic_data_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator(self.config)
            users_df, transactions_df = generator.generate_complete_dataset()
            generator.save_dataset(users_df, transactions_df)
            
            return users_df, transactions_df
        
        users_df = self.load_users(users_file, validate_schema=False)
        transactions_df = self.load_transactions(transactions_file, validate_schema=False)
        
        return users_df, transactions_df
    
    def _convert_transaction_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte tipos de dados das transações."""
        
        # Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Valores numéricos
        numeric_columns = ['amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Booleanos
        boolean_columns = ['is_fraud', 'is_weekend']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Categóricas
        categorical_columns = [
            'merchant_category', 'payment_method', 'device_type',
            'country', 'city'
        ]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def _convert_user_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte tipos de dados dos usuários."""
        
        # Valores numéricos
        numeric_columns = ['age', 'account_age_days', 'risk_score']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Booleanos
        boolean_columns = ['is_premium']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Categóricas
        categorical_columns = ['gender', 'income_level', 'country', 'city']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def _validate_transaction_data(self, df: pd.DataFrame) -> None:
        """Valida dados de transações."""
        
        # Verifica valores nulos em colunas críticas
        critical_columns = ['transaction_id', 'user_id', 'timestamp', 'amount']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(f"Coluna {col} possui {null_count} valores nulos")
        
        # Verifica valores negativos em amount
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            if negative_amounts > 0:
                logger.warning(f"Encontrados {negative_amounts} valores negativos em 'amount'")
        
        # Verifica duplicatas de transaction_id
        if 'transaction_id' in df.columns:
            duplicates = df['transaction_id'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Encontradas {duplicates} transações duplicadas")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Retorna resumo dos dados.
        
        Args:
            df: DataFrame a ser analisado
            
        Returns:
            Dicionário com estatísticas dos dados
        """
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Estatísticas específicas para transações
        if 'is_fraud' in df.columns:
            summary['fraud_rate'] = df['is_fraud'].mean()
            summary['fraud_count'] = df['is_fraud'].sum()
        
        if 'amount' in df.columns:
            summary['amount_stats'] = {
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            }
        
        return summary
    
    def save_processed_data(self, 
                           df: pd.DataFrame, 
                           filename: str,
                           format: str = 'parquet') -> Path:
        """
        Salva dados processados.
        
        Args:
            df: DataFrame a ser salvo
            filename: Nome do arquivo
            format: Formato do arquivo ('parquet', 'csv')
            
        Returns:
            Caminho do arquivo salvo
        """
        processed_path = Path(self.config['data']['processed_data_path'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            filepath = processed_path / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            filepath = processed_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Formato não suportado: {format}")
        
        logger.info(f"Dados salvos em: {filepath}")
        return filepath


def main():
    """Função principal para teste do carregador."""
    
    loader = DataLoader()
    
    try:
        # Tenta carregar dados sintéticos
        users_df, transactions_df = loader.load_synthetic_data()
        
        print("✅ Dados carregados com sucesso!")
        print(f"📊 Usuários: {len(users_df):,}")
        print(f"📊 Transações: {len(transactions_df):,}")
        
        # Mostra resumo
        summary = loader.get_data_summary(transactions_df)
        print(f"📈 Taxa de fraude: {summary.get('fraud_rate', 0):.2%}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")


if __name__ == "__main__":
    main()
