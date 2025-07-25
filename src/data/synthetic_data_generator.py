"""
Gerador de Dados Sintéticos para Transações Financeiras

Este módulo gera dados transacionais realistas para desenvolvimento,
testes e treinamento de modelos de detecção de fraudes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from faker import Faker
from pathlib import Path
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """
    Gerador de dados sintéticos para transações financeiras.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o gerador.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        self.fake = Faker('pt_BR')
        Faker.seed(self.config['data']['synthetic']['random_seed'])
        np.random.seed(self.config['data']['synthetic']['random_seed'])
        random.seed(self.config['data']['synthetic']['random_seed'])
        
        # Configurações
        self.n_samples = self.config['data']['synthetic']['n_samples']
        self.fraud_rate = self.config['data']['synthetic']['fraud_rate']
        
        # Dados de referência
        self._setup_reference_data()
    
    def _setup_reference_data(self):
        """Configura dados de referência para geração."""
        
        # Categorias de comerciantes
        self.merchant_categories = [
            'grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy',
            'hotel', 'airline', 'online', 'atm', 'bank', 'entertainment',
            'healthcare', 'education', 'government', 'utilities'
        ]
        
        # Métodos de pagamento
        self.payment_methods = [
            'credit_card', 'debit_card', 'pix', 'bank_transfer', 
            'digital_wallet', 'cash'
        ]
        
        # Tipos de dispositivo
        self.device_types = [
            'mobile', 'desktop', 'tablet', 'pos_terminal', 'atm'
        ]
        
        # Países e cidades
        self.countries = ['BR', 'US', 'AR', 'CL', 'CO', 'PE', 'UY']
        self.br_cities = [
            'São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador',
            'Fortaleza', 'Belo Horizonte', 'Manaus', 'Curitiba',
            'Recife', 'Porto Alegre'
        ]
        
        # Faixas de valores por categoria
        self.amount_ranges = {
            'grocery': (10, 500),
            'gas_station': (30, 300),
            'restaurant': (15, 200),
            'retail': (20, 1000),
            'pharmacy': (5, 150),
            'hotel': (100, 2000),
            'airline': (200, 5000),
            'online': (10, 800),
            'atm': (20, 1000),
            'bank': (50, 10000),
            'entertainment': (20, 300),
            'healthcare': (50, 1000),
            'education': (100, 5000),
            'government': (10, 500),
            'utilities': (50, 800)
        }
    
    def generate_user_profiles(self, n_users: int = 10000) -> pd.DataFrame:
        """
        Gera perfis de usuários.
        
        Args:
            n_users: Número de usuários a gerar
            
        Returns:
            DataFrame com perfis de usuários
        """
        logger.info(f"Gerando {n_users} perfis de usuários...")
        
        users = []
        for i in range(n_users):
            user = {
                'user_id': f'user_{i:06d}',
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
                'income_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
                'account_age_days': np.random.randint(1, 3650),  # 1 dia a 10 anos
                'country': np.random.choice(self.countries, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]),
                'city': np.random.choice(self.br_cities),
                'risk_score': np.random.beta(2, 5),  # Maioria com baixo risco
                'is_premium': np.random.choice([True, False], p=[0.15, 0.85])
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_transactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera transações baseadas nos perfis de usuários.
        
        Args:
            users_df: DataFrame com perfis de usuários
            
        Returns:
            DataFrame com transações
        """
        logger.info(f"Gerando {self.n_samples} transações...")
        
        transactions = []
        n_frauds = int(self.n_samples * self.fraud_rate)
        n_legitimate = self.n_samples - n_frauds
        
        # Gera transações legítimas
        for i in range(n_legitimate):
            transaction = self._generate_legitimate_transaction(users_df, i)
            transactions.append(transaction)
        
        # Gera transações fraudulentas
        for i in range(n_frauds):
            transaction = self._generate_fraudulent_transaction(users_df, n_legitimate + i)
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Embaralha as transações
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Adiciona features derivadas
        df = self._add_derived_features(df)
        
        return df
    
    def _generate_legitimate_transaction(self, users_df: pd.DataFrame, transaction_id: int) -> Dict:
        """Gera uma transação legítima."""
        
        user = users_df.sample(1).iloc[0]
        merchant_category = np.random.choice(self.merchant_categories)
        
        # Valor baseado na categoria e perfil do usuário
        min_amount, max_amount = self.amount_ranges[merchant_category]
        if user['income_level'] == 'high':
            max_amount *= 2
        elif user['income_level'] == 'low':
            max_amount *= 0.5
        
        amount = np.random.uniform(min_amount, max_amount)
        
        # Timestamp realista (últimos 90 dias)
        days_ago = np.random.randint(0, 90)
        base_time = datetime.now() - timedelta(days=days_ago)
        
        # Horário baseado em padrões normais
        if merchant_category in ['restaurant', 'entertainment']:
            hour = np.random.choice(range(18, 24), p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
        elif merchant_category in ['grocery', 'retail']:
            # 14 horas (8-22), probabilidades devem somar 1.0
            probs = [0.05, 0.08, 0.1, 0.12, 0.15, 0.15, 0.15, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002]
            hour = np.random.choice(range(8, 22), p=probs)
        else:
            hour = np.random.randint(6, 23)
        
        timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
        
        return {
            'transaction_id': f'txn_{transaction_id:08d}',
            'user_id': user['user_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'payment_method': np.random.choice(self.payment_methods),
            'device_type': np.random.choice(self.device_types),
            'country': user['country'],
            'city': user['city'],
            'is_weekend': timestamp.weekday() >= 5,
            'transaction_hour': hour,
            'is_fraud': False
        }
    
    def _generate_fraudulent_transaction(self, users_df: pd.DataFrame, transaction_id: int) -> Dict:
        """Gera uma transação fraudulenta com padrões suspeitos."""
        
        user = users_df.sample(1).iloc[0]
        
        # Padrões fraudulentos
        fraud_patterns = [
            'high_amount', 'unusual_time', 'foreign_country', 
            'multiple_attempts', 'unusual_merchant'
        ]
        
        pattern = np.random.choice(fraud_patterns)
        merchant_category = np.random.choice(self.merchant_categories)
        
        # Valor suspeito
        min_amount, max_amount = self.amount_ranges[merchant_category]
        if pattern == 'high_amount':
            amount = np.random.uniform(max_amount * 2, max_amount * 5)
        else:
            amount = np.random.uniform(min_amount, max_amount * 1.5)
        
        # Timestamp suspeito
        days_ago = np.random.randint(0, 30)  # Fraudes mais recentes
        base_time = datetime.now() - timedelta(days=days_ago)
        
        if pattern == 'unusual_time':
            # Horários suspeitos (madrugada)
            hour = np.random.choice([2, 3, 4, 5])
        else:
            hour = np.random.randint(0, 24)
        
        timestamp = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
        
        # País suspeito
        if pattern == 'foreign_country':
            country = np.random.choice(['XX', 'YY', 'ZZ'])  # Países fictícios suspeitos
            city = 'Unknown'
        else:
            country = user['country']
            city = user['city']
        
        return {
            'transaction_id': f'txn_{transaction_id:08d}',
            'user_id': user['user_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'payment_method': np.random.choice(self.payment_methods),
            'device_type': np.random.choice(self.device_types),
            'country': country,
            'city': city,
            'is_weekend': timestamp.weekday() >= 5,
            'transaction_hour': hour,
            'is_fraud': True,
            'fraud_pattern': pattern
        }
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features derivadas às transações."""
        
        logger.info("Adicionando features derivadas...")
        
        # Ordena por usuário e timestamp
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Features de velocidade por usuário
        df['days_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.days.fillna(0)
        
        # Features de agregação (últimos 30 dias)
        df['avg_transaction_amount_30d'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        df['transaction_count_30d'] = df.groupby('user_id').cumcount() + 1
        
        # Features de horário
        df['is_business_hour'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
        df['is_night'] = ((df['transaction_hour'] < 6) | (df['transaction_hour'] > 22)).astype(int)
        
        # Features de valor
        df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        return df
    
    def generate_complete_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gera dataset completo com usuários e transações.
        
        Returns:
            Tuple com (users_df, transactions_df)
        """
        logger.info("Iniciando geração de dataset completo...")
        
        # Gera usuários
        n_users = min(10000, self.n_samples // 10)  # ~10 transações por usuário
        users_df = self.generate_user_profiles(n_users)
        
        # Gera transações
        transactions_df = self.generate_transactions(users_df)
        
        logger.info(f"Dataset gerado: {len(users_df)} usuários, {len(transactions_df)} transações")
        logger.info(f"Taxa de fraude: {transactions_df['is_fraud'].mean():.2%}")
        
        return users_df, transactions_df
    
    def save_dataset(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                    output_dir: str = None) -> None:
        """
        Salva o dataset gerado.
        
        Args:
            users_df: DataFrame de usuários
            transactions_df: DataFrame de transações
            output_dir: Diretório de saída
        """
        if output_dir is None:
            output_dir = self.config['data']['synthetic_data_path']
        
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # Salva arquivos
        users_file = output_path / 'synthetic_users.csv'
        transactions_file = output_path / 'synthetic_transactions.csv'
        
        users_df.to_csv(users_file, index=False)
        transactions_df.to_csv(transactions_file, index=False)
        
        logger.info(f"Dataset salvo em:")
        logger.info(f"  - Usuários: {users_file}")
        logger.info(f"  - Transações: {transactions_file}")


def main():
    """Função principal para execução standalone."""
    
    print("🔄 Gerando dados sintéticos...")
    
    generator = SyntheticDataGenerator()
    users_df, transactions_df = generator.generate_complete_dataset()
    generator.save_dataset(users_df, transactions_df)
    
    print("✅ Dados sintéticos gerados com sucesso!")
    print(f"📊 Estatísticas:")
    print(f"   - Usuários: {len(users_df):,}")
    print(f"   - Transações: {len(transactions_df):,}")
    print(f"   - Taxa de fraude: {transactions_df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
