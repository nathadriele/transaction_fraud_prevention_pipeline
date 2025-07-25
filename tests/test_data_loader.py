"""
Testes para o módulo DataLoader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Testes para a classe DataLoader."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.loader = DataLoader()
    
    def test_initialization(self):
        """Testa inicialização do DataLoader."""
        assert self.loader is not None
        assert hasattr(self.loader, 'config')
        assert hasattr(self.loader, 'transaction_schema')
        assert hasattr(self.loader, 'user_schema')
    
    def test_transaction_schema(self):
        """Testa se o schema de transações está correto."""
        expected_columns = [
            'transaction_id', 'user_id', 'timestamp', 'amount',
            'merchant_category', 'payment_method', 'device_type',
            'country', 'city', 'is_fraud'
        ]
        
        for col in expected_columns:
            assert col in self.loader.transaction_schema
    
    def test_user_schema(self):
        """Testa se o schema de usuários está correto."""
        expected_columns = [
            'user_id', 'age', 'gender', 'income_level',
            'account_age_days', 'country', 'city'
        ]
        
        for col in expected_columns:
            assert col in self.loader.user_schema
    
    def test_load_synthetic_data(self):
        """Testa carregamento de dados sintéticos."""
        users_df, transactions_df = self.loader.load_synthetic_data()
        
        # Verifica se os DataFrames foram carregados
        assert isinstance(users_df, pd.DataFrame)
        assert isinstance(transactions_df, pd.DataFrame)
        
        # Verifica se não estão vazios
        assert len(users_df) > 0
        assert len(transactions_df) > 0
        
        # Verifica colunas essenciais
        assert 'user_id' in users_df.columns
        assert 'transaction_id' in transactions_df.columns
        assert 'is_fraud' in transactions_df.columns
        assert 'amount' in transactions_df.columns
    
    def test_data_summary(self):
        """Testa geração de resumo dos dados."""
        users_df, transactions_df = self.loader.load_synthetic_data()
        summary = self.loader.get_data_summary(transactions_df)
        
        # Verifica estrutura do resumo
        assert 'total_records' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'null_counts' in summary
        assert 'memory_usage_mb' in summary
        
        # Verifica valores
        assert summary['total_records'] == len(transactions_df)
        assert summary['total_records'] > 0
        
        # Se há coluna de fraude, deve ter estatísticas
        if 'is_fraud' in transactions_df.columns:
            assert 'fraud_rate' in summary
            assert 'fraud_count' in summary
            assert 0 <= summary['fraud_rate'] <= 1
    
    def test_convert_transaction_types(self):
        """Testa conversão de tipos de dados das transações."""
        # Cria DataFrame de teste
        test_data = pd.DataFrame({
            'transaction_id': ['txn_001', 'txn_002'],
            'amount': ['100.50', '200.75'],
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
            'is_fraud': [True, False],
            'merchant_category': ['grocery', 'retail']
        })
        
        converted_df = self.loader._convert_transaction_types(test_data)
        
        # Verifica tipos
        assert pd.api.types.is_numeric_dtype(converted_df['amount'])
        assert pd.api.types.is_datetime64_any_dtype(converted_df['timestamp'])
        assert converted_df['is_fraud'].dtype == bool
        assert converted_df['merchant_category'].dtype.name == 'category'
    
    def test_validate_transaction_data(self):
        """Testa validação de dados de transações."""
        # Cria DataFrame com problemas
        test_data = pd.DataFrame({
            'transaction_id': ['txn_001', 'txn_002', 'txn_001'],  # Duplicata
            'user_id': ['user_001', None, 'user_003'],  # Valor nulo
            'amount': [100.0, -50.0, 200.0],  # Valor negativo
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        # Não deve gerar exceção, apenas warnings
        try:
            self.loader._validate_transaction_data(test_data)
        except Exception as e:
            pytest.fail(f"Validação não deveria gerar exceção: {e}")
    
    def test_file_not_found_error(self):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_transactions("arquivo_inexistente.csv")
    
    def test_invalid_file_format(self):
        """Testa erro com formato de arquivo inválido."""
        # Cria arquivo temporário com extensão inválida
        temp_file = Path("temp_test.xyz")
        temp_file.write_text("test data")
        
        try:
            with pytest.raises(ValueError, match="Formato de arquivo não suportado"):
                self.loader.load_transactions(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestDataLoaderIntegration:
    """Testes de integração para DataLoader."""
    
    def test_full_data_pipeline(self):
        """Testa pipeline completo de carregamento de dados."""
        loader = DataLoader()
        
        # Carrega dados
        users_df, transactions_df = loader.load_synthetic_data()
        
        # Verifica integridade dos dados
        assert len(users_df) > 0
        assert len(transactions_df) > 0
        
        # Verifica relacionamento entre usuários e transações
        user_ids_in_users = set(users_df['user_id'].unique())
        user_ids_in_transactions = set(transactions_df['user_id'].unique())
        
        # Deve haver sobreposição (nem todos os usuários precisam ter transações)
        overlap = user_ids_in_users.intersection(user_ids_in_transactions)
        assert len(overlap) > 0
        
        # Verifica distribuição de fraudes
        fraud_rate = transactions_df['is_fraud'].mean()
        assert 0 < fraud_rate < 1  # Deve haver fraudes e não fraudes
        
        # Verifica valores de transações
        assert transactions_df['amount'].min() > 0  # Valores positivos
        assert transactions_df['amount'].max() < 1000000  # Valores razoáveis
        
        # Verifica timestamps
        assert transactions_df['timestamp'].notna().all()
        
        # Verifica categorias
        assert transactions_df['merchant_category'].notna().all()
        assert transactions_df['payment_method'].notna().all()


if __name__ == "__main__":
    # Executa testes se rodado diretamente
    pytest.main([__file__, "-v"])
