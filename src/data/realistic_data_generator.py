#!/usr/bin/env python3
"""
Gerador de dados realistas para sistema de prevenção de fraudes.
Implementa distribuições estatísticas apropriadas e padrões de comportamento reais.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class RealisticFraudDataGenerator:
    """Gerador de dados de fraude com distribuições realistas."""
    
    def __init__(self, seed: int = 42):
        """Inicializa o gerador com seed para reprodutibilidade."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Configurações realistas baseadas em dados do mercado
        self.fraud_rates = {
            'grocery': 0.005,      # 0.5%
            'restaurant': 0.01,    # 1%
            'gas_station': 0.008,  # 0.8%
            'pharmacy': 0.006,     # 0.6%
            'coffee_shop': 0.007,  # 0.7%
            'fast_food': 0.009,    # 0.9%
            'clothing': 0.015,     # 1.5%
            'online': 0.025,       # 2.5%
            'electronics': 0.03,   # 3%
            'travel': 0.02,        # 2%
            'jewelry': 0.08,       # 8%
            'luxury': 0.05         # 5%
        }
        
        self.country_fraud_multipliers = {
            'BR': 1.0,    # Base (1.5% média)
            'US': 1.2,    # 20% mais risco
            'CA': 0.8,    # 20% menos risco
            'UK': 0.9,    # 10% menos risco
            'DE': 0.7,    # 30% menos risco
            'CN': 2.5,    # 150% mais risco
            'RU': 3.0,    # 200% mais risco
            'XX': 4.0     # 300% mais risco (países de alto risco)
        }
        
        # Distribuições de valores por categoria (log-normal)
        self.value_distributions = {
            'grocery': {'mean': 4.5, 'sigma': 0.8},        # ~R$ 90
            'restaurant': {'mean': 4.8, 'sigma': 0.7},     # ~R$ 120
            'gas_station': {'mean': 4.2, 'sigma': 0.6},    # ~R$ 65
            'pharmacy': {'mean': 4.0, 'sigma': 0.9},       # ~R$ 55
            'coffee_shop': {'mean': 3.5, 'sigma': 0.5},    # ~R$ 30
            'fast_food': {'mean': 3.8, 'sigma': 0.6},      # ~R$ 45
            'clothing': {'mean': 5.5, 'sigma': 1.0},       # ~R$ 245
            'online': {'mean': 5.2, 'sigma': 1.2},         # ~R$ 180
            'electronics': {'mean': 6.5, 'sigma': 1.1},    # ~R$ 665
            'travel': {'mean': 7.0, 'sigma': 0.9},         # ~R$ 1100
            'jewelry': {'mean': 7.5, 'sigma': 1.3},        # ~R$ 1800
            'luxury': {'mean': 8.0, 'sigma': 1.0}          # ~R$ 3000
        }
        
        self.rules_patterns = [
            'none', 'high_amount', 'velocity_rule', 'pattern_anomaly',
            'suspicious_country', 'time_anomaly', 'merchant_risk',
            'card_testing', 'account_takeover'
        ]

    def calculate_fraud_probability(self, amount: float, category: str, 
                                  country: str, hour: int) -> float:
        """Calcula probabilidade de fraude baseada em múltiplos fatores."""
        base_rate = self.fraud_rates.get(category, 0.02)
        country_multiplier = self.country_fraud_multipliers.get(country, 1.0)
        
        # Fator de valor (valores muito altos ou muito baixos são mais suspeitos)
        if amount > 5000:
            value_factor = 1.5
        elif amount > 2000:
            value_factor = 1.2
        elif amount < 10:
            value_factor = 1.3
        else:
            value_factor = 1.0
        
        # Fator temporal
        if 2 <= hour <= 5:
            time_factor = 1.8
        elif 22 <= hour <= 23 or 0 <= hour <= 1:
            time_factor = 1.3
        else:
            time_factor = 1.0
        
        final_probability = base_rate * country_multiplier * value_factor * time_factor
        return min(final_probability, 0.95)  # Cap em 95%

    def generate_transaction_value(self, category: str, is_fraud: bool = False) -> float:
        """Gera valor de transação usando distribuição log-normal."""
        dist = self.value_distributions.get(category, {'mean': 5.0, 'sigma': 1.0})
        
        # Fraudes tendem a ser ligeiramente maiores, mas não sempre
        if is_fraud:
            # 30% das fraudes são de alto valor
            if np.random.random() < 0.3:
                value = np.random.lognormal(dist['mean'] + 0.5, dist['sigma'])
            else:
                value = np.random.lognormal(dist['mean'], dist['sigma'])
        else:
            value = np.random.lognormal(dist['mean'], dist['sigma'])
        
        return round(max(value, 5.0), 2)  # Mínimo R$ 5

    def generate_fraud_score(self, is_fraud: bool, amount: float, 
                           category: str, country: str) -> float:
        """Gera score de fraude realista."""
        if is_fraud:
            # Fraudes reais: score alto com alguma variação
            base_score = np.random.normal(0.85, 0.08)
            
            # Ajustes baseados em fatores
            if amount > 5000:
                base_score += 0.05
            if country in ['CN', 'RU', 'XX']:
                base_score += 0.03
            if category in ['jewelry', 'luxury']:
                base_score += 0.02
                
        else:
            # Transações legítimas: score baixo com alguns falsos positivos
            base_score = np.random.normal(0.15, 0.12)
            
            # Alguns falsos positivos
            if np.random.random() < 0.05:  # 5% de falsos positivos
                base_score = np.random.normal(0.6, 0.1)
        
        return round(np.clip(base_score, 0.01, 0.99), 3)

    def select_rule_triggered(self, is_fraud: bool, amount: float, 
                            category: str, country: str) -> str:
        """Seleciona regra acionada baseada no contexto."""
        if not is_fraud:
            return 'none'
        
        # Regras baseadas em padrões reais
        if amount > 5000:
            return np.random.choice(['high_amount', 'pattern_anomaly'], p=[0.7, 0.3])
        elif country in ['CN', 'RU', 'XX']:
            return np.random.choice(['suspicious_country', 'pattern_anomaly'], p=[0.6, 0.4])
        elif category in ['jewelry', 'luxury']:
            return np.random.choice(['merchant_risk', 'pattern_anomaly'], p=[0.5, 0.5])
        else:
            return np.random.choice([
                'velocity_rule', 'pattern_anomaly', 'time_anomaly', 
                'card_testing', 'account_takeover'
            ])

    def generate_realistic_dataset(self, n_transactions: int = 1000, 
                                 days_span: int = 30) -> pd.DataFrame:
        """Gera dataset completo com distribuições realistas."""
        transactions = []
        
        # Distribuição de categorias (baseada em frequência real)
        categories = list(self.fraud_rates.keys())
        category_weights = [
            0.25, 0.15, 0.12, 0.08, 0.06, 0.05,  # Frequentes
            0.08, 0.10, 0.06, 0.03, 0.01, 0.01   # Menos frequentes
        ]
        
        # Distribuição de países
        countries = list(self.country_fraud_multipliers.keys())
        country_weights = [0.4, 0.25, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]
        
        base_date = datetime.now() - timedelta(days=days_span)
        
        for i in range(n_transactions):
            # Seleciona categoria e país
            category = np.random.choice(categories, p=category_weights)
            country = np.random.choice(countries, p=country_weights)
            
            # Gera timestamp realista
            day_offset = np.random.randint(0, days_span)
            hour = np.random.choice(range(24), p=self._get_hourly_distribution())
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            timestamp = base_date + timedelta(
                days=day_offset, hours=hour, minutes=minute, seconds=second
            )
            
            # Gera valor primeiro
            amount = self.generate_transaction_value(category)
            
            # Determina se é fraude
            fraud_prob = self.calculate_fraud_probability(amount, category, country, hour)
            is_fraud = np.random.random() < fraud_prob
            
            # Se é fraude, pode ajustar o valor
            if is_fraud:
                amount = self.generate_transaction_value(category, is_fraud=True)
            
            # Gera outros campos
            fraud_score = self.generate_fraud_score(is_fraud, amount, category, country)
            rule_triggered = self.select_rule_triggered(is_fraud, amount, category, country)
            model_prediction = 1 if fraud_score > 0.5 else 0
            
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'user_id': f'USR_{np.random.randint(1, n_transactions//2):06d}',
                'amount': amount,
                'merchant_category': category,
                'country': country,
                'is_fraud': int(is_fraud),
                'fraud_score': fraud_score,
                'model_prediction': model_prediction,
                'rule_triggered': rule_triggered,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df

    def _get_hourly_distribution(self) -> List[float]:
        """Retorna distribuição realista de transações por hora."""
        # Baseado em padrões reais de uso
        hourly_probs = [
            0.01, 0.005, 0.003, 0.002, 0.003, 0.005,  # 0-5h (madrugada)
            0.01, 0.02, 0.04, 0.06, 0.07, 0.08,       # 6-11h (manhã)
            0.09, 0.08, 0.07, 0.06, 0.05, 0.06,       # 12-17h (tarde)
            0.07, 0.08, 0.07, 0.05, 0.03, 0.02        # 18-23h (noite)
        ]
        return hourly_probs

def main():
    """Função principal para gerar dados realistas."""
    generator = RealisticFraudDataGenerator()
    
    print("Gerando dataset realista...")
    df = generator.generate_realistic_dataset(n_transactions=1000, days_span=30)
    
    total_transactions = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_rate = (fraud_count / total_transactions) * 100
    
    print(f"Dataset gerado:")
    print(f"- Total de transações: {total_transactions:,}")
    print(f"- Fraudes detectadas: {fraud_count}")
    print(f"- Taxa de fraude: {fraud_rate:.2f}%")
    print(f"- Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    output_path = 'data/realistic_final_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")
    
    return df

if __name__ == "__main__":
    main()
