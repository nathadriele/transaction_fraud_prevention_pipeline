"""
Sistema de Regras de Negócio para Detecção de Fraudes

Implementa um engine de regras customizáveis que complementa
os modelos de ML na detecção de fraudes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import json
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger, get_fraud_logger
from src.utils.helpers import save_json, load_json, ensure_dir

logger = get_logger(__name__)
fraud_logger = get_fraud_logger(__name__)


class Rule:
    """
    Classe que representa uma regra de negócio individual.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 condition: Callable,
                 action: str = "flag",
                 priority: int = 1,
                 enabled: bool = True):
        """
        Inicializa uma regra.
        
        Args:
            name: Nome da regra
            description: Descrição da regra
            condition: Função que avalia a condição
            action: Ação a ser tomada ("flag", "block", "review")
            priority: Prioridade da regra (1 = alta, 5 = baixa)
            enabled: Se a regra está ativa
        """
        self.name = name
        self.description = description
        self.condition = condition
        self.action = action
        self.priority = priority
        self.enabled = enabled
        self.triggered_count = 0
        self.last_triggered = None
    
    def evaluate(self, transaction: Dict) -> bool:
        """
        Avalia se a regra é acionada para uma transação.
        
        Args:
            transaction: Dados da transação
            
        Returns:
            True se a regra for acionada
        """
        if not self.enabled:
            return False
        
        try:
            result = self.condition(transaction)
            if result:
                self.triggered_count += 1
                self.last_triggered = datetime.now()
                fraud_logger.log_rule_triggered(
                    self.name, 
                    transaction.get('transaction_id', 'unknown'),
                    self.description
                )
            return result
        except Exception as e:
            logger.error(f"Erro ao avaliar regra {self.name}: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Converte regra para dicionário."""
        return {
            'name': self.name,
            'description': self.description,
            'action': self.action,
            'priority': self.priority,
            'enabled': self.enabled,
            'triggered_count': self.triggered_count,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }


class BusinessRulesEngine:
    """
    Engine de regras de negócio para detecção de fraudes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o engine de regras.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        self.rules = []
        self.rule_stats = {}
        
        # Inicializa regras padrão
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Inicializa regras padrão baseadas na configuração."""
        
        rules_config = self.config.get('business_rules', {})
        
        # Regras de valor
        amount_rules = rules_config.get('amount_rules', {})
        if amount_rules.get('max_single_transaction'):
            self.add_rule(
                name="high_amount_transaction",
                description=f"Transação acima de {amount_rules['max_single_transaction']}",
                condition=lambda t: t.get('amount', 0) > amount_rules['max_single_transaction'],
                action="review",
                priority=2
            )
        
        if amount_rules.get('suspicious_round_amounts'):
            self.add_rule(
                name="round_amount_suspicious",
                description="Valor suspeito (múltiplo de 100)",
                condition=lambda t: t.get('amount', 0) % 100 == 0 and t.get('amount', 0) >= 1000,
                action="flag",
                priority=3
            )
        
        # Regras de velocidade
        velocity_rules = rules_config.get('velocity_rules', {})
        if velocity_rules.get('max_transactions_per_hour'):
            self.add_rule(
                name="high_velocity_user",
                description=f"Mais de {velocity_rules['max_transactions_per_hour']} transações por hora",
                condition=lambda t: t.get('transactions_last_hour', 0) > velocity_rules['max_transactions_per_hour'],
                action="block",
                priority=1
            )
        
        # Regras de localização
        location_rules = rules_config.get('location_rules', {})
        if location_rules.get('suspicious_countries'):
            suspicious_countries = location_rules['suspicious_countries']
            self.add_rule(
                name="suspicious_country",
                description=f"Transação de país suspeito: {suspicious_countries}",
                condition=lambda t: t.get('country') in suspicious_countries,
                action="review",
                priority=2
            )
        
        # Regras de tempo
        time_rules = rules_config.get('time_rules', {})
        if time_rules.get('suspicious_hours'):
            suspicious_hours = time_rules['suspicious_hours']
            self.add_rule(
                name="night_transaction",
                description=f"Transação em horário suspeito: {suspicious_hours}",
                condition=lambda t: t.get('transaction_hour', 12) in suspicious_hours,
                action="flag",
                priority=3
            )
        
        if time_rules.get('weekend_multiplier'):
            self.add_rule(
                name="weekend_high_amount",
                description="Transação de alto valor no fim de semana",
                condition=lambda t: (t.get('is_weekend', False) and 
                                   t.get('amount', 0) > amount_rules.get('max_single_transaction', 10000) * 0.5),
                action="review",
                priority=2
            )
        
        logger.info(f"Regras padrão inicializadas: {len(self.rules)} regras")
    
    def add_rule(self, 
                 name: str,
                 description: str,
                 condition: Callable,
                 action: str = "flag",
                 priority: int = 1,
                 enabled: bool = True) -> None:
        """
        Adiciona uma nova regra ao engine.
        
        Args:
            name: Nome da regra
            description: Descrição da regra
            condition: Função que avalia a condição
            action: Ação a ser tomada
            priority: Prioridade da regra
            enabled: Se a regra está ativa
        """
        rule = Rule(name, description, condition, action, priority, enabled)
        self.rules.append(rule)
        logger.info(f"Regra adicionada: {name}")
    
    def remove_rule(self, name: str) -> bool:
        """
        Remove uma regra do engine.
        
        Args:
            name: Nome da regra a ser removida
            
        Returns:
            True se a regra foi removida
        """
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                logger.info(f"Regra removida: {name}")
                return True
        return False
    
    def enable_rule(self, name: str) -> bool:
        """Ativa uma regra."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                logger.info(f"Regra ativada: {name}")
                return True
        return False
    
    def disable_rule(self, name: str) -> bool:
        """Desativa uma regra."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                logger.info(f"Regra desativada: {name}")
                return True
        return False
    
    def evaluate_transaction(self, transaction: Dict) -> Dict[str, Any]:
        """
        Avalia uma transação contra todas as regras.
        
        Args:
            transaction: Dados da transação
            
        Returns:
            Resultado da avaliação
        """
        triggered_rules = []
        highest_priority = 5  # Menor número = maior prioridade
        final_action = "allow"
        
        # Ordena regras por prioridade
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if rule.evaluate(transaction):
                triggered_rules.append({
                    'name': rule.name,
                    'description': rule.description,
                    'action': rule.action,
                    'priority': rule.priority
                })
                
                # Atualiza ação final baseada na prioridade
                if rule.priority < highest_priority:
                    highest_priority = rule.priority
                    final_action = rule.action
        
        result = {
            'transaction_id': transaction.get('transaction_id'),
            'triggered_rules': triggered_rules,
            'final_action': final_action,
            'risk_score': self._calculate_risk_score(triggered_rules),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_risk_score(self, triggered_rules: List[Dict]) -> float:
        """
        Calcula score de risco baseado nas regras acionadas.
        
        Args:
            triggered_rules: Lista de regras acionadas
            
        Returns:
            Score de risco (0-1)
        """
        if not triggered_rules:
            return 0.0
        
        # Peso baseado na prioridade e ação
        action_weights = {
            'flag': 0.3,
            'review': 0.6,
            'block': 1.0
        }
        
        priority_weights = {
            1: 1.0,
            2: 0.8,
            3: 0.6,
            4: 0.4,
            5: 0.2
        }
        
        total_score = 0.0
        for rule in triggered_rules:
            action_weight = action_weights.get(rule['action'], 0.5)
            priority_weight = priority_weights.get(rule['priority'], 0.5)
            total_score += action_weight * priority_weight
        
        # Normaliza para 0-1
        return min(total_score, 1.0)
    
    def evaluate_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Avalia um lote de transações.
        
        Args:
            transactions: Lista de transações
            
        Returns:
            Lista de resultados
        """
        logger.info(f"Avaliando lote de {len(transactions)} transações...")
        
        results = []
        for transaction in transactions:
            result = self.evaluate_transaction(transaction)
            results.append(result)
        
        # Atualiza estatísticas
        self._update_statistics(results)
        
        return results
    
    def _update_statistics(self, results: List[Dict]) -> None:
        """Atualiza estatísticas das regras."""
        
        total_transactions = len(results)
        flagged_transactions = sum(1 for r in results if r['triggered_rules'])
        
        self.rule_stats = {
            'total_transactions_evaluated': total_transactions,
            'flagged_transactions': flagged_transactions,
            'flag_rate': flagged_transactions / total_transactions if total_transactions > 0 else 0,
            'rule_performance': {}
        }
        
        # Estatísticas por regra
        for rule in self.rules:
            self.rule_stats['rule_performance'][rule.name] = {
                'triggered_count': rule.triggered_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                'enabled': rule.enabled
            }
    
    def get_rule_statistics(self) -> Dict:
        """Retorna estatísticas das regras."""
        return self.rule_stats
    
    def get_rules_summary(self) -> List[Dict]:
        """Retorna resumo de todas as regras."""
        return [rule.to_dict() for rule in self.rules]
    
    def save_rules_config(self, filepath: str = "config/business_rules.json") -> None:
        """
        Salva configuração das regras.
        
        Args:
            filepath: Caminho para salvar
        """
        rules_config = {
            'rules': self.get_rules_summary(),
            'statistics': self.rule_stats,
            'saved_at': datetime.now().isoformat()
        }
        
        ensure_dir(os.path.dirname(filepath))
        save_json(rules_config, filepath)
        logger.info(f"Configuração das regras salva em: {filepath}")
    
    def create_custom_rule(self, 
                          rule_definition: Dict) -> bool:
        """
        Cria uma regra customizada a partir de definição.
        
        Args:
            rule_definition: Definição da regra
            
        Returns:
            True se a regra foi criada com sucesso
        """
        try:
            # Exemplo de regra customizada simples
            name = rule_definition['name']
            description = rule_definition['description']
            field = rule_definition['field']
            operator = rule_definition['operator']
            value = rule_definition['value']
            action = rule_definition.get('action', 'flag')
            priority = rule_definition.get('priority', 3)
            
            # Cria função de condição baseada nos parâmetros
            if operator == 'greater_than':
                condition = lambda t: t.get(field, 0) > value
            elif operator == 'less_than':
                condition = lambda t: t.get(field, 0) < value
            elif operator == 'equals':
                condition = lambda t: t.get(field) == value
            elif operator == 'in':
                condition = lambda t: t.get(field) in value
            else:
                logger.error(f"Operador não suportado: {operator}")
                return False
            
            self.add_rule(name, description, condition, action, priority)
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar regra customizada: {e}")
            return False


def main():
    """Função principal para teste do sistema de regras."""
    
    from src.data.data_loader import DataLoader
    
    print("📋 Testando Sistema de Regras de Negócio...")
    
    # Carrega dados
    loader = DataLoader()
    users_df, transactions_df = loader.load_synthetic_data()
    
    # Inicializa engine de regras
    rules_engine = BusinessRulesEngine()
    
    # Converte algumas transações para teste
    test_transactions = transactions_df.head(100).to_dict('records')
    
    # Avalia transações
    results = rules_engine.evaluate_batch(test_transactions)
    
    # Estatísticas
    stats = rules_engine.get_rule_statistics()
    
    # Salva configuração
    rules_engine.save_rules_config()
    
    print("✅ Sistema de regras testado!")
    print(f"📊 Transações avaliadas: {stats['total_transactions_evaluated']}")
    print(f"📊 Transações sinalizadas: {stats['flagged_transactions']}")
    print(f"📊 Taxa de sinalização: {stats['flag_rate']:.2%}")
    
    # Mostra regras mais acionadas
    rule_performance = stats['rule_performance']
    top_rules = sorted(rule_performance.items(), key=lambda x: x[1]['triggered_count'], reverse=True)[:3]
    
    print("🔥 Top 3 regras mais acionadas:")
    for rule_name, perf in top_rules:
        print(f"   - {rule_name}: {perf['triggered_count']} vezes")


if __name__ == "__main__":
    main()
