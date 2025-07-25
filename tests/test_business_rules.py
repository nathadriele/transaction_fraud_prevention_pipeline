"""
Testes para o módulo BusinessRules
"""

import pytest
from datetime import datetime
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.business_rules import Rule, BusinessRulesEngine


class TestRule:
    """Testes para a classe Rule."""
    
    def test_rule_creation(self):
        """Testa criação de uma regra."""
        condition = lambda t: t.get('amount', 0) > 1000
        rule = Rule(
            name="high_amount",
            description="Transação de alto valor",
            condition=condition,
            action="review",
            priority=2
        )
        
        assert rule.name == "high_amount"
        assert rule.description == "Transação de alto valor"
        assert rule.action == "review"
        assert rule.priority == 2
        assert rule.enabled == True
        assert rule.triggered_count == 0
    
    def test_rule_evaluation_true(self):
        """Testa avaliação de regra que deve ser acionada."""
        condition = lambda t: t.get('amount', 0) > 1000
        rule = Rule("high_amount", "Alto valor", condition)
        
        transaction = {'amount': 1500, 'transaction_id': 'test_001'}
        result = rule.evaluate(transaction)
        
        assert result == True
        assert rule.triggered_count == 1
        assert rule.last_triggered is not None
    
    def test_rule_evaluation_false(self):
        """Testa avaliação de regra que não deve ser acionada."""
        condition = lambda t: t.get('amount', 0) > 1000
        rule = Rule("high_amount", "Alto valor", condition)
        
        transaction = {'amount': 500, 'transaction_id': 'test_001'}
        result = rule.evaluate(transaction)
        
        assert result == False
        assert rule.triggered_count == 0
        assert rule.last_triggered is None
    
    def test_rule_disabled(self):
        """Testa regra desabilitada."""
        condition = lambda t: t.get('amount', 0) > 1000
        rule = Rule("high_amount", "Alto valor", condition, enabled=False)
        
        transaction = {'amount': 1500, 'transaction_id': 'test_001'}
        result = rule.evaluate(transaction)
        
        assert result == False
        assert rule.triggered_count == 0
    
    def test_rule_error_handling(self):
        """Testa tratamento de erro na avaliação."""
        condition = lambda t: t['nonexistent_key'] > 1000
        rule = Rule("error_rule", "Regra com erro", condition)
        
        transaction = {'amount': 1500}
        result = rule.evaluate(transaction)
        
        assert result == False
        assert rule.triggered_count == 0
    
    def test_rule_to_dict(self):
        """Testa conversão de regra para dicionário."""
        condition = lambda t: t.get('amount', 0) > 1000
        rule = Rule("high_amount", "Alto valor", condition, action="block", priority=1)
        
        rule_dict = rule.to_dict()
        
        assert rule_dict['name'] == "high_amount"
        assert rule_dict['description'] == "Alto valor"
        assert rule_dict['action'] == "block"
        assert rule_dict['priority'] == 1
        assert rule_dict['enabled'] == True
        assert rule_dict['triggered_count'] == 0


class TestBusinessRulesEngine:
    """Testes para a classe BusinessRulesEngine."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.engine = BusinessRulesEngine()
    
    def test_engine_initialization(self):
        """Testa inicialização do engine."""
        assert self.engine is not None
        assert hasattr(self.engine, 'rules')
        assert hasattr(self.engine, 'config')
        assert len(self.engine.rules) > 0
    
    def test_add_rule(self):
        """Testa adição de nova regra."""
        initial_count = len(self.engine.rules)
        
        condition = lambda t: t.get('amount', 0) > 5000
        self.engine.add_rule(
            name="very_high_amount",
            description="Valor muito alto",
            condition=condition,
            action="block"
        )
        
        assert len(self.engine.rules) == initial_count + 1
        
        new_rule = self.engine.rules[-1]
        assert new_rule.name == "very_high_amount"
        assert new_rule.action == "block"
    
    def test_remove_rule(self):
        """Testa remoção de regra."""
        condition = lambda t: t.get('amount', 0) > 5000
        self.engine.add_rule("test_rule", "Teste", condition)
        
        initial_count = len(self.engine.rules)
        result = self.engine.remove_rule("test_rule")
        
        assert result == True
        assert len(self.engine.rules) == initial_count - 1
        
        result = self.engine.remove_rule("nonexistent_rule")
        assert result == False
    
    def test_enable_disable_rule(self):
        """Testa ativação e desativação de regras."""
        condition = lambda t: t.get('amount', 0) > 5000
        self.engine.add_rule("test_rule", "Teste", condition)
        
        result = self.engine.disable_rule("test_rule")
        assert result == True
        
        rule = next((r for r in self.engine.rules if r.name == "test_rule"), None)
        assert rule is not None
        assert rule.enabled == False
        
        result = self.engine.enable_rule("test_rule")
        assert result == True
        assert rule.enabled == True
    
    def test_evaluate_transaction_no_rules_triggered(self):
        """Testa avaliação de transação que não aciona regras."""
        transaction = {
            'transaction_id': 'test_001',
            'amount': 50.0,
            'merchant_category': 'grocery',
            'country': 'BR',
            'transaction_hour': 14,
            'is_weekend': False
        }
        
        result = self.engine.evaluate_transaction(transaction)
        
        assert result['transaction_id'] == 'test_001'
        assert result['final_action'] == 'allow'
        assert len(result['triggered_rules']) == 0
        assert result['risk_score'] == 0.0
    
    def test_evaluate_transaction_with_rules_triggered(self):
        """Testa avaliação de transação que aciona regras."""
        transaction = {
            'transaction_id': 'test_002',
            'amount': 15000.0,
            'merchant_category': 'online',
            'country': 'XX',
            'transaction_hour': 3,
            'is_weekend': True
        }
        
        result = self.engine.evaluate_transaction(transaction)
        
        assert result['transaction_id'] == 'test_002'
        assert len(result['triggered_rules']) > 0
        assert result['risk_score'] > 0.0
        assert result['final_action'] in ['flag', 'review', 'block']
    
    def test_evaluate_batch(self):
        """Testa avaliação em lote."""
        transactions = [
            {
                'transaction_id': 'batch_001',
                'amount': 100.0,
                'country': 'BR',
                'transaction_hour': 14
            },
            {
                'transaction_id': 'batch_002',
                'amount': 20000.0,
                'country': 'BR',
                'transaction_hour': 14
            }
        ]
        
        results = self.engine.evaluate_batch(transactions)
        
        assert len(results) == 2
        assert results[0]['transaction_id'] == 'batch_001'
        assert results[1]['transaction_id'] == 'batch_002'
        
        # Segunda transação deve ter score maior
        assert results[1]['risk_score'] >= results[0]['risk_score']
    
    def test_calculate_risk_score(self):
        """Testa cálculo de score de risco."""
        # Sem regras acionadas
        score = self.engine._calculate_risk_score([])
        assert score == 0.0
        
        # Com regras acionadas
        triggered_rules = [
            {'action': 'flag', 'priority': 3},
            {'action': 'review', 'priority': 2},
            {'action': 'block', 'priority': 1}
        ]
        
        score = self.engine._calculate_risk_score(triggered_rules)
        assert 0.0 < score <= 1.0
    
    def test_create_custom_rule(self):
        """Testa criação de regra customizada."""
        rule_definition = {
            'name': 'custom_amount_rule',
            'description': 'Valor customizado',
            'field': 'amount',
            'operator': 'greater_than',
            'value': 2000,
            'action': 'review',
            'priority': 2
        }
        
        result = self.engine.create_custom_rule(rule_definition)
        assert result == True
        
        # Verifica se a regra foi criada
        rule = next((r for r in self.engine.rules if r.name == 'custom_amount_rule'), None)
        assert rule is not None
        assert rule.action == 'review'
        assert rule.priority == 2
    
    def test_custom_rule_operators(self):
        """Testa diferentes operadores de regras customizadas."""
        rule_def = {
            'name': 'equals_test',
            'description': 'Teste equals',
            'field': 'country',
            'operator': 'equals',
            'value': 'US',
            'action': 'flag'
        }
        
        result = self.engine.create_custom_rule(rule_def)
        assert result == True
        
        rule_def = {
            'name': 'in_test',
            'description': 'Teste in',
            'field': 'merchant_category',
            'operator': 'in',
            'value': ['online', 'gambling'],
            'action': 'review'
        }
        
        result = self.engine.create_custom_rule(rule_def)
        assert result == True
        
        # Testa operador inválido
        rule_def = {
            'name': 'invalid_test',
            'description': 'Teste inválido',
            'field': 'amount',
            'operator': 'invalid_operator',
            'value': 1000,
            'action': 'flag'
        }
        
        result = self.engine.create_custom_rule(rule_def)
        assert result == False
    
    def test_get_rules_summary(self):
        """Testa obtenção de resumo das regras."""
        summary = self.engine.get_rules_summary()
        
        assert isinstance(summary, list)
        assert len(summary) > 0
        
        # Verifica estrutura do primeiro item
        first_rule = summary[0]
        assert 'name' in first_rule
        assert 'description' in first_rule
        assert 'action' in first_rule
        assert 'priority' in first_rule
        assert 'enabled' in first_rule
        assert 'triggered_count' in first_rule


class TestBusinessRulesIntegration:
    """Testes de integração para BusinessRules."""
    
    def test_full_evaluation_pipeline(self):
        """Testa pipeline completo de avaliação."""
        engine = BusinessRulesEngine()
        
        # Transação normal
        normal_transaction = {
            'transaction_id': 'normal_001',
            'amount': 150.0,
            'merchant_category': 'grocery',
            'payment_method': 'credit_card',
            'country': 'BR',
            'transaction_hour': 14,
            'is_weekend': False
        }
        
        # Transação suspeita
        suspicious_transaction = {
            'transaction_id': 'suspicious_001',
            'amount': 25000.0,  # Muito alto
            'merchant_category': 'online',
            'payment_method': 'credit_card',
            'country': 'XX',  # País suspeito
            'transaction_hour': 3,  # Madrugada
            'is_weekend': True
        }
        
        # Avalia ambas
        normal_result = engine.evaluate_transaction(normal_transaction)
        suspicious_result = engine.evaluate_transaction(suspicious_transaction)
        
        # Transação normal deve ter score baixo
        assert normal_result['risk_score'] < 0.5
        assert normal_result['final_action'] in ['allow', 'flag']
        
        # Transação suspeita deve ter score alto
        assert suspicious_result['risk_score'] > normal_result['risk_score']
        assert len(suspicious_result['triggered_rules']) > 0
        assert suspicious_result['final_action'] in ['review', 'block']


if __name__ == "__main__":
    # Executa testes se rodado diretamente
    pytest.main([__file__, "-v"])
