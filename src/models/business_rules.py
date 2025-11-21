"""
Sistema de Regras de Negócio para Detecção de Fraudes.

Engine de regras customizáveis que complementa os modelos de ML.
"""

import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.utils.config import load_config
from src.utils.logger import get_fraud_logger, get_logger
from src.utils.helpers import ensure_dir, save_json

# Garante acesso ao diretório raiz do projeto
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

logger = get_logger(__name__)
fraud_logger = get_fraud_logger(__name__)


class Rule:
    """Regra de negócio individual."""

    def __init__(
        self,
        name: str,
        description: str,
        condition: Callable,
        action: str = "flag",
        priority: int = 1,
        enabled: bool = True,
    ):
        self.name = name
        self.description = description
        self.condition = condition
        self.action = action
        self.priority = priority
        self.enabled = enabled
        self.triggered_count = 0
        self.last_triggered: Optional[datetime] = None

    def evaluate(self, transaction: Dict) -> bool:
        """Avalia se a regra é acionada para uma transação."""
        if not self.enabled:
            return False

        try:
            result = self.condition(transaction)
            if result:
                self.triggered_count += 1
                self.last_triggered = datetime.now()
                fraud_logger.log_rule_triggered(
                    self.name,
                    transaction.get("transaction_id", "unknown"),
                    self.description,
                )
            return result
        except Exception as e:
            logger.error(f"Erro ao avaliar regra {self.name}: {e}")
            return False

    def to_dict(self) -> Dict:
        """Converte a regra para dicionário."""
        return {
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "priority": self.priority,
            "enabled": self.enabled,
            "triggered_count": self.triggered_count,
            "last_triggered": self.last_triggered.isoformat()
            if self.last_triggered
            else None,
        }


class BusinessRulesEngine:
    """Engine de regras de negócio para detecção de fraudes."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        self.rules: List[Rule] = []
        self.rule_stats: Dict[str, Any] = {}
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        rules_config = self.config.get("business_rules", {})

        # Valor
        amount_rules = rules_config.get("amount_rules", {})
        if amount_rules.get("max_single_transaction"):
            self.add_rule(
                name="high_amount_transaction",
                description=(
                    f"Transação acima de {amount_rules['max_single_transaction']}"
                ),
                condition=lambda t: t.get("amount", 0)
                > amount_rules["max_single_transaction"],
                action="review",
                priority=2,
            )

        if amount_rules.get("suspicious_round_amounts"):
            self.add_rule(
                name="round_amount_suspicious",
                description="Valor suspeito (múltiplo de 100)",
                condition=lambda t: t.get("amount", 0) % 100 == 0
                and t.get("amount", 0) >= 1000,
                action="flag",
                priority=3,
            )

        # Velocidade
        velocity_rules = rules_config.get("velocity_rules", {})
        if velocity_rules.get("max_transactions_per_hour"):
            self.add_rule(
                name="high_velocity_user",
                description=(
                    f"Mais de {velocity_rules['max_transactions_per_hour']} "
                    f"transações por hora"
                ),
                condition=lambda t: t.get("transactions_last_hour", 0)
                > velocity_rules["max_transactions_per_hour"],
                action="block",
                priority=1,
            )

        # Localização
        location_rules = rules_config.get("location_rules", {})
        if location_rules.get("suspicious_countries"):
            suspicious_countries = location_rules["suspicious_countries"]
            self.add_rule(
                name="suspicious_country",
                description=f"Transação de país suspeito: {suspicious_countries}",
                condition=lambda t: t.get("country") in suspicious_countries,
                action="review",
                priority=2,
            )

        # Tempo
        time_rules = rules_config.get("time_rules", {})
        if time_rules.get("suspicious_hours"):
            suspicious_hours = time_rules["suspicious_hours"]
            self.add_rule(
                name="night_transaction",
                description=f"Transação em horário suspeito: {suspicious_hours}",
                condition=lambda t: t.get("transaction_hour", 12) in suspicious_hours,
                action="flag",
                priority=3,
            )

        if time_rules.get("weekend_multiplier"):
            self.add_rule(
                name="weekend_high_amount",
                description="Transação de alto valor no fim de semana",
                condition=lambda t: (
                    t.get("is_weekend", False)
                    and t.get("amount", 0)
                    > amount_rules.get("max_single_transaction", 10000) * 0.5
                ),
                action="review",
                priority=2,
            )

        logger.info("Regras padrão inicializadas: %d regras", len(self.rules))

    def add_rule(
        self,
        name: str,
        description: str,
        condition: Callable,
        action: str = "flag",
        priority: int = 1,
        enabled: bool = True,
    ) -> None:
        rule = Rule(name, description, condition, action, priority, enabled)
        self.rules.append(rule)
        logger.info("Regra adicionada: %s", name)

    def remove_rule(self, name: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                logger.info("Regra removida: %s", name)
                return True
        return False

    def enable_rule(self, name: str) -> bool:
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                logger.info("Regra ativada: %s", name)
                return True
        return False

    def disable_rule(self, name: str) -> bool:
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                logger.info("Regra desativada: %s", name)
                return True
        return False

    def evaluate_transaction(self, transaction: Dict) -> Dict[str, Any]:
        """Avalia uma transação contra todas as regras."""
        triggered_rules: List[Dict[str, Any]] = []
        highest_priority = 5  # menor número = maior prioridade
        final_action = "allow"

        for rule in sorted(self.rules, key=lambda r: r.priority):
            if rule.evaluate(transaction):
                triggered_rules.append(
                    {
                        "name": rule.name,
                        "description": rule.description,
                        "action": rule.action,
                        "priority": rule.priority,
                    }
                )
                if rule.priority < highest_priority:
                    highest_priority = rule.priority
                    final_action = rule.action

        return {
            "transaction_id": transaction.get("transaction_id"),
            "triggered_rules": triggered_rules,
            "final_action": final_action,
            "risk_score": self._calculate_risk_score(triggered_rules),
            "evaluation_timestamp": datetime.now().isoformat(),
        }

    def _calculate_risk_score(self, triggered_rules: List[Dict]) -> float:
        """Calcula score de risco com base nas regras acionadas."""
        if not triggered_rules:
            return 0.0

        action_weights = {"flag": 0.3, "review": 0.6, "block": 1.0}
        priority_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}

        total_score = 0.0
        for rule in triggered_rules:
            action_weight = action_weights.get(rule["action"], 0.5)
            priority_weight = priority_weights.get(rule["priority"], 0.5)
            total_score += action_weight * priority_weight

        return min(total_score, 1.0)

    def evaluate_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Avalia um lote de transações."""
        logger.info("Avaliando lote de %d transações...", len(transactions))

        results = [self.evaluate_transaction(t) for t in transactions]
        self._update_statistics(results)
        return results

    def _update_statistics(self, results: List[Dict]) -> None:
        total_transactions = len(results)
        flagged_transactions = sum(1 for r in results if r["triggered_rules"])

        self.rule_stats = {
            "total_transactions_evaluated": total_transactions,
            "flagged_transactions": flagged_transactions,
            "flag_rate": (
                flagged_transactions / total_transactions if total_transactions > 0 else 0
            ),
            "rule_performance": {},
        }

        for rule in self.rules:
            self.rule_stats["rule_performance"][rule.name] = {
                "triggered_count": rule.triggered_count,
                "last_triggered": rule.last_triggered.isoformat()
                if rule.last_triggered
                else None,
                "enabled": rule.enabled,
            }

    def get_rule_statistics(self) -> Dict:
        return self.rule_stats

    def get_rules_summary(self) -> List[Dict]:
        return [rule.to_dict() for rule in self.rules]

    def save_rules_config(self, filepath: str = "config/business_rules.json") -> None:
        rules_config = {
            "rules": self.get_rules_summary(),
            "statistics": self.rule_stats,
            "saved_at": datetime.now().isoformat(),
        }

        ensure_dir(os.path.dirname(filepath))
        save_json(rules_config, filepath)
        logger.info("Configuração das regras salva em: %s", filepath)

    def create_custom_rule(self, rule_definition: Dict) -> bool:
        """Cria uma regra customizada a partir de um dicionário de definição."""
        try:
            name = rule_definition["name"]
            description = rule_definition["description"]
            field = rule_definition["field"]
            operator = rule_definition["operator"]
            value = rule_definition["value"]
            action = rule_definition.get("action", "flag")
            priority = rule_definition.get("priority", 3)

            if operator == "greater_than":
                condition = lambda t: t.get(field, 0) > value
            elif operator == "less_than":
                condition = lambda t: t.get(field, 0) < value
            elif operator == "equals":
                condition = lambda t: t.get(field) == value
            elif operator == "in":
                condition = lambda t: t.get(field) in value
            else:
                logger.error("Operador não suportado: %s", operator)
                return False

            self.add_rule(name, description, condition, action, priority)
            return True

        except Exception as e:
            logger.error("Erro ao criar regra customizada: %s", e)
            return False


def main() -> None:
    """Execução de teste simples do sistema de regras."""
    from src.data.data_loader import DataLoader

    print("Testando Sistema de Regras de Negócio...")

    loader = DataLoader()
    _, transactions_df = loader.load_synthetic_data()

    rules_engine = BusinessRulesEngine()

    test_transactions = transactions_df.head(100).to_dict("records")
    results = rules_engine.evaluate_batch(test_transactions)

    stats = rules_engine.get_rule_statistics()
    rules_engine.save_rules_config()

    print("Sistema de regras testado!")
    print(f"Transações avaliadas: {stats['total_transactions_evaluated']}")
    print(f"Transações sinalizadas: {stats['flagged_transactions']}")
    print(f"Taxa de sinalização: {stats['flag_rate']:.2%}")

    rule_performance = stats["rule_performance"]
    top_rules = sorted(
        rule_performance.items(),
        key=lambda x: x[1]["triggered_count"],
        reverse=True,
    )[:3]

    print("Top 3 regras mais acionadas:")
    for rule_name, perf in top_rules:
        print(f"   - {rule_name}: {perf['triggered_count']} vezes")


if __name__ == "__main__":
    main()
