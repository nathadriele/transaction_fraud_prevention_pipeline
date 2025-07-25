"""
Utilitários de Logging

Sistema de logging configurável para o projeto.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log (opcional)
        format_string: Formato personalizado das mensagens
    """
    # Remove handlers padrão do loguru
    loguru_logger.remove()
    
    # Formato padrão
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Handler para console
    loguru_logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Handler para arquivo (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger configurado.
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        Logger configurado
    """
    # Cria um logger padrão do Python que redireciona para loguru
    logger = logging.getLogger(name)
    
    # Remove handlers existentes
    logger.handlers.clear()
    
    # Cria handler personalizado que redireciona para loguru
    class LoguruHandler(logging.Handler):
        def emit(self, record):
            # Obtém o nível correspondente no loguru
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Obtém informações do frame
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            # Log usando loguru
            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Adiciona o handler personalizado
    logger.addHandler(LoguruHandler())
    logger.setLevel(logging.DEBUG)
    
    return logger


class FraudDetectionLogger:
    """
    Logger especializado para detecção de fraudes.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_transaction_processed(self, transaction_id: str, is_fraud: bool, score: float):
        """Log de transação processada."""
        status = "FRAUD" if is_fraud else "LEGITIMATE"
        self.logger.info(
            f"Transaction processed | ID: {transaction_id} | "
            f"Status: {status} | Score: {score:.4f}"
        )
    
    def log_model_prediction(self, model_name: str, prediction: bool, confidence: float):
        """Log de predição do modelo."""
        self.logger.debug(
            f"Model prediction | Model: {model_name} | "
            f"Prediction: {prediction} | Confidence: {confidence:.4f}"
        )
    
    def log_rule_triggered(self, rule_name: str, transaction_id: str, details: str):
        """Log de regra de negócio acionada."""
        self.logger.warning(
            f"Business rule triggered | Rule: {rule_name} | "
            f"Transaction: {transaction_id} | Details: {details}"
        )
    
    def log_performance_metrics(self, metrics: dict):
        """Log de métricas de performance."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Performance metrics | {metrics_str}")
    
    def log_data_quality_issue(self, issue: str, details: str):
        """Log de problema de qualidade de dados."""
        self.logger.error(f"Data quality issue | Issue: {issue} | Details: {details}")
    
    def log_model_drift(self, metric: str, current_value: float, threshold: float):
        """Log de drift do modelo."""
        self.logger.warning(
            f"Model drift detected | Metric: {metric} | "
            f"Current: {current_value:.4f} | Threshold: {threshold:.4f}"
        )


# Instância global do logger especializado
fraud_logger = None


def get_fraud_logger(name: str) -> FraudDetectionLogger:
    """
    Obtém o logger especializado para detecção de fraudes.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger especializado
    """
    global fraud_logger
    if fraud_logger is None:
        fraud_logger = FraudDetectionLogger(name)
    return fraud_logger
