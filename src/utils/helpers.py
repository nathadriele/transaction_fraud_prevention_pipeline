"""
Utilitários Gerais

Funções auxiliares utilizadas em todo o projeto.
"""

import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        path: Caminho do diretório
        
    Returns:
        Path object do diretório
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Salva um modelo usando joblib.
    
    Args:
        model: Modelo a ser salvo
        filepath: Caminho para salvar o modelo
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    joblib.dump(model, filepath)


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Carrega um modelo usando joblib.
    
    Args:
        filepath: Caminho do modelo
        
    Returns:
        Modelo carregado
    """
    return joblib.load(filepath)


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Salva dados em formato JSON.
    
    Args:
        data: Dados a serem salvos
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Carrega dados de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dados carregados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_hash(filepath: Union[str, Path]) -> str:
    """
    Calcula hash MD5 de um arquivo.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Hash MD5 do arquivo
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_currency(value: float, currency: str = "BRL") -> str:
    """
    Formata valor monetário.
    
    Args:
        value: Valor a ser formatado
        currency: Código da moeda
        
    Returns:
        Valor formatado
    """
    if currency == "BRL":
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        return f"{currency} {value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como percentual.
    
    Args:
        value: Valor a ser formatado (0.1 = 10%)
        decimals: Número de casas decimais
        
    Returns:
        Valor formatado como percentual
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calcula número de dias úteis entre duas datas.
    
    Args:
        start_date: Data inicial
        end_date: Data final
        
    Returns:
        Número de dias úteis
    """
    return pd.bdate_range(start_date, end_date).size


def get_time_features(timestamp: pd.Timestamp) -> Dict[str, int]:
    """
    Extrai features temporais de um timestamp.
    
    Args:
        timestamp: Timestamp para extrair features
        
    Returns:
        Dicionário com features temporais
    """
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'quarter': timestamp.quarter,
        'is_weekend': timestamp.dayofweek >= 5,
        'is_business_hour': 9 <= timestamp.hour <= 17,
        'is_night': timestamp.hour < 6 or timestamp.hour > 22
    }


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando método IQR.
    
    Args:
        data: Série de dados
        multiplier: Multiplicador para definir outliers
        
    Returns:
        Série booleana indicando outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detecta outliers usando Z-score.
    
    Args:
        data: Série de dados
        threshold: Threshold do Z-score
        
    Returns:
        Série booleana indicando outliers
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def memory_usage_mb() -> float:
    """
    Retorna uso atual de memória em MB.
    
    Returns:
        Uso de memória em MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def execution_time(func):
    """
    Decorator para medir tempo de execução de funções.
    
    Args:
        func: Função a ser decorada
        
    Returns:
        Função decorada
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_duration = end_time - start_time
        
        print(f"⏱️  {func.__name__} executada em {execution_duration.total_seconds():.2f}s")
        return result
    
    return wrapper


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000):
    """
    Divide DataFrame em chunks menores.
    
    Args:
        df: DataFrame a ser dividido
        chunk_size: Tamanho de cada chunk
        
    Yields:
        Chunks do DataFrame
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisão segura que evita divisão por zero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se denominador for zero
        
    Returns:
        Resultado da divisão ou valor padrão
    """
    return numerator / denominator if denominator != 0 else default


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Valida se DataFrame possui colunas obrigatórias.
    
    Args:
        df: DataFrame a ser validado
        required_columns: Lista de colunas obrigatórias
        
    Returns:
        True se válido, False caso contrário
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
        return False
    return True


def get_project_root() -> Path:
    """
    Retorna o diretório raiz do projeto.
    
    Returns:
        Path do diretório raiz
    """
    current_path = Path(__file__).resolve()
    
    # Procura pelo arquivo README.md ou requirements.txt
    for parent in current_path.parents:
        if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
            return parent
    
    # Se não encontrar, retorna o diretório atual
    return Path.cwd()
