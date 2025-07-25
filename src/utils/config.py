"""
Utilitários de Configuração

Funções para carregar e gerenciar configurações do sistema.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carrega configurações do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com as configurações
        
    Raises:
        FileNotFoundError: Se o arquivo não for encontrado
        yaml.YAMLError: Se houver erro no parsing do YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Tenta carregar o arquivo de exemplo se o principal não existir
        example_path = "config/config.example.yaml"
        if Path(example_path).exists():
            print(f"⚠️  Arquivo {config_path} não encontrado. Usando {example_path}")
            config_file = Path(example_path)
        else:
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {config_path}\n"
                f"Copie o arquivo de exemplo: cp {example_path} {config_path}"
            )
    
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        # Validação básica da configuração
        _validate_config(config)
        
        # Expansão de variáveis de ambiente
        config = _expand_env_vars(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Erro ao carregar configuração: {e}")


def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """
    Salva configurações em arquivo YAML.
    
    Args:
        config: Dicionário com as configurações
        config_path: Caminho para salvar o arquivo
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Valida a estrutura básica da configuração.
    
    Args:
        config: Configuração a ser validada
        
    Raises:
        ValueError: Se a configuração for inválida
    """
    required_sections = ['data', 'models', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")
    
    # Validações específicas
    if 'synthetic' in config['data']:
        synthetic_config = config['data']['synthetic']
        if synthetic_config.get('fraud_rate', 0) > 0.5:
            raise ValueError("Taxa de fraude não pode ser maior que 50%")


def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expande variáveis de ambiente na configuração.
    
    Args:
        config: Configuração original
        
    Returns:
        Configuração com variáveis expandidas
    """
    def expand_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        return value
    
    return expand_value(config)


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Obtém valor da configuração usando notação de ponto.
    
    Args:
        config: Configuração
        key_path: Caminho da chave (ex: 'data.synthetic.n_samples')
        default: Valor padrão se a chave não existir
        
    Returns:
        Valor da configuração ou default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def update_config_value(config: Dict[str, Any], key_path: str, new_value) -> Dict[str, Any]:
    """
    Atualiza valor na configuração usando notação de ponto.
    
    Args:
        config: Configuração
        key_path: Caminho da chave (ex: 'data.synthetic.n_samples')
        new_value: Novo valor
        
    Returns:
        Configuração atualizada
    """
    keys = key_path.split('.')
    current = config
    
    # Navega até o penúltimo nível
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Atualiza o valor final
    current[keys[-1]] = new_value
    
    return config
