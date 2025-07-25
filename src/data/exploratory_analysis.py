"""
Análise Exploratória de Dados (EDA)

Módulo para análise exploratória e detecção de padrões em dados transacionais.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import sys
import os

# Importações opcionais para visualização
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.helpers import detect_outliers_iqr, detect_outliers_zscore, format_currency, format_percentage

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class ExploratoryDataAnalysis:
    """
    Classe para análise exploratória de dados transacionais.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o analisador EDA.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config or load_config()
        
        # Configurações de visualização (se disponível)
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            except:
                pass  # Usa configuração padrão se houver erro

        # Configurações do plotly
        self.plotly_template = "plotly_white" if PLOTLY_AVAILABLE else None
        
    def generate_data_overview(self, df: pd.DataFrame) -> Dict:
        """
        Gera visão geral dos dados.
        
        Args:
            df: DataFrame com dados transacionais
            
        Returns:
            Dicionário com estatísticas gerais
        """
        logger.info("Gerando visão geral dos dados...")
        
        overview = {
            'total_transactions': len(df),
            'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None,
                'days': (df['timestamp'].max() - df['timestamp'].min()).days if 'timestamp' in df.columns else 0
            },
            'fraud_statistics': {
                'total_frauds': df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
                'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
                'legitimate_transactions': (~df['is_fraud']).sum() if 'is_fraud' in df.columns else len(df)
            },
            'amount_statistics': {
                'total_volume': df['amount'].sum() if 'amount' in df.columns else 0,
                'avg_transaction': df['amount'].mean() if 'amount' in df.columns else 0,
                'median_transaction': df['amount'].median() if 'amount' in df.columns else 0,
                'max_transaction': df['amount'].max() if 'amount' in df.columns else 0,
                'min_transaction': df['amount'].min() if 'amount' in df.columns else 0
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_transactions': df.duplicated().sum(),
                'negative_amounts': (df['amount'] < 0).sum() if 'amount' in df.columns else 0
            }
        }
        
        return overview
    
    def analyze_fraud_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analisa padrões de fraude nos dados.
        
        Args:
            df: DataFrame com dados transacionais
            
        Returns:
            Dicionário com análises de fraude
        """
        logger.info("Analisando padrões de fraude...")
        
        if 'is_fraud' not in df.columns:
            logger.warning("Coluna 'is_fraud' não encontrada")
            return {}
        
        fraud_df = df[df['is_fraud'] == True]
        legit_df = df[df['is_fraud'] == False]
        
        patterns = {
            'amount_analysis': {
                'fraud_avg_amount': fraud_df['amount'].mean(),
                'legit_avg_amount': legit_df['amount'].mean(),
                'fraud_median_amount': fraud_df['amount'].median(),
                'legit_median_amount': legit_df['amount'].median(),
                'amount_difference_ratio': fraud_df['amount'].mean() / legit_df['amount'].mean() if legit_df['amount'].mean() > 0 else 0
            },
            'temporal_analysis': {},
            'categorical_analysis': {},
            'geographical_analysis': {}
        }
        
        # Análise temporal
        if 'transaction_hour' in df.columns:
            fraud_by_hour = fraud_df['transaction_hour'].value_counts().sort_index()
            legit_by_hour = legit_df['transaction_hour'].value_counts().sort_index()
            
            patterns['temporal_analysis'] = {
                'fraud_peak_hours': fraud_by_hour.nlargest(3).index.tolist(),
                'fraud_low_hours': fraud_by_hour.nsmallest(3).index.tolist(),
                'hourly_fraud_rate': (fraud_by_hour / (fraud_by_hour + legit_by_hour)).fillna(0).to_dict()
            }
        
        # Análise categórica
        categorical_cols = ['merchant_category', 'payment_method', 'device_type']
        for col in categorical_cols:
            if col in df.columns:
                fraud_dist = fraud_df[col].value_counts(normalize=True)
                legit_dist = legit_df[col].value_counts(normalize=True)
                
                patterns['categorical_analysis'][col] = {
                    'fraud_distribution': fraud_dist.to_dict(),
                    'legit_distribution': legit_dist.to_dict(),
                    'risk_by_category': (fraud_df[col].value_counts() / df[col].value_counts()).fillna(0).to_dict()
                }
        
        # Análise geográfica
        if 'country' in df.columns:
            fraud_by_country = fraud_df['country'].value_counts()
            total_by_country = df['country'].value_counts()
            
            patterns['geographical_analysis'] = {
                'high_risk_countries': (fraud_by_country / total_by_country).nlargest(5).to_dict(),
                'fraud_by_country': fraud_by_country.to_dict()
            }
        
        return patterns
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detecta anomalias nos dados.
        
        Args:
            df: DataFrame com dados transacionais
            
        Returns:
            Dicionário com anomalias detectadas
        """
        logger.info("Detectando anomalias...")
        
        anomalies = {}
        
        # Anomalias em valores
        if 'amount' in df.columns:
            amount_outliers_iqr = detect_outliers_iqr(df['amount'])
            amount_outliers_zscore = detect_outliers_zscore(df['amount'])
            
            anomalies['amount_anomalies'] = {
                'outliers_iqr_count': amount_outliers_iqr.sum(),
                'outliers_iqr_percentage': amount_outliers_iqr.mean(),
                'outliers_zscore_count': amount_outliers_zscore.sum(),
                'outliers_zscore_percentage': amount_outliers_zscore.mean(),
                'suspicious_round_amounts': (df['amount'] % 100 == 0).sum(),
                'very_high_amounts': (df['amount'] > df['amount'].quantile(0.99)).sum()
            }
        
        # Anomalias temporais
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            night_transactions = ((df['hour'] < 6) | (df['hour'] > 22)).sum()
            weekend_transactions = df['is_weekend'].sum() if 'is_weekend' in df.columns else 0
            
            anomalies['temporal_anomalies'] = {
                'night_transactions': night_transactions,
                'night_percentage': night_transactions / len(df),
                'weekend_transactions': weekend_transactions,
                'weekend_percentage': weekend_transactions / len(df)
            }
        
        # Anomalias de velocidade (se disponível)
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            user_transaction_counts = df['user_id'].value_counts()
            high_velocity_users = (user_transaction_counts > user_transaction_counts.quantile(0.95)).sum()
            
            anomalies['velocity_anomalies'] = {
                'high_velocity_users': high_velocity_users,
                'max_transactions_per_user': user_transaction_counts.max(),
                'avg_transactions_per_user': user_transaction_counts.mean()
            }
        
        return anomalies
    
    def create_fraud_visualizations(self, df: pd.DataFrame) -> Dict:
        """
        Cria visualizações relacionadas a fraudes.

        Args:
            df: DataFrame com dados transacionais

        Returns:
            Dicionário com figuras plotly
        """
        logger.info("Criando visualizações de fraude...")

        if 'is_fraud' not in df.columns:
            return {}

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly não disponível. Visualizações não serão criadas.")
            return {}

        figures = {}
        
        # 1. Distribuição de valores por tipo de transação
        fig_amount = px.box(
            df, 
            x='is_fraud', 
            y='amount',
            title='Distribuição de Valores: Fraude vs Legítima',
            labels={'is_fraud': 'É Fraude?', 'amount': 'Valor (R$)'},
            template=self.plotly_template
        )
        figures['amount_distribution'] = fig_amount
        
        # 2. Taxa de fraude por hora
        if 'transaction_hour' in df.columns:
            hourly_fraud = df.groupby('transaction_hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            hourly_fraud['fraud_rate'] = hourly_fraud['mean']
            
            fig_hourly = px.line(
                hourly_fraud,
                x='transaction_hour',
                y='fraud_rate',
                title='Taxa de Fraude por Hora do Dia',
                labels={'transaction_hour': 'Hora', 'fraud_rate': 'Taxa de Fraude'},
                template=self.plotly_template
            )
            figures['hourly_fraud_rate'] = fig_hourly
        
        # 3. Fraude por categoria de comerciante
        if 'merchant_category' in df.columns:
            category_fraud = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            category_fraud = category_fraud.sort_values('mean', ascending=False)
            
            fig_category = px.bar(
                category_fraud,
                x='merchant_category',
                y='mean',
                title='Taxa de Fraude por Categoria de Comerciante',
                labels={'merchant_category': 'Categoria', 'mean': 'Taxa de Fraude'},
                template=self.plotly_template
            )
            fig_category.update_xaxis(tickangle=45)
            figures['category_fraud_rate'] = fig_category
        
        # 4. Heatmap de correlação
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title='Matriz de Correlação - Features Numéricas',
                template=self.plotly_template,
                color_continuous_scale='RdBu_r'
            )
            figures['correlation_matrix'] = fig_corr
        
        return figures
    
    def create_general_visualizations(self, df: pd.DataFrame) -> Dict:
        """
        Cria visualizações gerais dos dados.

        Args:
            df: DataFrame com dados transacionais

        Returns:
            Dicionário com figuras plotly
        """
        logger.info("Criando visualizações gerais...")

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly não disponível. Visualizações não serão criadas.")
            return {}

        figures = {}
        
        # 1. Distribuição de valores
        if 'amount' in df.columns:
            fig_amount_hist = px.histogram(
                df,
                x='amount',
                nbins=50,
                title='Distribuição de Valores das Transações',
                labels={'amount': 'Valor (R$)', 'count': 'Frequência'},
                template=self.plotly_template
            )
            figures['amount_histogram'] = fig_amount_hist
        
        # 2. Transações ao longo do tempo
        if 'timestamp' in df.columns:
            daily_transactions = df.set_index('timestamp').resample('D').size().reset_index()
            daily_transactions.columns = ['date', 'transaction_count']
            
            fig_timeline = px.line(
                daily_transactions,
                x='date',
                y='transaction_count',
                title='Volume de Transações ao Longo do Tempo',
                labels={'date': 'Data', 'transaction_count': 'Número de Transações'},
                template=self.plotly_template
            )
            figures['transaction_timeline'] = fig_timeline
        
        # 3. Top categorias de comerciantes
        if 'merchant_category' in df.columns:
            category_counts = df['merchant_category'].value_counts().head(10)
            
            fig_categories = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title='Top 10 Categorias de Comerciantes',
                labels={'x': 'Categoria', 'y': 'Número de Transações'},
                template=self.plotly_template
            )
            figures['top_categories'] = fig_categories
        
        # 4. Métodos de pagamento
        if 'payment_method' in df.columns:
            payment_counts = df['payment_method'].value_counts()
            
            fig_payment = px.pie(
                values=payment_counts.values,
                names=payment_counts.index,
                title='Distribuição por Método de Pagamento',
                template=self.plotly_template
            )
            figures['payment_methods'] = fig_payment
        
        return figures
    
    def generate_statistical_report(self, df: pd.DataFrame) -> Dict:
        """
        Gera relatório estatístico detalhado.
        
        Args:
            df: DataFrame com dados transacionais
            
        Returns:
            Dicionário com estatísticas detalhadas
        """
        logger.info("Gerando relatório estatístico...")
        
        report = {
            'descriptive_statistics': {},
            'distribution_tests': {},
            'correlation_analysis': {},
            'hypothesis_tests': {}
        }
        
        # Estatísticas descritivas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                report['descriptive_statistics'][col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'skewness': stats.skew(df[col].dropna()) if SCIPY_AVAILABLE else None,
                    'kurtosis': stats.kurtosis(df[col].dropna()) if SCIPY_AVAILABLE else None
                }
        
        # Testes de normalidade
        if 'amount' in df.columns and SCIPY_AVAILABLE:
            shapiro_stat, shapiro_p = stats.shapiro(df['amount'].sample(min(5000, len(df))))

            report['distribution_tests']['amount_normality'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Análise de correlação
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Encontra correlações mais fortes
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            # Ordena por correlação absoluta
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
            
            report['correlation_analysis'] = {
                'strongest_correlations': corr_pairs[:10],
                'correlation_matrix': corr_matrix.to_dict()
            }
        
        # Testes de hipótese (se há fraudes)
        if 'is_fraud' in df.columns and 'amount' in df.columns and SCIPY_AVAILABLE:
            fraud_amounts = df[df['is_fraud'] == True]['amount']
            legit_amounts = df[df['is_fraud'] == False]['amount']

            # Teste t para diferença de médias
            t_stat, t_p = stats.ttest_ind(fraud_amounts, legit_amounts)

            # Teste Mann-Whitney U (não paramétrico)
            u_stat, u_p = stats.mannwhitneyu(fraud_amounts, legit_amounts, alternative='two-sided')

            report['hypothesis_tests']['amount_difference'] = {
                't_test': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < 0.05},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_p, 'significant': u_p < 0.05}
            }
        
        return report
    
    def save_analysis_results(self, 
                            overview: Dict, 
                            patterns: Dict, 
                            anomalies: Dict, 
                            report: Dict,
                            output_path: str = "data/processed/eda_results.json") -> None:
        """
        Salva resultados da análise.
        
        Args:
            overview: Visão geral dos dados
            patterns: Padrões de fraude
            anomalies: Anomalias detectadas
            report: Relatório estatístico
            output_path: Caminho para salvar resultados
        """
        from src.utils.helpers import save_json, ensure_dir
        from pathlib import Path
        
        results = {
            'overview': overview,
            'fraud_patterns': patterns,
            'anomalies': anomalies,
            'statistical_report': report,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Converte valores numpy para tipos Python nativos
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results = convert_numpy(results)
        
        ensure_dir(Path(output_path).parent)
        save_json(results, output_path)
        
        logger.info(f"Resultados da análise salvos em: {output_path}")


def main():
    """Função principal para execução standalone."""
    
    from src.data.data_loader import DataLoader
    
    print("🔍 Iniciando Análise Exploratória de Dados...")
    
    # Carrega dados
    loader = DataLoader()
    users_df, transactions_df = loader.load_synthetic_data()
    
    # Executa EDA
    eda = ExploratoryDataAnalysis()
    
    # Análises
    overview = eda.generate_data_overview(transactions_df)
    patterns = eda.analyze_fraud_patterns(transactions_df)
    anomalies = eda.detect_anomalies(transactions_df)
    report = eda.generate_statistical_report(transactions_df)
    
    # Salva resultados
    eda.save_analysis_results(overview, patterns, anomalies, report)
    
    # Mostra resumo
    print("✅ Análise Exploratória concluída!")
    print(f"📊 Total de transações: {overview['total_transactions']:,}")
    print(f"📊 Taxa de fraude: {format_percentage(overview['fraud_statistics']['fraud_rate'])}")
    print(f"📊 Volume total: {format_currency(overview['amount_statistics']['total_volume'])}")
    print(f"📊 Anomalias detectadas: {anomalies.get('amount_anomalies', {}).get('outliers_iqr_count', 0):,}")


if __name__ == "__main__":
    main()
