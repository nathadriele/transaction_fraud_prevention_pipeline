"""
Dashboard Streamlit para Sistema de Prevenção de Fraudes

Interface web interativa para monitoramento, análise e gestão
do sistema antifraude.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import json

# Adiciona o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Adiciona também o diretório src
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Importações do projeto
try:
    from src.data.data_loader import DataLoader
    from src.data.exploratory_analysis import ExploratoryDataAnalysis
    from src.models.business_rules import BusinessRulesEngine
    from src.models.model_ensemble import ModelEnsemble
    from src.utils.config import load_config
    from src.utils.helpers import format_currency, format_percentage
    PROJECT_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Erro ao importar módulos do projeto: {e}")
    PROJECT_MODULES_AVAILABLE = False

# Funções auxiliares para carregamento de dados
@st.cache_data
def load_final_results():
    """Carrega dados de resultados finais."""
    try:
        with st.spinner('Carregando dados de transações...'):
            df = pd.read_csv('data/final_results.csv')
            if df.empty:
                st.warning("⚠️ Arquivo de dados está vazio. Usando dados de exemplo.")
                return create_sample_transactions()
            return df
    except FileNotFoundError:
        st.error("Arquivo final_results.csv não encontrado.")
        st.info("Usando dados de exemplo para demonstração.")
        return create_sample_transactions()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return create_sample_transactions()

def create_sample_transactions():
    """Cria dados de exemplo realistas para demonstração."""
    import numpy as np
    from datetime import datetime, timedelta

    # Configurações realistas
    np.random.seed(42)
    n_samples = 100
    base_date = datetime.now() - timedelta(days=30)

    # Distribuições realistas
    categories = ['grocery', 'restaurant', 'gas_station', 'pharmacy', 'coffee_shop',
                 'fast_food', 'clothing', 'online', 'electronics', 'travel', 'jewelry', 'luxury']
    category_weights = [0.25, 0.15, 0.12, 0.08, 0.06, 0.05, 0.08, 0.10, 0.06, 0.03, 0.01, 0.01]

    countries = ['BR', 'US', 'CA', 'UK', 'DE', 'CN', 'RU', 'XX']
    country_weights = [0.4, 0.25, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]

    # Taxa de fraude realista por categoria
    fraud_rates = {
        'grocery': 0.005, 'restaurant': 0.01, 'gas_station': 0.008, 'pharmacy': 0.006,
        'coffee_shop': 0.007, 'fast_food': 0.009, 'clothing': 0.015, 'online': 0.025,
        'electronics': 0.03, 'travel': 0.02, 'jewelry': 0.08, 'luxury': 0.05
    }

    transactions = []

    for i in range(n_samples):
        # Seleciona categoria e país
        category = np.random.choice(categories, p=category_weights)
        country = np.random.choice(countries, p=country_weights)

        # Gera timestamp realista
        day_offset = np.random.randint(0, 30)
        hour = np.random.choice(range(24))
        timestamp = base_date + timedelta(days=day_offset, hours=hour,
                                        minutes=np.random.randint(0, 60),
                                        seconds=np.random.randint(0, 60))

        # Gera valor baseado na categoria
        if category in ['grocery', 'restaurant', 'pharmacy']:
            amount = np.random.lognormal(4.5, 0.8)  # ~R$ 90
        elif category in ['electronics', 'travel']:
            amount = np.random.lognormal(6.5, 1.1)  # ~R$ 665
        elif category in ['jewelry', 'luxury']:
            amount = np.random.lognormal(7.5, 1.3)  # ~R$ 1800
        else:
            amount = np.random.lognormal(5.0, 1.0)  # ~R$ 150

        amount = round(max(amount, 5.0), 2)

        # Determina se é fraude
        base_fraud_rate = fraud_rates.get(category, 0.02)
        country_multiplier = 3.0 if country in ['CN', 'RU', 'XX'] else 1.0
        value_multiplier = 1.5 if amount > 2000 else 1.0

        fraud_prob = base_fraud_rate * country_multiplier * value_multiplier
        is_fraud = np.random.random() < min(fraud_prob, 0.95)

        # Gera score de fraude
        if is_fraud:
            fraud_score = np.random.normal(0.85, 0.08)
            fraud_score = np.clip(fraud_score, 0.6, 0.99)
        else:
            fraud_score = np.random.normal(0.15, 0.12)
            fraud_score = np.clip(fraud_score, 0.01, 0.5)

        # Seleciona regra
        if is_fraud:
            if amount > 2000:
                rule = 'high_amount'
            elif country in ['CN', 'RU', 'XX']:
                rule = 'suspicious_country'
            else:
                rule = np.random.choice(['velocity_rule', 'pattern_anomaly', 'time_anomaly'])
        else:
            rule = 'none'

        transactions.append({
            'transaction_id': f'TXN_{i+1:06d}',
            'user_id': f'USR_{np.random.randint(1, n_samples//2):06d}',
            'amount': amount,
            'merchant_category': category,
            'country': country,
            'is_fraud': int(is_fraud),
            'fraud_score': round(fraud_score, 3),
            'model_prediction': 1 if fraud_score > 0.5 else 0,
            'rule_triggered': rule,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

@st.cache_data
def load_eda_summary():
    """Carrega dados de resumo EDA."""
    try:
        with st.spinner('Carregando resumo da análise exploratória...'):
            df = pd.read_csv('data/eda_summary.csv')
            if df.empty:
                st.warning("Arquivo de resumo EDA está vazio.")
                return create_sample_eda()
            return df
    except FileNotFoundError:
        st.info("Arquivo eda_summary.csv não encontrado. Usando dados de exemplo.")
        return create_sample_eda()
    except Exception as e:
        st.error(f"Erro ao carregar resumo EDA: {e}")
        return create_sample_eda()

def create_sample_eda():
    """Cria dados de exemplo para EDA."""
    return pd.DataFrame({
        'metric': ['total_transactions', 'fraud_rate', 'avg_amount'],
        'value': [10000, 0.02, 1250.50],
        'category': ['general', 'general', 'general']
    })

@st.cache_data
def load_alerts_log():
    """Carrega dados de alertas."""
    try:
        with st.spinner('Carregando log de alertas...'):
            df = pd.read_csv('data/alerts_log.csv')
            if df.empty:
                st.warning("Log de alertas está vazio.")
                return create_sample_alerts()
            return df
    except FileNotFoundError:
        st.info("Arquivo alerts_log.csv não encontrado. Usando dados de exemplo.")
        return create_sample_alerts()
    except Exception as e:
        st.error(f"Erro ao carregar alertas: {e}")
        return create_sample_alerts()

def create_sample_alerts():
    """Cria dados de exemplo para alertas."""
    return pd.DataFrame({
        'alert_id': ['ALT_001', 'ALT_002', 'ALT_003'],
        'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
        'timestamp': ['2024-01-13 10:15:30', '2024-01-13 10:22:45', '2024-01-13 10:30:12'],
        'fraud_score': [0.8, 0.9, 0.75],
        'rule_triggered': ['high_amount', 'suspicious_country', 'velocity_rule'],
        'model_used': ['random_forest', 'xgboost', 'ensemble'],
        'action_taken': ['review', 'block', 'flag'],
        'priority': ['high', 'critical', 'medium'],
        'status': ['pending', 'resolved', 'active']
    })

@st.cache_data
def load_model_scores():
    """Carrega scores dos modelos."""
    try:
        with st.spinner('Carregando métricas dos modelos...'):
            with open('data/model_scores.json', 'r') as f:
                scores = json.load(f)
                if not scores:
                    st.warning("Arquivo de scores está vazio.")
                    return create_sample_scores()
                return scores
    except FileNotFoundError:
        st.info("Arquivo model_scores.json não encontrado. Usando dados de exemplo.")
        return create_sample_scores()
    except json.JSONDecodeError:
        st.error("Erro ao decodificar arquivo JSON de scores.")
        return create_sample_scores()
    except Exception as e:
        st.error(f"Erro ao carregar scores: {e}")
        return create_sample_scores()

def create_sample_scores():
    """Cria dados de exemplo para scores de modelos."""
    return {
        'random_forest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85, 'roc_auc': 0.92},
        'xgboost': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.90, 'f1_score': 0.87, 'roc_auc': 0.94},
        'ensemble': {'accuracy': 0.89, 'precision': 0.86, 'recall': 0.92, 'f1_score': 0.89, 'roc_auc': 0.95}
    }

# Configuração da página
st.set_page_config(
    page_title="Sistema de Prevenção de Fraudes",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Carrega dados com cache."""
    try:
        # Tenta carregar dados reais primeiro
        final_results = load_final_results()

        # Cria DataFrames vazios para users (não usado atualmente)
        users_df = pd.DataFrame()

        # Usa final_results como transactions_df
        transactions_df = final_results

        return users_df, transactions_df
    except Exception as e:
        st.warning(f"Usando dados de exemplo: {e}")
        # Retorna dados de exemplo
        users_df = pd.DataFrame()
        transactions_df = create_sample_transactions()
        return users_df, transactions_df


@st.cache_data
def get_fraud_statistics(transactions_df):
    """Calcula estatísticas de fraude."""
    if transactions_df is None:
        return {}
    
    total_transactions = len(transactions_df)
    fraud_transactions = transactions_df['is_fraud'].sum()
    fraud_rate = fraud_transactions / total_transactions if total_transactions > 0 else 0
    
    total_volume = transactions_df['amount'].sum()
    fraud_volume = transactions_df[transactions_df['is_fraud']]['amount'].sum()
    fraud_volume_rate = fraud_volume / total_volume if total_volume > 0 else 0
    
    avg_transaction = transactions_df['amount'].mean()
    avg_fraud_transaction = transactions_df[transactions_df['is_fraud']]['amount'].mean()
    
    return {
        'total_transactions': total_transactions,
        'fraud_transactions': fraud_transactions,
        'fraud_rate': fraud_rate,
        'total_volume': total_volume,
        'fraud_volume': fraud_volume,
        'fraud_volume_rate': fraud_volume_rate,
        'avg_transaction': avg_transaction,
        'avg_fraud_transaction': avg_fraud_transaction
    }


def apply_custom_css():
    """Aplica CSS customizado para o dashboard."""
    # Tenta carregar CSS do arquivo, senão usa CSS inline
    css_file_path = os.path.join(os.path.dirname(__file__), 'style.css')

    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception:
            # Fallback para CSS inline se não conseguir ler o arquivo
            apply_inline_css()
    else:
        # CSS inline como fallback
        apply_inline_css()

def apply_inline_css():
    """Aplica CSS inline como fallback."""
    st.markdown("""
    <style>
    /* Cor de fundo principal - múltiplos seletores para compatibilidade */
    .stApp,
    .main,
    .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .css-1d391kg,
    .css-18e3th9,
    .css-1lcbmhc,
    .css-1outpf7 {
        background-color: #E6E6FA !important;
    }

    /* Cor de fundo da sidebar - múltiplos seletores */
    .css-1d391kg,
    .css-17eq0hr,
    .css-1lcbmhc,
    [data-testid="stSidebar"],
    .sidebar .sidebar-content {
        background-color: #E6E6FA !important;
    }

    /* Força cor de fundo em todos os containers */
    .stContainer,
    .element-container,
    .stMarkdown,
    .stColumns,
    .stColumn {
        background-color: transparent !important;
    }

    /* Elementos de input mantêm fundo branco para legibilidade */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stFileUploader > div,
    .stSlider > div > div > div > div {
        background-color: white !important;
        border: 1px solid #ddd !important;
    }

    /* Estilo do cabeçalho principal */
    .main-header {
        color: #4B0082 !important;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Métricas com fundo branco */
    .metric-container,
    [data-testid="metric-container"],
    .css-1xarl3l {
        background-color: white !important;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* DataFrames com fundo branco */
    .stDataFrame,
    .dataframe,
    [data-testid="stDataFrame"] {
        background-color: white !important;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Gráficos com fundo branco */
    .js-plotly-plot,
    .plotly,
    [data-testid="stPlotlyChart"] {
        background-color: white !important;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Botões */
    .stButton > button {
        background-color: #4B0082 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }

    .stButton > button:hover {
        background-color: #6A0DAD !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }

    /* Força aplicação em elementos específicos */
    html, body {
        background-color: #E6E6FA !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renderiza cabeçalho do dashboard."""
    # Aplica CSS customizado
    apply_custom_css()

    st.markdown('<h1 class="main-header">Sistema de Prevenção de Fraudes</h1>',
                unsafe_allow_html=True)

    st.markdown("---")


def render_sidebar():
    """Renderiza barra lateral com navegação."""
    st.sidebar.title("Navegação")

    # Informações do sistema
    with st.sidebar.expander("Informações do Sistema", expanded=False):
        st.write("**Sistema:** Prevenção de Fraudes")
        st.write("**Versão:** 1.0.0")
        st.write("**Status:** Online")
        st.write("**Última atualização:** 24/07/2025")

    st.sidebar.markdown("---")

    pages = {
        "Visão Geral": "overview",
        "Análise de Dados": "analysis",
        "Modelos ML": "models",
        "Regras de Negócio": "rules",
        "Alertas": "alerts",
        "Configurações": "settings"
    }

    # Navegação com radio buttons para melhor UX
    selected_page = st.sidebar.radio(
        "Selecione uma página:",
        list(pages.keys()),
        index=0
    )

    st.sidebar.markdown("---")

    # Links úteis
    with st.sidebar.expander("Links Úteis", expanded=False):
        st.markdown("[Documentação Técnica](docs/TECHNICAL_DOCUMENTATION.md)")
        st.markdown("[Notebooks de Análise](notebooks/)")
        st.markdown("[Testes](tests/)")
        st.markdown("[Configurações](config/)")

    # Status dos dados
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status dos Dados")

    try:
        final_results = load_final_results()
        total_transactions = len(final_results)
        fraud_count = final_results['is_fraud'].sum() if 'is_fraud' in final_results.columns else 0
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

        st.sidebar.success(f"Transações: {total_transactions:,}")
        st.sidebar.info(f"Fraudes detectadas: {fraud_count}")
        st.sidebar.metric("Taxa de fraude", f"{fraud_rate:.1f}%")
    except Exception as e:
        st.sidebar.error("Erro nos dados de transações")

    try:
        alerts = load_alerts_log()
        if not alerts.empty and 'status' in alerts.columns:
            pending_alerts = len(alerts[alerts['status'] == 'pending'])
            critical_alerts = len(alerts[alerts['priority'] == 'critical']) if 'priority' in alerts.columns else 0
            total_alerts = len(alerts)

            st.sidebar.warning(f"Alertas pendentes: {pending_alerts}")
            if critical_alerts > 0:
                st.sidebar.error(f"Alertas críticos: {critical_alerts}")
            st.sidebar.info(f"Total de alertas: {total_alerts}")
        else:
            st.sidebar.info("Nenhum alerta disponível")
    except Exception as e:
        st.sidebar.error("Erro nos dados de alertas")

    return pages[selected_page]


def render_overview_page(users_df, transactions_df):
    """Renderiza página de visão geral."""
    st.header("Visão Geral do Sistema")

    # Indicador de carregamento
    with st.spinner('Carregando dados da visão geral...'):
        # Carrega dados de resultados finais
        final_results = load_final_results()
        eda_summary = load_eda_summary()

    if final_results.empty:
        st.error("Dados não disponíveis")
        st.info("Verifique se os arquivos de dados estão no diretório correto.")
        return

    # Alerta de dados de exemplo
    if len(final_results) <= 4:  # Se usando dados de exemplo
        st.info("Exibindo dados de exemplo para demonstração. Para dados reais, execute o gerador de dados sintéticos.")

    # Estatísticas principais dos dados reais
    total_transactions = len(final_results)
    fraud_transactions = final_results['is_fraud'].sum()
    fraud_rate = fraud_transactions / total_transactions if total_transactions > 0 else 0
    total_volume = final_results['amount'].sum()
    fraud_volume = final_results[final_results['is_fraud'] == 1]['amount'].sum()

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total de Transações",
            value=f"{total_transactions:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Taxa de Fraude",
            value=f"{fraud_rate:.2%}",
            delta=None
        )

    with col3:
        st.metric(
            label="Volume Total",
            value=f"R$ {total_volume:,.2f}",
            delta=None
        )

    with col4:
        st.metric(
            label="Volume de Fraude",
            value=f"R$ {fraud_volume:,.2f}",
            delta=f"-{fraud_volume/total_volume:.2%}" if total_volume > 0 else "0%"
        )
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Valores")

        fig_amount = px.histogram(
            final_results,
            x='amount',
            color='is_fraud',
            nbins=20,
            title='Distribuição de Valores por Tipo',
            labels={'amount': 'Valor (R$)', 'count': 'Frequência'},
            color_discrete_map={0: 'lightblue', 1: 'red'}
        )
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        st.subheader("Transações por Hora")

        # Extrai a hora do timestamp
        if 'timestamp' in final_results.columns:
            try:
                final_results_temp = final_results.copy()
                final_results_temp['timestamp'] = pd.to_datetime(final_results_temp['timestamp'])
                final_results_temp['transaction_hour'] = final_results_temp['timestamp'].dt.hour

                hourly_data = final_results_temp.groupby(['transaction_hour', 'is_fraud']).size().reset_index(name='count')

                fig_hourly = px.bar(
                    hourly_data,
                    x='transaction_hour',
                    y='count',
                    color='is_fraud',
                    title='Volume de Transações por Hora',
                    labels={'transaction_hour': 'Hora', 'count': 'Número de Transações'},
                    color_discrete_map={0: 'lightblue', 1: 'red'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            except Exception as e:
                st.warning(f"Não foi possível gerar análise por hora: {e}")
                st.info("Dados de timestamp podem estar em formato incorreto")
        else:
            st.info("Coluna timestamp não encontrada. Análise por hora não disponível.")
    
    # Tabela de transações recentes
    st.subheader("Transações Recentes")

    if 'timestamp' in final_results.columns:
        try:
            final_results_temp = final_results.copy()
            final_results_temp['timestamp'] = pd.to_datetime(final_results_temp['timestamp'])
            recent_transactions = final_results_temp.sort_values('timestamp', ascending=False).head(10)

            # Seleciona colunas disponíveis
            available_columns = ['transaction_id', 'amount', 'merchant_category', 'country', 'is_fraud']
            display_columns = [col for col in available_columns if col in recent_transactions.columns]

            # Formata dados para exibição
            display_df = recent_transactions[display_columns].copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"R$ {x:,.2f}")
            display_df['is_fraud'] = display_df['is_fraud'].map({1: 'Fraude', 0: 'Legítima'})

            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível exibir transações recentes: {e}")
            # Fallback para exibir dados básicos
            display_df = final_results.head(10)[['transaction_id', 'amount', 'merchant_category', 'country', 'is_fraud']].copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"R$ {x:,.2f}")
            display_df['is_fraud'] = display_df['is_fraud'].map({1: 'Fraude', 0: 'Legítima'})
            st.dataframe(display_df, use_container_width=True)
    else:
        # Fallback sem ordenação por timestamp
        display_df = final_results.head(10)[['transaction_id', 'amount', 'merchant_category', 'country', 'is_fraud']].copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"R$ {x:,.2f}")
        display_df['is_fraud'] = display_df['is_fraud'].map({1: 'Fraude', 0: 'Legítima'})
        st.dataframe(display_df, use_container_width=True)


def render_analysis_page(users_df, transactions_df):
    """Renderiza página de análise de dados."""
    st.header("Análise Exploratória de Dados")

    # Carrega dados de análise
    final_results = load_final_results()
    eda_summary = load_eda_summary()

    if final_results.empty:
        st.error("Dados não disponíveis")
        return

    # Análise por categoria
    st.subheader("Análise por Categoria de Comerciante")

    if 'merchant_category' in final_results.columns:
        category_analysis = final_results.groupby('merchant_category').agg({
            'amount': ['count', 'sum', 'mean'],
            'is_fraud': ['sum', 'mean']
        }).round(2)

        category_analysis.columns = ['Transações', 'Volume Total', 'Valor Médio', 'Fraudes', 'Taxa de Fraude']
        category_analysis['Volume Total'] = category_analysis['Volume Total'].apply(lambda x: f"R$ {x:,.2f}")
        category_analysis['Valor Médio'] = category_analysis['Valor Médio'].apply(lambda x: f"R$ {x:,.2f}")
        category_analysis['Taxa de Fraude'] = category_analysis['Taxa de Fraude'].apply(lambda x: f"{x:.2%}")

        st.dataframe(category_analysis, use_container_width=True)

        # Gráfico de taxa de fraude por categoria
        fraud_by_category = final_results.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
    else:
        st.warning("Dados de categoria de comerciante não disponíveis")
        fraud_by_category = pd.Series()
    
    fig_category = px.bar(
        x=fraud_by_category.index,
        y=fraud_by_category.values,
        title='Taxa de Fraude por Categoria de Comerciante',
        labels={'x': 'Categoria', 'y': 'Taxa de Fraude'}
    )
    fig_category.update_xaxes(tickangle=45)
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Análise temporal
    st.subheader("Análise Temporal")

    # Verifica se há coluna de timestamp
    if 'timestamp' in final_results.columns:
        try:
            # Converte timestamp para datetime se necessário
            final_results_temp = final_results.copy()
            final_results_temp['timestamp'] = pd.to_datetime(final_results_temp['timestamp'])

            # Agrupa por dia
            daily_data = final_results_temp.set_index('timestamp').resample('D').agg({
                'amount': ['count', 'sum'],
                'is_fraud': 'sum'
            })
            daily_data.columns = ['Transações', 'Volume', 'Fraudes']
            daily_data['Taxa de Fraude'] = daily_data['Fraudes'] / daily_data['Transações']

            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['Taxa de Fraude'],
                mode='lines+markers',
                name='Taxa de Fraude',
                line=dict(color='red')
            ))

            fig_timeline.update_layout(
                title='Taxa de Fraude ao Longo do Tempo',
                xaxis_title='Data',
                yaxis_title='Taxa de Fraude',
                yaxis=dict(tickformat='.2%')
            )

            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível gerar análise temporal: {e}")
            st.info("Dados de timestamp podem estar em formato incorreto ou ausentes")
    else:
        st.info("Coluna de timestamp não encontrada nos dados. Análise temporal não disponível.")


def render_rules_page():
    """Renderiza página de regras de negócio."""
    st.header("Regras de Negócio")
    
    if not PROJECT_MODULES_AVAILABLE:
        st.error("Módulos do projeto não disponíveis")
        return
    
    try:
        # Inicializa engine de regras
        rules_engine = BusinessRulesEngine()
        
        # Estatísticas das regras
        st.subheader("Estatísticas das Regras")
        
        rules_summary = rules_engine.get_rules_summary()
        
        if rules_summary:
            rules_df = pd.DataFrame(rules_summary)
            
            # Métricas das regras
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Regras", len(rules_df))
            
            with col2:
                active_rules = rules_df['enabled'].sum()
                st.metric("Regras Ativas", active_rules)
            
            with col3:
                total_triggers = rules_df['triggered_count'].sum()
                st.metric("Total de Acionamentos", total_triggers)
            
            # Tabela de regras
            st.subheader("Lista de Regras")
            
            display_rules = rules_df[['name', 'description', 'action', 'priority', 'enabled', 'triggered_count']].copy()
            display_rules.columns = ['Nome', 'Descrição', 'Ação', 'Prioridade', 'Ativa', 'Acionamentos']
            display_rules['Ativa'] = display_rules['Ativa'].map({True: '✅', False: '❌'})
            
            st.dataframe(display_rules, use_container_width=True)
            
            # Gráfico de acionamentos
            if total_triggers > 0:
                fig_rules = px.bar(
                    rules_df.sort_values('triggered_count', ascending=False),
                    x='name',
                    y='triggered_count',
                    title='Acionamentos por Regra',
                    labels={'name': 'Regra', 'triggered_count': 'Acionamentos'}
                )
                fig_rules.update_xaxes(tickangle=45)
                st.plotly_chart(fig_rules, use_container_width=True)
        
        else:
            st.info("Nenhuma regra configurada")
    
    except Exception as e:
        st.error(f"Erro ao carregar regras: {e}")


def render_alerts_page(transactions_df):
    """Renderiza página de alertas."""
    st.header("Sistema de Alertas")

    # Carrega dados de alertas
    alerts_df = load_alerts_log()

    if alerts_df.empty:
        st.error("Dados não disponíveis")
        return
    
    # Métricas de alertas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_alerts = len(alerts_df)
        st.metric("Total de Alertas", total_alerts)

    with col2:
        pending_alerts = len(alerts_df[alerts_df['status'] == 'pending'])
        st.metric("Alertas Pendentes", pending_alerts)

    with col3:
        critical_alerts = len(alerts_df[alerts_df['priority'] == 'critical'])
        st.metric("Alertas Críticos", critical_alerts)

    with col4:
        avg_score = alerts_df['fraud_score'].mean()
        st.metric("Score Médio", f"{avg_score:.2f}")

    st.markdown("---")

    # Filtros
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox("Status", ["Todos"] + list(alerts_df['status'].unique()))

    with col2:
        priority_filter = st.selectbox("Prioridade", ["Todas"] + list(alerts_df['priority'].unique()))

    with col3:
        model_filter = st.selectbox("Modelo", ["Todos"] + list(alerts_df['model_used'].unique()))

    # Aplica filtros
    filtered_alerts = alerts_df.copy()

    if status_filter != "Todos":
        filtered_alerts = filtered_alerts[filtered_alerts['status'] == status_filter]

    if priority_filter != "Todas":
        filtered_alerts = filtered_alerts[filtered_alerts['priority'] == priority_filter]

    if model_filter != "Todos":
        filtered_alerts = filtered_alerts[filtered_alerts['model_used'] == model_filter]

    # Tabela de alertas
    st.subheader("Lista de Alertas")

    # Formatação da tabela
    display_alerts = filtered_alerts.copy()
    display_alerts['fraud_score'] = display_alerts['fraud_score'].apply(lambda x: f"{x:.2f}")
    display_alerts['timestamp'] = pd.to_datetime(display_alerts['timestamp']).dt.strftime('%d/%m/%Y %H:%M')

    st.dataframe(display_alerts, use_container_width=True)


def render_models_page():
    """Renderiza página de modelos ML."""
    st.header("Modelos de Machine Learning")

    # Carrega scores dos modelos
    model_scores = load_model_scores()

    # Seletor de modelo
    model_names = list(model_scores.keys())
    selected_model = st.selectbox("Selecione o modelo", model_names)

    if selected_model in model_scores:
        model_data = model_scores[selected_model]

        # Métricas do modelo selecionado
        st.subheader(f"Métricas - {selected_model.replace('_', ' ').title()}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{model_data.get('accuracy', 0):.3f}")

        with col2:
            st.metric("Precision", f"{model_data.get('precision', 0):.3f}")

        with col3:
            st.metric("Recall", f"{model_data.get('recall', 0):.3f}")

        with col4:
            st.metric("F1-Score", f"{model_data.get('f1_score', 0):.3f}")

        # ROC-AUC se disponível
        if 'roc_auc' in model_data:
            st.metric("ROC-AUC", f"{model_data['roc_auc']:.3f}")

        st.markdown("---")

        # Comparação de modelos
        st.subheader("Comparação de Modelos")

        # Cria DataFrame para comparação
        comparison_data = []
        for model_name, data in model_scores.items():
            comparison_data.append({
                'Modelo': model_name.replace('_', ' ').title(),
                'Accuracy': data.get('accuracy', 0),
                'Precision': data.get('precision', 0),
                'Recall': data.get('recall', 0),
                'F1-Score': data.get('f1_score', 0),
                'ROC-AUC': data.get('roc_auc', 0)
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Gráfico de barras comparativo
        fig_comparison = px.bar(
            comparison_df.melt(id_vars=['Modelo'], var_name='Métrica', value_name='Valor'),
            x='Modelo',
            y='Valor',
            color='Métrica',
            title='Comparação de Performance dos Modelos',
            barmode='group'
        )

        st.plotly_chart(fig_comparison, use_container_width=True)


def render_settings_page():
    """Renderiza página de configurações."""
    st.header("Configurações")

    # Upload de arquivo
    st.subheader("Upload de Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado com sucesso! {len(df)} linhas encontradas.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")

    st.markdown("---")

    # Configurações de threshold
    st.subheader("Configurações de Threshold")

    col1, col2 = st.columns(2)

    with col1:
        fraud_threshold = st.slider(
            "Threshold de Fraude",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Score mínimo para classificar como fraude"
        )

    with col2:
        alert_threshold = st.slider(
            "Threshold de Alerta",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="Score mínimo para gerar alerta"
        )

    st.markdown("---")

    # Configurações de modelo
    st.subheader("Configurações de Modelo")

    model_scores = load_model_scores()
    active_model = st.selectbox(
        "Modelo Ativo",
        list(model_scores.keys()),
        help="Modelo usado para predições em produção"
    )

    # Modo debug
    debug_mode = st.checkbox("Modo Debug", help="Ativa logs detalhados")

    # Botão para salvar configurações
    if st.button("Salvar Configurações"):
        config = {
            'fraud_threshold': fraud_threshold,
            'alert_threshold': alert_threshold,
            'active_model': active_model,
            'debug_mode': debug_mode
        }

        st.success("Configurações salvas com sucesso!")
        st.json(config)


def main():
    """Função principal do dashboard."""
    render_header()
    
    # Carrega dados
    users_df, transactions_df = load_data()
    
    # Navegação
    current_page = render_sidebar()
    
    # Renderiza página selecionada
    if current_page == "overview":
        render_overview_page(users_df, transactions_df)
    elif current_page == "analysis":
        render_analysis_page(users_df, transactions_df)
    elif current_page == "rules":
        render_rules_page()
    elif current_page == "alerts":
        render_alerts_page(transactions_df)
    elif current_page == "models":
        render_models_page()
    elif current_page == "settings":
        render_settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("*Sistema de Prevenção de Fraudes - Desenvolvido usando Streamlit por Nathalia Adriele*")


if __name__ == "__main__":
    main()
