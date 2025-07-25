#!/usr/bin/env python3
"""
Script para analisar os dados realistas gerados.
"""

import pandas as pd
import numpy as np

def analyze_final_results():
    """Analisa o arquivo final_results.csv"""
    print("ANÁLISE DOS DADOS REALISTAS")
    print("=" * 50)
    
    # Carrega dados
    df = pd.read_csv('data/final_results.csv')
    
    # Estatísticas gerais
    total_transactions = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_rate = (fraud_count / total_transactions) * 100
    
    print(f"ESTATÍSTICAS GERAIS:")
    print(f"   Total de transações: {total_transactions:,}")
    print(f"   Fraudes detectadas: {fraud_count}")
    print(f"   Taxa de fraude: {fraud_rate:.2f}%")
    print()
    
    # Análise por categoria
    print(f"ANÁLISE POR CATEGORIA:")
    category_stats = df.groupby('merchant_category').agg({
        'amount': ['count', 'mean'],
        'is_fraud': ['sum', 'mean']
    }).round(3)
    
    category_stats.columns = ['Transações', 'Valor Médio', 'Fraudes', 'Taxa Fraude']
    category_stats['Taxa Fraude'] = (category_stats['Taxa Fraude'] * 100).round(2)
    
    for category, row in category_stats.iterrows():
        print(f"   {category:12}: {row['Transações']:3.0f} trans, "
              f"R$ {row['Valor Médio']:7.2f} médio, "
              f"{row['Fraudes']:2.0f} fraudes ({row['Taxa Fraude']:4.1f}%)")
    print()
    
    # Análise por país
    print(f"ANÁLISE POR PAÍS:")
    country_stats = df.groupby('country').agg({
        'amount': ['count', 'mean'],
        'is_fraud': ['sum', 'mean']
    }).round(3)
    
    country_stats.columns = ['Transações', 'Valor Médio', 'Fraudes', 'Taxa Fraude']
    country_stats['Taxa Fraude'] = (country_stats['Taxa Fraude'] * 100).round(2)
    
    for country, row in country_stats.iterrows():
        print(f"   {country:3}: {row['Transações']:3.0f} trans, "
              f"R$ {row['Valor Médio']:7.2f} médio, "
              f"{row['Fraudes']:2.0f} fraudes ({row['Taxa Fraude']:4.1f}%)")
    print()
    
    # Análise de valores
    print(f"ANÁLISE DE VALORES:")
    fraud_amounts = df[df['is_fraud'] == 1]['amount']
    legit_amounts = df[df['is_fraud'] == 0]['amount']
    
    print(f"   Fraudes - Média: R$ {fraud_amounts.mean():.2f}, "
          f"Mediana: R$ {fraud_amounts.median():.2f}, "
          f"Min: R$ {fraud_amounts.min():.2f}, "
          f"Max: R$ {fraud_amounts.max():.2f}")
    
    print(f"   Legítimas - Média: R$ {legit_amounts.mean():.2f}, "
          f"Mediana: R$ {legit_amounts.median():.2f}, "
          f"Min: R$ {legit_amounts.min():.2f}, "
          f"Max: R$ {legit_amounts.max():.2f}")
    print()
    
    # Análise de scores
    print(f"ANÁLISE DE SCORES:")
    fraud_scores = df[df['is_fraud'] == 1]['fraud_score']
    legit_scores = df[df['is_fraud'] == 0]['fraud_score']
    
    print(f"   Fraudes - Média: {fraud_scores.mean():.3f}, "
          f"Min: {fraud_scores.min():.3f}, "
          f"Max: {fraud_scores.max():.3f}")
    
    print(f"   Legítimas - Média: {legit_scores.mean():.3f}, "
          f"Min: {legit_scores.min():.3f}, "
          f"Max: {legit_scores.max():.3f}")
    print()
    
    # Análise temporal
    print(f"ANÁLISE TEMPORAL:")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    date_range = df['timestamp'].max() - df['timestamp'].min()
    print(f"   Período: {date_range.days} dias")
    print(f"   Data inicial: {df['timestamp'].min().strftime('%Y-%m-%d')}")
    print(f"   Data final: {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print()
    
    return df

def analyze_alerts():
    """Analisa o arquivo alerts_log.csv"""
    print("ANÁLISE DOS ALERTAS")
    print("=" * 30)
    
    alerts = pd.read_csv('data/alerts_log.csv')
    
    total_alerts = len(alerts)
    print(f"   Total de alertas: {total_alerts}")
    
    # Por prioridade
    priority_counts = alerts['priority'].value_counts()
    print(f"   Por prioridade:")
    for priority, count in priority_counts.items():
        print(f"     {priority}: {count} ({count/total_alerts*100:.1f}%)")
    
    # Por status
    status_counts = alerts['status'].value_counts()
    print(f"   Por status:")
    for status, count in status_counts.items():
        print(f"     {status}: {count} ({count/total_alerts*100:.1f}%)")
    
    print()

def main():
    """Função principal"""
    try:
        df = analyze_final_results()
        analyze_alerts()
        
        print("✅ AVALIAÇÃO DE REALISMO:")
        fraud_rate = (df['is_fraud'].sum() / len(df)) * 100
        
        if fraud_rate <= 5:
            print(f"   ✅ Taxa de fraude realista: {fraud_rate:.2f}%")
        else:
            print(f"   ⚠️ Taxa de fraude ainda alta: {fraud_rate:.2f}%")
        
        # Verifica distribuição geográfica
        fraud_by_country = df[df['is_fraud'] == 1]['country'].value_counts()
        if len(fraud_by_country) > 1:
            print(f"   ✅ Fraudes distribuídas geograficamente")
        else:
            print(f"   ⚠️ Fraudes concentradas em poucos países")
        
        # Verifica variação de valores
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        if fraud_amounts.std() > fraud_amounts.mean() * 0.5:
            print(f"   ✅ Valores de fraude variados")
        else:
            print(f"   ⚠️ Valores de fraude pouco variados")
        
        print("\n🎯 DADOS PRONTOS PARA USO NO DASHBOARD!")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    main()
