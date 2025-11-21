#!/usr/bin/env python3

import pandas as pd
import numpy as np


def analyze_final_results():
    """Análise consolidada de final_results.csv."""
    print("ANÁLISE DOS DADOS REALISTAS")
    print("=" * 50)

    df = pd.read_csv("data/final_results.csv")

    # Estatísticas gerais
    total = len(df)
    fraud = df["is_fraud"].sum()
    fraud_rate = (fraud / total) * 100

    print("ESTATÍSTICAS GERAIS:")
    print(f"   Transações: {total:,}")
    print(f"   Fraudes: {fraud}")
    print(f"   Taxa de fraude: {fraud_rate:.2f}%\n")

    # Por categoria
    print("ANÁLISE POR CATEGORIA:")
    category_stats = (
        df.groupby("merchant_category")
        .agg(
            {
                "amount": ["count", "mean"],
                "is_fraud": ["sum", "mean"],
            }
        )
        .round(3)
    )
    category_stats.columns = ["Transações", "Valor Médio", "Fraudes", "Taxa Fraude"]
    category_stats["Taxa Fraude"] = (category_stats["Taxa Fraude"] * 100).round(2)

    for category, row in category_stats.iterrows():
        print(
            f"   {category:12}: {row['Transações']:3.0f} trans, "
            f"R$ {row['Valor Médio']:7.2f}, "
            f"{row['Fraudes']:2.0f} fraudes ({row['Taxa Fraude']:4.1f}%)"
        )
    print()

    # Por país
    print("ANÁLISE POR PAÍS:")
    country_stats = (
        df.groupby("country")
        .agg(
            {
                "amount": ["count", "mean"],
                "is_fraud": ["sum", "mean"],
            }
        )
        .round(3)
    )
    country_stats.columns = ["Transações", "Valor Médio", "Fraudes", "Taxa Fraude"]
    country_stats["Taxa Fraude"] = (country_stats["Taxa Fraude"] * 100).round(2)

    for country, row in country_stats.iterrows():
        print(
            f"   {country:3}: {row['Transações']:3.0f} trans, "
            f"R$ {row['Valor Médio']:7.2f}, "
            f"{row['Fraudes']:2.0f} fraudes ({row['Taxa Fraude']:4.1f}%)"
        )
    print()

    # Valores
    print("ANÁLISE DE VALORES:")
    fraud_vals = df[df["is_fraud"] == 1]["amount"]
    legit_vals = df[df["is_fraud"] == 0]["amount"]

    print(
        f"   Fraudes   - Média: R$ {fraud_vals.mean():.2f}, "
        f"Mediana: R$ {fraud_vals.median():.2f}, "
        f"Min: R$ {fraud_vals.min():.2f}, "
        f"Max: R$ {fraud_vals.max():.2f}"
    )
    print(
        f"   Legítimas - Média: R$ {legit_vals.mean():.2f}, "
        f"Mediana: R$ {legit_vals.median():.2f}, "
        f"Min: R$ {legit_vals.min():.2f}, "
        f"Max: R$ {legit_vals.max():.2f}\n"
    )

    # Scores
    print("ANÁLISE DE SCORES:")
    fraud_scores = df[df["is_fraud"] == 1]["fraud_score"]
    legit_scores = df[df["is_fraud"] == 0]["fraud_score"]

    print(
        f"   Fraudes   - Média: {fraud_scores.mean():.3f}, "
        f"Min: {fraud_scores.min():.3f}, Max: {fraud_scores.max():.3f}"
    )
    print(
        f"   Legítimas - Média: {legit_scores.mean():.3f}, "
        f"Min: {legit_scores.min():.3f}, Max: {legit_scores.max():.3f}\n"
    )

    # Temporal
    print("ANÁLISE TEMPORAL:")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"   Período: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")
    print(f"   Início: {df['timestamp'].min():%Y-%m-%d}")
    print(f"   Fim:    {df['timestamp'].max():%Y-%m-%d}\n")

    return df


def analyze_alerts():
    """Análise de alerts_log.csv."""
    print("ANÁLISE DOS ALERTAS")
    print("=" * 30)

    alerts = pd.read_csv("data/alerts_log.csv")
    total = len(alerts)

    print(f"   Total de alertas: {total}")

    print("   Por prioridade:")
    for p, c in alerts["priority"].value_counts().items():
        print(f"     {p}: {c} ({c/total*100:.1f}%)")

    print("   Por status:")
    for s, c in alerts["status"].value_counts().items():
        print(f"     {s}: {c} ({c/total*100:.1f}%)")

    print()


def main():
    try:
        df = analyze_final_results()
        analyze_alerts()

        print("AVALIAÇÃO DE REALISMO")
        print("=" * 30)

        fraud_rate = (df["is_fraud"].sum() / len(df)) * 100
        print(
            f"   Taxa de fraude: {fraud_rate:.2f}% "
            f"{'✅' if fraud_rate <= 5 else '⚠️'}"
        )

        fraud_geo = df[df["is_fraud"] == 1]["country"].nunique()
        print(
            f"   Distribuição geográfica: "
            f"{'adequada' if fraud_geo > 1 else 'concentrada'}"
        )

        fraud_vals = df[df["is_fraud"] == 1]["amount"]
        varied = fraud_vals.std() > fraud_vals.mean() * 0.5
        print(f"   Variação de valores: {'adequada' if varied else 'baixa'}")

        print("\nDADOS PRONTOS PARA O DASHBOARD")

    except Exception as e:
        print(f"Erro na análise: {e}")


if __name__ == "__main__":
    main()
