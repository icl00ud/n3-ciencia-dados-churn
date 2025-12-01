"""
Script auxiliar para fazer previsões de churn usando o modelo treinado.
Demonstra como o modelo pode ser usado em produção.
"""

import joblib
import pandas as pd


def load_model(model_path: str = 'modelo_final.pkl'):
    """Carrega o modelo treinado."""
    return joblib.load(model_path)


def predict_churn(model, customer_data: pd.DataFrame) -> dict:
    """
    Faz previsão de churn para um cliente.
    
    Args:
        model: Modelo treinado carregado
        customer_data: DataFrame com os dados do cliente (mesmo formato do treino)
    
    Returns:
        dict com previsão e probabilidades
    """
    prediction = model.predict(customer_data)[0]
    probabilities = model.predict_proba(customer_data)[0]
    
    return {
        'churn': bool(prediction),
        'churn_label': 'Sim' if prediction == 1 else 'Não',
        'prob_no_churn': round(probabilities[0], 4),
        'prob_churn': round(probabilities[1], 4),
        'risco': 'ALTO' if probabilities[1] > 0.5 else 'BAIXO'
    }


if __name__ == '__main__':
    # Exemplo de uso
    print("Carregando modelo...")
    model = load_model('../modelo_final.pkl')
    
    print("Modelo carregado com sucesso!")
    print(f"Tipo do modelo: {type(model).__name__}")
