"""
Modulo para carregar o dataset Wine Quality Red da UCI.
"""
import pandas as pd
from typing import Tuple


def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega o dataset Wine Quality Red diretamente do CSV.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features) e y (quality original)
    """
    # URL direta do dataset Wine Quality Red
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Carregar dados (o separador eh ';')
    df = pd.read_csv(url, sep=';')
    
    # Separar features e target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    print(f"Dataset carregado da URL oficial")
    print(f"Shape: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Colunas: {list(X.columns)}")
    print(f"Valores de quality (original): {sorted(y.unique())}")
    print(f"Distribuicao original:\n{y.value_counts().sort_index()}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_wine_data()
    print(f"\nDados carregados com sucesso!")