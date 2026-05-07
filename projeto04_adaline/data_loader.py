from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Tuple


def load_ionosphere_data() -> Tuple[pd.DataFrame, pd.Series]:

    ionosphere = fetch_ucirepo(id=52)
    
    # Features (34 atributos continuos)
    X = ionosphere.data.features
    
    # Target original (good/bad)
    y = ionosphere.data.targets
    
    # Garantir que y seja uma Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    print(f"Dataset: Ionosphere")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Colunas: {list(X.columns)}")
    print(f"Valores unicos em y: {y.unique()}")
    print(f"Distribuicao original:\n{y.value_counts()}")
    
    return X, y


def get_dataset_info() -> dict:
    ionosphere = fetch_ucirepo(id=52)
    return {
        "metadata": ionosphere.metadata,
        "variables": ionosphere.variables
    }


if __name__ == "__main__":
    X, y = load_ionosphere_data()
    print(f"\nDados carregados com sucesso!")