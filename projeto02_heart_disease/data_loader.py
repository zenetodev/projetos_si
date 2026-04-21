
from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Tuple


def load_heart_disease_data() -> Tuple[pd.DataFrame, pd.Series]:
    heart_disease = fetch_ucirepo(id=45)
    
    X = heart_disease.data.features
    
    y = heart_disease.data.targets
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    return X, y


def get_dataset_info() -> dict:
    heart_disease = fetch_ucirepo(id=45)
    return {
        "metadata": heart_disease.metadata,
        "variables": heart_disease.variables
    }


if __name__ == "__main__":
    X, y = load_heart_disease_data()
    print(f" Dados carregados com sucesso!")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Colunas: {list(X.columns)}")
    print(f"Valores únicos em y: {y.unique()}")