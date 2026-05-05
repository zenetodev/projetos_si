import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def recode_quality(y: pd.Series) -> pd.Series:
    y_recode = y.copy()
    
    y_recode = pd.cut(y_recode, 
                      bins=[-float('inf'), 5, 7, float('inf')],
                      labels=[0, 1, 2])
    
    y_recode = y_recode.astype(int)
    
    print("\nDistribuicao original:")
    print(y.value_counts().sort_index().to_dict())
    
    print("\nDistribuicao apos recodificacao:")
    print(f"  Ruim (0) - notas <=5: {(y_recode == 0).sum()} amostras")
    print(f"  Medio (1) - notas 6-7: {(y_recode == 1).sum()} amostras")
    print(f"  Bom (2) - notas >=8: {(y_recode == 2).sum()} amostras")
    
    return y_recode


def normalize_features(X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, StandardScaler]:
    if method == 'standard':
        scaler = StandardScaler()
        X_normalized = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print(f"\nNormalizacao aplicada: Z-score (media=0, desvio=1)")
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_normalized = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print(f"\nNormalizacao aplicada: Min-Max (0-1)")
    else:
        raise ValueError("method deve ser 'standard' ou 'minmax'")
    
    return X_normalized, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nSplit treino/teste: {test_size*100:.0f}/{100-test_size*100:.0f}")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    print("\nProporcao de classes no treino:")
    print(f"  Ruim: {(y_train == 0).mean():.2%}")
    print(f"  Medio: {(y_train == 1).mean():.2%}")
    print(f"  Bom: {(y_train == 2).mean():.2%}")
    
    print("\nProporcao de classes no teste:")
    print(f"  Ruim: {(y_test == 0).mean():.2%}")
    print(f"  Medio: {(y_test == 1).mean():.2%}")
    print(f"  Bom: {(y_test == 2).mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def run_preprocessing(X: pd.DataFrame, y: pd.Series):
    print("\n" + "="*60)
    print("TAREFA 1 - PREPARACAO DOS DADOS")
    print("="*60)
    
    print("\n1a) Recodificacao da variavel quality:")
    y_recode = recode_quality(y)
    
    print("\n1b) Normalizacao das features:")
    print("Justificativa: O k-NN utiliza distancia euclidiana.")
    print("Features em escalas diferentes (ex: alcool ~10-15, densidade ~0.99-1.00)")
    print("fariam com que o alcool dominasse o calculo da distancia.")
    print("Normalizacao garante que cada feature contribua igualmente.")
    
    X_normalized, scaler = normalize_features(X, method='standard')
    
    print("\nEstatisticas apos normalizacao:")
    for col in X_normalized.columns[:3]:  # Mostrar apenas 3 exemplos
        print(f"  {col}: media={X_normalized[col].mean():.2f}, std={X_normalized[col].std():.2f}")
    
    print("\n1c) Divisao treino/teste com estratificacao:")
    X_train, X_test, y_train, y_test = split_data(X_normalized, y_recode)
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from data_loader import load_wine_data
    X, y = load_wine_data()
    run_preprocessing(X, y)