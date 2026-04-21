import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    # criar uma copia explicita para evitar warnings
    X_clean = X.copy()
    
    # colunas que podem ter valores nulos
    cols_with_nulls = ['ca', 'thal']
    
    for col in cols_with_nulls:
        if col in X_clean.columns:
            nulos = X_clean[col].isnull().sum()
            if nulos > 0:
                mediana = X_clean[col].median()
                # Metodo seguro sem inplace
                X_clean[col] = X_clean[col].fillna(mediana)
                print(f"   • {col}: {nulos} valores nulos preenchidos com mediana={mediana}")
    
    return X_clean


def binarize_target(y: pd.Series) -> pd.Series:
    y_binary = (y > 0).astype(int)
    
    print(f"\n   Distribuicao original:")
    print(f"   {y.value_counts().sort_index().to_dict()}")
    print(f"\n   Distribuicao binarizada:")
    print(f"   0 (sem doenca): {(y_binary == 0).sum()} pacientes")
    print(f"   1 (com doenca): {(y_binary == 1).sum()} pacientes")
    
    return y_binary


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n   Split treino/teste: {test_size*100:.0f}/{100-test_size*100:.0f}")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    print(f"   Proporcao de doenca no treino: {y_train.mean():.2%}")
    print(f"   Proporcao de doenca no teste: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # verificar quais features existem no dataset
    features_to_scale = [f for f in continuous_features if f in X_train.columns]
    
    print(f"\n   Features escalonadas: {features_to_scale}")
    
    # criar copias
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # aplicar StandardScaler
    scaler = StandardScaler()
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler


def check_missing_values(X: pd.DataFrame, stage: str):
    nulos = X.isnull().sum()
    if nulos.sum() > 0:
        print(f"   ATENCAO - Valores nulos encontrados em {stage}:")
        for col in nulos[nulos > 0].index:
            print(f"      {col}: {nulos[col]} nulos")
        return False
    else:
        print(f"   OK - Sem valores nulos em {stage}")
        return True


def run_preprocessing(X: pd.DataFrame, y: pd.Series):
    print("\n" + "="*60)
    print("TAREFA 1 - PRE-PROCESSAMENTO")
    print("="*60)
    
    print("\n1a) Tratamento de valores nulos:")
    check_missing_values(X, "dados originais")
    X_clean = handle_missing_values(X)
    check_missing_values(X_clean, "dados apos tratamento")
    
    print("\n1b) Binarizacao da variavel target:")
    y_binary = binarize_target(y)
    
    print("\n1c) Divisao treino/teste com stratify:")
    X_train, X_test, y_train, y_test = split_data(X_clean, y_binary)
    
    print("\n1d) Escalonamento de features continuas:")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Verificacao final
    print("\nVERIFICACAO FINAL:")
    check_missing_values(X_train_scaled, "X_train escalonado")
    check_missing_values(X_test_scaled, "X_test escalonado")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    X, y = load_heart_disease_data()
    run_preprocessing(X, y)