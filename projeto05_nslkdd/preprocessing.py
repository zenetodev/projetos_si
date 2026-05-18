import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Aplicar one-hot encoding no treino
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols)
    
    # Aplicar no teste (garantindo as mesmas colunas)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols)
    
    # Alinhar colunas do teste com o treino (preencher com 0 onde faltar)
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    
    # Garantir mesma ordem de colunas
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    print(f"\nOne-hot encoding aplicado:")
    print(f"  Colunas originais: {X_train.shape[1]}")
    print(f"  Colunas apos encoding: {X_train_encoded.shape[1]}")
    
    return X_train_encoded, X_test_encoded


def binarize_target(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    y_train_binary = (y_train != 'normal').astype(int).values
    y_test_binary = (y_test != 'normal').astype(int).values
    
    print(f"\nBinarizacao do target:")
    print(f"  Treino - normal: {sum(y_train_binary == 0)} ({sum(y_train_binary == 0)/len(y_train_binary)*100:.2f}%)")
    print(f"  Treino - ataque: {sum(y_train_binary == 1)} ({sum(y_train_binary == 1)/len(y_train_binary)*100:.2f}%)")
    print(f"  Teste - normal: {sum(y_test_binary == 0)} ({sum(y_test_binary == 0)/len(y_test_binary)*100:.2f}%)")
    print(f"  Teste - ataque: {sum(y_test_binary == 1)} ({sum(y_test_binary == 1)/len(y_test_binary)*100:.2f}%)")
    
    return y_train_binary, y_test_binary


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nNormalizacao Z-score aplicada:")
    print(f"  Media de cada feature (treino): ~0")
    print(f"  Desvio padrao de cada feature (treino): ~1")
    
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "="*60)
    print("TAREFA 1 - PREPARACAO DOS DADOS")
    print("="*60)
    
    # Separar features e target
    X_train = train_df.drop('attack_type', axis=1)
    y_train = train_df['attack_type']
    
    X_test = test_df.drop('attack_type', axis=1)
    y_test = test_df['attack_type']
    
    print(f"\nFeatures originais: {X_train.shape[1]}")
    
    # One-hot encoding
    print("\n1a) One-hot encoding das variaveis categoricas:")
    X_train_encoded, X_test_encoded = one_hot_encode(X_train, X_test)
    
    # Binarizacao do target
    print("\n1b) Binarizacao do target (normal=0, ataque=1):")
    y_train_binary, y_test_binary = binarize_target(y_train, y_test)
    
    # Normalizacao Z-score
    print("\n1c) Normalizacao Z-score:")
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train_encoded, X_test_encoded)
    
    print("\nPre-processamento concluido!")
    print(f"  X_train shape: {X_train_scaled.shape}")
    print(f"  X_test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_binary, y_test_binary, scaler


if __name__ == "__main__":
    from data_loader import load_nslkdd_data
    
    train_df, test_df = load_nslkdd_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(train_df, test_df)