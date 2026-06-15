import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple


def clean_dependents_column(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    if 'Dependents' in df_clean.columns:
        # Converter valores como '3+' para '3'
        df_clean['Dependents'] = df_clean['Dependents'].replace('3+', '3')
        # Converter para numerico, forcar erros para NaN
        df_clean['Dependents'] = pd.to_numeric(df_clean['Dependents'], errors='coerce')
        print("  Dependents: valores convertidos (3+ → 3)")
    
    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    print("\nTratamento de valores ausentes:")
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            nulos = df_clean[col].isnull().sum()
            
            if df_clean[col].dtype in ['int64', 'float64']:
                # Coluna numerica: usar mediana
                mediana = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(mediana)
                print(f"  {col}: {nulos} nulos preenchidos com mediana={mediana}")
            else:
                # Coluna categorica: usar moda
                moda = df_clean[col].mode()[0]
                df_clean.loc[:, col] = df_clean[col].fillna(moda)
                print(f"  {col}: {nulos} nulos preenchidos com moda='{moda}'")
    
    return df_clean


def encode_categorical_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    X_encoded = X.copy()
    encoders = {}
    
    print("\nLabel Encoding das variaveis categoricas:")
    
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            
            # Exibir mapeamento
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"  {col}: {mapping}")
    
    return X_encoded, encoders


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Dependents', 'Credit_History']
    
    # Verificar quais colunas existem
    cols_to_scale = [col for col in numeric_cols if col in X_train.columns]
    
    print(f"\nNormalizacao Min-Max (0-1) aplicada nas colunas:")
    print(f"  {cols_to_scale}")
    
    # Garantir que as colunas sao numericas
    for col in cols_to_scale:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Criar scaler e aplicar
    scaler = MinMaxScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit no treino, transform em ambos
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    print("\nExemplo de normalizacao (ApplicantIncome):")
    print(f"  Min original: {X_train['ApplicantIncome'].min():.2f}")
    print(f"  Max original: {X_train['ApplicantIncome'].max():.2f}")
    print(f"  Min apos normalizacao: {X_train_scaled['ApplicantIncome'].min():.4f}")
    print(f"  Max apos normalizacao: {X_train_scaled['ApplicantIncome'].max():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nDivisao treino/teste: {test_size*100:.0f}/{100-test_size*100:.0f}")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    print("\nProporcao de classes no treino:")
    print(f"  Aprovado (1): {(y_train == 1).mean()*100:.2f}%")
    print(f"  Rejeitado (0): {(y_train == 0).mean()*100:.2f}%")
    
    print("\nProporcao de classes no teste:")
    print(f"  Aprovado (1): {(y_test == 1).mean()*100:.2f}%")
    print(f"  Rejeitado (0): {(y_test == 0).mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def run_preprocessing(df: pd.DataFrame):
    print("\n" + "="*60)
    print("TAREFA 1 - PREPARACAO DOS DADOS")
    print("="*60)
    
    # Verificar dados originais
    print("\nDados originais:")
    print(f"  Shape: {df.shape}")
    
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        print("  Coluna Loan_ID removida (nao util para predicao)")
    
    # Limpar coluna Dependents
    print("\n0) Limpeza da coluna Dependents:")
    df = clean_dependents_column(df)
    
    # Tratar valores ausentes
    print("\n1a) Tratamento de valores ausentes:")
    df_clean = handle_missing_values(df)
    
    # Separar features e target
    X = df_clean.drop('Loan_Status', axis=1)
    y = df_clean['Loan_Status']
    
    # Codificar target: Y -> 1 (aprovado), N -> 0 (rejeitado)
    y = y.map({'Y': 1, 'N': 0})
    
    # Label encoding para features categoricas
    print("\n1b) Conversao de atributos categoricos para numericos:")
    X_encoded, encoders = encode_categorical_features(X)
    
    # Divisao treino/teste
    print("\n1c) Divisao treino/teste:")
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    
    # Normalizacao
    print("\n1d) Normalizacao das features numericas:")
    print("Justificativa: O k-NN calcula distancia euclidiana entre pontos.")
    print("Sem normalizacao, variaveis como ApplicantIncome (0-50.000)")
    print("dominariam variaveis como Dependents (0-5).")
    print("Min-Max scaling coloca todas as features na escala [0,1].")
    
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    print("\nPre-processamento concluido!")
    print(f"  X_train shape: {X_train_scaled.shape}")
    print(f"  X_test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders


if __name__ == "__main__":
    from data_loader import load_loan_data
    
    df = load_loan_data()
    X_train, X_test, y_train, y_test, scaler, encoders = run_preprocessing(df)