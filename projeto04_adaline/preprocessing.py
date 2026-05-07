"""
Modulo de pre-processamento:
- Recodificacao do target para +1 (good) e -1 (bad)
- Remocao de atributos constantes
- Normalizacao Z-score
- Divisao treino/teste com stratify
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def recode_target(y: pd.Series) -> pd.Series:
    """
    Recodifica o target: 'g' (good) -> +1, 'b' (bad) -> -1.
    
    Args:
        y: Series com valores 'g' e 'b'
    
    Returns:
        Series com valores +1 e -1 (inteiros)
    """
    # Mapeamento correto para os valores originais
    y_recode = y.map({'g': 1, 'b': -1})
    
    # Verificar se houve mapeamento correto
    if y_recode.isnull().any():
        print(f"  ATENCAO: Valores nao mapeados encontrados: {y[y_recode.isnull()].unique()}")
        # Fallback: converter string para mapeamento generico
        unique_vals = y.unique()
        if len(unique_vals) == 2:
            y_recode = y.map({unique_vals[0]: 1, unique_vals[1]: -1})
    
    # Converter para int (agora sem NaN)
    y_recode = y_recode.astype(int)
    
    print("\nDistribuicao apos recodificacao:")
    print(f"  good (+1): {(y_recode == 1).sum()} amostras")
    print(f"  bad (-1): {(y_recode == -1).sum()} amostras")
    
    return y_recode


def remove_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove atributos que sao constantes (variancia zero).
    
    Args:
        X: DataFrame com features
    
    Returns:
        DataFrame sem features constantes
    """
    X_clean = X.copy()
    variancias = X_clean.var()
    
    constant_cols = variancias[variancias < 1e-10].index.tolist()
    
    if constant_cols:
        print(f"\nAtributos constantes removidos: {len(constant_cols)} atributos")
        print(f"  Colunas removidas: {constant_cols[:5]}..." if len(constant_cols) > 5 else f"  Colunas removidas: {constant_cols}")
        X_clean = X_clean.drop(columns=constant_cols)
    else:
        print("\nNenhum atributo constante encontrado.")
    
    print(f"Shape apos remocao: {X_clean.shape}")
    
    return X_clean


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """
    Normaliza as features usando StandardScaler (Z-score).
    
    Args:
        X_train: Features de treino
        X_test: Features de teste
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("\nNormalizacao Z-score aplicada:")
    print(f"  Media de cada feature (treino): ~0")
    print(f"  Desvio padrao de cada feature (treino): ~1")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Divide dados em treino e teste com estratificacao.
    """
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
    print(f"  good (+1): {(y_train == 1).mean():.2%}")
    print(f"  bad (-1): {(y_train == -1).mean():.2%}")
    
    print("\nProporcao de classes no teste:")
    print(f"  good (+1): {(y_test == 1).mean():.2%}")
    print(f"  bad (-1): {(y_test == -1).mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def run_preprocessing(X: pd.DataFrame, y: pd.Series):
    """
    Executa todo o pipeline de pre-processamento.
    """
    print("\n" + "="*60)
    print("TAREFA 1 - PREPARACAO DOS DADOS")
    print("="*60)
    
    print("\n1a) Recodificacao do target (+1/-1):")
    y_bipolar = recode_target(y)
    
    print("\n1b) Remocao de atributos constantes:")
    X_clean = remove_constant_features(X)
    
    print("\nVerificacao de valores ausentes:")
    nulos = X_clean.isnull().sum().sum()
    print(f"  Total de valores nulos: {nulos}")
    
    print("\n1c) Divisao treino/teste com estratificacao:")
    X_train, X_test, y_train, y_test = split_data(X_clean, y_bipolar)
    
    print("\n1d) Normalizacao Z-score:")
    print("Justificativa algebraica:")
    print("  A regra Delta atualiza pesos: Δw = η * (d - y) * x")
    print("  Se x tem escalas diferentes, os gradientes sao desproporcionais")
    print("  Z-score (media=0, std=1) iguala a magnitude de todos os atributos")
    print("  Garantindo estabilidade e convergencia mais rapida")
    
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Converter para numpy arrays para o Adaline
    X_train_array = X_train_scaled.values
    X_test_array = X_test_scaled.values
    y_train_array = y_train.values
    y_test_array = y_test.values
    
    return X_train_array, X_test_array, y_train_array, y_test_array, X_train_scaled, X_test_scaled


if __name__ == "__main__":
    from data_loader import load_ionosphere_data
    X, y = load_ionosphere_data()
    X_train, X_test, y_train, y_test, _, _ = run_preprocessing(X, y)
    print(f"\nPre-processamento concluido!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")