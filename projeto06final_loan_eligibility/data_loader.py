import pandas as pd
import os
import kagglehub
from typing import Tuple


def download_loan_dataset():
    print("Baixando dataset Eligibility Prediction for Loan do Kaggle...")
    path = kagglehub.dataset_download("devzohaib/eligibility-prediction-for-loan")
    print(f"Dataset baixado em: {path}")
    return path


def load_loan_data(file_path: str = None) -> pd.DataFrame:
    if file_path is not None and os.path.exists(file_path):
        print(f"Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        print(f"\nDataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
        return df
    
    # Tentar encontrar em locais comuns
    possible_paths = [
        "Loan_Data.csv",
        "data/Loan_Data.csv",
        "../Loan_Data.csv",
        "loan_data/Loan_Data.csv",
        "eligibility-prediction-for-loan/Loan_Data.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Carregando dados de: {path}")
            df = pd.read_csv(path)
            print(f"\nDataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
            return df
    
    # Se nao encontrou, baixar do Kaggle
    try:
        dataset_path = download_loan_dataset()
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Carregando dados de: {file_path}")
                    df = pd.read_csv(file_path)
                    print(f"\nDataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
                    return df
        
        raise FileNotFoundError("Arquivo CSV nao encontrado no dataset baixado.")
        
    except Exception as e:
        raise FileNotFoundError(
            f"Nao foi possivel baixar o dataset: {e}\n"
            "Baixe manualmente do Kaggle: https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan\n"
            "E coloque o arquivo Loan_Data.csv na pasta do projeto."
        )


def get_dataset_info(df: pd.DataFrame):
    print("\n" + "="*60)
    print("INFORMACOES DO DATASET")
    print("="*60)
    
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    print("\nTipos de dados:")
    print(df.dtypes)
    
    print("\nValores ausentes:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, val in missing_cols.items():
            print(f"  {col}: {val} valores ausentes ({val/len(df)*100:.2f}%)")
    else:
        print("  Nenhum valor ausente")
    
    print("\nDistribuicao da variavel alvo (Loan_Status):")
    target_dist = df['Loan_Status'].value_counts()
    print(f"  Y (aprovado): {target_dist.get('Y', 0)} ({target_dist.get('Y', 0)/len(df)*100:.2f}%)")
    print(f"  N (rejeitado): {target_dist.get('N', 0)} ({target_dist.get('N', 0)/len(df)*100:.2f}%)")
    
    print("\nEstatisticas das variaveis numericas:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe())


if __name__ == "__main__":
    df = load_loan_data()
    get_dataset_info(df)