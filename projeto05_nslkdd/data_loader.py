import pandas as pd
import os
import kagglehub
from typing import Tuple


def get_column_names():
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"
    ]
    return col_names


def download_nslkdd_dataset():
    print("Baixando dataset NSL-KDD do Kaggle...")
    path = kagglehub.dataset_download("hassan06/nslkdd")
    print(f"Dataset baixado em: {path}")
    return path


def load_nslkdd_data(train_path: str = None, test_path: str = None):
    col_names = get_column_names()
    
    if train_path is None or test_path is None:
        # Primeiro, tentar encontrar em locais comuns
        possible_paths = [
            "KDDTrain+.txt",
            "KDDTest+.txt",
            "data/KDDTrain+.txt",
            "data/KDDTest+.txt",
            "../KDDTrain+.txt",
            "../KDDTest+.txt",
            "nslkdd/KDDTrain+.txt",
            "nslkdd/KDDTest+.txt"
        ]
        
        local_train = None
        local_test = None
        
        for i in range(0, len(possible_paths), 2):
            if os.path.exists(possible_paths[i]):
                local_train = possible_paths[i]
            if i+1 < len(possible_paths) and os.path.exists(possible_paths[i+1]):
                local_test = possible_paths[i+1]
            if local_train and local_test:
                break
        
        if local_train and local_test:
            train_path = local_train
            test_path = local_test
            print(f"Usando arquivos locais: {train_path}, {test_path}")
        else:
            # Baixar do Kaggle
            try:
                dataset_path = download_nslkdd_dataset()
                train_path = os.path.join(dataset_path, "KDDTrain+.txt")
                test_path = os.path.join(dataset_path, "KDDTest+.txt")
                
                if not os.path.exists(train_path):
                    # Tentar encontrar os arquivos na pasta baixada
                    for root, dirs, files in os.walk(dataset_path):
                        for file in files:
                            if file == "KDDTrain+.txt":
                                train_path = os.path.join(root, file)
                            elif file == "KDDTest+.txt":
                                test_path = os.path.join(root, file)
                
                print(f"Arquivo de treino: {train_path}")
                print(f"Arquivo de teste: {test_path}")
                
            except Exception as e:
                raise FileNotFoundError(
                    f"Nao foi possivel baixar o dataset: {e}\n"
                    "Baixe manualmente do Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd\n"
                    "E coloque os arquivos KDDTrain+.txt e KDDTest+.txt na pasta do projeto."
                )
    
    # Carregar dados
    print(f"\nCarregando treino de: {train_path}")
    train_df = pd.read_csv(train_path, names=col_names)
    
    print(f"Carregando teste de: {test_path}")
    test_df = pd.read_csv(test_path, names=col_names)
    
    # Descartar coluna difficulty_level
    train_df = train_df.drop('difficulty_level', axis=1)
    test_df = test_df.drop('difficulty_level', axis=1)
    
    print(f"\nTreino: {train_df.shape[0]} amostras, {train_df.shape[1]} colunas")
    print(f"Teste: {test_df.shape[0]} amostras, {test_df.shape[1]} colunas")
    
    return train_df, test_df


def get_dataset_info(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "="*60)
    print("INFORMACOES DO DATASET")
    print("="*60)
    
    print("\nColunas categoricas:")
    categorical = ['protocol_type', 'service', 'flag']
    for col in categorical:
        print(f"  {col}: {train_df[col].nunique()} valores unicos")
    
    print("\nDistribuicao de classes no treino:")
    attack_counts = train_df['attack_type'].value_counts()
    normal_count = attack_counts.get('normal', 0)
    attack_count = attack_counts.sum() - normal_count
    print(f"  normal: {normal_count} ({normal_count/len(train_df)*100:.2f}%)")
    print(f"  ataque: {attack_count} ({attack_count/len(train_df)*100:.2f}%)")
    
    print("\nDistribuicao de classes no teste:")
    attack_counts_test = test_df['attack_type'].value_counts()
    normal_count_test = attack_counts_test.get('normal', 0)
    attack_count_test = attack_counts_test.sum() - normal_count_test
    print(f"  normal: {normal_count_test} ({normal_count_test/len(test_df)*100:.2f}%)")
    print(f"  ataque: {attack_count_test} ({attack_count_test/len(test_df)*100:.2f}%)")


if __name__ == "__main__":
    train_df, test_df = load_nslkdd_data()
    get_dataset_info(train_df, test_df)