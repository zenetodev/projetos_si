import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       batch_size: int = 256, val_split: float = 0.2,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Se validacao nao foi fornecida, dividir do treino
    if X_val is None:
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
            X_train_tensor, y_train_tensor,
            test_size=val_split,
            random_state=random_state,
            stratify=y_train
        )
    else:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Criar TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders criados:")
    print(f"  Treino: {len(train_dataset)} amostras, {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  Validacao: {len(val_dataset)} amostras, {len(val_loader)} batches")
    
    return train_loader, val_loader


def create_test_loader(X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 256) -> DataLoader:
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoader de teste criado:")
    print(f"  Teste: {len(test_dataset)} amostras, {len(test_loader)} batches")
    
    return test_loader


if __name__ == "__main__":
    import numpy as np
    from preprocessing import run_preprocessing
    from data_loader import load_nslkdd_data
    
    train_df, test_df = load_nslkdd_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(train_df, test_df)
    
    train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=256)
    test_loader = create_test_loader(X_test, y_test, batch_size=256)