import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001,
                early_stopping_patience=10, device='cpu'):
    print("\n" + "="*60)
    print("TAREFA 3 - TREINAMENTO E MONITORAMENTO")
    print("="*60)
    
    # Configurar criterio e otimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Mover modelo para o dispositivo
    model = model.to(device)
    
    # Historico
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nIniciando treinamento por {epochs} epocas...")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Treinar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validar
        val_loss = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        # Print a cada 5 epocas
        if (epoch + 1) % 5 == 0:
            print(f"Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping na epoca {epoch+1}")
                print(f"Melhor val loss: {best_val_loss:.6f}")
                break
    
    # Carregar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("\nTreinamento concluido!")
    print(f"Melhor loss de validacao: {best_val_loss:.6f}")
    
    return history, model


def plot_loss_curves(history: Dict, save_path: str = "loss_curves.png"):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Treino')
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=1.5, label='Validacao')
    
    plt.xlabel('Epocas', fontsize=12)
    plt.ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)
    plt.title('Curvas de Loss - MLP para Deteccao de Intrusao', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nCurvas de loss salvas como '{save_path}'")


def train_with_topology(X_train, y_train, X_val, y_val, input_dim, topology, device='cpu'):
    from dataset import create_dataloaders
    from model import create_model
    
    # Criar DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=256
    )
    
    # Criar modelo
    model = create_model(input_dim, topology)
    
    # Treinar
    history, _ = train_model(model, train_loader, val_loader, epochs=50, device=device)
    
    return history


if __name__ == "__main__":
    import numpy as np
    from preprocessing import run_preprocessing
    from data_loader import load_nslkdd_data
    from dataset import create_dataloaders
    from model import create_model
    
    # Carregar e processar dados
    train_df, test_df = load_nslkdd_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(train_df, test_df)
    
    # Criar DataLoaders (divide 20% para validacao)
    train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=256, val_split=0.2)
    
    # Criar modelo
    input_dim = X_train.shape[1]
    model = create_model(input_dim, 'medium')
    
    # Treinar
    history, best_model = train_model(model, train_loader, val_loader, epochs=50)
    
    # Plotar curvas
    plot_loss_curves(history)