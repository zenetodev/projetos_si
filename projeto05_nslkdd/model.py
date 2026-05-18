import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float = 0.3):
        super(MLP, self).__init__()
        
        # Construir camadas
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Camada de saida (sem ativacao, pois BCEWithLogitsLoss ja inclui sigmoide)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def get_model_topologies():
    topologies = {
        'small': [64, 32],
        'medium': [128, 64, 32],
        'large': [256, 128, 64, 32],
        'very_large': [512, 256, 128, 64],
        'shallow': [128],  # Apenas uma camada oculta
        'deep': [256, 128, 64, 32, 16]  # Muitas camadas
    }
    return topologies


def create_model(input_dim: int, topology: str = 'medium', dropout_rate: float = 0.3) -> MLP:
    topologies = get_model_topologies()
    
    if topology not in topologies:
        raise ValueError(f"Topologia invalida. Escolha entre: {list(topologies.keys())}")
    
    hidden_dims = topologies[topology]
    model = MLP(input_dim, hidden_dims, dropout_rate)
    
    print(f"\nModelo MLP criado (topologia: {topology}):")
    print(f"  Camadas ocultas: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Total de parametros: {sum(p.numel() for p in model.parameters())}")
    
    return model


if __name__ == "__main__":
    # Teste com dados simulados
    input_dim = 120  # Exemplo apos one-hot encoding
    model = create_model(input_dim, 'medium')
    
    # Forward pass teste
    x = torch.randn(32, input_dim)
    output = model(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sample: {output[0, 0].item():.4f}")