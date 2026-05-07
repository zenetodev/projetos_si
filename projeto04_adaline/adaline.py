import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class Adaline:
    
    def __init__(self, eta: float = 0.0001, n_iterations: int = 200, random_state: int = 42):
        self.eta = eta
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.cost_history_ = []  # Lista para historico de custo
        
    def _initialize_weights(self, n_features: int):

        np.random.seed(self.random_state)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
    
    def _activation(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        linear_output = self._activation(X)
        return np.where(linear_output >= threshold, 1, -1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        self._initialize_weights(n_features)
        
        # Historico de custo
        self.cost_history_ = []
        
        # Loop de treinamento
        for epoch in range(self.n_iterations):
            # Calcular saida linear para todas as amostras
            linear_output = self._activation(X)
            
            # Calcular erro (d - y) onde d = target, y = saida linear
            errors = y - linear_output
            
            # Atualizar pesos (regra Delta)
            self.weights += self.eta * np.dot(X.T, errors)
            self.bias += self.eta * np.sum(errors)
            
            # Calcular custo (erro quadratico medio)
            cost = np.mean(errors ** 2) / 2
            
            # Verificar overflow/inf/nan
            if np.isnan(cost) or np.isinf(cost):
                print(f"  Overflow detectado na epoca {epoch + 1}. Parando treinamento.")
                break
                
            self.cost_history_.append(cost)
            
            # Criterio de parada (se custo estabilizou)
            if epoch > 10 and abs(self.cost_history_[-1] - self.cost_history_[-2]) < 1e-8:
                break
        
        return self.cost_history_
    
    def get_cost_history(self) -> List[float]:
        return self.cost_history_


def plot_convergence_curves(results: Dict, save_path: str = "convergence_curves.png"):
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'orange']
    
    for i, (eta, history) in enumerate(results.items()):
        color = colors[i % len(colors)]
        if history and len(history) > 0:
            # Filtrar valores invalidos para plotagem
            valid_history = [h for h in history if not (np.isnan(h) or np.isinf(h))]
            if valid_history:
                plt.plot(valid_history, color=color, linewidth=1.5, label=f'η = {eta:.0e}')
    
    plt.xlabel('Epocas', fontsize=12)
    plt.ylabel('Custo (Erro Quadratico Medio)', fontsize=12)
    plt.title('Curvas de Convergencia - Rede Adaline\nIonosphere Dataset', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nGrafico de convergencia salvo como '{save_path}'")


if __name__ == "__main__":
    # Teste simples
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_test = np.array([-1, -1, -1, 1])  # AND gate bipolar
    
    adaline = Adaline(eta=0.01, n_iterations=50)
    history = adaline.fit(X_test, y_test)
    print(f"Predicoes: {adaline.predict(X_test)}")
    print(f"Historico de custo: {len(history)} epocas")