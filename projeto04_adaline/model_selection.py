import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from adaline import Adaline, plot_convergence_curves


def study_learning_rates(X_train: np.ndarray, y_train: np.ndarray,
                        learning_rates: List[float] = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
                        n_iterations: int = 200) -> Tuple[float, Dict]:

    print("\n" + "="*60)
    print("TAREFA 2 - ESTUDO DA TAXA DE APRENDIZAGEM")
    print("="*60)
    
    results = {}
    best_eta = None
    best_final_cost = float('inf')
    
    print(f"\nTestando {len(learning_rates)} taxas de aprendizagem:")
    print(f"Maximo de epocas: {n_iterations}")
    print("-" * 50)
    
    for eta in learning_rates:
        print(f"\nη = {eta:.0e}:")
        
        # Criar e treinar Adaline
        adaline = Adaline(eta=eta, n_iterations=n_iterations, random_state=42)
        history = adaline.fit(X_train, y_train)
        
        results[eta] = adaline.get_cost_history()
        final_cost = results[eta][-1]
        
        print(f"  Custo final: {final_cost:.6f}")
        print(f"  Epocas executadas: {len(results[eta])}")
        
        if final_cost < best_final_cost:
            best_final_cost = final_cost
            best_eta = eta
    
    print("\n" + "="*50)
    print(f"Melhor taxa de aprendizagem: η = {best_eta:.0e}")
    print(f"Menor custo final: {best_final_cost:.6f}")
    
    return best_eta, results


def analyze_convergence(results: Dict) -> None:
    print("\n" + "="*60)
    print("ANALISE DAS CURVAS DE CONVERGENCIA")
    print("="*60)
    
    for eta, history in results.items():
        print(f"\nη = {eta:.0e}:")
        print(f"  Custo inicial: {history[0]:.6f}")
        print(f"  Custo final: {history[-1]:.6f}")
        print(f"  Reducao: {(1 - history[-1]/history[0])*100:.2f}%")
        
        # Verificar se convergiu
        if len(history) < 200:
            print(f"  Convergiu antes da epoca 200")
        else:
            # Verificar se ainda esta decaindo
            recent_slope = (history[-1] - history[-10]) / 9
            if abs(recent_slope) < 1e-6:
                print(f"  Convergiu (custo estabilizado)")
            else:
                print(f"  Pode precisar de mais epocas")
    
    print("\nRecomendacao:")
    print("  Taxas muito baixas (1e-4, 1e-3): convergencia lenta")
    print("  Taxas moderadas (1e-2, 5e-2): boa convergencia")
    print("  Taxas altas (1e-1): pode oscilar ou divergir")


if __name__ == "__main__":
    from data_loader import load_ionosphere_data
    from preprocessing import run_preprocessing
    
    X, y = load_ionosphere_data()
    X_train, X_test, y_train, y_test, _, _ = run_preprocessing(X, y)
    
    best_eta, results = study_learning_rates(X_train, y_train)
    plot_convergence_curves(results)
    analyze_convergence(results)