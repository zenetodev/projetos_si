import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from typing import Tuple, List


def select_best_k(X_train, y_train, k_values: List[int] = [1, 3, 5, 7, 11, 15], cv: int = 5):
    print("\n" + "="*60)
    print("TAREFA 3 - SELECAO DO MELHOR K")
    print("="*60)
    
    mean_scores = []
    std_scores = []
    
    print(f"\nValidacao cruzada com {cv} folds para cada k:")
    print("-" * 50)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        
        print(f"k = {k:2d} -> Acuracia media: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    best_k = k_values[np.argmax(mean_scores)]
    best_score = max(mean_scores)
    
    print(f"\nMelhor k: {best_k} (acuracia media: {best_score:.4f})")
    
    return best_k, mean_scores, std_scores


def plot_k_selection(k_values: List[int], mean_scores: List[float], std_scores: List[float], 
                     save_path: str = "k_selection_plot.png"):
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, mean_scores, 'b-', marker='o', linewidth=2, markersize=8)
    
    # Adicionar barras de erro
    plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='none', ecolor='gray', capsize=5, elinewidth=2)
    
    # Destacar melhor k
    best_idx = np.argmax(mean_scores)
    plt.plot(k_values[best_idx], mean_scores[best_idx], 'ro', markersize=12, 
             label=f'Melhor k = {k_values[best_idx]}')
    
    plt.xlabel('Valor de k (numero de vizinhos)', fontsize=12, fontweight='bold')
    plt.ylabel('Acuracia Media (Validacao Cruzada 5-folds)', fontsize=12, fontweight='bold')
    plt.title('Selecao do Melhor k para k-NN - Classificacao de Credito', fontsize=14, fontweight='bold')
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    
    # Adicionar anotacoes
    y_min = max(0.6, min(mean_scores) - 0.05)
    y_max = min(0.9, max(mean_scores) + 0.05)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nGrafico salvo como '{save_path}'")


def analyze_k_behavior(k_values: List[int], mean_scores: List[float]) -> None:
    print("\n" + "="*60)
    print("ANALISE DO COMPORTAMENTO DO k-NN")
    print("="*60)
    
    print("\nEfeito de valores muito baixos de k (ex: k=1):")
    print("  - Modelo sofre OVERFITTING")
    print("  - Memoriza os dados de treino")
    print("  - Muito sensivel a ruido e outliers")
    print("  - Alta variancia, baixo viés")
    
    print("\nEfeito de valores muito altos de k (ex: k=15):")
    print("  - Modelo sofre UNDERFITTING")
    print("  - Fronteira de decisao muito suavizada")
    print("  - Perde detalhes importantes dos dados")
    print("  - Baixa variancia, alto viés")
    
    print("\nNeste problema:")
    best_idx = np.argmax(mean_scores)
    print(f"  Melhor k = {k_values[best_idx]} com acuracia = {mean_scores[best_idx]:.4f}")
    print(f"  k=1 acuracia = {mean_scores[0]:.4f}")
    print(f"  k=15 acuracia = {mean_scores[-1]:.4f}")


if __name__ == "__main__":
    from data_loader import load_loan_data
    from preprocessing import run_preprocessing
    
    df = load_loan_data()
    X_train, X_test, y_train, y_test, scaler, encoders = run_preprocessing(df)
    
    k_values = [1, 3, 5, 7, 11, 15]
    best_k, mean_scores, std_scores = select_best_k(X_train, y_train, k_values)
    plot_k_selection(k_values, mean_scores, std_scores)
    analyze_k_behavior(k_values, mean_scores)