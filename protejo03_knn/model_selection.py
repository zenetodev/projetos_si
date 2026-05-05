import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from typing import List


def select_best_k(X_train, y_train, k_values: List[int] = [1, 3, 5, 7, 11], cv: int = 5):
    print("\n" + "="*60)
    print("TAREFA 3 - SELECAO DO MELHOR K")
    print("="*60)
    
    mean_scores = []
    std_scores = []
    
    print(f"\nValidacao cruzada com {cv} folds para cada k:")
    print("-" * 50)
    
    for k in k_values:
        # Criar modelo k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Cross-validation
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        
        print(f"k = {k:2d} -> Acuracia media: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Encontrar melhor k
    best_k = k_values[np.argmax(mean_scores)]
    best_score = max(mean_scores)
    
    print(f"\nMelhor k: {best_k} (acuracia media: {best_score:.4f})")
    
    return best_k, mean_scores, std_scores


def plot_k_selection(k_values: List[int], mean_scores: List[float], std_scores: List[float], save_path: str = "k_selection_plot.png"):

    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, mean_scores, 'b-', marker='o', linewidth=2, markersize=8)
    
    plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='none', ecolor='gray', capsize=5, elinewidth=2)
    
    best_idx = np.argmax(mean_scores)
    plt.plot(k_values[best_idx], mean_scores[best_idx], 'ro', markersize=12, label=f'Melhor k = {k_values[best_idx]}')
    
    plt.xticks(k_values, [str(k) for k in k_values], fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    y_min = max(0, min(mean_scores) - 0.05)
    y_max = min(1.0, max(mean_scores) + 0.05)
    plt.ylim(y_min, y_max)
    
    plt.xlabel('Valor de k (numero de vizinhos)', fontsize=12, fontweight='bold')
    plt.ylabel('Acuracia Media (Validacao Cruzada 5-folds)', fontsize=12, fontweight='bold')
    plt.title('Selecao do Melhor k para k-NN - Classificacao de Vinhos', fontsize=14, fontweight='bold')
    
    if len(k_values) >= 2:
        plt.annotate('k baixo = overfitting\n(sensivel a ruido)', 
                     xy=(k_values[0], mean_scores[0]), 
                     xytext=(k_values[0] + 1, mean_scores[0] - 0.1),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        plt.annotate('k alto = underfitting\n(fronteira suavizada)', 
                     xy=(k_values[-1], mean_scores[-1]), 
                     xytext=(k_values[-1] - 3, mean_scores[-1] - 0.1),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGrafico salvo como '{save_path}'")