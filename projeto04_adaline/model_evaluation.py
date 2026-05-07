import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score)
from adaline import Adaline
from typing import Dict, Tuple


def train_final_model(X_train: np.ndarray, y_train: np.ndarray, eta: float, n_iterations: int = 200) -> Adaline:
    """
    Treina o modelo final com a melhor taxa de aprendizagem.
    """
    print("\n" + "="*60)
    print("TAREFA 3 - TREINAMENTO DO MODELO FINAL")
    print("="*60)
    
    print(f"\nTreinando Adaline com η = {eta:.0e}")
    print(f"Maximo de epocas: {n_iterations}")
    
    adaline = Adaline(eta=eta, n_iterations=n_iterations, random_state=42)
    history = adaline.fit(X_train, y_train)
    
    print(f"\nResultado do treinamento:")
    print(f"  Epocas executadas: {len(history)}")
    if history and not np.isnan(history[-1]) and not np.isinf(history[-1]):
        print(f"  Custo final: {history[-1]:.6f}")
    else:
        print(f"  Custo final: valor invalido (nan/inf)")
    
    return adaline


def plot_final_cost_curve(cost_history: list, eta: float, save_path: str = "final_cost_curve.png"):
    """
    Plota a curva de convergencia do modelo final.
    """
    if not cost_history:
        print("Sem historico de custo para plotar.")
        return
    
    # Filtrar valores validos
    valid_history = [h for h in cost_history if not (np.isnan(h) or np.isinf(h))]
    
    if not valid_history:
        print("Nenhum valor valido no historico de custo.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_history, 'b-', linewidth=1.5)
    plt.xlabel('Epocas', fontsize=12)
    plt.ylabel('Custo (Erro Quadratico Medio)', fontsize=12)
    plt.title(f'Curva de Convergencia - Adaline (η = {eta:.0e})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # So usar escala log se nao houver valores negativos
    if min(valid_history) > 0:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nCurva de convergencia salva como '{save_path}'")


def evaluate_model(adaline: Adaline, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    print("\n" + "="*60)
    print("TAREFA 3 - AVALIACAO DO MODELO")
    print("="*60)
    
    # Predicoes
    y_pred = adaline.predict(X_test)
    
    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    
    print(f"\nAcuracia Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nMetricas para classe good (+1):")
    print(f"  Precisao: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    # Matriz de confusao formatada
    print("\nMatriz de Confusao:")
    print("-" * 40)
    print(f"{'':<15} {'Previsto good':<15} {'Previsto bad':<15}")
    print(f"{'Real good':<15} {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"{'Real bad':<15} {cm[1,0]:<15} {cm[1,1]:<15}")
    
    # Erros
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"\nAnalise detalhada:")
    print(f"  True Positives (good → good): {tp}")
    print(f"  True Negatives (bad → bad): {tn}")
    print(f"  False Positives (bad → good): {fp}")
    print(f"  False Negatives (good → bad): {fn}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "false_negatives": fn,
        "false_positives": fp
    }


def analyze_critical_errors(metrics: Dict) -> None:
    print("\n" + "="*60)
    print("TAREFA 4 - ANALISE DE ERROS CRITICOS")
    print("="*60)
    
    fn = metrics['false_negatives']  # good classificado como bad
    fp = metrics['false_positives']  # bad classificado como good
    
    print("\nCenario real de monitoramento ionosferico:")
    print("  - good = estrutura ionosferica detectada (sinal valido)")
    print("  - bad = sem estrutura coerente (falha/interferencia)")
    
    print(f"\nResultados do modelo:")
    print(f"  Falsos Negativos (good → bad): {fn}")
    print(f"  Falsos Positivos (bad → good): {fp}")
    
    print("\nAnalise de criticidade:")
    print("  - Falso Negativo: Sistema classifica sinal valido como falha")
    print("    -> Consequencia: Perda de dados uteis, alarme falso, interrupcao desnecessaria")
    print("    -> Nivel de criticidade: ALTO (perda de informacao valida)")
    
    print("  - Falso Positivo: Sistema classifica falha como sinal valido")
    print("    -> Consequencia: Dado incorreto usado em analises, mascara problemas reais")
    print("    -> Nivel de criticidade: ALTO tambem, mas com impacto diferente")
    
    print("\nConclusao - Qual erro mais critico?")
    if fn > fp:
        print("  Neste modelo, FALSOS NEGATIVOS sao mais frequentes")
        print("  Em aplicacoes de monitoramento, este erro e critico pois:")
        print("  - Pode levar a ignorar tempestades geomagneticas iminentes")
        print("  - Compromete sistemas de navegacao que dependem da previsao ionosferica")
    else:
        print("  Neste modelo, FALSOS POSITIVOS sao mais frequentes")
        print("  Em aplicacoes de monitoramento, este erro e critico pois:")
        print("  - Pode gerar alertas falsos de disturbios ionosfericos")
        print("  - Causa intervencoes desnecessarias em sistemas de comunicacao")
    
    print("\nRecomendacao:")
    print("  Para minimizar o erro mais critico, pode-se ajustar o limiar de decisao")
    print("  ou utilizar funcao de custo assimetrica no treinamento.")


def visualize_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    
    classes = ['good (+1)', 'bad (-1)']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    
    plt.xlabel('Previsto', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title('Matriz de Confusao - Adaline', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nMatriz de confusao salva como '{save_path}'")


if __name__ == "__main__":
    from data_loader import load_ionosphere_data
    from preprocessing import run_preprocessing
    from model_selection import study_learning_rates
    
    X, y = load_ionosphere_data()
    X_train, X_test, y_train, y_test, _, _ = run_preprocessing(X, y)
    
    best_eta, _ = study_learning_rates(X_train, y_train, [1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
    
    adaline = train_final_model(X_train, y_train, best_eta)
    metrics = evaluate_model(adaline, X_test, y_test)
    analyze_critical_errors(metrics)
    visualize_confusion_matrix(metrics['confusion_matrix'])