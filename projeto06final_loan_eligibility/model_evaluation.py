import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, ConfusionMatrixDisplay)
from typing import Dict, Tuple


def evaluate_model(model, X_test, y_test) -> Dict:
    print("\n" + "="*60)
    print("TAREFA 4 - AVALIACAO DO MODELO")
    print("="*60)
    
    # Predicoes
    y_pred = model.predict(X_test)
    
    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    
    print(f"\nAcuracia Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precisao: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-score: {f1:.4f} ({f1*100:.2f}%)")
    
    # Matriz de confusao
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print("\nMatriz de Confusao:")
    print("-" * 45)
    print(f"{'':<15} {'Previsto Aprovado':<18} {'Previsto Rejeitado':<18}")
    print(f"{'Real Aprovado':<15} {tp:<18} {fn:<18}")
    print(f"{'Real Rejeitado':<15} {fp:<18} {tn:<18}")
    
    print("\nAnalise detalhada:")
    print(f"  True Positives (aprovado → aprovado): {tp}")
    print(f"  True Negatives (rejeitado → rejeitado): {tn}")
    print(f"  False Positives (rejeitado → aprovado): {fp}")
    print(f"  False Negatives (aprovado → rejeitado): {fn}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn
    }


def analyze_classification_errors(metrics: Dict) -> None:
    print("\n" + "="*60)
    print("ANALISE DOS ERROS DE CLASSIFICACAO")
    print("="*60)
    
    fp = metrics['false_positives']  # Rejeitado classificado como Aprovado
    fn = metrics['false_negatives']  # Aprovado classificado como Rejeitado
    
    print("\nCenario real: Concessao de Credito Bancario")
    print("  - Aprovado: Cliente recebe o emprestimo")
    print("  - Rejeitado: Cliente nao recebe o emprestimo")
    
    print(f"\nResultados do modelo:")
    print(f"  Falsos Positivos (rejeitado → aprovado): {fp}")
    print(f"  Falsos Negativos (aprovado → rejeitado): {fn}")
    
    print("\nAnalise de Impacto:")
    print("  - Falso Positivo: Cliente REJEITADO recebeu emprestimo")
    print("    -> Consequencia: Risco de inadimplencia, prejuizo financeiro")
    print("    -> Impacto: ALTO (custo direto para o banco)")
    
    print("  - Falso Negativo: Cliente APROVADO foi rejeitado")
    print("    -> Consequencia: Perda de cliente potencial")
    print("    -> Impacto: Custo de oportunidade (cliente buscaria outro banco)")
    
    print("\nConclusao:")
    if fp > fn:
        print("  O modelo apresenta MAIOR DIFICULDADE em aprovar clientes que seriam rejeitados")
        print("  (Falsos Positivos sao mais frequentes)")
        print("  Isso representa risco financeiro direto para o banco")
    elif fn > fp:
        print("  O modelo apresenta MAIOR DIFICULDADE em rejeitar clientes que seriam aprovados")
        print("  (Falsos Negativos sao mais frequentes)")
        print("  Isso representa perda de oportunidade de negocios")
    else:
        print("  O modelo tem equilibrio entre ambos os tipos de erro")


def visualize_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Aprovado (1)', 'Rejeitado (0)'])
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title('Matriz de Confusao - k-NN para Concessao de Credito', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nMatriz de confusao salva como '{save_path}'")


def print_model_interpretation():
    print("\n" + "="*60)
    print("CONCLUSOES SOBRE A ADEQUACAO DO k-NN")
    print("="*60)
    
    print("\nPontos Positivos do k-NN para este problema:")
    print("  - Implementacao simples e intuitiva")
    print("  - Nao requer treinamento pesado (lazy learning)")
    print("  - Funciona bem com dados de tamanho moderado (614 amostras)")
    print("  - Facil interpretacao (vizinhança proxima)")
    
    print("\nPontos Negativos / Limitacoes:")
    print("  - Escolha do k requer validacao cruzada")
    print("  - Sensivel a features irrelevantes")
    print("  - Requer normalizacao obrigatoria")
    print("  - Pior desempenho com dados desbalanceados")
    
    print("\nRecomendacao para uso real:")
    print("  - O modelo e ADEQUADO como ferramenta de apoio a decisao")
    print("  - Deve ser usado em conjunto com analise humana")
    print("  - Para producao, recomenda-se atualizacao periodica do modelo")
    print("  - Considerar outros algoritmos (Random Forest, XGBoost) para comparacao")


if __name__ == "__main__":
    from data_loader import load_loan_data
    from preprocessing import run_preprocessing
    from model_selection import select_best_k
    
    df = load_loan_data()
    X_train, X_test, y_train, y_test, scaler, encoders = run_preprocessing(df)
    
    k_values = [1, 3, 5, 7, 11, 15]
    best_k, mean_scores, std_scores = select_best_k(X_train, y_train, k_values)
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    metrics = evaluate_model(knn, X_test, y_test)
    analyze_classification_errors(metrics)
    visualize_confusion_matrix(metrics['confusion_matrix'])
    print_model_interpretation()