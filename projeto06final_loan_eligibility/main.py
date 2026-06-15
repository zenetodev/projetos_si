from data_loader import load_loan_data, get_dataset_info
from preprocessing import run_preprocessing
from model_selection import select_best_k, plot_k_selection, analyze_k_behavior
from model_evaluation import (evaluate_model, analyze_classification_errors, 
                              visualize_confusion_matrix, print_model_interpretation)
from sklearn.neighbors import KNeighborsClassifier  # <--- ADICIONAR ESTA LINHA


def main():
    print("\n" + "="*60)
    print("PROJETO: CLASSIFICACAO DE ELEGIBILIDADE PARA EMPRESTIMO (k-NN)")
    print("="*60)
    
    # 1. Carregar dados
    print("\nCarregando dataset Eligibility Prediction for Loan...")
    df = load_loan_data()
    get_dataset_info(df)
    
    # 2. Pre-processamento (Tarefa 1)
    X_train, X_test, y_train, y_test, scaler, encoders = run_preprocessing(df)
    
    # 3. Selecao do melhor k (Tarefa 3)
    k_values = [1, 3, 5, 7, 11, 15]
    best_k, mean_scores, std_scores = select_best_k(X_train, y_train, k_values)
    
    # 4. Plotar grafico k vs acuracia
    plot_k_selection(k_values, mean_scores, std_scores)
    
    # 5. Analisar comportamento do k
    analyze_k_behavior(k_values, mean_scores)
    
    # 6. Treinar modelo final com melhor k
    print(f"\nTreinando modelo final com k={best_k}...")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # 7. Avaliacao final no teste (Tarefa 4)
    metrics = evaluate_model(knn, X_test, y_test)
    
    # 8. Analisar erros
    analyze_classification_errors(metrics)
    
    # 9. Visualizar matriz de confusao
    visualize_confusion_matrix(metrics['confusion_matrix'])
    
    # 10. Interpretacao final
    print_model_interpretation()
    
    # 11. Resumo final
    print("\n" + "="*60)
    print("PROJETO CONCLUIDO COM SUCESSO!")
    print("="*60)
    print(f"\nRESUMO FINAL:")
    print(f"  Melhor k: {best_k}")
    print(f"  Acuracia no teste: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precisao: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Falsos Positivos (rejeitado → aprovado): {metrics['false_positives']}")
    print(f"  Falsos Negativos (aprovado → rejeitado): {metrics['false_negatives']}")
    print("\nArquivos gerados:")
    print("  - k_selection_plot.png (grafico de acuracia vs k)")
    print("  - confusion_matrix.png (matriz de confusao)")


if __name__ == "__main__":
    main()