from data_loader import load_wine_data
from preprocessing import run_preprocessing
from model_selection import select_best_k, plot_k_selection
from model_evaluation import evaluate_model, analyze_confusion_matrix, visualize_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def main():
    print("\n" + "="*60)
    print("PROJETO: CLASSIFICACAO DE VINHOS (k-NN)")
    print("="*60)
    
    # Carrega dados
    print("\nCarregando dataset Wine Quality Red (UCI ID=109)...")
    X, y = load_wine_data()
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Pre-processamento (Tarefa 1)
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(X, y)
    
    # Selecao do melhor k (Tarefa 3)
    k_values = [1, 3, 5, 7, 11]
    best_k, mean_scores, std_scores = select_best_k(X_train, y_train, k_values)
    
    # Plotar grafico k vs acuracia
    plot_k_selection(k_values, mean_scores, std_scores)
    
    # Resposta sobre k baixo vs k alto
    print("\n" + "="*60)
    print("INTERPRETACAO - Efeito do k no modelo")
    print("="*60)
    print("Para valores muito baixos de k (ex: k=1):")
    print("  - Modelo sofre overfitting")
    print("  - Sensivel a ruido e outliers")
    print("  - Alta variancia, baixo viés")
    print("\nPara valores muito altos de k (ex: k=11):")
    print("  - Modelo sofre underfitting")
    print("  - Fronteira de decisao muito suavizada")
    print("  - Baixa variancia, alto viés")
    
    # Treinar modelo final com melhor k
    print(f"\nTreinando modelo final com k={best_k}...")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # Avaliacao final no teste (Tarefa 4)
    metrics = evaluate_model(knn, X_test, y_test)
    
    # Analisar matriz de confusao
    analyze_confusion_matrix(metrics['confusion_matrix'], ['Ruim', 'Medio', 'Bom'])
    
    # Visualizar matriz de confusao
    visualize_confusion_matrix(metrics['confusion_matrix'], ['Ruim', 'Medio', 'Bom'])
    
    # Resumo final
    print("\n" + "="*60)
    print("PROJETO CONCLUIDO COM SUCESSO!")
    print("="*60)
    print(f"\nRESUMO FINAL:")
    print(f"  Melhor k: {best_k}")
    print(f"  Acuracia no teste: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-score macro: {(metrics['f1'].mean()):.4f}")
    print("\nArquivos gerados:")
    print("  - k_selection_plot.png (grafico de acuracia vs k)")
    print("  - confusion_matrix.png (matriz de confusao)")


if __name__ == "__main__":
    main()