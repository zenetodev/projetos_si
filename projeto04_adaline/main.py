from data_loader import load_ionosphere_data
from preprocessing import run_preprocessing
from model_selection import study_learning_rates, plot_convergence_curves, analyze_convergence
from model_evaluation import (train_final_model, evaluate_model, 
                              analyze_critical_errors, visualize_confusion_matrix,
                              plot_final_cost_curve)


def main():
    print("\n" + "="*60)
    print("PROJETO: CLASSIFICACAO IONOSFERICA - REDE ADALINE")
    print("="*60)
    
    # Carrega dados
    print("\nCarregando dataset Ionosphere (UCI ID=52)...")
    X, y = load_ionosphere_data()
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Pre-processamento (Tarefa 1)
    X_train, X_test, y_train, y_test, X_train_df, X_test_df = run_preprocessing(X, y)
    
    # Estudo das taxas de aprendizagem (Tarefa 2)
    learning_rates = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    best_eta, results = study_learning_rates(X_train, y_train, learning_rates)
    
    # Plotar curvas de convergencia
    plot_convergence_curves(results)
    analyze_convergence(results)
    
    # Treinar modelo final com melhor eta (Tarefa 3)
    adaline = train_final_model(X_train, y_train, best_eta)
    plot_final_cost_curve(adaline.get_cost_history(), best_eta)
    
    # Avaliar no teste (Tarefa 3)
    metrics = evaluate_model(adaline, X_test, y_test)
    visualize_confusion_matrix(metrics['confusion_matrix'])
    
    # Analise de erros criticos (Tarefa 4)
    analyze_critical_errors(metrics)
    
    # Resumo final
    print("\n" + "="*60)
    print("="*60)
    print(f"\nRESUMO FINAL:")
    print(f"  Melhor taxa de aprendizagem: η = {best_eta:.0e}")
    print(f"  Acuracia no teste: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-score (good): {metrics['f1']:.4f}")
    print(f"  Falsos Negativos (good → bad): {metrics['false_negatives']}")
    print(f"  Falsos Positivos (bad → good): {metrics['false_positives']}")
    print("\nArquivos gerados:")
    print("  - convergence_curves.png (curvas para diferentes η)")
    print("  - final_cost_curve.png (curva do modelo final)")
    print("  - confusion_matrix.png (matriz de confusao)")


if __name__ == "__main__":
    main()