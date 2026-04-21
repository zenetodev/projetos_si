from data_loader import load_heart_disease_data
from preprocessing import run_preprocessing
from model_training import train_logistic_regression
from model_evaluation import evaluate_model, plot_roc_curve


def main():
    print("\n" + "="*60)
    print(" PROJETO: PREVISÃO DE DOENÇA CARDÍACA")
    print("="*60)
    
    # carrega dados
    print("\n Carregando dataset Heart Disease Cleveland (UCI ID=45)...")
    X, y = load_heart_disease_data()
    print(f" Dataset carregado: {X.shape[0]} pacientes, {X.shape[1]} features")
    
    # pré-processamento (Tarefa 1)
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(X, y)
    
    # treinamento (Tarefa 2)
    model, n_iterations = train_logistic_regression(X_train, y_train)
    
    # avaliação (Tarefa 3)
    metrics = evaluate_model(model, X_test, y_test)
    
    # curva ROC
    plot_roc_curve(y_test, metrics["y_pred_proba"])
    
    # resumo final
    print("\n" + "="*60)
    print(" PROJETO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\n RESUMO DAS MÉTRICAS:")
    print(f"   • Acurácia:  {metrics['accuracy']:.4f}")
    print(f"   • Precisão:  {metrics['precision']:.4f}")
    print(f"   • Recall:    {metrics['recall']:.4f}")
    print(f"   • F1-score:  {metrics['f1_score']:.4f}")
    print(f"   • AUC:       {metrics['auc']:.4f}")
    print(f"\n Modelo convergiu em {n_iterations} iterações")


if __name__ == "__main__":
    main()