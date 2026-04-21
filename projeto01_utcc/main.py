from projeto01_utcc.data_loader import load_data
from projeto01_utcc.correlation_analysis import task1_correlation_with_target, task2_correlation_among_features
from projeto01_utcc.regression_model import task3_build_model, task4_evaluate_model

def main():
    print("\n" + "="*60)
    print(" PROJETO: PREVISÃO DE POTÊNCIA ELÉTRICA (UTCC)")
    print("="*60)
    
    # Carrega dados
    print("\n Carregando dataset UCI CCPP...")
    X, y = load_data()
    print(f" Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Tarefa 1
    task1_correlation_with_target(X, y)
    
    # Tarefa 2
    task2_correlation_among_features(X)
    
    # Tarefa 3
    model, X_train, X_test, y_train, y_test, y_pred = task3_build_model(X, y)
    
    # Tarefa 4
    metrics = task4_evaluate_model(y_test, y_pred)
    
    print("\n" + "="*60)
    print(" PROJETO CONCLUÍDO COM SUCESSO!")
    print("="*60)

if __name__ == "__main__":
    main()