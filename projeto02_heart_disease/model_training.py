from sklearn.linear_model import LogisticRegression
import pandas as pd


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    print("\n" + "="*60)
    print("TAREFA 2 - TREINAMENTO DO MODELO")
    print("="*60)
    
    # parâmetros conforme especificação
    model = LogisticRegression(
        solver='liblinear',  
        random_state=42,     
        max_iter=1000        
    )
    
    print("\n Parâmetros do modelo:")
    print(f"   • solver: liblinear")
    print(f"   • random_state: 42")
    print(f"   • max_iter: 1000")
    
    # treinar modelo
    model.fit(X_train, y_train)
    
    # numero de iterações necessárias
    n_iterations = model.n_iter_[0]
    
    print(f"\n Resultado do treinamento:")
    print(f"   • Iterações necessárias para convergência: {n_iterations}")
    print(f"   • Coeficientes (pesos) das features: {model.coef_[0].shape[0]} variáveis")
    print(f"   • Intercept: {model.intercept_[0]:.4f}")
    
    # exibir coeficientes
    print(f"\n Coeficientes do modelo:")
    for feature, coef in zip(X_train.columns, model.coef_[0]):
        print(f"   • {feature}: {coef:.4f}")
    
    return model, n_iterations


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    from preprocessing import run_preprocessing
    
    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(X, y)
    model, n_iter = train_logistic_regression(X_train, y_train)