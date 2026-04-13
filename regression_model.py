# regression_model.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

def task3_build_model(X, y, test_size=0.2, random_state=42):
    """Constrói modelo de regressão linear múltipla (80% treino, 20% teste)"""
    
    # Garantir que X e y estão no formato correto (numpy arrays 1D/2D)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values.ravel()  # ravel() garante 1D
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Treinamento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("TAREFA 3 - Modelo de Regressão Linear Múltipla")
    print("="*60)
    print(f" Dados de treino: {X_train.shape[0]} amostras")
    print(f" Dados de teste: {X_test.shape[0]} amostras")
    
    # CORREÇÃO DEFINITIVA: Garantir que coef_ e intercept_ são escalares/lista
    coefs = model.coef_
    intercept = model.intercept_
    
    # Se for array 2D, achatar; se for 1D, manter; se for escalar, virar lista
    if hasattr(coefs, 'ndim'):
        if coefs.ndim > 1:
            coefs = coefs.flatten()
        elif coefs.ndim == 0:  # escalar
            coefs = [coefs]
    elif not hasattr(coefs, '__iter__'):
        coefs = [coefs]
    
    # Garantir que intercept é escalar
    if hasattr(intercept, 'ndim') and intercept.ndim > 0:
        intercept = intercept[0] if len(intercept) > 0 else 0.0
    elif hasattr(intercept, '__iter__') and not isinstance(intercept, (int, float)):
        intercept = intercept[0] if len(intercept) > 0 else 0.0
    
    print(f"\n Coeficientes do modelo:")
    for i, feature in enumerate(feature_names):
        coef_val = coefs[i] if i < len(coefs) else 0.0
        print(f"   • {feature}: {float(coef_val):.6f}")
    print(f"   • Intercept: {float(intercept):.6f}")
    
    # Equação do modelo
    print(f"\n Equação do modelo:")
    eq = f"PE = {float(intercept):.4f}"
    for i, feature in enumerate(feature_names):
        coef_val = coefs[i] if i < len(coefs) else 0.0
        signal = "+" if coef_val >= 0 else "-"
        eq += f" {signal} {abs(float(coef_val)):.4f} * {feature}"
    print(f"   {eq}")
    
    # Converter X_test e y_test de volta para formato amigável (opcional)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def task4_evaluate_model(y_test, y_pred):
    """Avalia o modelo com métricas R², RMSE e MAE"""
    
    # Garantir que são arrays 1D e valores numéricos puros
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*60)
    print("TAREFA 4 - Avaliação do Modelo")
    print("="*60)
    print(f" R² (Coeficiente de Determinação): {float(r2):.6f}")
    print(f" RMSE (Raiz do Erro Quadrático Médio): {float(rmse):.6f} MW")
    print(f" MAE (Erro Absoluto Médio): {float(mae):.6f} MW")
    
    # Interpretação
    print("\n" + "="*60)
    print(" INTERPRETAÇÃO DOS RESULTADOS")
    print("="*60)
    
    if r2 > 0.9:
        print(f" R² = {r2:.4f} → Modelo EXCELENTE, explica >90% da variabilidade")
    elif r2 > 0.8:
        print(f" R² = {r2:.4f} → Modelo BOM, explica >80% da variabilidade")
    elif r2 > 0.7:
        print(f" R² = {r2:.4f} → Modelo RAZOÁVEL, explica >70% da variabilidade")
    else:
        print(f" R² = {r2:.4f} → Modelo FRACO, explica <70% da variabilidade")
    
    print(f"\n RMSE = {rmse:.4f} MW → Em média, o erro da previsão é de {rmse:.4f} MW")
    print(f" MAE = {mae:.4f} MW → Em média, o erro absoluto é de {mae:.4f} MW")
    
    # Conclusão sobre adequação
    print("\n" + "="*60)
    print(" CONCLUSÃO SOBRE ADEQUAÇÃO DO MODELO")
    print("="*60)
    
    if r2 > 0.85 and rmse < 5:
        print(" O modelo é MUITO ADEQUADO para prever a produção de energia.")
        print("   A usina pode usar essas previsões para otimização do despacho energético.")
    elif r2 > 0.7:
        print(" O modelo é ADEQUADO, mas com limitações.")
        print("   Pode ser usado como referência, mas recomenda-se validação adicional.")
    else:
        print(" O modelo NÃO É ADEQUADO para uso prático.")
        print("   Considere: features adicionais, modelos não-lineares ou mais dados.")
    
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)}

if __name__ == "__main__":
    from data_loader import load_data
    X, y = load_data()
    model, X_train, X_test, y_train, y_test, y_pred = task3_build_model(X, y)
    metrics = task4_evaluate_model(y_test, y_pred)