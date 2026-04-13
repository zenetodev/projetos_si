# correlation_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def task1_correlation_with_target(X, y):
    """calcula a correlação entre variáveis independentes e a variável alvo"""
    df = X.copy()
    df['PE'] = y
    correlations = df.corr()['PE'].drop('PE').sort_values(ascending=False)
    
    print("\n" + "="*60)
    print("TAREFA 1 - Correlação com a variável alvo (PE)")
    print("="*60)
    print(correlations)
    print("\n Variável com maior influência:", correlations.index[0])
    print(" Variável com menor influência:", correlations.index[-1])
    
    return correlations

def task2_correlation_among_features(X):
    """calcula correlação entre variáveis independentes (detecta multicolinearidade)"""
    corr_matrix = X.corr()
    
    print("\n" + "="*60)
    print("TAREFA 2 - Correlação entre variáveis independentes")
    print("="*60)
    print(corr_matrix)
    
    # detecta pares com alta correlação (>0.7 ou <-0.7)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                  corr_matrix.iloc[i, j]))
    
    if high_corr:
        print("\n ALERTA - Possível multicolinearidade detectada:")
        for var1, var2, corr_val in high_corr:
            print(f"   • {var1} x {var2}: {corr_val:.3f}")
    else:
        print("\n Nenhuma correlação forte (>0.7) entre variáveis independentes.")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriz de Correlação entre Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("\n Heatmap salvo como 'correlation_heatmap.png'")
    
    return corr_matrix

if __name__ == "__main__":
    from data_loader import load_data
    X, y = load_data()
    task1_correlation_with_target(X, y)
    task2_correlation_among_features(X)