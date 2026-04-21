import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix
)
import pandas as pd


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    print("\n" + "="*60)
    print("TAREFA 3 - AVALIAÇÃO DO MODELO")
    print("="*60)
    
    # predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade para classe 1
    
    # métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n📊 3a) Acurácia:")
    print(f"   • {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📊 3b) Precisão:")
    print(f"   • {precision:.4f} ({precision*100:.2f}%)")
    
    print("\n📊 3c) Recall:")
    print(f"   • {recall:.4f} ({recall*100:.2f}%)")
    
    print("\n📊 3d) F1-score:")
    print(f"   • {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n📊 3e) Curva ROC e AUC:")
    print(f"   • AUC (Área sob a curva): {auc:.4f}")
    
    # matriz de confusão formatada
    print("\n📋 Matriz de Confusão:")
    print("                 Previsto")
    print("               Negativo  Positivo")
    print(f"   Real Negativo    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"        Positivo    {cm[1,0]:3d}       {cm[1,1]:3d}")
    
    # interpretação
    print("\n" + "="*60)
    print("🔍 INTERPRETAÇÃO DOS RESULTADOS")
    print("="*60)
    
    # interpretação do AUC
    if auc >= 0.9:
        print(f"✅ AUC = {auc:.4f} → Modelo EXCELENTE (discriminação muito boa)")
    elif auc >= 0.8:
        print(f"✅ AUC = {auc:.4f} → Modelo BOM (discriminação boa)")
    elif auc >= 0.7:
        print(f"⚠️ AUC = {auc:.4f} → Modelo RAZOÁVEL")
    else:
        print(f"❌ AUC = {auc:.4f} → Modelo FRACO")
    
    # interpretação do Recall (sensibilidade)
    print(f"\n📌 Recall (Sensibilidade) = {recall:.4f}")
    print(f"   → O modelo identificou {recall*100:.1f}% dos pacientes que realmente tinham doença")
    
    # interpretação da Precisão
    print(f"\n📌 Precisão = {precision:.4f}")
    print(f"   → Quando o modelo prevê doença, acerta em {precision*100:.1f}% dos casos")
    
    # interpretação do F1-score
    print(f"\n📌 F1-score = {f1:.4f}")
    print(f"   → Média harmônica entre precisão e recall")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_pred_proba": y_pred_proba
    }


def plot_roc_curve(y_test: pd.Series, y_pred_proba: np.ndarray, save_path: str = "roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificidade)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilidade)', fontsize=12)
    plt.title('Curva ROC - Regressão Logística para Doença Cardíaca', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n📊 Curva ROC salva como '{save_path}'")


if __name__ == "__main__":
    from data_loader import load_heart_disease_data
    from preprocessing import run_preprocessing
    from model_training import train_logistic_regression
    
    X, y = load_heart_disease_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(X, y)
    model, n_iter = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    plot_roc_curve(y_test, metrics["y_pred_proba"])