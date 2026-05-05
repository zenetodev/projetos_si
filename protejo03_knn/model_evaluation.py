import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, ConfusionMatrixDisplay)
from typing import Dict, List


def evaluate_model(model, X_test, y_test) -> Dict:
    print("\n" + "="*60)
    print("TAREFA 4 - AVALIACAO DO MODELO")
    print("="*60)
    
    # Predicoes
    y_pred = model.predict(X_test)
    
    # Acuracia
    accuracy = accuracy_score(y_test, y_pred)
    
    # Metricas por classe
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Macros (media das classes)
    precision_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[0]
    recall_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[1]
    f1_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
    
    # Matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAcuracia Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nMetricas por Classe:")
    print("-" * 50)
    print(f"{'Classe':<10} {'Precisao':<12} {'Recall':<12} {'F1-score':<12} {'Suporte':<10}")
    print("-" * 50)
    
    class_names = ['Ruim (0)', 'Medio (1)', 'Bom (2)']
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:.4f}        {recall[i]:.4f}        {f1[i]:.4f}        {support[i]:<10}")
    
    print("-" * 50)
    print(f"{'Media':<10} {precision_macro:.4f}        {recall_macro:.4f}        {f1_macro:.4f}        {sum(support):<10}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def analyze_confusion_matrix(cm: np.ndarray, class_names: List[str] = None) -> None:
    if class_names is None:
        class_names = ['Ruim', 'Medio', 'Bom']
    
    print("\nMatriz de Confusao:")
    print("-" * 30)
    print(f"{'':<12}", end="")
    for pred in class_names:
        print(f"{pred:>10}", end="")
    print()
    
    for i, true in enumerate(class_names):
        print(f"{true:<12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    # Identificar erros mais frequentes
    print("\nAnalise de Erros:")
    print("-" * 30)
    
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append({
                    "true": class_names[i],
                    "pred": class_names[j],
                    "count": cm[i, j]
                })
    
    errors.sort(key=lambda x: x['count'], reverse=True)
    
    print("Pares de classe com mais erros de classificacao:")
    for err in errors[:3]:
        print(f"  {err['true']} -> {err['pred']}: {err['count']} amostras")
    
    if len(errors) >= 2:
        print(f"\nResposta: O modelo erra com mais frequencia entre {errors[0]['true']} e {errors[0]['pred']}.")


def visualize_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = "confusion_matrix.png"):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Configurar ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Classe Prevista',
               ylabel='Classe Verdadeira')
        
        # Rotacionar ticks do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar valores nas celulas
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_title('Matriz de Confusao - Classificacao de Vinhos (k-NN)', fontsize=14)
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close() 
        print(f"\nMatriz de confusao salva como '{save_path}'")
        
    except Exception as e:
        print(f"\nErro ao gerar matriz de confusao: {e}")
        print("Tentando metodo alternativo...")
        
        try:
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Classe Prevista')
            plt.ylabel('Classe Verdadeira')
            plt.title('Matriz de Confusao - Classificacao de Vinhos (k-NN)')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"\nMatriz de confusao salva como '{save_path}' (metodo seaborn)")
        except ImportError:
            print("Seaborn nao instalado. Pule geracao da figura.")