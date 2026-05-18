import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score)
from typing import Dict, Tuple


def evaluate_model(model, test_loader, device='cpu', threshold=0.5):
    print("\n" + "="*60)
    print("TAREFA 4 - AVALIACAO FINAL")
    print("="*60)
    
    model.eval()
    model = model.to(device)
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    # Concatenar resultados
    y_true = np.concatenate(all_targets).flatten()
    y_scores = np.concatenate(all_outputs).flatten()
    
    # Aplicar sigmoide e limiar
    y_proba = 1 / (1 + np.exp(-y_scores))  # Sigmoid
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calcular metricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nLimiar de decisao: {threshold}")
    print(f"\nAcuracia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precisao: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-score: {f1:.4f} ({f1*100:.2f}%)")
    
    print("\nMatriz de Confusao:")
    print("-" * 40)
    print(f"{'':<15} {'Previsto Normal':<18} {'Previsto Ataque':<18}")
    print(f"{'Real Normal':<15} {cm[0,0]:<18} {cm[0,1]:<18}")
    print(f"{'Real Ataque':<15} {cm[1,0]:<18} {cm[1,1]:<18}")
    
    # Analise de erros
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"\nAnalise detalhada:")
    print(f"  True Negatives (normal → normal): {tn}")
    print(f"  True Positives (ataque → ataque): {tp}")
    print(f"  False Positives (normal → ataque): {fp}")
    print(f"  False Negatives (ataque → normal): {fn}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "false_positives": fp,
        "false_negatives": fn
    }


def analyze_critical_errors(metrics: Dict) -> None:
    print("\n" + "="*60)
    print("ANALISE DE ERROS CRITICOS EM IDS")
    print("="*60)
    
    fp = metrics['false_positives']  # normal classificado como ataque
    fn = metrics['false_negatives']  # ataque classificado como normal
    
    print("\nCenario real: Sistema de Deteccao de Intrusao (IDS)")
    print("  - Normal: conexao legitima")
    print("  - Ataque: tentativa de invasao (DoS, Probe, R2L, U2R)")
    
    print(f"\nResultados do modelo:")
    print(f"  Falsos Positivos (normal → ataque): {fp}")
    print(f"  Falsos Negativos (ataque → normal): {fn}")
    
    print("\nAnalise de criticidade:")
    print("  - Falso Negativo: Ataque classificado como normal")
    print("    -> Consequencia: A invasao NAO E DETECTADA")
    print("    -> Atacante pode prosseguir sem ser notado")
    print("    -> Dano potencial: EXTREMAMENTE ALTO")
    
    print("  - Falso Positivo: Normal classificado como ataque")
    print("    -> Consequencia: Alarme FALSO, interrupcao desnecessaria")
    print("    -> Dano potencial: Moderado (sobrecarga da equipe de seguranca)")
    
    print("\nConclusao - Qual erro mais critico?")
    print("  Em um IDS, FALSOS NEGATIVOS sao MUITO MAIS CRITICOS")
    print("  Um ataque nao detectado pode causar danos reais ao sistema")
    print("  Falsos positivos geram inconveniencia, mas nao comprometem a seguranca")
    
    print("\nImpacto na escolha do limiar:")
    print("  - Limiar mais baixo (ex: 0.3): Menos falsos negativos, mais falsos positivos")
    print("  - Limiar mais alto (ex: 0.7): Menos falsos positivos, mais falsos negativos")
    print("  - Recomendacao: Usar limiar BAIXO para priorizar deteccao de ataques")


def visualize_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    
    classes = ['Normal (0)', 'Ataque (1)']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    
    plt.xlabel('Previsto', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title('Matriz de Confusao - MLP para Deteccao de Intrusao', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nMatriz de confusao salva como '{save_path}'")


def test_different_thresholds(model, test_loader, device='cpu'):
    print("\n" + "="*60)
    print("ANALISE DE DIFERENTES LIMIARES")
    print("="*60)
    
    model.eval()
    model = model.to(device)
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_true = np.concatenate(all_targets).flatten()
    y_scores = np.concatenate(all_outputs).flatten()
    y_proba = 1 / (1 + np.exp(-y_scores))
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n{'Limiar':<10} {'Acuracia':<12} {'Precisao':<12} {'Recall':<12} {'F1':<12} {'FP':<8} {'FN':<8}")
    print("-" * 80)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fp = confusion_matrix(y_true, y_pred)[0, 1]
        fn = confusion_matrix(y_true, y_pred)[1, 0]
        
        print(f"{thresh:<10.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fp:<8} {fn:<8}")
    
    print("\nRecomendacao: Para IDS, priorizar Recall (detectar ataques)")
    print("  Limiar mais baixo (0.3-0.4) geralmente e preferivel")


if __name__ == "__main__":
    import numpy as np
    from data_loader import load_nslkdd_data
    from preprocessing import run_preprocessing
    from dataset import create_dataloaders, create_test_loader
    from model import create_model
    from train import train_model
    
    # Carregar dados
    train_df, test_df = load_nslkdd_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(train_df, test_df)
    
    # Criar DataLoaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=256, val_split=0.2)
    test_loader = create_test_loader(X_test, y_test, batch_size=256)
    
    # Criar e treinar modelo
    input_dim = X_train.shape[1]
    model = create_model(input_dim, 'medium')
    history, best_model = train_model(model, train_loader, val_loader, epochs=50)
    
    # Avaliar
    metrics = evaluate_model(best_model, test_loader)
    analyze_critical_errors(metrics)
    visualize_confusion_matrix(metrics['confusion_matrix'])
    test_different_thresholds(best_model, test_loader)