import torch
import numpy as np
from data_loader import load_nslkdd_data, get_dataset_info
from preprocessing import run_preprocessing
from dataset import create_dataloaders, create_test_loader
from model import create_model, get_model_topologies
from train import train_model, plot_loss_curves
from evaluate import evaluate_model, analyze_critical_errors, visualize_confusion_matrix, test_different_thresholds


def main():
    print("\n" + "="*60)
    print("PROJETO: DETECCAO DE INTRUSAO EM REDES (MLP)")
    print("="*60)
    
    # Verificar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device}")
    
    # 1. Carregar dados
    print("\nCarregando dataset NSL-KDD...")
    train_df, test_df = load_nslkdd_data()
    get_dataset_info(train_df, test_df)
    
    # 2. Pre-processamento
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(train_df, test_df)
    
    # 3. Criar DataLoaders
    print("\n" + "="*60)
    print("PREPARACAO DOS DATALOADERS")
    print("="*60)
    train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=256, val_split=0.2)
    test_loader = create_test_loader(X_test, y_test, batch_size=256)
    
    # 4. Experimentar diferentes topologias
    print("\n" + "="*60)
    print("EXPERIMENTACAO COM TOPOLOGIAS")
    print("="*60)
    
    input_dim = X_train.shape[1]
    topologies = ['small', 'medium', 'large', 'shallow', 'deep']
    
    best_val_loss = float('inf')
    best_topology = None
    best_model = None
    best_history = None
    
    # Para teste completo, rodar com todas as topologias
    use_fast_mode = True  # Mude para False para testar todas
    
    if use_fast_mode:
        print("\nModo rapido: testando apenas topologia 'medium'")
        topologies_to_test = ['medium']
    else:
        print("\nModo completo: testando todas as topologias")
        topologies_to_test = topologies
    
    for topology in topologies_to_test:
        print(f"\n--- Testando topologia: {topology} ---")
        
        # Recriar DataLoaders para evitar vazamento entre topologias
        train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=256, val_split=0.2)
        
        # Criar modelo
        model = create_model(input_dim, topology)
        
        # Treinar
        history, trained_model = train_model(model, train_loader, val_loader, epochs=50, device=device)
        
        # Verificar melhor loss de validacao
        final_val_loss = history['val_loss'][-1]
        min_val_loss = min(history['val_loss'])
        
        print(f"  Melhor val loss: {min_val_loss:.6f}")
        print(f"  Final val loss: {final_val_loss:.6f}")
        
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_topology = topology
            best_model = trained_model
            best_history = history
    
    print("\n" + "="*60)
    print("MELHOR TOPOLOGIA ENCONTRADA")
    print("="*60)
    print(f"Topologia: {best_topology}")
    print(f"Melhor loss de validacao: {best_val_loss:.6f}")
    
    # 5. Plotar curvas de loss da melhor topologia
    plot_loss_curves(best_history)
    
    # 6. Avaliacao final no teste
    metrics = evaluate_model(best_model, test_loader, device=device)
    
    # 7. Analise de erros criticos
    analyze_critical_errors(metrics)
    
    # 8. Visualizar matriz de confusao
    visualize_confusion_matrix(metrics['confusion_matrix'])
    
    # 9. Testar diferentes limiares
    test_different_thresholds(best_model, test_loader, device=device)
    
    # 10. Resumo final
    print("\n" + "="*60)
    print("="*60)
    print(f"\nRESUMO FINAL:")
    print(f"  Melhor topologia: {best_topology}")
    print(f"  Acuracia no teste: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precisao: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Falsos Negativos (ataque → normal): {metrics['false_negatives']}")
    print(f"  Falsos Positivos (normal → ataque): {metrics['false_positives']}")
    print("\nArquivos gerados:")
    print("  - loss_curves.png (curvas de loss de treino e validacao)")
    print("  - confusion_matrix.png (matriz de confusao)")


if __name__ == "__main__":
    main()