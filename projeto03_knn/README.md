# Projeto 3 - Classificacao de Vinhos (k-NN)

## Descricao
Classificacao de vinhos tintos em 3 categorias de qualidade (ruim, medio, bom) usando o algoritmo k-Nearest Neighbors (k-NN).

**Disciplina:** Sistemas Inteligentes

## Dataset
- **Fonte:** UCI Wine Quality Red
- **Amostras:** 1.599 vinhos tintos
- **Features:** 11 atributos fisico-quimicos
- **Target:** Quality (0-10) recodificado para 3 classes

## Pre-processamento
1. Recodificacao da variavel quality:
   - Notas ≤ 5: Ruim (0) - 744 amostras
   - Notas 6-7: Medio (1) - 837 amostras
   - Notas ≥ 8: Bom (2) - 18 amostras

2. Normalizacao Z-score (obrigatoria para k-NN)
   - Justificativa: k-NN utiliza distancia euclidiana
   - Features em escalas diferentes distorceriam o calculo

3. Divisao treino/teste: 80/20 com estratificacao
   - Treino: 1.279 amostras
   - Teste: 320 amostras

## Resultados

### Selecao do Melhor k (Validacao Cruzada 5-folds)
| k | Acuracia Media | Desvio Padrao |
|---|----------------|---------------|
| 1 | 0.7295 | ± 0.0110 |
| 3 | 0.7044 | ± 0.0141 |
| 5 | 0.7084 | ± 0.0093 |
| 7 | 0.7029 | ± 0.0208 |
| 11 | 0.7068 | ± 0.0157 |

**Melhor k:** 1

### Avaliacao no Teste
| Metrica | Valor |
|---------|-------|
| Acuracia | 71.88% |
| F1-score (macro) | 59.25% |

### Metricas por Classe
| Classe | Precisao | Recall | F1-score | Suporte |
|--------|----------|--------|----------|---------|
| Ruim (0) | 0.6987 | 0.7315 | 0.7148 | 149 |
| Medio (1) | 0.7407 | 0.7186 | 0.7295 | 167 |
| Bom (2) | 0.5000 | 0.2500 | 0.3333 | 4 |

### Matriz de Confusao
|            | Ruim | Medio | Bom |
|------------|------|-------|-----|
| Ruim (0)   | 109  | 40    | 0   |
| Medio (1)  | 46   | 120   | 1   |
| Bom (2)    | 1    | 2     | 1   |

## Analise dos Erros
O modelo erra com mais frequencia entre as classes:
- **Medio -> Ruim:** 46 amostras classificadas erroneamente
- **Ruim -> Medio:** 40 amostras classificadas erroneamente

Isso indica que a fronteira entre vinhos de qualidade "Ruim" e "Media" nao e bem definida pelos atributos quimicos disponiveis.

## Efeito do valor de k
- **k baixo (k=1):** Overfitting, sensivel a ruido, alta variancia. Foi o melhor valor neste caso.
- **k alto (k=11):** Underfitting, fronteira suavizada, perde detalhes importantes.

## Como executar

```bash
cd projeto03_wine_quality
pip install -r requirements.txt
python main.py
```

## Arquivos Gerados

- **k_selection_plot.png** - Grafico de acuracia vs diferentes valores de k

- **confusion_matrix.png** - Matriz de confusao do modelo final


## Estrutura do Projeto

```bash
projeto03_knn/
├── data_loader.py      # Carregamento do dataset
├── preprocessing.py    # Recodificacao, normalizacao, split
├── model_selection.py  # Selecao do melhor k com cross-validation
├── model_evaluation.py # Avaliacao final no teste
├── main.py             # Orquestrador principal
├── requirements.txt    # Dependencias
└── README.md           # Documentacao
``` 

## Conclusão do projeto

O k-NN apresentou acuracia de 71.88% no conjunto de teste, com melhor desempenho para k=1. O modelo e adequado, mas tem limitacoes, especialmente para distinguir vinhos de qualidade "Ruim" e "Media". A classe "Bom" teve desempenho prejudicado pelo baixo numero de amostras (desequilibrio de classes).

## Projeto concluído em direção a matéria Sistemas Inteligentes