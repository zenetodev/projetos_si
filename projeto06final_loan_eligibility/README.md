# Projeto 6 - Classificacao de Elegibilidade para Emprestimo (k-NN)

## Descricao
Classificacao binaria de solicitacoes de emprestimo (aprovado/rejeitado) usando k-Nearest Neighbors.

**Disciplina:** PAM0466 - Sistemas Inteligentes (2026.1)
**Projeto Final - Apresentacao ao Professor**

## Dataset
- **Fonte:** Kaggle - Eligibility Prediction for Loan
- **Autor:** Devzohaib
- **Amostras:** 614 solicitacoes de emprestimo
- **Features:** 11 variaveis (financeiras + pessoais)
- **Target:** Loan_Status (Y = aprovado, N = rejeitado)

## Distribuicao das Classes
| Classe | Quantidade | Percentual |
|--------|------------|------------|
| Aprovado (Y) | 422 | 68.73% |
| Rejeitado (N) | 192 | 31.27% |

## Pre-processamento
1. Remocao da coluna Loan_ID (nao util para predicao)
2. Limpeza da coluna Dependents (conversao de '3+' para 3)
3. Tratamento de valores ausentes:
   - Categoricas: moda (Gender, Married, Self_Employed)
   - Numericas: mediana (Dependents, LoanAmount, Loan_Amount_Term, Credit_History)
4. Label Encoding para variaveis categoricas (5 variaveis)
5. Normalizacao Min-Max (6 variaveis numericas)
6. Divisao treino/teste: 80/20 com estratificacao

## Resultados

### Selecao do Melhor k (Cross-validation 5-folds)
| k | Acuracia Media | Desvio |
|---|----------------|--------|
| 1 | 68.04% | ± 3.90% |
| 3 | 75.97% | ± 2.58% |
| **5** | **78.41%** | **± 1.20%** |
| 7 | 77.39% | ± 1.01% |
| 11 | 76.38% | ± 1.70% |
| 15 | 76.17% | ± 1.47% |

**Melhor k = 5**

### Avaliacao no Teste (k=5)
| Metrica | Valor |
|---------|-------|
| Acuracia | 82.93% |
| Precisao | 82.00% |
| Recall | 96.47% |
| F1-score | 88.65% |

### Matriz de Confusao
| Real \ Previsto | Aprovado | Rejeitado |
|-----------------|----------|-----------|
| Aprovado (85) | 20 | 18 |
| Rejeitado (38) | 3 | 82 |

### Analise dos Erros
| Tipo | Quantidade | Impacto |
|------|------------|---------|
| Falsos Positivos (rejeitado → aprovado) | 3 | Risco financeiro (baixo) |
| Falsos Negativos (aprovado → rejeitado) | 18 | Perda de oportunidade (alto) |

**Conclusao:** O modelo apresenta maior dificuldade em rejeitar clientes que seriam aprovados (18 falsos negativos vs 3 falsos positivos).

## Adequacao do k-NN
**Pontos Positivos:**
- Implementacao simples e intuitiva
- Nao requer treinamento pesado (lazy learning)
- Funciona bem com dados de tamanho moderado
- Facil interpretacao

**Pontos Negativos:**
- Requer normalizacao obrigatoria
- Sensivel a features irrelevantes
- Pior desempenho com dados desbalanceados

**Recomendacao:** O modelo e ADEQUADO como ferramenta de apoio a decisao, devendo ser usado em conjunto com analise humana.

## Como executar

```bash
cd projeto06_loan_eligibility
pip install -r requirements.txt
python main.py
```

## Arquivos gerados

- k_selection_plot.png - Grafico de acuracia vs k

- confusion_matrix.png - Matriz de confusão

## Estrutura do Projeto

```bash
projeto06_loan_eligibility/
├── data_loader.py      # Carregamento do CSV via kagglehub
├── preprocessing.py    # Missing values, encoding, normalizacao, split
├── model_selection.py  # Selecao do melhor k
├── model_evaluation.py # Avaliacao final
├── main.py             # Orquestrador principal
├── requirements.txt
└── README.md
```

## Grupo 
```bash
José Ferreira Sousa Neto
Túlio Gomes de Araújo Feitosa
Thiago Geovane da Costa Nunes
```