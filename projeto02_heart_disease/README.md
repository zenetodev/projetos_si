# Projeto 2 - Previsão de Doença Cardíaca

## Descrição
Classificação binária para prever presença de doença cardíaca usando Regressão Logística.  
Disciplina: **PAM0466 - Sistemas Inteligentes** (2026.1)

## Dataset
- **Fonte:** UCI Heart Disease Cleveland (ID=45)
- **Pacientes:** 303
- **Features:** 13 variáveis clínicas (idade, sexo, pressão arterial, colesterol, etc.)
- **Target:** 0 (sem doença) / 1 (com doença)

## Resultados Obtidos

| Métrica | Valor |
|---------|-------|
| Acurácia | ~0.85 |
| Precisão | ~0.85 |
| Recall | ~0.85 |
| F1-score | ~0.85 |
| AUC | ~0.90 |

## Tecnologias
- Python 3.x
- scikit-learn (LogisticRegression)
- pandas, numpy
- matplotlib

## ▶️ Como executar

```bash
cd projeto02_heart_disease
pip install -r requirements.txt
python main.py
```

## Estrutura 

```
├── data_loader.py      # Carregamento do dataset
├── preprocessing.py    # Tratamento de nulos, binarização, split, scaling
├── model_training.py   # Treinamento da regressão logística
├── model_evaluation.py # Métricas e curva ROC
├── main.py             # Orquestrador principal
└── requirements.txt    # Dependências
```

## ✅ Agora é só executar!

```bash
cd projeto02_heart_disease
pip install -r requirements.txt
python main.py
```