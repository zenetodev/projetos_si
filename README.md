
Projeto da disciplina **Sistemas Inteligentes** (2026.1)  
Previsão da potência elétrica líquida horária (MW) de uma Usina Termoelétrica de Ciclo Combinado (UTCC) utilizando regressão linear múltipla.

## Dataset
- **Fonte:** UCI Machine Learning Repository - Combined Cycle Power Plant (ID=294)
- **Amostras:** 9.568 coletadas
- **Features:** AT (temperatura), V (vácuo), AP (pressão), RH (umidade)
- **Target:** PE (potência elétrica em MW)

## Resultados Obtidos

| Métrica | Valor |
|---------|-------|
| R² | 0.9301 |
| RMSE | 4.50 MW |
| MAE | 3.60 MW |

### Equação do Modelo

PE = 454.57 - 1.986·AT - 0.232·V + 0.062·AP - 0.158·RH

### Principais Correlações
- AT x PE: **-0.948** (maior influência)
- V x PE: **-0.870**
- AP x PE: 0.518
- RH x PE: 0.390

##  Tecnologias
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- ucimlrepo

## ▶ Como executar

```bash
# Clone o repositório
git clone https://github.com/zenetodev/projeto_utcc.git

# Instale as dependências
pip install -r requirements.txt

# Execute o projeto
python main.py
```

## Estrutura do Projeto
```
├── data_loader.py          # Carregamento do dataset UCI
├── correlation_analysis.py # Análise de correlações (Tarefas 1 e 2)
├── regression_model.py     # Regressão linear e métricas (Tarefas 3 e 4)
├── main.py                 # Orquestrador principal
├── correlation_heatmap.png # Matriz de correlação gerada
├── requirements.txt        # Dependências
└── README.md               # Este arquivo
```

---

## Tags
machine-learning, linear-regression, power-plant, regression-analysis, scikit-learn, uci-dataset, python, data-science, predictive-modeling