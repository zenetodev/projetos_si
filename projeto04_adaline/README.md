# Projeto 4 - Classificacao Ionosferica (Rede Adaline)

## Descricao
Classificacao binaria de retornos de radar ionosferico usando Rede Adaline (Adaptive Linear Neuron) com regra Delta.

**Disciplina:** Sistemas Inteligentes (2026.1)

## Dataset
- **Fonte:** UCI Ionosphere (ID=52)
- **Amostras:** 351 retornos de radar
- **Features:** 34 atributos continuos originais (1 removido por ser constante)
- **Target:** good (+1) / bad (-1)

## Pre-processamento
1. Recodificacao do target: 'g' → +1 (good), 'b' → -1 (bad)
2. Remocao de atributo constante (Attribute2)
3. Normalizacao Z-score (media=0, desvio=1)

**Justificativa da normalizacao:**
A regra Delta atualiza pesos por: Δw = η * (d - y) * x
Features em escalas diferentes geram gradientes desproporcionais. Z-score iguala as magnitudes, garantindo estabilidade e convergencia mais rapida.

## Estudo da Taxa de Aprendizagem (η)

| η | Comportamento | Custo Final |
|---|---------------|-------------|
| 1e-04 | Convergencia suave e estavel | 0.1907 |
| 1e-03 | Divergencia (overflow) | ~1.34e+44 |
| 1e-02 | Divergencia (overflow) | inf |
| 5e-02 | Divergencia (overflow) | inf |
| 1e-01 | Divergencia (overflow) | inf |

**Melhor η:** 1e-04

## Resultados da Avaliacao (Teste)

### Metricas Globais
| Metrica | Valor |
|---------|-------|
| Acuracia | 95.77% |
| Precisao (good) | 93.88% |
| Recall (good) | 100.00% |
| F1-score (good) | 96.84% |

### Matriz de Confusao
| Real \ Previsto | good (+1) | bad (-1) |
|-----------------|-----------|----------|
| good (+1)       | 46        | 0        |
| bad (-1)        | 3         | 22       |

### Analise Detalhada
- True Positives (good → good): 46
- True Negatives (bad → bad): 22
- False Positives (bad → good): 3
- False Negatives (good → bad): 0

## Analise de Erros Criticos

**Cenario real:** Sistema de monitoramento ionosferico para telecomunicacoes e GPS.

**Tipos de erro:**
- **Falso Negativo (good → bad):** 0 ocorrencias
  - Sinal valido classificado como falha
  - Consequencia: Perda de dados uteis, alarme falso

- **Falso Positivo (bad → good):** 3 ocorrencias
  - Falha classificado como sinal valido
  - Consequencia: Dado incorreto usado em analises, mascara problemas reais

**Conclusão:** Ambos os erros sao criticos, mas o modelo apresentou 0 falsos negativos, o que e excelente para aplicações de monitoramento onde perder um sinal válido pode significar ignorar uma tempestade geomagnética iminente.

## Como executar

```bash
cd projeto04_adaline
pip install -r requirements.txt
python main.py
```

## Arquivos Gerados

- **convergence_curves.png** - Curvas de convergencia para diferentes η

- **final_cost_curve.png** - Curva do modelo final (η = 1e-04)

- **confusion_matrix.png** - Matriz de confusao do modelo final

## Estrutura do Projeto

```bash
projeto04_adaline/
├── data_loader.py      # Carregamento do dataset
├── preprocessing.py    # Recodificacao, remocao constantes, normalizacao, split
├── adaline.py          # Implementacao da Rede Adaline
├── model_selection.py  # Estudo da taxa de aprendizagem
├── model_evaluation.py # Avaliacao final
├── main.py             # Orquestrador principal
├── requirements.txt
└── README.md
```

## Respostas para o Relatório

1) Por que a normalizacao e obrigatoria?

A regra Delta utiliza o gradiente Δw = η * (d - y) * x. Se as features tem escalas diferentes, a atualização dos pesos sera dominada pelas features de maior magnitude, causando instabilidade e convergencia lenta.

2) Qual o melhor η e por que?

η = 1e-04 foi o único que convergiu estavelmente. Taxas maiores causaram overflow numerico porque os gradientes ficaram muito grandes.

3) O modelo e adequado para uso real?

Sim. Com 95.77% de acuracia e 100% de recall para a classe good, o modelo e excelente para detectar estrutura ionosferica valida. Os 3 falsos positivos podem ser aceitaveis dependendo da tolerancia da aplicacao.

4) Qual erro e mais critico?

Em monitoramento ionosferico, ambos os erros sao problematicos. Este modelo tem 0 falsos negativos, o que e ideal para aplicações onde não se pode perder um evento real.