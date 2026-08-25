# Aula 14 — Métricas de classificação: matriz de confusão, precision, recall e F1

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 13 deste módulo  
**Objetivo central:** Entender métricas condicionais e escolher a que representa o custo dos erros.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar TP, FP, TN e FN.
- Calcular precision, recall, specificity e F1.
- Entender por que accuracy falha em classes raras.
- Distinguir macro, micro e weighted averages.
- Relacionar métrica ao custo operacional.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender métricas condicionais e escolher a que representa o custo dos erros.**

## 2. Ideias fundamentais

### 1. Matriz de confusão

Toda métrica de classificação binária pode ser entendida a partir de TP, FP, TN e FN.

### 2. Precision

Entre previsões positivas, quantas eram realmente positivas? Importa quando falso positivo é caro.

### 3. Recall

Entre positivos reais, quantos foram detectados? Importa quando falso negativo é caro.

### 4. F1

Média harmônica de precision e recall. É útil quando queremos equilíbrio, mas não incorpora TN.

## 3. Equação para guardar

$$
F1=2\frac{precision\cdot recall}{precision+recall}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em detecção de fraude, recall alto captura mais fraudes, mas pode aumentar bloqueios indevidos. A decisão depende do custo e da capacidade de revisão.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

## 6. Conexão com o AI Systems Laboratory

Para o projeto longitudinal, aplique este conceito a um dataset real e salve:
- configuração do experimento;
- baseline;
- métricas de validação;
- análise de erros;
- limitações;
- evidência de que o teste não contaminou o treinamento.

Ao longo do M4, esses artefatos serão acumulados até formar o **Gate II**.

## 7. Armadilhas comuns

- Usar accuracy em dataset 99/1.
- Escolher F1 sem justificar custo.
- Confundir precision com recall.
- Ignorar averaging em multiclasse.

## 8. Exercícios

1. Calcule precision e recall para TP=80, FP=20, FN=40.
2. Qual métrica priorizaria em triagem de doença?
3. Quando accuracy pode ser adequada?
4. Explique macro vs micro F1.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Powers (2011) — Evaluation: From Precision, Recall and F-Measure to ROC.
- scikit-learn — Classification metrics.
- Saito & Rehmsmeier (2015) — Precision-Recall plot.
- ISLP — Classification evaluation.

## Próxima aula

**ROC, Precision-Recall, thresholds e calibração**
