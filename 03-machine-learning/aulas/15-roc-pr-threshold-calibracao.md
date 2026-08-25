# Aula 15 — ROC, Precision-Recall, thresholds e calibração

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 14 deste módulo  
**Objetivo central:** Separar ranking, threshold e qualidade probabilística.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Construir curvas ROC e PR.
- Entender AUC como métrica de ranking.
- Escolher threshold por custo.
- Avaliar calibração.
- Entender Brier score e calibration curves.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **separar ranking, threshold e qualidade probabilística.**

## 2. Ideias fundamentais

### 1. ROC

Varia threshold e plota TPR contra FPR. ROC-AUC mede capacidade de ranking sob uma interpretação probabilística por pares.

### 2. Precision-Recall

Mais informativa quando classe positiva é rara e nos interessa a qualidade das detecções positivas.

### 3. Threshold

Threshold é uma política de decisão, não parte intrínseca do ranker. Pode ser adaptado ao contexto.

### 4. Calibração

Se previsões 0.8 ocorrem, aproximadamente 80% desses casos deveriam ser positivos em um modelo bem calibrado.

## 3. Equação para guardar

$$
Brier=\frac{1}{n}\sum_i(p_i-y_i)^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Dois modelos podem ter ROC-AUC semelhante, mas um produzir probabilidades muito mal calibradas, prejudicando decisões baseadas em risco.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

roc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)

print(roc, ap)
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

- Chamar AUC de accuracy.
- Assumir que bom ranking implica boa calibração.
- Escolher threshold no conjunto de teste.
- Preferir ROC em classe extremamente rara sem olhar PR.

## 8. Exercícios

1. Explique ranking vs decisão.
2. Quando PR-AUC é mais informativa?
3. O que significa uma probabilidade calibrada?
4. Como escolher threshold por custo?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Fawcett (2006) — An Introduction to ROC Analysis.
- Saito & Rehmsmeier (2015) — PR vs ROC for imbalanced data.
- Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities.
- scikit-learn — Probability calibration.

## Próxima aula

**Classes desbalanceadas: amostragem, pesos e avaliação correta**
