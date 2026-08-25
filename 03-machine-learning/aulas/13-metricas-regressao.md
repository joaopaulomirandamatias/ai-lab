# Aula 13 — Métricas de regressão: MAE, MSE, RMSE, R² e erro relativo

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 12 deste módulo  
**Objetivo central:** Aprender a escolher métricas de regressão de acordo com a pergunta e o custo do erro.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Calcular e interpretar MAE, MSE e RMSE.
- Entender R² e suas limitações.
- Reconhecer sensibilidade a outliers.
- Evitar MAPE quando o target pode ser zero.
- Usar múltiplas métricas de forma coerente.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **aprender a escolher métricas de regressão de acordo com a pergunta e o custo do erro.**

## 2. Ideias fundamentais

### 1. MAE

Penaliza erros linearmente e é interpretável na unidade do target. É relativamente mais robusta a erros extremos que MSE.

### 2. MSE/RMSE

Erros quadráticos penalizam fortemente grandes desvios. RMSE volta à unidade original.

### 3. R²

Compara a soma de quadrados residual a um baseline baseado na média. Pode ser negativo fora da amostra.

### 4. Métricas relativas

Percentuais podem ser úteis, mas ficam instáveis quando o denominador se aproxima de zero.

## 3. Equação para guardar

$$
RMSE=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em previsão de demanda, MAE responde ao erro absoluto típico; RMSE destaca falhas grandes que podem causar ruptura de estoque.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print(mae, rmse, r2)
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

- Usar apenas R².
- Comparar RMSE entre targets em escalas diferentes sem contexto.
- Usar MAPE com zeros.
- Escolher métrica depois de ver qual favorece o modelo.

## 8. Exercícios

1. Quando MAE é preferível a RMSE?
2. R² pode ser negativo? Explique.
3. Por que MAPE falha com y=0?
4. Escolha métricas para previsão de tempo de atendimento.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP — model assessment.
- scikit-learn — Regression metrics.
- Hyndman & Koehler (2006) — Another Look at Measures of Forecast Accuracy.
- Murphy — predictive evaluation.

## Próxima aula

**Métricas de classificação: matriz de confusão, precision, recall e F1**
