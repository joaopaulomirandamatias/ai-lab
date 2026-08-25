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

## Aprofundamento — cada métrica responde a uma pergunta diferente

Defina resíduos $e_i=y_i-\hat y_i$. MAE estima a média de $|e_i|$ e sua minimização está ligada à mediana condicional. MSE estima a média de $e_i^2$, amplificando grandes desvios, e sua minimização está ligada à média condicional. RMSE apenas retorna a raiz para a unidade do target.

O coeficiente de determinação fora da amostra é

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y_{train})^2}.
$$

Use o baseline calculado no treino. $R^2<0$ significa que o modelo perde para essa referência no conjunto avaliado. MAPE divide pelo valor real e explode perto de zero; também trata de forma assimétrica erros de sobre e subprevisão. Escolha a métrica antes do resultado e complemente-a com distribuição dos resíduos por faixa, tempo e grupo relevante.

## 3. Equação para guardar

$$
RMSE=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em previsão de demanda, MAE responde ao erro absoluto típico; RMSE destaca falhas grandes que podem causar ruptura de estoque.

## Exemplo numérico resolvido

$y=[10,12,18]$ e $\hat y=[9,15,17]$. Erros absolutos $[1,3,1]$:

$$
MAE=5/3\approx1{,}667.
$$

Erros quadráticos $[1,9,1]$:

$$
MSE=11/3\approx3{,}667,\qquad RMSE\approx1{,}915.
$$

Como $\bar y=13{,}333$ e $SST\approx34{,}667$, temos $R^2=1-11/34{,}667\approx0{,}683$. As quatro medidas descrevem o mesmo conjunto sob lentes diferentes.

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

### Investigação adicional

Crie previsões com erro normal e injete 1%, 5% e 10% de outliers. Trace MAE, RMSE e $R^2$ contra a contaminação. Estratifique resíduos por quantis do target e produza um gráfico previsto × real com linha $y=x$.

## Laboratório guiado completo

Use os mesmos resíduos para comparar métricas e depois injete um outlier.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y = np.array([10., 12., 18., 20., 22.])
pred = np.array([9., 15., 17., 19., 23.])

def report(actual, forecast):
    return {
        "MAE": mean_absolute_error(actual, forecast),
        "RMSE": mean_squared_error(actual, forecast) ** 0.5,
        "R2": r2_score(actual, forecast),
        "residuos": actual - forecast,
    }

print("original", report(y, pred))
pred_outlier = pred.copy(); pred_outlier[-1] = 60
print("com outlier", report(y, pred_outlier))
```

**Entregue:** cálculo manual; análise de unidade; curva de métricas contra tamanho do outlier; residual plots por faixa do target.

### Protocolo investigativo obrigatório

O laboratório não termina quando o código executa. Para transformar execução em aprendizagem e evidência:

1. escreva uma hipótese antes de rodar o experimento;
2. mantenha um baseline e altere uma decisão por vez;
3. use o mesmo split ou os mesmos folds nas comparações;
4. reporte a distribuição das métricas, não apenas o melhor número;
5. inspecione pelo menos cinco erros ou casos extremos;
6. registre seed, versões, hiperparâmetros e tempo de execução;
7. conclua com **o que os resultados sustentam** e **o que não sustentam**.

Salve um relatório curto em Markdown, a configuração em JSON e o código executável. Uma execução sem interpretação não satisfaz o critério de domínio.

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

## Exercícios de aprofundamento e rubrica

### Nível A — reconstrução conceitual

Feche o material e explique o problema, as hipóteses, cada símbolo das equações e a diferença entre treinamento, seleção e avaliação. Desenhe o fluxo de dados sem consultar o texto. Se uma definição depender de palavras vagas como “melhor” ou “parecido”, torne-a operacional.

### Nível B — cálculo e implementação

Refaça o exemplo numérico com valores diferentes e confira manualmente o resultado do código. Implemente a operação matemática central com NumPy ou Python básico antes de usar a abstração do scikit-learn. Compare tolerâncias e explique qualquer diferença numérica.

### Nível C — contraprova experimental

Crie deliberadamente um cenário em que o método falha: ruído, outlier, escala incompatível, shift, grupos repetidos, classe rara ou leakage. Formule antes o comportamento esperado, execute a ablação e confronte hipótese e resultado.

### Nível D — transferência para sistema real

Aplique o conceito a um problema do AI Systems Laboratory. Declare unidade, instante de predição, dados disponíveis, baseline, métrica, custo dos erros e threat to validity. Produza um artefato que outra pessoa consiga auditar.

### Rubrica de 0 a 4

- **0 — reconhecimento:** identifica o nome, mas não explica o mecanismo;
- **1 — reprodução:** executa exemplo pronto;
- **2 — compreensão:** deriva/calcula e interpreta o resultado;
- **3 — diagnóstico:** prevê falhas, escolhe protocolo e analisa erros;
- **4 — transferência:** projeta, implementa e defende um experimento novo e reproduzível.

**Carga sugerida:** 45 min de leitura ativa, 45 min de derivação/cálculo, 90 min de laboratório, 30 min de análise de erros e 30 min de relatório. Avance somente ao atingir pelo menos nível 3.

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

## Leitura orientada e fontes verificadas

- scikit-learn — [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
- James et al. — [ISLP](https://www.statlearning.com/), avaliação de regressão.
- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), losses e decisão estatística.
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), erro de generalização.

## Próxima aula

**Métricas de classificação: matriz de confusão, precision, recall e F1**
