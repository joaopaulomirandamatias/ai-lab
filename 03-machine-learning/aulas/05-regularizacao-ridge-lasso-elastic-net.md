# Aula 05 — Regularização: Ridge, Lasso e Elastic Net

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 04 deste módulo  
**Objetivo central:** Entender como penalizar complexidade reduz variância, melhora estabilidade e pode realizar seleção de features.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar o papel de lambda/alpha.
- Distinguir penalizações L1 e L2.
- Entender shrinkage.
- Relacionar regularização a viés-variância.
- Aplicar Ridge, Lasso e ElasticNet corretamente.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como penalizar complexidade reduz variância, melhora estabilidade e pode realizar seleção de features.**

## 2. Ideias fundamentais

### 1. Ridge / L2

Adiciona $\lambda\|\beta\|_2^2$ à loss. Tende a reduzir coeficientes de forma suave, sendo útil com multicolinearidade e muitas features correlacionadas.

### 2. Lasso / L1

Adiciona $\lambda\|\beta\|_1$. A geometria L1 favorece soluções esparsas e pode zerar coeficientes.

### 3. Elastic Net

Combina L1 e L2, oferecendo esparsidade e estabilidade quando há grupos de features correlacionadas.

### 4. Regularização e scaling

Como a penalidade depende da magnitude dos coeficientes, features devem estar em escalas comparáveis quando a penalização é aplicada.

## 3. Equação para guardar

$$
\min_\beta \|X\beta-y\|_2^2+\lambda\|\beta\|_2^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Com centenas de features correlacionadas, regressão sem penalização pode produzir coeficientes instáveis. Ridge reduz sua magnitude e geralmente melhora estabilidade fora da amostra.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(X_train, y_train)
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

- Comparar alphas sem padronizar features.
- Escolher regularização pelo teste final.
- Interpretar coeficiente zero do Lasso como prova de irrelevância causal.
- Esquecer que regularização adiciona viés deliberadamente.

## 8. Exercícios

1. Explique por que Ridge ajuda com multicolinearidade.
2. Qual método tende a gerar solução esparsa?
3. Por que scaling importa?
4. Descreva o trade-off de aumentar lambda.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP, cap. 6 — Linear Model Selection and Regularization.
- Tibshirani (1996) — Regression Shrinkage and Selection via the Lasso.
- Zou & Hastie (2005) — Regularization and Variable Selection via the Elastic Net.
- scikit-learn — Linear models.

## Próxima aula

**Regressão logística e classificação probabilística**
