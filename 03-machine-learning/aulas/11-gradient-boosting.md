# Aula 11 — Boosting e Gradient Boosting: aprendendo com os erros anteriores

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 10 deste módulo  
**Objetivo central:** Entender boosting como construção sequencial de um modelo aditivo que corrige resíduos/gradientes.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Diferenciar bagging e boosting.
- Explicar modelo aditivo.
- Entender learning rate e número de árvores.
- Interpretar gradient boosting como descida no espaço de funções.
- Conhecer XGBoost/LightGBM/CatBoost conceitualmente.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender boosting como construção sequencial de um modelo aditivo que corrige resíduos/gradientes.**

## 2. Ideias fundamentais

### 1. Sequencialidade

Boosting adiciona novos weak learners para corrigir padrões ainda mal modelados.

### 2. Gradient Boosting

Cada nova árvore é ajustada aos pseudo-resíduos, relacionados ao gradiente negativo da loss.

### 3. Shrinkage

Learning rate pequeno exige mais árvores, mas frequentemente melhora generalização.

### 4. Implementações modernas

XGBoost, LightGBM e CatBoost acrescentam engenharia eficiente, regularização, estratégias para histogramas e tratamento de categorias.

## 3. Equação para guardar

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em regressão, a primeira árvore aproxima o target; as próximas modelam sistematicamente os resíduos ainda existentes.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.ensemble import HistGradientBoostingClassifier

gb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    random_state=42
)
gb.fit(X_train, y_train)
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

- Usar muitas árvores e learning rate alto sem validação.
- Comparar bibliotecas com defaults diferentes como se fossem equivalentes.
- Ignorar calibration em classificação.
- Fazer tuning no teste.

## 8. Exercícios

1. Diferencie bagging de boosting.
2. Qual o papel do learning rate?
3. Por que boosting é sequencial?
4. Explique pseudo-resíduos em linguagem simples.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Friedman (2001) — Greedy Function Approximation: A Gradient Boosting Machine.
- Chen & Guestrin (2016) — XGBoost.
- ISLP — Boosting.
- scikit-learn — Gradient Boosting.

## Próxima aula

**Support Vector Machines: margem máxima e kernels**
