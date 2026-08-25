# Aula 02 — Do problema ao experimento: features, target, splits e baseline

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 01 deste módulo  
**Objetivo central:** Aprender a transformar uma pergunta de negócio ou pesquisa em um experimento de ML mensurável e sem vazamento.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Definir unidade de análise, features e target.
- Separar treino, validação e teste com papéis distintos.
- Construir baselines úteis.
- Escolher métricas compatíveis com o custo do erro.
- Entender por que o conjunto de teste deve permanecer intocado.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **aprender a transformar uma pergunta de negócio ou pesquisa em um experimento de ml mensurável e sem vazamento.**

## 2. Ideias fundamentais

### 1. Unidade de análise

Antes de falar em features, defina o que cada linha representa: pessoa, transação, processo, documento, janela temporal, equipamento etc. Uma unidade mal definida cria duplicação, dependência indevida e leakage.

### 2. Train / validation / test

Treino ajusta parâmetros; validação orienta escolhas; teste estima desempenho final. Quando cross-validation é usada, o papel da validação é incorporado aos folds, mas o teste externo continua separado.

### 3. Baseline

Um modelo só é útil se supera uma referência plausível. Em regressão, prever a média é um baseline; em classificação, classe majoritária ou uma regra operacional existente podem ser referências.

### 4. Métrica orientada ao objetivo

Accuracy pode ser inadequada quando falsos negativos e falsos positivos têm custos diferentes. A métrica deve refletir o uso real do sistema.

## 3. Equação para guardar

$$
\text{ganho}=\text{métrica(modelo)}-\text{métrica(baseline)}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever atraso de entrega. Uma linha = uma entrega. O target deve usar apenas informação disponível até o instante em que a previsão será feita. Um baseline pode ser a taxa histórica de atraso por rota.

## 5. Laboratório em Python / scikit-learn

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# X, y devem estar definidos
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# baseline = DummyClassifier(strategy="prior")
# baseline.fit(X_train, y_train)
# pred = baseline.predict(X_test)
# print(balanced_accuracy_score(y_test, pred))
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

- Usar o teste repetidamente durante desenvolvimento.
- Criar features com informação posterior ao momento da previsão.
- Não definir um baseline antes do tuning.
- Fazer split aleatório em dados temporais ou grupos dependentes.

## 8. Exercícios

1. Defina unidade de análise e target para um problema de detecção de fraude.
2. Explique a diferença entre validação e teste.
3. Crie dois baselines para um problema de classificação desbalanceada.
4. Dê um exemplo em que split aleatório seria metodologicamente incorreto.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP, cap. 5 — Resampling Methods.
- Kaufman et al. — Leakage in Data Mining: Formulation, Detection, and Avoidance.
- scikit-learn — Model selection and evaluation.
- Google Rules of ML — orientação prática sobre baselines e métricas.

## Próxima aula

**Pré-processamento, pipelines e data leakage**
