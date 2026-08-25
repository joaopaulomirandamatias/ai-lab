# Aula 18 — Hyperparameter tuning: Grid Search, Random Search e validação aninhada

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 17 deste módulo  
**Objetivo central:** Otimizar hiperparâmetros sem contaminar a estimativa final de desempenho.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir parâmetros e hiperparâmetros.
- Usar GridSearchCV e RandomizedSearchCV.
- Entender espaços logarítmicos.
- Evitar overfitting à validação.
- Conhecer nested cross-validation.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **otimizar hiperparâmetros sem contaminar a estimativa final de desempenho.**

## 2. Ideias fundamentais

### 1. Parâmetro vs hiperparâmetro

Parâmetros são aprendidos pelo fit; hiperparâmetros controlam o processo/modelo e são escolhidos externamente.

### 2. Random Search

Quando poucos hiperparâmetros realmente importam, random search explora valores úteis de forma mais eficiente que uma grade cartesiana rígida.

### 3. Nested CV

Loop interno escolhe hiperparâmetros; loop externo estima generalização. É importante em comparações científicas com datasets pequenos.

### 4. Orçamento

Tuning é um problema de busca sob orçamento. Mais trials não corrigem um protocolo metodológico ruim.

## 3. Equação para guardar

$$
\lambda^*=\arg\max_{\lambda\in\Lambda}\widehat{Score}_{CV}(\lambda)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Tunar C e gamma de uma SVM dentro dos folds e usar outro loop externo para estimar desempenho reduz o otimismo de selecionar e avaliar nos mesmos folds.

## 5. Laboratório em Python / scikit-learn

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

search = RandomizedSearchCV(
    SVC(),
    param_distributions={
        "C": loguniform(1e-3, 1e3),
        "gamma": loguniform(1e-4, 1e1)
    },
    n_iter=50,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
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

- Tunar no teste.
- Usar distribuição linear para hiperparâmetros que variam em ordens de magnitude.
- Reportar melhor score interno como desempenho final.
- Comparar modelos com budgets de tuning muito diferentes sem transparência.

## 8. Exercícios

1. Diferencie Grid e Random Search.
2. O que nested CV resolve?
3. Por que C costuma ser pesquisado em escala log?
4. Explique overfitting à validação.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization.
- Cawley & Talbot (2010) — On Over-fitting in Model Selection.
- scikit-learn — Tuning hyper-parameters.
- ISLP — model selection.

## Próxima aula

**Feature engineering e seleção de variáveis**
