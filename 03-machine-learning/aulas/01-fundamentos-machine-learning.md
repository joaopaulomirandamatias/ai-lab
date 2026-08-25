# Aula 01 — Fundamentos de Machine Learning: problemas, paradigmas e generalização

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Módulos 01 Matemática e 02 Probabilidade/Estatística concluídos  
**Objetivo central:** Entender o que torna um problema de Machine Learning diferente de uma regra programada manualmente e distinguir treinamento, inferência e generalização.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir aprendizado supervisionado, não supervisionado e semi/self-supervisionado.
- Definir amostra, feature, target, modelo, parâmetro e hiperparâmetro.
- Explicar treinamento, inferência e generalização.
- Reconhecer underfitting, overfitting e erro irredutível.
- Formular um problema de ML antes de escolher um algoritmo.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender o que torna um problema de machine learning diferente de uma regra programada manualmente e distinguir treinamento, inferência e generalização.**

## 2. Ideias fundamentais

### 1. Do programa explícito ao modelo aprendido

Em programação tradicional, regras + dados produzem saídas. Em ML supervisionado, exemplos de entrada e saída são usados para aprender uma função aproximada $\hat f(x;\theta)$ capaz de generalizar para exemplos não vistos.

### 2. Treinamento e inferência

Treinamento é o processo de ajustar parâmetros $\theta$ para reduzir uma função de perda. Inferência é usar os parâmetros já ajustados para produzir previsões. Misturar os dois conceitos leva a erros de arquitetura e avaliação.

### 3. Generalização

O objetivo real não é minimizar o erro nos dados de treinamento, mas o risco esperado em dados provenientes da distribuição de interesse. Um modelo que memoriza o treino pode ter baixo erro empírico e alto erro fora da amostra.

### 4. Viés e variância

Modelos simples podem ter alto viés; modelos muito flexíveis podem ter alta variância. O ponto útil depende do tamanho dos dados, ruído, regularização e desenho experimental.

## 3. Equação para guardar

$$
\hat{\theta}=\arg\min_{\theta}\frac{1}{n}\sum_{i=1}^{n}L(f_\theta(x_i),y_i)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Classificar e-mails como spam/não spam. Features podem incluir representações do texto; target é a classe. O modelo é treinado em exemplos rotulados e avaliado em mensagens que não participaram do ajuste.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
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

- Escolher o algoritmo antes de definir claramente target, unidade de análise e métrica.
- Avaliar o modelo nos mesmos dados usados para treiná-lo.
- Confundir desempenho médio histórico com garantia de comportamento futuro.
- Ignorar mudança de distribuição entre treino e produção.

## 8. Exercícios

1. Dê um exemplo de problema que não precisa de ML e explique por quê.
2. Explique com suas palavras a diferença entre parâmetro e hiperparâmetro.
3. Por que erro de treino baixo não garante generalização?
4. Classifique três problemas seus como regressão, classificação ou não supervisionado.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- James et al. — An Introduction to Statistical Learning with Applications in Python (ISLP), caps. 1–2.
- Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, caps. 1–2.
- Murphy — Probabilistic Machine Learning: An Introduction, cap. 1.
- scikit-learn User Guide — Supervised learning.

## Próxima aula

**Do problema ao experimento: features, target, splits e baseline**
