# Aula 06 — Regressão logística e classificação probabilística

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 05 deste módulo  
**Objetivo central:** Entender como um modelo linear produz probabilidades de classe por meio da função sigmoide e log-loss.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar logits e probabilidades.
- Entender sigmoide e log-odds.
- Relacionar regressão logística a cross-entropy.
- Separar probabilidade prevista de decisão por threshold.
- Interpretar coeficientes em termos de odds.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como um modelo linear produz probabilidades de classe por meio da função sigmoide e log-loss.**

## 2. Ideias fundamentais

### 1. Logit

A combinação linear $z=w^Tx+b$ não é probabilidade. A sigmoide converte o logit em valor entre 0 e 1.

### 2. Log-odds

A regressão logística assume que o logaritmo das odds é linear nas features: $\log(p/(1-p))=w^Tx+b$.

### 3. Cross-entropy

O treinamento por máxima verossimilhança para Bernoulli leva à binary cross-entropy.

### 4. Threshold

O modelo estima score/probabilidade; a regra de decisão pode usar 0.5 ou outro limiar definido pelo custo operacional.

## 3. Equação para guardar

$$
p(y=1\mid x)=\sigma(w^Tx+b)=\frac{1}{1+e^{-(w^Tx+b)}}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever inadimplência. O modelo pode estimar 0.18 para um cliente e 0.73 para outro. O threshold de ação deve ser escolhido conforme custo de risco e falso alarme.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred_30 = (proba >= 0.30).astype(int)
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

- Tratar predict() como única saída relevante.
- Escolher threshold pelo teste final.
- Chamar score não calibrado de probabilidade sem verificar.
- Interpretar associação do coeficiente como causalidade.

## 8. Exercícios

1. Calcule a sigmoide de z=0 e interprete.
2. Qual a diferença entre score, probabilidade e classe?
3. Por que threshold 0.5 não é universal?
4. Explique a relação entre regressão logística e cross-entropy.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP, cap. 4 — Classification.
- Hastie et al. — Linear Methods for Classification.
- Murphy — Logistic Regression.
- scikit-learn — LogisticRegression.

## Próxima aula

**K-Nearest Neighbors: distâncias e maldição da dimensionalidade**
