# Aula 08 — Naive Bayes: probabilidade condicional aplicada à classificação

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 07 deste módulo  
**Objetivo central:** Aplicar Bayes em classificação e entender por que uma hipótese de independência irrealista ainda pode produzir bons classificadores.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Derivar a regra de decisão do Naive Bayes.
- Distinguir Gaussian, Multinomial e Bernoulli NB.
- Entender a hipótese de independência condicional.
- Relacionar NB a classificação de texto.
- Reconhecer diferenças entre boa classificação e boa calibração.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **aplicar bayes em classificação e entender por que uma hipótese de independência irrealista ainda pode produzir bons classificadores.**

## 2. Ideias fundamentais

### 1. Regra de Bayes

Escolhemos a classe que maximiza $P(y)\prod_jP(x_j\mid y)$. O denominador comum entre classes pode ser omitido na decisão.

### 2. Hipótese naive

As features são tratadas como condicionalmente independentes dado y. Essa suposição costuma ser falsa, mas pode simplificar muito a estimação.

### 3. Texto

MultinomialNB funciona bem como baseline para contagens TF/TF-IDF em documentos.

### 4. Logs

Produtos de muitas probabilidades pequenas sofrem underflow; implementações usam somas de log-probabilidades.

## 3. Equação para guardar

$$
\hat y=\arg\max_y P(y)\prod_j P(x_j\mid y)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em spam detection, tokens como 'grátis', 'promoção' e 'urgente' alteram a evidência condicional de cada classe.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)
model.fit(text_train, y_train)
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

- Interpretar independência naive como descrição real do mundo.
- Usar GaussianNB em dados que não combinam com sua hipótese sem diagnóstico.
- Comparar probabilidades sem considerar calibração.
- Ignorar smoothing em contagens raras.

## 8. Exercícios

1. Por que usamos logs na implementação?
2. Qual variante de NB é natural para contagem de palavras?
3. Explique a hipótese de independência condicional.
4. Por que um modelo com hipótese errada ainda pode classificar bem?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Murphy — Naive Bayes classifiers.
- ISLP — Classification.
- scikit-learn — Naive Bayes.
- Manning, Raghavan & Schütze — Introduction to Information Retrieval, text classification.

## Próxima aula

**Árvores de decisão: partições, impureza e interpretabilidade**
