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

## Aprofundamento — decisão Bayesiana e suavização

Pela regra de Bayes,

$$
P(y=c\mid x)\propto P(y=c)P(x\mid y=c).
$$

Naive Bayes fatora a likelihood como $P(x\mid c)=\prod_jP(x_j\mid c)$. Para evitar underflow e transformar produtos em somas:

$$
\log score(c)=\log P(c)+\sum_j\log P(x_j\mid c).
$$

No Multinomial NB, probabilidades de tokens são estimadas por contagens. Sem suavização, um token nunca visto zera todo o produto. A suavização de Laplace/Lidstone adiciona $\alpha$ às contagens. A independência condicional costuma ser falsa; ainda assim, a ordem dos scores pode ser útil para classificação. Probabilidades extremas do Naive Bayes não devem ser assumidas calibradas.

## 3. Equação para guardar

$$
\hat y=\arg\max_y P(y)\prod_j P(x_j\mid y)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em spam detection, tokens como 'grátis', 'promoção' e 'urgente' alteram a evidência condicional de cada classe.

## Exemplo numérico resolvido

Suponha $P(spam)=0{,}4$ e $P(normal)=0{,}6$. Para as palavras “grátis” e “urgente”:

$$
P(grátis\mid spam)=0{,}5,\quad P(urgente\mid spam)=0{,}4,
$$

$$
P(grátis\mid normal)=0{,}05,\quad P(urgente\mid normal)=0{,}1.
$$

Scores não normalizados: spam $=0{,}4\cdot0{,}5\cdot0{,}4=0{,}08$; normal $=0{,}6\cdot0{,}05\cdot0{,}1=0{,}003$. Normalizando, a posterior aproximada de spam é $0{,}08/(0{,}083)\approx0{,}964$. O número alto depende fortemente da hipótese naive.

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

### Investigação adicional

Calcule priors, contagens com suavização e log-scores manualmente para um corpus de dez documentos. Compare `CountVectorizer` e TF-IDF com `MultinomialNB`, varie `alpha` e avalie accuracy, macro-F1 e curva de calibração.

## Laboratório guiado completo

Construa um classificador de texto mínimo e inspecione as probabilidades condicionais aprendidas.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

texts = np.array([
    "grátis promoção urgente", "prêmio grátis agora", "oferta exclusiva urgente",
    "reunião projeto amanhã", "ata da reunião", "revisão do relatório",
    "promoção limitada grátis", "cronograma do projeto", "urgente ganhe prêmio",
    "relatório técnico final", "oferta prêmio", "reunião técnica amanhã",
])
y = np.array([1,1,1,0,0,0,1,0,1,0,1,0])
tr, te = train_test_split(np.arange(len(y)), test_size=0.33, stratify=y, random_state=42)
vec = CountVectorizer()
Xtr = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te])
for alpha in [0.1, 1.0, 10.0]:
    model = MultinomialNB(alpha=alpha).fit(Xtr, y[tr])
    print("alpha", alpha)
    print(classification_report(y[te], model.predict(Xte), zero_division=0))
    print(dict(zip(vec.get_feature_names_out(), model.feature_log_prob_[1])))
```

**Entregue:** cálculo manual de um documento; efeito de `alpha`; comparação Count/TF-IDF; reliability diagram em corpus maior.

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

- Interpretar independência naive como descrição real do mundo.
- Usar GaussianNB em dados que não combinam com sua hipótese sem diagnóstico.
- Comparar probabilidades sem considerar calibração.
- Ignorar smoothing em contagens raras.

## 8. Exercícios

1. Por que usamos logs na implementação?
2. Qual variante de NB é natural para contagem de palavras?
3. Explique a hipótese de independência condicional.
4. Por que um modelo com hipótese errada ainda pode classificar bem?

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

- Murphy — Naive Bayes classifiers.
- ISLP — Classification.
- scikit-learn — Naive Bayes.
- Manning, Raghavan & Schütze — Introduction to Information Retrieval, text classification.

## Leitura orientada e fontes verificadas

- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), modelos generativos e Naive Bayes.
- Manning, Raghavan e Schütze — [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), classificação de texto.
- scikit-learn — [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
- scikit-learn — [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

## Próxima aula

**Árvores de decisão: partições, impureza e interpretabilidade**
