# Aula 19 — Feature engineering e seleção de variáveis

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 18 deste módulo  
**Objetivo central:** Criar representações úteis sem vazar target e selecionar features de forma compatível com validação.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir transformação e seleção de features.
- Criar interações e features temporais.
- Conhecer filtros, wrappers e métodos embedded.
- Aplicar seleção dentro do pipeline.
- Entender regularização como seleção/controle.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **criar representações úteis sem vazar target e selecionar features de forma compatível com validação.**

## 2. Ideias fundamentais

### 1. Representação

Modelos aprendem sobre as features fornecidas. Uma boa representação pode tornar um problema difícil quase linear.

### 2. Filtros

Métodos univariados avaliam cada feature individualmente e são rápidos, mas ignoram interações.

### 3. Embedded

Lasso e árvores fazem seleção/ponderação durante o próprio treinamento.

### 4. Leakage

Qualquer seleção orientada por y precisa ser ajustada apenas no treino/fold.

## Aprofundamento — representação incorpora hipóteses

$\phi(x)$ não é neutra. Log-transform pressupõe que razões são mais relevantes que diferenças; interações permitem que efeito de uma feature dependa de outra; janelas temporais definem memória; embeddings definem uma geometria.

Separe três famílias de seleção:

- **filter**: score feature-target independente do estimador;
- **wrapper**: avalia subconjuntos treinando modelos, com custo alto;
- **embedded**: seleção durante o fit, como L1 ou árvores.

Toda decisão supervisionada de seleção pertence ao pipeline e ao loop interno de validação. Usar todos os dados para escolher $k$ features e depois fazer CV mede um pipeline que já viu os folds. Features temporais precisam de “as-of join”: cada valor deve existir no instante de predição.

Representações cíclicas evitam que 23h e 0h pareçam distantes: $\sin(2\pi h/24)$ e $\cos(2\pi h/24)$.

## 3. Equação para guardar

$$
x_{\text{novo}}=\phi(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Uma timestamp pode gerar hora, dia da semana, feriado e tempo desde último evento; mas nenhuma feature pode usar informações posteriores ao instante de predição.

## Exemplo numérico resolvido

Na codificação bruta, distância entre 23h e 0h é 23. Na codificação cíclica:

$$
\phi(23)=(-0{,}259,0{,}966),\qquad \phi(0)=(0,1).
$$

A distância Euclidiana é aproximadamente $0{,}261$, coerente com horários vizinhos. A transformação muda o que o modelo consegue aprender com uma fronteira simples.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    SelectKBest(mutual_info_classif, k=20),
    LogisticRegression(max_iter=1000)
)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Crie features temporais cíclicas, razão e interação usando `FunctionTransformer`/`ColumnTransformer`. Compare baseline, engenharia e seleção dentro da mesma CV. Faça um teste negativo incluindo deliberadamente uma feature posterior a $t_0$ e documente o salto enganoso.

## Laboratório guiado completo

Teste se codificação cíclica melhora uma relação periódica sob validação.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(42)
hour = rng.integers(0, 24, 1200)
y = ((hour >= 22) | (hour <= 2)).astype(int)
y = np.where(rng.random(len(y)) < 0.08, 1-y, y)
raw = hour[:,None]
cyclic = np.c_[np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, X in {"raw": raw, "cyclic": cyclic}.items():
    model = make_pipeline(StandardScaler(), PolynomialFeatures(2),
                          LogisticRegression(max_iter=2000))
    s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(name, s.mean(), s.std())
```

**Entregue:** desenho de $\phi(x)$; comparação raw/cíclica; seleção dentro do pipeline; auditoria temporal de disponibilidade das features.

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

- Selecionar features antes do CV.
- Criar agregações futuras.
- Confiar em importância univariada para relações complexas.
- Adicionar centenas de features sem avaliar estabilidade.

## 8. Exercícios

1. Dê três features derivadas de timestamp.
2. Por que seleção deve estar dentro do pipeline?
3. Diferencie filtro, wrapper e embedded.
4. Explique uma feature que causaria leakage.

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

- Guyon & Elisseeff (2003) — An Introduction to Variable and Feature Selection.
- ISLP — model selection and feature engineering.
- scikit-learn — Feature selection.
- Murphy — feature selection.

## Leitura orientada e fontes verificadas

- scikit-learn — [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) e [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html).
- scikit-learn — [Pipelines](https://scikit-learn.org/stable/modules/compose.html).
- James et al. — [ISLP](https://www.statlearning.com/), seleção de modelos e bases não lineares.
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), representação e seleção.

## Próxima aula

**Interpretabilidade: coeficientes, permutation importance e SHAP**
