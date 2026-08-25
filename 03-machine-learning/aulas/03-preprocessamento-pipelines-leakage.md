# Aula 03 — Pré-processamento, pipelines e data leakage

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 02 deste módulo  
**Objetivo central:** Construir transformações reproduzíveis sem deixar estatísticas do conjunto de validação/teste vazarem para o treinamento.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Tratar dados numéricos e categóricos com pipelines.
- Entender imputação, scaling e encoding.
- Aplicar fit apenas nos dados de treinamento.
- Usar ColumnTransformer e Pipeline.
- Identificar leakage de target, tempo e pré-processamento.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **construir transformações reproduzíveis sem deixar estatísticas do conjunto de validação/teste vazarem para o treinamento.**

## 2. Ideias fundamentais

### 1. Fit versus transform

Transformações como padronização aprendem estatísticas dos dados. A média e o desvio usados no StandardScaler devem ser estimados somente no treino.

### 2. Imputação

Valores ausentes não devem ser preenchidos usando estatísticas calculadas com treino+teste. Além disso, o próprio padrão de ausência pode carregar informação e deve ser investigado.

### 3. Encoding

One-hot encoding é apropriado para categorias nominais em muitos modelos. OrdinalEncoder só deve impor ordem quando ela fizer sentido ou quando o algoritmo tolerar a codificação.

### 4. Pipeline

Pipeline encapsula pré-processamento e modelo numa única unidade, tornando cross-validation e tuning muito mais seguros.

## Aprofundamento — transformadores também aprendem parâmetros

Um scaler, imputador, encoder orientado por frequência ou seletor de features não é uma função fixa: ele possui estado aprendido. Para uma transformação $T_{\hat\phi}$,

$$
\hat\phi=fit(X_{train}),\qquad Z_{train}=T_{\hat\phi}(X_{train}),\qquad Z_{test}=T_{\hat\phi}(X_{test}).
$$

O erro metodológico é estimar $\hat\phi$ com treino e teste. A mesma regra vale dentro de cross-validation: cada fold precisa ajustar seu próprio preprocessing apenas na parte de treino. `Pipeline` automatiza essa fronteira e permite tunar transformação e modelo como uma unidade.

Leakage pode ser **estatístico** (média global), **temporal** (informação futura), **por target** (encoding ou seleção usando $y$ fora do fold), **por duplicação** ou **por agregação**. Uma pipeline resolve a primeira classe, mas não corrige um dataset cuja feature já contém o futuro.

## 3. Equação para guardar

$$
z=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Num dataset com idade, renda e estado civil, idade/renda podem ser imputadas e padronizadas; estado civil pode receber one-hot encoding. Tudo é ajustado apenas com X_train.

## Exemplo numérico resolvido

Treino: $[1,2,3]$; teste: $[100]$. O scaler correto aprende $\mu_{train}=2$ e $\sigma_{train}\approx0{,}816$. Assim, o valor de teste vira aproximadamente $120{,}0$ desvios do centro de treino — um caso extremo real.

Se o scaler for ajustado em todos os dados, a média vira $26{,}5$ e o desvio cresce para cerca de $42{,}44$; o teste vira apenas $1{,}73$. A informação do próprio teste o tornou artificialmente menos extremo. A métrica resultante é otimista porque o preprocessing conheceu a distribuição que deveria permanecer invisível.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_cols = ["idade", "renda"]
cat_cols = ["estado_civil"]

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Construa duas avaliações: (A) scaler e seleção antes do CV; (B) ambos dentro de `Pipeline`. Use dados sintéticos com 1.000 features aleatórias e poucas amostras. Compare a diferença e explique por que selecionar features no dataset inteiro consegue “descobrir” correlações espúrias do fold de validação.

## Laboratório guiado completo

O exemplo usa dados mistos, missing values e CV. Todo estado aprendido permanece dentro do pipeline.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "idade": rng.normal(40, 12, 600),
    "renda": rng.lognormal(8.5, 0.5, 600),
    "estado": rng.choice(["AP", "SP", "PA"], 600),
})
df.loc[rng.choice(600, 50, replace=False), "renda"] = np.nan
y = ((df["idade"] > 45) | (df["estado"] == "SP")).astype(int)

num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))])
prep = ColumnTransformer([("num", num, ["idade", "renda"]), ("cat", cat, ["estado"])])
pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_validate(pipe, df, y, cv=cv, scoring=["roc_auc", "average_precision"])
print({k: (v.mean(), v.std()) for k, v in scores.items() if k.startswith("test_")})
```

**Entregue:** versão correta; uma versão com leakage intencional; diferença entre elas; diagrama de quais operações executam `fit` em cada fold.

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

- Executar scaler antes do train_test_split.
- Selecionar features olhando todo o dataset.
- Imputar com estatísticas globais.
- Criar agregações que incluem o futuro do registro.

## 8. Exercícios

1. Explique por que StandardScaler pode causar leakage.
2. Monte um pipeline para dados mistos.
3. Dê três exemplos de leakage temporal.
4. Explique por que pipeline é importante durante cross-validation.

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

- scikit-learn — Pipelines and composite estimators.
- Kaufman et al. — Leakage in Data Mining.
- ISLP, capítulos de model assessment e preprocessing.
- Géron — Hands-On Machine Learning, capítulos de end-to-end ML projects.

## Leitura orientada e fontes verificadas

- scikit-learn — [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
- scikit-learn — [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- scikit-learn — [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
- James et al. — [ISLP](https://www.statlearning.com/), laboratórios de preprocessing e resampling.

## Próxima aula

**Regressão linear e mínimos quadrados**
