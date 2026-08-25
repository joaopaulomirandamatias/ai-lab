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

## 3. Equação para guardar

$$
z=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Num dataset com idade, renda e estado civil, idade/renda podem ser imputadas e padronizadas; estado civil pode receber one-hot encoding. Tudo é ajustado apenas com X_train.

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

## Próxima aula

**Regressão linear e mínimos quadrados**
