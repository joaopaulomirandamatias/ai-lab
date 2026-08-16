# 02 · Estatística, probabilidade e dados

**Mês M3** · Roadmap: Níveis 1 (parte) e 2

---

## O que dominar

**Probabilidade** — eventos, variáveis aleatórias, probabilidade condicional,
independência, Teorema de Bayes, esperança, variância, distribuições (Bernoulli,
binomial, normal, Poisson, categorical).

**Estatística** — descritiva, média, mediana, variância, correlação, inferência,
intervalos de confiança, testes de hipótese, p-value, **tamanho de efeito**, regressão,
ANOVA, bootstrap, análise experimental.

**Teoria da informação** — entropia, cross-entropy, KL divergence, informação mútua,
perplexidade. Muito importante para IA moderna: cross-entropy *é* a loss de todo LLM,
e perplexidade é como se mede um modelo de linguagem.

**Dados** (Nível 2) — coleta, limpeza, normalização, transformação, missing values,
outliers, encoding, feature engineering, **data leakage**, balanceamento, amostragem,
train/validation/test split, versionamento de datasets, pipelines, qualidade,
*provenance*, *lineage*, governança, LGPD aplicada a datasets.
Também: SQL avançado, ETL/ELT, Parquet, Data Lake, Data Warehouse.

---

## Fonte principal

⭐ **An Introduction to Statistical Learning (ISLP — edição Python)** — grátis em
`statlearning.com`. É a ponte entre estatística e ML.

Complemento fortíssimo: **Statistical Rethinking** (McElreath) — livro + curso em vídeo
gratuito. Ensina a *pensar* estatisticamente em vez de aplicar receita, que é exatamente
o que o M18 vai exigir de você.

Destravar conceito: StatQuest no YouTube.

---

## Projeto

**P3 — Experimento estatístico honesto (M3)**
Pegue um dataset real e responda uma pergunta com teste de hipótese.

*Pronto quando* há intervalo de confiança, tamanho de efeito e uma frase declarando
o que o resultado **não** prova.

---

## Por que isso importa mais do que parece

Este é o mês que decide se o M18 vai funcionar. A diferença entre "meu sistema
multiagente parece melhor" e "meu sistema multiagente é melhor com p < 0.01 e efeito
médio de 0.6" está inteiramente aqui.

Duas coisas para levar a sério agora, porque voltam depois:

- **`provenance` e `lineage`** — de onde veio cada dado e por onde passou. É o mesmo
  problema conceitual da camada de *Evidence* do M15 e do PROV-O do
  [`11-interop-semantica/`](../11-interop-semantica/). Você vai reencontrar isso três vezes.
- **Tamanho de efeito** — p-value diz se há diferença; tamanho de efeito diz se ela
  importa. Papers ruins reportam só o primeiro.
