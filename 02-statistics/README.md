# 02 · Estatística, probabilidade e dados

**Mês M3** · Roadmap: Níveis 1 (parte) e 2

> Trilha detalhada: [`aulas/README.md`](aulas/README.md) — **24 aulas** de Probabilidade → Estatística → Teoria da Informação → Capstone P3.

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
e perplexidade é uma métrica clássica para modelos de linguagem, devendo ser interpretada
com cuidado quando tokenizadores e domínios diferem.

**Dados** (Nível 2) — coleta, limpeza, normalização, transformação, missing values,
outliers, encoding, feature engineering, **data leakage**, balanceamento, amostragem,
train/validation/test split, versionamento de datasets, pipelines, qualidade,
*provenance*, *lineage*, governança, LGPD aplicada a datasets.
Também: SQL avançado, ETL/ELT, Parquet, Data Lake, Data Warehouse.

---

## Sequência didática do M3

A trilha foi organizada em quatro blocos:

1. **Probabilidade (Aulas 01–10)** — incerteza, Bayes, variáveis aleatórias, distribuições, LLN/TCL e Monte Carlo.
2. **Estatística (Aulas 11–20)** — descritiva, amostragem, MLE/MAP, intervalos, testes, tamanho de efeito, bootstrap, FDR, causalidade e desenho experimental.
3. **Teoria da Informação (Aulas 21–23)** — entropia, cross-entropy, perplexidade, KL divergence e informação mútua.
4. **Capstone P3 (Aula 24)** — experimento estatístico honesto, reproduzível e auditável.

O índice completo, objetivos e referências por aula estão em [`aulas/README.md`](aulas/README.md).

---

## Fonte principal

⭐ **An Introduction to Statistical Learning (ISLP — edição Python)** — grátis em
`statlearning.com`. É a ponte entre estatística e ML.

Complementos principais:

- **Introduction to Probability — Blitzstein & Hwang** para probabilidade e Bayes;
- **OpenIntro Statistics** para inferência e experimentação;
- **All of Statistics — Larry Wasserman** como referência compacta;
- **Probabilistic Machine Learning — Kevin Murphy** para a ponte com IA;
- **Statistical Rethinking — Richard McElreath** para raciocínio estatístico;
- **Elements of Information Theory — Cover & Thomas** e **Information Theory, Inference, and Learning Algorithms — David MacKay** para teoria da informação;
- **NIST/SEMATECH e-Handbook of Statistical Methods** para prática experimental;
- **StatQuest** para destravar conceitos visualmente.

---

## Projeto

**P3 — Experimento estatístico honesto (M3)**
Pegue um dataset real e responda uma pergunta com método estatístico justificado.

*Pronto quando* há:

- pergunta e hipótese definidas antes da análise;
- estimativa pontual + intervalo de confiança;
- tamanho de efeito;
- teste ou resampling apropriado;
- análise de premissas/sensibilidade;
- seed e versões de dados/código;
- provenance e lineage;
- limitações e threats to validity;
- uma frase declarando explicitamente o que o resultado **não** prova.

---

## Por que isso importa mais do que parece

Este é o mês que decide se o M18 vai funcionar. A diferença entre "meu sistema
multiagente parece melhor" e "há evidência de melhoria, com efeito quantificado,
incerteza reportada e limitações explícitas" está inteiramente aqui.

Duas coisas para levar a sério agora, porque voltam depois:

- **`provenance` e `lineage`** — de onde veio cada dado e por onde passou. É o mesmo
  problema conceitual da camada de *Evidence* do M15 e do PROV-O do
  [`11-interop-semantica/`](../11-interop-semantica/).
- **Tamanho de efeito** — p-value, quando apropriado, fala sobre incompatibilidade entre dados e um modelo nulo; tamanho de efeito ajuda a responder se a diferença é relevante na prática.
