# Aula 21 — Clustering: K-Means, hierárquico e DBSCAN

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 20 deste módulo  
**Objetivo central:** Aprender agrupamento não supervisionado e reconhecer quando clusters são artefatos da métrica e do pré-processamento.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Entender objetivo do K-Means.
- Conhecer clustering hierárquico.
- Entender DBSCAN e densidade.
- Avaliar silhouette com cautela.
- Reconhecer sensibilidade a scaling e geometria.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **aprender agrupamento não supervisionado e reconhecer quando clusters são artefatos da métrica e do pré-processamento.**

## 2. Ideias fundamentais

### 1. K-Means

Minimiza a soma das distâncias quadráticas aos centroides. Favorece clusters aproximadamente esféricos e exige escolher k.

### 2. Hierárquico

Constrói uma árvore de agrupamentos por fusão ou divisão. Dendrograma permite examinar diferentes cortes.

### 3. DBSCAN

Define clusters por regiões densas e pode marcar ruído, sem exigir k, mas depende de eps/min_samples.

### 4. Sem ground truth

Cluster não é automaticamente uma categoria real. Validação deve combinar métricas internas, estabilidade e utilidade de domínio.

## 3. Equação para guardar

$$
\min_{C_1,\dots,C_k}\sum_{j=1}^{k}\sum_{x_i\in C_j}\|x_i-\mu_j\|^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Segmentar clientes por comportamento. Os clusters devem ser avaliados por estabilidade e utilidade, não apenas por um gráfico 2D atraente.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

clusterer = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=4, n_init="auto", random_state=42)
)

labels = clusterer.fit_predict(X)
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

- Rodar K-Means sem scaling.
- Escolher k apenas pelo elbow de forma mecânica.
- Interpretar cluster como classe natural sem validação.
- Usar t-SNE/UMAP 2D para provar separação original.

## 8. Exercícios

1. Qual geometria K-Means favorece?
2. Quando DBSCAN é interessante?
3. Por que cluster não é necessariamente categoria real?
4. Como verificar estabilidade de clustering?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Hastie et al. — Unsupervised Learning.
- Ester et al. (1996) — DBSCAN.
- Rousseeuw (1987) — Silhouettes.
- scikit-learn — Clustering.

## Próxima aula

**Redução de dimensionalidade em ML: PCA, t-SNE e UMAP com responsabilidade**
