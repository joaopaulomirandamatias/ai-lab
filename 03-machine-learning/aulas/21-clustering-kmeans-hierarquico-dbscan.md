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

## Aprofundamento — clustering é uma hipótese exploratória

K-Means alterna duas etapas: atribuir cada ponto ao centroide mais próximo e atualizar cada centroide pela média. A objective não convexa pode convergir a mínimos locais; `n_init` e seed importam. Distância quadrática e centróides favorecem grupos aproximadamente convexos/esféricos e sensibilidade a escala/outliers.

Clustering hierárquico depende da distância e do linkage (single, complete, average, Ward). DBSCAN define ponto central por pelo menos `min_samples` na vizinhança de raio `eps`, expande conectividade por densidade e marca ruído. Densidades variáveis desafiam um único `eps`.

Silhouette mede coesão/separação na geometria escolhida, não “verdade”. Avalie estabilidade por reamostragem, concordância entre seeds, separação em dados originais e utilidade de domínio. Não batize clusters como personas sem validação externa.

## 3. Equação para guardar

$$
\min_{C_1,\dots,C_k}\sum_{j=1}^{k}\sum_{x_i\in C_j}\|x_i-\mu_j\|^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Segmentar clientes por comportamento. Os clusters devem ser avaliados por estabilidade e utilidade, não apenas por um gráfico 2D atraente.

## Exemplo numérico resolvido

Pontos unidimensionais $[0,1,9,10]$ e $k=2$. Com clusters $C_1=[0,1]$ e $C_2=[9,10]$, centróides são $0{,}5$ e $9{,}5$. A SSE é

$$
(0-0{,}5)^2+(1-0{,}5)^2+(9-9{,}5)^2+(10-9{,}5)^2=1.
$$

Um agrupamento $[0,1,9]$ e $[10]$ tem centroide $10/3$ no primeiro grupo e SSE muito maior. O exemplo também mostra por que outliers deslocam médias.

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

### Investigação adicional

Use blobs esféricos, luas e dados com ruído. Compare K-Means, aglomerativo e DBSCAN após scaling. Varie seeds/parâmetros, calcule silhouette e estabilidade por Adjusted Rand Index entre reamostragens. Descreva onde cada hipótese geométrica falha.

## Laboratório guiado completo

Compare hipóteses geométricas em blobs e luas e avalie estabilidade.

```python
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, truth = make_moons(n_samples=800, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)
models = {
    "kmeans": KMeans(n_clusters=2, n_init=20, random_state=42),
    "dbscan": DBSCAN(eps=0.25, min_samples=8),
}
for name, model in models.items():
    labels = model.fit_predict(X)
    keep = labels != -1
    n_clusters = len(set(labels[keep]))
    sil = silhouette_score(X[keep], labels[keep]) if n_clusters > 1 else float("nan")
    print(name, "clusters", n_clusters, "ruído", (~keep).sum(), "silhouette", sil)
```

**Entregue:** vários seeds e `eps`; silhouette e estabilidade por ARI; dendrograma aglomerativo; interpretação de domínio sem batizar automaticamente os clusters.

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

- Rodar K-Means sem scaling.
- Escolher k apenas pelo elbow de forma mecânica.
- Interpretar cluster como classe natural sem validação.
- Usar t-SNE/UMAP 2D para provar separação original.

## 8. Exercícios

1. Qual geometria K-Means favorece?
2. Quando DBSCAN é interessante?
3. Por que cluster não é necessariamente categoria real?
4. Como verificar estabilidade de clustering?

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

- Hastie et al. — Unsupervised Learning.
- Ester et al. (1996) — DBSCAN.
- Rousseeuw (1987) — Silhouettes.
- scikit-learn — Clustering.

## Leitura orientada e fontes verificadas

- Ester et al. (1996) — [artigo original do DBSCAN](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf).
- James et al. — [ISLP](https://www.statlearning.com/), aprendizagem não supervisionada.
- scikit-learn — [Clustering](https://scikit-learn.org/stable/modules/clustering.html).
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), cap. 14.

## Próxima aula

**Redução de dimensionalidade em ML: PCA, t-SNE e UMAP com responsabilidade**
