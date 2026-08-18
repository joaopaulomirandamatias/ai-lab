# Aula 17 — PCA: redução de dimensionalidade preservando variância

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear aplicada  
**Pré-requisitos:** autovalores, ortogonalidade e SVD  
**Objetivo:** entender PCA geometricamente, implementar o algoritmo e reconhecer quando reduzir dimensionalidade é útil — e quando pode destruir informação importante.

## 1. O problema

Imagine dados com 100 características. Talvez eles não ocupem de verdade todas as 100 direções. Podem estar concentrados perto de um subespaço muito menor.

PCA — **Principal Component Analysis** — procura novas direções ortogonais que expliquem a maior variância possível dos dados.

> PCA não escolhe simplesmente algumas colunas originais. Ele cria novas coordenadas como combinações lineares das características.

## 2. Intuição em 2D

Suponha uma nuvem de pontos alongada na diagonal:

```text
      • •
    • •
  • •
 •
```

Os eixos x e y não estão alinhados com a estrutura principal. PCA encontra uma direção nova, seguindo o eixo longo da nuvem.

Essa primeira direção é a **primeira componente principal**.

A segunda componente é ortogonal à primeira e explica a maior parte da variância restante.

## 3. Centralização

Antes de PCA, normalmente centralizamos os dados:

\[
X_c=X-\mu
\]

onde \(\mu\) é a média de cada característica.

Por que? Porque PCA procura direções de variação em torno do centro dos dados.

Sem centralização, a origem arbitrária do sistema de coordenadas pode contaminar a análise.

## 4. Matriz de covariância

Para dados centralizados, uma convenção comum é:

\[
\Sigma=\frac{1}{n-1}X_c^TX_c
\]

A diagonal contém variâncias individuais. Os termos fora da diagonal descrevem covariação entre características.

Como \(\Sigma\) é simétrica, possui uma base ortonormal de autovetores.

## 5. Autovetores e autovalores em PCA

Se:

\[
\Sigma v_i=\lambda_i v_i
\]

então:

- \(v_i\): direção principal;
- \(\lambda_i\): variância dos dados nessa direção.

Ordenamos:

\[
\lambda_1\ge\lambda_2\ge\cdots\ge0
\]

A primeira componente é a direção com maior variância.

## 6. Variância explicada

A proporção explicada pela componente \(i\) é:

\[
\boxed{\frac{\lambda_i}{\sum_j\lambda_j}}
\]

Se os três primeiros autovalores explicam 95% da variância, podemos considerar representar os dados apenas nessas três direções.

Isso reduz dimensionalidade com uma perda controlada segundo o critério de variância.

## 7. Projeção

Se \(W_k\) contém as \(k\) componentes principais:

\[
W_k=[v_1\ \cdots\ v_k]
\]

as novas coordenadas são:

\[
\boxed{Z=X_cW_k}
\]

Se \(X\) era \(n\times d\), agora:

\[
Z\in\mathbb{R}^{n\times k}
\]

com \(k<d\).

## 8. Reconstrução aproximada

Podemos voltar aproximadamente ao espaço original:

\[
\hat X=ZW_k^T+\mu
\]

Quanto menor \(k\), maior a compressão e potencialmente maior o erro de reconstrução.

## 9. PCA via SVD

Em vez de formar explicitamente a covariância, podemos usar:

\[
X_c=U\Sigma V^T
\]

As colunas de \(V\) são as direções principais. Os valores singulares se relacionam à variância explicada.

Na prática, calcular PCA via SVD é uma abordagem muito comum e numericamente atraente.

## 10. Exemplo NumPy

```python
import numpy as np

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0]
])

Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

pc1 = Vt[0]
Z1 = Xc @ pc1[:, None]
Xhat = Z1 @ pc1[None, :] + X.mean(axis=0)

print("primeira direção:", pc1)
print("dados 1D:", Z1.ravel())
```

## 11. PCA com scikit-learn

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print(pca.components_)
print(pca.explained_variance_ratio_)
```

Use a implementação de biblioteca em projetos reais; implemente do zero no laboratório para entender a matemática.

## 12. Escala das características

PCA é sensível à escala. Se uma variável está em quilômetros e outra em milímetros, a de maior variância numérica pode dominar.

Dependendo do problema, pode ser apropriado padronizar:

\[
z=\frac{x-\mu}{\sigma}
\]

Mas padronizar também muda a pergunta estatística. Não deve ser um ritual automático.

## 13. PCA não entende “importância semântica”

PCA preserva variância, não necessariamente informação relevante para uma tarefa.

Uma característica de baixa variância pode ser decisiva para classificação. Por isso PCA é uma técnica não supervisionada e precisa ser avaliada no contexto do objetivo final.

## 14. Data leakage

Se PCA fizer parte de um pipeline de Machine Learning, ajuste o PCA apenas nos dados de treino.

Errado:

```text
todo dataset → PCA → split
```

Correto:

```text
split
  ↓
fit PCA no treino
  ↓
transform treino e teste com o mesmo PCA
```

Caso contrário, informações do conjunto de teste vazam para o pré-processamento.

## 15. Conexões com IA

- visualização de embeddings em 2D/3D;
- compressão de features;
- redução de ruído;
- análise exploratória;
- diagnóstico de dimensionalidade efetiva;
- preparação conceitual para SVD, latent spaces e representações.

Para embeddings modernos, PCA é uma ferramenta de análise útil, mas projeções 2D nunca mostram toda a geometria do espaço original.

## Exercícios

1. Por que centralizamos os dados antes de PCA?
2. O que significa um autovalor grande na matriz de covariância?
3. Diferencie “selecionar features” de “criar componentes principais”.
4. Se 5 componentes explicam 98% da variância, o que ganhamos e o que ainda precisamos avaliar?
5. Por que PCA pode causar data leakage?
6. Explique a relação entre PCA e SVD.

## Referências

- JOLLIFFE, I. T.; CADIMA, J. *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A, 2016. DOI: 10.1098/rsta.2015.0202.
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 10 — PCA. https://mml-book.github.io/
- STRANG, G. *MIT 18.06 Linear Algebra*. SVD, projections e eigenvectors. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

## Próxima aula

**Aula 18 — Mínimos quadrados, pseudoinversa e condicionamento.**