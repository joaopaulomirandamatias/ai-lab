# Aula 15 — Decomposições matriciais: LU, QR, espectral e por que decompor matrizes

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisitos:** eliminação de Gauss, ortogonalidade e autovalores  
**Objetivo:** compreender decomposição matricial como estratégia para transformar um problema difícil em operações estruturadas mais simples.

## 1. Por que decompor uma matriz?

Uma matriz grande pode parecer opaca. A ideia das decomposições é escrevê-la como produto de matrizes com propriedades especiais:

\[
A=BC
\]

ou:

\[
A=BCD
\]

Cada fator revela uma estrutura diferente.

> Decompor uma matriz é semelhante a fatorar um número: a representação fatorada facilita certas perguntas.

## 2. LU

A decomposição LU escreve:

\[
\boxed{A=LU}
\]

onde:

- \(L\): triangular inferior;
- \(U\): triangular superior.

Ela está intimamente ligada à eliminação de Gauss.

## 3. Exemplo conceitual de LU

Quando eliminamos elementos abaixo dos pivôs, transformamos \(A\) em uma matriz triangular \(U\). Os multiplicadores usados na eliminação podem ser armazenados em \(L\).

Para resolver:

\[
Ax=b
\]

substituímos:

\[
LUx=b
\]

Primeiro resolvemos:

\[
Ly=b
\]

por substituição direta; depois:

\[
Ux=y
\]

por substituição retroativa.

## 4. Por que LU é útil?

Se precisamos resolver vários sistemas com a mesma matriz \(A\), mas diferentes vetores \(b\), fatoramos \(A\) uma vez e reutilizamos \(L\) e \(U\).

Isso evita repetir toda a eliminação.

## 5. QR revisitado

\[
\boxed{A=QR}
\]

onde:

- \(Q\): colunas ortonormais;
- \(R\): triangular superior.

QR é especialmente importante para mínimos quadrados e problemas numéricos em que formar \(A^TA\) pode piorar o condicionamento.

## 6. Decomposição espectral

Para uma matriz simétrica:

\[
\boxed{A=Q\Lambda Q^T}
\]

Ela separa a transformação em três etapas:

1. mudar para a base de autovetores: \(Q^T\);
2. escalar cada direção: \(\Lambda\);
3. voltar para a base original: \(Q\).

Essa interpretação é extremamente didática: uma transformação simétrica é uma rotação/reflexão para uma base especial, seguida de escalas independentes e retorno.

## 7. Cholesky

Para matrizes simétricas positivas definidas:

\[
\boxed{A=LL^T}
\]

Essa é a decomposição de Cholesky. Ela é muito usada em estatística, otimização e modelos Gaussianos por ser eficiente quando as condições se aplicam.

## 8. A próxima grande decomposição: SVD

A SVD funciona para qualquer matriz real \(m\times n\):

\[
\boxed{A=U\Sigma V^T}
\]

Ela combina ideias que já estudamos:

- bases ortonormais;
- mudança de coordenadas;
- escalas;
- rank;
- compressão.

A próxima aula será dedicada inteiramente a ela.

## 9. Qual decomposição usar?

Não existe uma decomposição “melhor” universalmente.

- LU → sistemas lineares quadrados;
- QR → mínimos quadrados e ortogonalização;
- Cholesky → matrizes simétricas positivas definidas;
- espectral → matrizes diagonalizáveis, especialmente simétricas;
- SVD → análise geral de rank, compressão e pseudoinversa.

A estrutura do problema determina a ferramenta.

## 10. NumPy e SciPy

```python
import numpy as np
from scipy.linalg import lu

A = np.array([
    [4., 3.],
    [6., 3.]
])

P, L, U = lu(A)
print(P)
print(L)
print(U)
print(P @ L @ U)
```

Na prática, pivotamento gera uma matriz de permutação \(P\). Por isso bibliotecas podem representar a fatoração como:

\[
A=PLU
\]

ou convenções equivalentes.

## 11. Estrutura e eficiência

Decomposições são um tema central em computação científica porque exploram estrutura para melhorar:

- custo computacional;
- estabilidade numérica;
- interpretabilidade;
- reutilização de cálculos.

Esse pensamento reaparece em IA moderna: não queremos apenas uma fórmula correta; queremos uma forma de calculá-la de modo eficiente e estável.

## 12. Conexão com IA

- PCA pode ser calculado via SVD;
- regressão linear pode usar QR/SVD;
- modelos Gaussianos usam fatorações de covariância;
- compressão de matrizes usa decomposições de baixo rank;
- análise de pesos de LLMs frequentemente usa espectro e valores singulares.

## Exercícios

1. Qual decomposição está diretamente associada à eliminação de Gauss?
2. Por que QR é atraente em mínimos quadrados?
3. Explique \(A=Q\Lambda Q^T\) em três transformações geométricas.
4. Quando Cholesky pode ser usada?
5. Por que SVD é mais geral que eigendecomposition?
6. Dê um exemplo de por que reutilizar uma fatoração economiza computação.

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Fatorações LU, QR, diagonalização e SVD. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 4 — Matrix Decompositions. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 16 — SVD: a decomposição que revela rank, energia e compressão.**