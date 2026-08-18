# Aula 11 — Os quatro subespaços fundamentais

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 10 — Independência, base e dimensão  
**Objetivo:** compreender column space, row space, null space e left null space e conectar esses espaços a rank, soluções de sistemas e perda de informação.

## 1. Uma matriz possui uma geometria escondida

Considere:

\[
A\in\mathbb{R}^{m\times n}
\]

A matriz não é apenas uma tabela. Ela define uma transformação:

\[
A:\mathbb{R}^n\rightarrow\mathbb{R}^m
\]

Essa transformação organiza quatro subespaços fundamentais:

- espaço das colunas;
- espaço das linhas;
- núcleo (null space);
- núcleo à esquerda (left null space).

Entender esses espaços permite responder: o que a matriz consegue produzir? O que ela destrói? Quais informações são redundantes?

## 2. Column space

O **espaço das colunas** é:

\[
\operatorname{Col}(A)=\operatorname{span}(a_1,\ldots,a_n)
\]

onde \(a_i\) são as colunas de \(A\).

Como:

\[
Ax=x_1a_1+x_2a_2+\cdots+x_na_n
\]

qualquer saída \(Ax\) pertence ao espaço das colunas.

> Column space = conjunto de todas as saídas que a transformação consegue gerar.

## 3. Quando Ax=b possui solução?

A equação:

\[
Ax=b
\]

possui solução exatamente quando:

\[
\boxed{b\in\operatorname{Col}(A)}
\]

Essa é uma interpretação geométrica poderosa. Resolver um sistema é perguntar se \(b\) pode ser construído por uma combinação das colunas de \(A\).

## 4. Row space

O espaço das linhas é o span das linhas de \(A\):

\[
\operatorname{Row}(A)
\]

Ele vive em \(\mathbb{R}^n\), o mesmo espaço da entrada.

Uma propriedade fundamental é:

\[
\dim(\operatorname{Row}(A))=\dim(\operatorname{Col}(A))=rank(A)
\]

O rank é simultaneamente o número de direções independentes nas linhas e nas colunas.

## 5. Null space

O núcleo é:

\[
\operatorname{Null}(A)=\{x:Ax=0\}
\]

Ele contém as direções da entrada que desaparecem depois da transformação.

Exemplo:

\[
A=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

Para qualquer:

\[
x=\begin{bmatrix}0\\t\end{bmatrix}
\]

temos:

\[
Ax=0
\]

A direção vertical é invisível para a transformação.

## 6. Por que o null space representa perda de informação?

Se \(z\in\operatorname{Null}(A)\), então:

\[
Az=0
\]

Logo, para qualquer \(x\):

\[
A(x+z)=Ax+Az=Ax
\]

Ou seja, \(x\) e \(x+z\) produzem a mesma saída. A matriz não consegue distinguir entradas que diferem por uma direção do null space.

## 7. Left null space

O núcleo à esquerda é:

\[
\operatorname{Null}(A^T)
\]

Ele vive em \(\mathbb{R}^m\).

Geometricamente, contém vetores ortogonais ao espaço das colunas. Mais adiante, ao estudar mínimos quadrados, o vetor de erro residual aparecerá justamente nessa direção ortogonal.

## 8. Ortogonalidade entre os subespaços

Existem relações elegantes:

\[
\operatorname{Row}(A)\perp\operatorname{Null}(A)
\]

\[
\operatorname{Col}(A)\perp\operatorname{Null}(A^T)
\]

Assim, o espaço de entrada pode ser decomposto em:

\[
\mathbb{R}^n=\operatorname{Row}(A)\oplus\operatorname{Null}(A)
\]

E o espaço de saída em:

\[
\mathbb{R}^m=\operatorname{Col}(A)\oplus\operatorname{Null}(A^T)
\]

A notação \(\oplus\) indica soma direta de subespaços.

## 9. Rank-nullity revisitado

Para \(A\in\mathbb{R}^{m\times n}\):

\[
\boxed{rank(A)+nullity(A)=n}
\]

Isso divide as \(n\) dimensões da entrada em:

- direções que a matriz preserva de alguma forma;
- direções que são enviadas para zero.

## 10. Exemplo

Considere:

\[
A=\begin{bmatrix}
1&2&3\\
2&4&6
\end{bmatrix}
\]

A segunda linha é o dobro da primeira, então:

\[
rank(A)=1
\]

A possui 3 colunas, logo:

\[
nullity(A)=3-1=2
\]

Existem duas direções independentes de entrada que desaparecem.

## 11. Relação com compressão e representações

Quando uma camada realiza:

\[
z=Wx
\]

se \(W\) possui rank baixo, a representação de saída está restrita a um subespaço de baixa dimensão.

Isso ajuda a entender:

- compressão linear;
- gargalos em autoencoders;
- aproximações low-rank;
- LoRA;
- PCA/SVD.

## 12. NumPy

O NumPy calcula rank diretamente:

```python
import numpy as np

A = np.array([
    [1., 2., 3.],
    [2., 4., 6.]
])

print(np.linalg.matrix_rank(A))  # 1
```

Para obter uma base do null space, o SciPy oferece:

```python
from scipy.linalg import null_space

N = null_space(A)
print(N)
```

Cada coluna de `N` é uma direção independente anulada por \(A\).

## 13. Pergunta para IA

Se um embedding tem 1024 dimensões, mas após determinada transformação todas as saídas pertencem a um subespaço de dimensão 32, quantas direções efetivamente sobrevivem?

A resposta está no rank: no máximo 32.

## Exercícios

1. Para \(A=[[1,0],[0,0]]\), descreva Col(A) e Null(A).
2. Explique por que \(Ax=b\) só pode ser resolvido se \(b\in Col(A)\).
3. Se \(A\) é \(5\times8\) e rank 5, qual a dimensão do null space?
4. O que significa uma direção estar no left null space?
5. Explique por que Row(A) e Null(A) são ortogonais.
6. Relacione null space a perda de informação em uma camada neural linear.

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Unidades sobre os quatro subespaços fundamentais. https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 2. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 12 — Ortogonalidade, projeções, Gram-Schmidt e decomposição QR.**