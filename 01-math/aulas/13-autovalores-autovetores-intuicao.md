# Aula 13 — Autovalores e autovetores: direções especiais de uma transformação

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** transformações, base e independência linear  
**Objetivo:** desenvolver uma intuição geométrica sólida para autovalores e autovetores antes de aprender o cálculo algébrico.

## 1. A pergunta central

Uma matriz geralmente muda tanto o tamanho quanto a direção de um vetor:

\[
x\rightarrow Ax
\]

Mas algumas direções especiais não mudam de direção. A matriz apenas estica, comprime ou inverte o vetor.

Se:

\[
Av=\lambda v
\]

então \(v\) é um **autovetor** e \(\lambda\) seu **autovalor**.

## 2. Interpretação geométrica

A equação:

\[
Av=\lambda v
\]

significa:

- \(A\) aplica a transformação;
- o resultado permanece na mesma reta de \(v\);
- \(\lambda\) informa quanto a direção foi escalada.

Casos:

- \(\lambda>1\): expansão;
- \(0<\lambda<1\): contração;
- \(\lambda<0\): inversão de sentido + escala;
- \(\lambda=0\): direção colapsada ao zero.

## 3. Exemplo diagonal

Considere:

\[
A=\begin{bmatrix}3&0\\0&2\end{bmatrix}
\]

Para:

\[
e_1=\begin{bmatrix}1\\0\end{bmatrix}
\]

temos:

\[
Ae_1=\begin{bmatrix}3\\0\end{bmatrix}=3e_1
\]

Então \(e_1\) é autovetor com autovalor 3.

Da mesma forma:

\[
Ae_2=2e_2
\]

Logo \(e_2\) é autovetor com autovalor 2.

## 4. Nem todo vetor é autovetor

Considere:

\[
x=\begin{bmatrix}1\\1\end{bmatrix}
\]

Então:

\[
Ax=\begin{bmatrix}3\\2\end{bmatrix}
\]

O resultado não é múltiplo de \([1,1]^T\). A direção mudou. Portanto \(x\) não é autovetor.

## 5. Autovetores descrevem eixos naturais da transformação

Se conseguirmos encontrar uma base formada por autovetores, a transformação fica extremamente simples nessa base: cada coordenada é apenas multiplicada pelo autovalor correspondente.

Essa é a motivação por trás da diagonalização.

```text
base original
    ↓ mudança de base
base de autovetores
    ↓
apenas escalas independentes
```

## 6. Dinâmica repetida

Considere aplicar \(A\) várias vezes:

\[
x,\ Ax,\ A^2x,\ A^3x,\ldots
\]

Se \(x\) puder ser escrito como combinação de autovetores:

\[
x=c_1v_1+c_2v_2
\]

então:

\[
A^kx=c_1\lambda_1^kv_1+c_2\lambda_2^kv_2
\]

O autovalor de maior magnitude tende a dominar o comportamento quando \(k\) cresce.

Essa ideia aparece em processos iterativos, cadeias de Markov, métodos de potência e análise de estabilidade.

## 7. Exemplo

Se:

\[
\lambda_1=2,\qquad \lambda_2=0.5
\]

então depois de muitas aplicações:

\[
2^k
\]

cresce enquanto:

\[
0.5^k
\]

tende a zero. A direção associada a \(\lambda_1\) passa a dominar.

## 8. Relação com PCA

Em PCA, buscamos direções que expliquem a maior variância dos dados. Quando usamos a matriz de covariância, essas direções são seus autovetores e os autovalores medem a variância capturada em cada direção.

Portanto:

```text
autovetor → direção principal
autovalor → importância/variância nessa direção
```

Mais adiante veremos por que SVD costuma ser uma forma numericamente conveniente de chegar ao mesmo objetivo.

## 9. Matrizes simétricas

Uma matriz simétrica satisfaz:

\[
A=A^T
\]

Ela possui propriedades especialmente agradáveis:

- autovalores reais;
- autovetores associados a autovalores distintos podem ser escolhidos ortogonais;
- existe uma base ortonormal de autovetores.

Isso é fundamental porque matrizes de covariância são simétricas.

## 10. Autovalor zero

Se:

\[
Av=0
\]

para algum \(v\neq0\), então:

\[
\lambda=0
\]

é um autovalor. Isso significa que existe uma direção não nula no null space. Logo a matriz é singular:

\[
\det(A)=0
\]

Temos uma nova ligação entre autovalores, determinante e invertibilidade.

## 11. Produto dos autovalores

Para uma matriz quadrada, contando multiplicidades:

\[
\det(A)=\prod_i\lambda_i
\]

Então se algum autovalor é zero, o determinante é zero.

A soma dos autovalores também se relaciona ao traço:

\[
\operatorname{tr}(A)=\sum_i\lambda_i
\]

Essas identidades serão úteis para construir intuição.

## 12. NumPy

```python
import numpy as np

A = np.array([
    [3., 0.],
    [0., 2.]
])

values, vectors = np.linalg.eig(A)

print("autovalores:", values)
print("autovetores:")
print(vectors)
```

As colunas de `vectors` são os autovetores retornados.

Teste:

```python
v = vectors[:, 0]
lmbda = values[0]
print(A @ v)
print(lmbda * v)
```

Os dois resultados devem coincidir numericamente.

## 13. Onde isso aparece em IA?

- PCA e análise de covariância;
- métodos espectrais;
- embeddings espectrais de grafos;
- análise de estabilidade de sistemas;
- compreensão de decomposições matriciais;
- métodos iterativos e PageRank como exemplo clássico de autovetor dominante.

## Exercícios

1. Para \(A=diag(4,1)\), quais são os autovetores óbvios?
2. O que significa geometricamente \(\lambda=-2\)?
3. Por que um autovalor zero implica não invertibilidade?
4. Se \(x=c_1v_1+c_2v_2\), escreva \(A^3x\).
5. Explique por que autovetores podem ser vistos como “direções naturais” de uma transformação.
6. Qual a ligação conceitual entre autovalores e PCA?

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Tópicos: eigenvalues, eigenvectors e diagonalization. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 4. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 14 — Cálculo de autovalores, diagonalização e matrizes simétricas.**