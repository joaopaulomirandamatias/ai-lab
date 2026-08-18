# Aula 14 — Cálculo de autovalores, diagonalização e matrizes simétricas

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 13 — Autovalores e autovetores  
**Objetivo:** aprender a calcular autovalores/autovetores, entender diagonalização e reconhecer por que matrizes simétricas são tão importantes em IA.

## 1. Partindo de Av=λv

Queremos vetores não nulos \(v\) que satisfaçam:

\[
Av=\lambda v
\]

Reorganize:

\[
(A-\lambda I)v=0
\]

Para existir uma solução não nula, a matriz \(A-\lambda I\) precisa ser singular:

\[
\boxed{\det(A-\lambda I)=0}
\]

Essa é a **equação característica**.

## 2. Exemplo 2×2

Considere:

\[
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
\]

Então:

\[
A-\lambda I=
\begin{bmatrix}2-\lambda&1\\1&2-\lambda\end{bmatrix}
\]

O determinante é:

\[
(2-\lambda)^2-1=0
\]

Logo:

\[
\lambda^2-4\lambda+3=0
\]

Portanto:

\[
\boxed{\lambda_1=3,\quad\lambda_2=1}
\]

## 3. Encontrando os autovetores

Para \(\lambda=3\):

\[
(A-3I)v=0
\]

\[
\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}=0
\]

Isso exige:

\[
x=y
\]

Logo podemos usar:

\[
v_1=\begin{bmatrix}1\\1\end{bmatrix}
\]

Para \(\lambda=1\):

\[
x=-y
\]

Então:

\[
v_2=\begin{bmatrix}1\\-1\end{bmatrix}
\]

## 4. Diagonalização

Se uma matriz possui uma base completa de autovetores, podemos construir:

\[
P=[v_1\ v_2\ \cdots\ v_n]
\]

E:

\[
D=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)
\]

Então:

\[
\boxed{A=PDP^{-1}}
\]

Na base de autovetores, a transformação vira simplesmente uma matriz diagonal.

## 5. Por que isso é útil?

Potências ficam fáceis:

\[
A^k=(PDP^{-1})^k=PD^kP^{-1}
\]

Como \(D\) é diagonal:

\[
D^k=\operatorname{diag}(\lambda_1^k,\ldots,\lambda_n^k)
\]

Isso simplifica análise de sistemas dinâmicos e métodos iterativos.

## 6. Nem toda matriz é diagonalizável

Para diagonalizar uma matriz \(n\times n\), precisamos de \(n\) autovetores linearmente independentes.

Algumas matrizes não possuem autovetores independentes suficientes. Esse detalhe mostra por que “encontrar autovalores” e “diagonalizar” não são exatamente a mesma coisa.

## 7. Matrizes simétricas

Se:

\[
A=A^T
\]

então o teorema espectral garante uma estrutura muito melhor. Podemos escrever:

\[
\boxed{A=Q\Lambda Q^T}
\]

onde:

- \(Q\) possui colunas ortonormais;
- \(\Lambda\) é diagonal com autovalores reais;
- \(Q^{-1}=Q^T\).

Essa é uma das decomposições mais importantes da Álgebra Linear.

## 8. Covariância

Matrizes de covariância são simétricas. Por isso PCA pode encontrar uma base ortonormal de direções principais.

Se:

\[
\Sigma q_i=\lambda_i q_i
\]

então:

- \(q_i\): direção principal;
- \(\lambda_i\): variância nessa direção.

## 9. Matrizes positivas semidefinidas

Uma matriz simétrica é positiva semidefinida se:

\[
x^TAx\ge0
\]

para todo \(x\).

Matrizes de covariância possuem essa propriedade. Seus autovalores são não negativos.

Isso conecta autovalores à curvatura, variância e otimização.

## 10. Ordenação espectral

Frequentemente ordenamos:

\[
\lambda_1\ge\lambda_2\ge\cdots
\]

Em PCA, manter os maiores autovalores significa preservar as direções de maior variância.

Em outros contextos, autovalores dominantes ajudam a identificar modos principais de uma transformação.

## 11. NumPy: use eigh quando a matriz é simétrica

```python
import numpy as np

A = np.array([
    [2., 1.],
    [1., 2.]
])

values, Q = np.linalg.eigh(A)

print(values)
print(Q)
print(Q.T @ Q)
print(Q @ np.diag(values) @ Q.T)
```

`np.linalg.eigh` aproveita a simetria e retorna autovalores reais com melhor adequação ao problema do que uma rotina geral `eig`.

## 12. Cuidados numéricos

Em aplicações reais, raramente calculamos raízes do polinômio característico manualmente para matrizes grandes. Algoritmos numéricos especializados são usados.

A equação característica é essencial para a teoria; bibliotecas numéricas são essenciais para a prática.

## 13. Conexão com IA

- PCA usa estrutura espectral da covariância;
- Hessianas em otimização são simétricas sob condições regulares e seus autovalores descrevem curvatura;
- métodos espectrais aparecem em grafos e clustering;
- decomposições de matrizes de pesos ajudam a analisar compressibilidade e estabilidade.

## Exercícios

1. Calcule os autovalores de \(diag(5,2)\).
2. Para \(A=[[2,1],[1,2]]\), verifique que \([1,1]^T\) é autovetor.
3. O que é necessário para uma matriz \(n\times n\) ser diagonalizável?
4. Por que matrizes simétricas são especiais?
5. Explique \(A=Q\Lambda Q^T\) geometricamente.
6. Por que autovalores de uma matriz de covariância não podem ser negativos?

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Eigenvalues, symmetric matrices e positive definite matrices. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 4. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 15 — Decomposições matriciais: LU, QR, espectral e por que decompor matrizes.**