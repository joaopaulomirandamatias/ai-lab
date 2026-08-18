# Aula 09 — Sistemas lineares, eliminação de Gauss e posto da matriz

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisitos:** matrizes, determinante e matriz inversa  
**Objetivo:** entender sistemas lineares como problemas matriciais, aprender eliminação de Gauss e compreender o **posto (rank)** como medida da quantidade de informação independente em uma matriz.

## 1. O problema central

Na aula anterior estudamos:

\[
Ax=b
\]

e vimos que, quando \(A\) é quadrada e invertível:

\[
x=A^{-1}b
\]

Mas em Machine Learning é comum a matriz não ser quadrada, ser singular ou possuir muito mais equações que incógnitas. Por exemplo, podemos ter \(1\,000\,000\) observações e apenas 50 variáveis:

\[
A\in\mathbb{R}^{1\,000\,000\times 50}
\]

Nesse caso não existe inversa tradicional. Precisamos de ferramentas mais gerais.

## 2. Sistema linear como matriz

Considere:

\[
\begin{cases}
2x+y=5\\
x+y=3
\end{cases}
\]

Podemos escrever:

\[
Ax=b
\]

com:

\[
A=\begin{bmatrix}2&1\\1&1\end{bmatrix},\quad
x=\begin{bmatrix}x\\y\end{bmatrix},\quad
b=\begin{bmatrix}5\\3\end{bmatrix}
\]

Cada linha de \(A\) representa uma equação. Geometricamente, em 2D cada equação linear representa uma reta; a solução é o ponto em que as retas se cruzam.

Um sistema pode ter:

- uma solução única;
- infinitas soluções;
- nenhuma solução.

## 3. Matriz aumentada

Para resolver o sistema sem calcular a inversa, usamos a **eliminação de Gauss**.

Juntamos \(A\) e \(b\):

\[
\left[\begin{array}{cc|c}
2&1&5\\
1&1&3
\end{array}\right]
\]

Essa é a matriz aumentada.

## 4. Operações elementares de linha

Podemos realizar três operações que preservam o conjunto de soluções:

1. Trocar duas linhas:

\[
L_1\leftrightarrow L_2
\]

2. Multiplicar uma linha por um escalar não nulo:

\[
L_1\leftarrow cL_1
\]

3. Somar um múltiplo de uma linha a outra:

\[
L_2\leftarrow L_2+cL_1
\]

## 5. Exemplo de eliminação de Gauss

Começamos com:

\[
\left[\begin{array}{cc|c}
2&1&5\\
1&1&3
\end{array}\right]
\]

Troque as linhas:

\[
\left[\begin{array}{cc|c}
1&1&3\\
2&1&5
\end{array}\right]
\]

Depois:

\[
L_2\leftarrow L_2-2L_1
\]

resultando em:

\[
\left[\begin{array}{cc|c}
1&1&3\\
0&-1&-1
\end{array}\right]
\]

Da segunda linha:

\[
y=1
\]

Substituindo na primeira:

\[
x=2
\]

Logo:

\[
\boxed{x=2,\;y=1}
\]

## 6. Forma escalonada, pivôs e Gauss-Jordan

O objetivo da eliminação é obter uma forma como:

\[
\begin{bmatrix}
*&*&*\\
0&*&*\\
0&0&*
\end{bmatrix}
\]

Essa é a **forma escalonada**.

O primeiro elemento não nulo de cada linha é um **pivô**.

Se continuarmos eliminando também os elementos acima dos pivôs, chegamos à **forma escalonada reduzida** (RREF — Reduced Row Echelon Form). A eliminação de Gauss normalmente para na forma escalonada; Gauss-Jordan continua até a forma reduzida.

## 7. Rank: quantas direções independentes existem?

Considere:

\[
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\]

A segunda linha é duas vezes a primeira. Aplicando:

\[
L_2\leftarrow L_2-2L_1
\]

obtemos:

\[
\begin{bmatrix}1&2\\0&0\end{bmatrix}
\]

Existe apenas um pivô. Portanto:

\[
\boxed{rank(A)=1}
\]

Uma forma prática de pensar é:

\[
\boxed{rank(A)=\text{número de pivôs}}
\]

O rank também corresponde ao número de linhas ou colunas linearmente independentes.

## 8. Full rank e dimensionalidade

Para:

\[
A\in\mathbb{R}^{m\times n}
\]

sempre:

\[
\boxed{rank(A)\leq\min(m,n)}
\]

Uma matriz possui **full rank** quando o rank é o máximo possível.

Exemplo:

\[
A\in\mathbb{R}^{5\times3},\quad rank(A)=3
\]

possui full column rank.

Para matriz quadrada:

\[
A\in\mathbb{R}^{n\times n}
\]

vale:

\[
\boxed{\det(A)\neq0\iff rank(A)=n}
\]

Logo:

```text
det(A) ≠ 0
     ↓
full rank
     ↓
invertível
```

Enquanto:

```text
det(A) = 0
     ↓
rank(A) < n
     ↓
dependência linear
     ↓
sem inversa
```

## 9. Rank como quantidade de informação independente

Essa interpretação é central para IA:

> **O rank mede quantas direções independentes de informação a transformação consegue preservar ou representar.**

Por exemplo, uma matriz \(100\times100\) pode ter rank 5. Apesar de possuir 100 componentes na saída, sua imagem vive em um subespaço de no máximo 5 dimensões independentes.

Isso conecta rank a:

- redundância;
- compressão;
- PCA;
- SVD;
- aproximações low-rank;
- LoRA.

## 10. Primeira conexão com LoRA

Uma atualização de pesos pode ser escrita como:

\[
\Delta W=BA
\]

com uma dimensão interna pequena \(r\):

\[
A\in\mathbb{R}^{r\times k},\qquad
B\in\mathbb{R}^{d\times r}
\]

com:

\[
r\ll d,k
\]

O rank da atualização é limitado por \(r\). Essa é a origem matemática do nome **Low-Rank Adaptation**.

## 11. Sistemas determinados, superdeterminados e subdeterminados

### Determinado

Mesmo número de equações e incógnitas, com informação independente suficiente. Pode haver solução única.

### Superdeterminado

Mais equações que incógnitas:

\[
A\in\mathbb{R}^{1000\times10}
\]

Isso aparece constantemente em Machine Learning.

### Subdeterminado

Menos equações que incógnitas:

\[
A\in\mathbb{R}^{3\times10}
\]

Se consistente, normalmente existem infinitas soluções e variáveis livres.

## 12. Mínimos quadrados

Em regressão podemos ter:

\[
Xw=y
\]

com milhares de observações. Em geral não existe um \(w\) que satisfaça todas as equações exatamente. Então buscamos:

\[
\boxed{\min_w\|Xw-y\|^2}
\]

Esse é o problema de **mínimos quadrados (least squares)**.

## 13. Variáveis pivô e variáveis livres

Depois do escalonamento, colunas com pivô correspondem a **variáveis pivô**; as demais são **variáveis livres**.

Exemplo:

\[
x+2y+3z=5
\]

Podemos escrever:

\[
x=5-2y-3z
\]

Aqui \(y\) e \(z\) são livres.

## 14. Detectando inconsistência

Considere:

\[
\left[\begin{array}{cc|c}
1&2&3\\
0&0&5
\end{array}\right]
\]

A segunda linha significa:

\[
0=5
\]

Isso é impossível. O sistema não possui solução.

Para \(Ax=b\), compare:

\[
rank(A)
\]

com:

\[
rank([A|b])
\]

O sistema possui solução se:

\[
\boxed{rank(A)=rank([A|b])}
\]

Se:

\[
rank(A)=rank([A|b])=n
\]

há solução única. Se:

\[
rank(A)=rank([A|b])<n
\]

há infinitas soluções.

## 15. Null space

Considere:

\[
Ax=0
\]

O conjunto de todas as soluções é o **núcleo** ou **null space**.

Exemplo:

\[
A=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

Para:

\[
x=\begin{bmatrix}0\\7\end{bmatrix}
\]

obtemos:

\[
Ax=0
\]

A direção vertical foi completamente destruída pela transformação.

## 16. Rank-nullity theorem

Se \(A\) possui \(n\) colunas:

\[
\boxed{rank(A)+nullity(A)=n}
\]

Isso fornece uma interpretação poderosa:

> a dimensão de entrada se divide entre direções preservadas e direções enviadas ao zero.

Exemplo:

\[
A\in\mathbb{R}^{5\times8},\quad rank(A)=5
\]

Logo:

\[
nullity(A)=8-5=3
\]

## 17. Rank, embeddings e matrizes de pesos

Um embedding possuir 768 dimensões não implica que seus dados usem todas essas direções de forma igualmente importante. A dimensionalidade efetiva pode ser muito menor.

Da mesma forma, uma matriz de pesos:

\[
W\in\mathbb{R}^{4096\times4096}
\]

pode possuir estrutura aproximável por rank menor:

\[
W\approx UV^T
\]

Essa ideia está por trás de técnicas de compressão e adaptação eficiente.

## 18. NumPy

### Resolver um sistema

```python
import numpy as np

A = np.array([
    [2.0, 1.0],
    [1.0, 1.0]
])

b = np.array([5.0, 3.0])

x = np.linalg.solve(A, b)
print(x)
```

Resultado:

```text
[2. 1.]
```

### Calcular rank

```python
A = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])

print(np.linalg.matrix_rank(A))
```

Resultado:

```text
1
```

### Mínimos quadrados

```python
x, residuals, rank, s = np.linalg.lstsq(
    A,
    b,
    rcond=None
)
```

A função procura uma solução que minimize:

\[
\|Ax-b\|^2
\]

## 19. Estabilidade numérica e pivotamento

Em teoria, um pivô muito pequeno ainda é diferente de zero. Em computação, porém, dividir por um valor como \(10^{-12}\) pode amplificar erros.

Por isso algoritmos reais usam estratégias como **pivotamento parcial**, escolhendo pivôs numericamente mais seguros.

Essa é uma conexão importante entre Álgebra Linear e Computação Numérica.

## 20. Conectando com \(y=Wx+b\)

Se:

\[
rank(W)
\]

é pequeno, a transformação possui dimensionalidade efetiva menor que o número aparente de componentes.

Por exemplo:

\[
W\in\mathbb{R}^{100\times100},\quad rank(W)=10
\]

A saída tem 100 posições, mas vive em um subespaço de no máximo 10 direções independentes.

## 21. Caminho para PCA e SVD

Se dados com 500 características estiverem aproximadamente concentrados em um subespaço de 20 dimensões, técnicas como PCA podem identificar essas direções principais.

Já a Singular Value Decomposition escreve:

\[
\boxed{A=U\Sigma V^T}
\]

e revela:

- rank;
- direções importantes;
- quantidade de informação por direção;
- possibilidades de compressão.

## 22. Mapa mental

```text
Ax = b
  ↓
matriz aumentada
  ↓
eliminação de Gauss
  ↓
forma escalonada
  ↓
pivôs
  ↓
rank
  │
  ├─ rank completo → informação independente
  └─ rank reduzido → redundância / compressão / perda dimensional
```

## Exercícios da Aula 09

1. Resolva por eliminação:

\[
\begin{cases}
x+y=5\\
2x+y=8
\end{cases}
\]

2. Para:

\[
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\]

qual é \(rank(A)\)? Explique sem software.

3. Para:

\[
A=\begin{bmatrix}1&0&2\\0&1&3\\0&0&0\end{bmatrix}
\]

quantos pivôs existem e qual é o rank?

4. Uma matriz \(7\times4\) pode ter rank máximo igual a quanto?

5. Uma matriz \(1000\times10\) representando 1000 equações e 10 incógnitas é determinada, superdeterminada ou subdeterminada?

6. O sistema representado por:

\[
\left[\begin{array}{cc|c}1&2&4\\0&0&3\end{array}\right]
\]

possui solução?

7. Se:

\[
A\in\mathbb{R}^{5\times8},\quad rank(A)=5
\]

qual é a nullity?

8. Explique o que significa dizer que duas colunas são linearmente dependentes.

9. Se:

\[
W\in\mathbb{R}^{1000\times1000},\quad rank(W)=20
\]

qual é a interpretação sobre a dimensionalidade efetiva da transformação?

10. Explique a diferença entre dimensão da matriz e rank. Exemplo:

\[
A\in\mathbb{R}^{100\times100},\quad rank(A)=5
\]

O que isso significa geometricamente e em termos de informação?

## Mini laboratório

```python
import numpy as np

A = np.array([
    [1., 2.],
    [2., 4.]
])

B = np.array([
    [1., 2.],
    [3., 4.]
])

print("Rank A:", np.linalg.matrix_rank(A))
print("Rank B:", np.linalg.matrix_rank(B))
print("Det A:", np.linalg.det(A))
print("Det B:", np.linalg.det(B))
```

Antes de executar, tente prever os resultados e relacione determinante e rank para matrizes quadradas.

## O que você precisa sair sabendo

Ao olhar para:

\[
Ax=b
\]

pense: **estou procurando uma entrada que, transformada por \(A\), produza \(b\)**.

Eliminação de Gauss reorganiza o sistema para revelar sua estrutura. Os pivôs evidenciam as direções independentes. E:

\[
\boxed{rank(A)=\text{quantidade de dimensões independentes preservadas pela matriz}}
\]

## Próxima aula

### Aula 10 — Independência linear, base, span e dimensão

Vamos aprofundar:

- independência linear;
- span;
- base;
- dimensão;
- vetores redundantes;
- mudança de base;
- relação entre base e rank.

Esse conteúdo prepara o caminho para autovalores, autovetores, PCA e SVD.
