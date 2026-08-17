# Aula 06 — Transformações geométricas: escala, rotação, reflexão e projeção

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 05 — Matrizes como transformações  
**Objetivo:** enxergar matrizes como operações geométricas sobre o espaço e compreender como escala, rotação, reflexão e projeção ajudam a formar a intuição necessária para redes neurais e Transformers.

## 1. A pergunta central desta aula

Na aula anterior aprendemos:

\[
\boxed{y=Wx+b}
\]

e interpretamos \(W\) como uma transformação.

Agora queremos responder:

> **O que uma transformação matricial realmente pode fazer com um vetor e com o espaço?**

Uma matriz pode aumentar, diminuir, rotacionar, refletir, projetar, deformar e combinar diferentes efeitos.

## 2. Um vetor antes e depois da transformação

Considere:

\[
x=\begin{bmatrix}2\\1\end{bmatrix}
\]

Aplicamos uma matriz \(W\) e obtemos:

\[
y=Wx
\]

A matriz decide onde o vetor irá parar. Em termos gerais:

\[
\boxed{W:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}}
\]

pode ser interpretada como uma função que transforma vetores.

## 3. A matriz transforma o espaço inteiro

Uma matriz não define uma regra apenas para um vetor específico. Ela define uma transformação para todo o espaço. Uma grade pode ficar esticada, comprimida, inclinada, girada ou refletida. A transformação de um vetor é consequência da transformação do espaço inteiro.

## 4. Matriz identidade

\[
I=\begin{bmatrix}1&0\\0&1\end{bmatrix}
\]

Para:

\[
x=\begin{bmatrix}3\\2\end{bmatrix}
\]

teremos:

\[
Ix=\begin{bmatrix}3\\2\end{bmatrix}
\]

Nada mudou. Por isso \(I\) é a transformação identidade.

## 5. Escala

Considere:

\[
W=\begin{bmatrix}2&0\\0&2\end{bmatrix}
\]

Para:

\[
x=\begin{bmatrix}1\\2\end{bmatrix}
\]

obtemos:

\[
Wx=\begin{bmatrix}2\\4\end{bmatrix}
\]

O vetor ficou duas vezes maior.

### Escala diferente em cada eixo

\[
W=\begin{bmatrix}3&0\\0&1\end{bmatrix}
\]

faz o eixo \(x\) ser multiplicado por 3 e preserva o eixo \(y\).

Para:

\[
x=\begin{bmatrix}2\\2\end{bmatrix}
\]

resultado:

\[
Wx=\begin{bmatrix}6\\2\end{bmatrix}
\]

A forma geral da matriz de escala 2D é:

\[
\boxed{S=\begin{bmatrix}s_x&0\\0&s_y\end{bmatrix}}
\]

## 6. Rotação

A matriz de rotação em 2D é:

\[
\boxed{R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}}
\]

### Rotação de 90°

Para \(\theta=90^\circ\):

\[
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\]

Aplicando em:

\[
x=\begin{bmatrix}1\\0\end{bmatrix}
\]

obtemos:

\[
Rx=\begin{bmatrix}0\\1\end{bmatrix}
\]

O vetor foi girado 90° no sentido anti-horário.

Uma rotação pura preserva comprimento e distância. Antes e depois, a norma continua igual.

### Rotação de 45°

\[
R=\begin{bmatrix}0.707&-0.707\\0.707&0.707\end{bmatrix}
\]

Aplicando em \([1,0]^T\), obtemos aproximadamente:

\[
\begin{bmatrix}0.707\\0.707\end{bmatrix}
\]

A direção mudou, mas a magnitude permaneceu aproximadamente 1.

### NumPy

```python
import numpy as np

theta = np.radians(45)

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

x = np.array([1.0, 0.0])
y = R @ x

print(y)
```

## 7. Reflexão

Uma matriz também pode espelhar o espaço.

Reflexão em relação ao eixo horizontal:

\[
W=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
\]

Para:

\[
x=\begin{bmatrix}2\\3\end{bmatrix}
\]

obtemos:

\[
Wx=\begin{bmatrix}2\\-3\end{bmatrix}
\]

### Reflexão no eixo y

\[
W=\begin{bmatrix}-1&0\\0&1\end{bmatrix}
\]

transforma \([2,3]^T\) em \([-2,3]^T\).

### Reflexão na reta y=x

\[
W=\begin{bmatrix}0&1\\1&0\end{bmatrix}
\]

transforma:

\[
\begin{bmatrix}2\\5\end{bmatrix}
\rightarrow
\begin{bmatrix}5\\2\end{bmatrix}
\]

As componentes foram trocadas.

## 8. Projeção

Considere:

\[
x=\begin{bmatrix}3\\4\end{bmatrix}
\]

Queremos projetá-lo sobre o eixo \(x\). Usamos:

\[
P=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

Então:

\[
Px=\begin{bmatrix}3\\0\end{bmatrix}
\]

A componente vertical foi descartada.

Projetar significa responder aproximadamente:

> **Quanto desse vetor existe em determinada direção ou subespaço?**

### Projeção sobre um vetor unitário

Para um vetor unitário \(u\):

\[
\boxed{proj_u(x)=(x\cdot u)u}
\]

Exemplo:

\[
x=\begin{bmatrix}3\\4\end{bmatrix},\quad u=\begin{bmatrix}1\\0\end{bmatrix}
\]

Temos:

\[
x\cdot u=3
\]

Logo:

\[
proj_u(x)=3\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}3\\0\end{bmatrix}
\]

### Matriz de projeção

Quando \(u\) é unitário:

\[
\boxed{P=uu^T}
\]

Se:

\[
u=\begin{bmatrix}1\\0\end{bmatrix}
\]

então:

\[
P=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

## 9. Projeções e IA

Em Transformers veremos:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

As matrizes \(W_Q\), \(W_K\) e \(W_V\) transformam a representação original \(X\) em novos espaços. Não são necessariamente projeções geométricas simples como a projeção ortogonal estudada aqui, mas preservam a ideia central de mapear uma representação para outra.

## 10. Transformações gerais e shear

Uma matriz pode combinar vários efeitos. Por exemplo:

\[
W=\begin{bmatrix}2&1\\0&1\end{bmatrix}
\]

pode deformar o espaço de maneira mais geral.

Um cisalhamento (shear) clássico é:

\[
W=\begin{bmatrix}1&1\\0&1\end{bmatrix}
\]

Para:

\[
x=\begin{bmatrix}2\\3\end{bmatrix}
\]

obtemos:

\[
Wx=\begin{bmatrix}5\\3\end{bmatrix}
\]

## 11. Vetores da base

No plano 2D temos:

\[
e_1=\begin{bmatrix}1\\0\end{bmatrix},\quad e_2=\begin{bmatrix}0\\1\end{bmatrix}
\]

Considere:

\[
W=\begin{bmatrix}2&1\\0&3\end{bmatrix}
\]

Então:

\[
We_1=\begin{bmatrix}2\\0\end{bmatrix}
\]

\[
We_2=\begin{bmatrix}1\\3\end{bmatrix}
\]

As colunas de \(W\) são exatamente os vetores \(We_1\) e \(We_2\).

Portanto:

> **As colunas de uma matriz mostram para onde os vetores da base são enviados.**

Qualquer vetor pode ser escrito como:

\[
x=ae_1+be_2
\]

Logo:

\[
Wx=aWe_1+bWe_2
\]

## 12. Composição de transformações

Se aplicamos primeiro \(A\) e depois \(B\):

\[
x\xrightarrow{A}Ax\xrightarrow{B}B(Ax)
\]

Então:

\[
\boxed{BAx}
\]

Em geral:

\[
\boxed{AB\neq BA}
\]

A ordem importa.

Uma regra útil: em \(ABCx\), leia da direita para a esquerda — primeiro \(C\), depois \(B\), depois \(A\).

## 13. Ligação com redes neurais

Uma rede com várias matrizes pode ser vista como uma sequência de mudanças de representação:

```text
entrada
  ↓
transformação 1
  ↓
representação 1
  ↓
transformação 2
  ↓
representação 2
  ↓
transformação 3
  ↓
representação 3
```

Se tivermos apenas:

\[
W_3W_2W_1x
\]

podemos combinar tudo em uma única transformação linear. Por isso redes neurais incluem não linearidades, por exemplo:

\[
\boxed{y=W_2ReLU(W_1x)}
\]

A não linearidade impede que todas as camadas sejam reduzidas a uma única matriz.

## 14. Reorganizando o espaço das representações

Uma forma útil de pensar em Deep Learning é:

> **as camadas aprendem transformações que reorganizam o espaço para tornar padrões mais fáceis de reconhecer ou separar.**

Dados inicialmente misturados podem, após transformações sucessivas, adquirir uma representação em que classes e estruturas se tornam mais separáveis.

## 15. Mudança de dimensionalidade

Podemos ter:

\[
W\in\mathbb{R}^{512\times768}
\]

Então:

\[
x\in\mathbb{R}^{768}\rightarrow Wx\in\mathbb{R}^{512}
\]

Também é possível expandir dimensões. O mesmo princípio vale em 768, 4096 ou dezenas de milhares de dimensões, mesmo quando não conseguimos visualizar o espaço.

## 16. NumPy: escala, rotação e composição

```python
import numpy as np

x = np.array([1.0, 1.0])

S = np.array([
    [2.0, 0.0],
    [0.0, 1.0]
])

R = np.array([
    [0.0, -1.0],
    [1.0,  0.0]
])

print("Original:", x)
print("Escala:", S @ x)
print("Rotação:", R @ x)
print("Escala + rotação:", R @ S @ x)
print("Rotação + escala:", S @ R @ x)
```

Comparar \(R S x\) com \(S R x\) mostra na prática que, em geral:

\[
RS\neq SR
\]

## 17. Ligação com Transformers

Mais adiante veremos:

\[
XW_Q,\quad XW_K,\quad XW_V
\]

seguido de:

\[
QK^T
\]

A sequência conceitual é:

```text
representação X
      ↓
transformações matriciais
      ↓
Q, K e V
      ↓
produtos escalares
      ↓
scores de atenção
```

Os conceitos das aulas anteriores se unem aqui.

## 18. Mapa conceitual

```text
vetor x
  ↓
matriz W
  ↓
transformação do espaço
  ├─ escala
  ├─ rotação
  ├─ reflexão
  ├─ projeção
  ├─ cisalhamento
  └─ mudança de dimensão
  ↓
composição de transformações
  ↓
novas representações
  ↓
redes neurais / Transformers
```

## Exercícios da Aula 06

### 1. Escala

\[
S=\begin{bmatrix}2&0\\0&3\end{bmatrix},\quad x=\begin{bmatrix}2\\1\end{bmatrix}
\]

Calcule \(Sx\) e explique geometricamente o que aconteceu.

### 2. Rotação

\[
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad x=\begin{bmatrix}1\\0\end{bmatrix}
\]

Calcule \(Rx\). Qual transformação ocorreu?

### 3. Reflexão

\[
W=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad x=\begin{bmatrix}3\\2\end{bmatrix}
\]

Calcule \(Wx\) e identifique a reflexão.

### 4. Projeção

\[
P=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad x=\begin{bmatrix}5\\7\end{bmatrix}
\]

Calcule \(Px\). O que aconteceu com a componente vertical?

### 5. Base

\[
W=\begin{bmatrix}3&2\\1&4\end{bmatrix}
\]

Determine \(We_1\) e \(We_2\) sem realizar uma multiplicação completa.

### 6. Composição

Se:

\[
y=BAx
\]

qual transformação acontece primeiro: \(A\) ou \(B\)? Explique.

### 7. Questão principal

Explique com suas palavras a frase:

> **“Uma rede neural aprende a reorganizar o espaço das representações.”**

Relacione sua resposta com transformações como \(W_1x\), \(W_2h\) e novas representações intermediárias.

## O que você precisa guardar

Não pense em matriz apenas como uma tabela de números. Pense:

\[
\boxed{W=\text{uma regra para reorganizar o espaço}}
\]

Ela pode alterar posição, direção, escala, dimensão e representação.

Em IA, os valores de \(W\) são aprendidos para produzir transformações úteis dos dados.

## Próxima aula

### Aula 07 — Determinante: área, volume, orientação e perda de informação

Vamos estudar:

\[
\boxed{\det(W)}
\]

com interpretação geométrica: mudança de área e volume, orientação, matrizes invertíveis e por que \(\det(W)=0\) indica colapso de dimensão e perda de informação.