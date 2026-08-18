# Aula 08 — Matriz inversa: desfazendo transformações

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 07 — Determinante  
**Objetivo:** compreender matriz inversa como a operação que desfaz uma transformação, conectando invertibilidade, determinante, sistemas lineares, perda de informação e computação numérica.

## 1. A pergunta central

Nas últimas aulas construímos a visão:

\[
x \rightarrow Ax
\]

Uma matriz \(A\) transforma uma representação. Agora surge a pergunta:

> **É possível desfazer essa transformação e recuperar o vetor original?**

Se:

\[
y=Ax
\]

conhecemos \(A\) e \(y\), e queremos recuperar \(x\). Quando existe uma transformação que desfaz \(A\), ela é chamada de matriz inversa:

\[
\boxed{A^{-1}}
\]

## 2. A ideia mais simples

Se uma transformação numérica faz:

\[
x\rightarrow2x
\]

podemos desfazê-la multiplicando por \(1/2\). Em matrizes queremos:

\[
\boxed{A^{-1}A=I}
\]

onde \(I\) é a matriz identidade.

Conceitualmente:

```text
x
↓ A
Ax
↓ A⁻¹
x
```

## 3. Exemplo geométrico simples

Considere:

\[
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\]

Ela dobra a componente horizontal e triplica a vertical. A inversa deve fazer o contrário:

\[
A^{-1}=\begin{bmatrix}1/2&0\\0&1/3\end{bmatrix}
\]

Assim:

\[
A^{-1}A=I
\]

Se:

\[
x=\begin{bmatrix}4\\6\end{bmatrix}
\]

então:

\[
Ax=\begin{bmatrix}8\\18\end{bmatrix}
\]

Aplicando \(A^{-1}\), recuperamos:

\[
\begin{bmatrix}4\\6\end{bmatrix}
\]

## 4. Definição formal

Uma matriz quadrada \(A\) é invertível se existe uma matriz \(A^{-1}\) tal que:

\[
\boxed{A^{-1}A=AA^{-1}=I}
\]

## 5. Nem toda matriz possui inversa

Na Aula 07 vimos:

\[
\det(A)=0
\]

significa que a transformação colapsou alguma dimensão e perdeu informação. Para matrizes quadradas:

\[
\boxed{A\text{ invertível}\iff\det(A)\neq0}
\]

## 6. Exemplo sem inversa: projeção

Considere:

\[
P=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

Ela transforma:

\[
\begin{bmatrix}x\\y\end{bmatrix}
\rightarrow
\begin{bmatrix}x\\0\end{bmatrix}
\]

Assim:

\[
\begin{bmatrix}3\\2\end{bmatrix}
\rightarrow
\begin{bmatrix}3\\0\end{bmatrix}
\]

mas também:

\[
\begin{bmatrix}3\\500\end{bmatrix}
\rightarrow
\begin{bmatrix}3\\0\end{bmatrix}
\]

Duas entradas diferentes geraram a mesma saída. A informação da componente vertical foi destruída, logo \(P^{-1}\) não existe. De fato:

\[
\det(P)=0
\]

## 7. Invertibilidade e preservação de informação

Uma transformação invertível pode esticar, girar, refletir ou deformar o espaço, mas preserva informação suficiente para existir um caminho de volta.

Uma transformação singular pode fazer:

```text
plano → linha
espaço 3D → plano
```

Nesse caso, diferenças entre vetores desaparecem.

## 8. Fórmula da inversa 2×2

Para:

\[
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
\]

com:

\[
ad-bc\neq0
\]

vale:

\[
\boxed{A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}}
\]

A fórmula é útil didaticamente. Em matrizes maiores, normalmente usamos algoritmos numéricos.

## 9. Exemplo completo

Considere:

\[
A=\begin{bmatrix}2&1\\1&1\end{bmatrix}
\]

Primeiro:

\[
\det(A)=2(1)-1(1)=1
\]

Logo, a matriz é invertível. Sua inversa é:

\[
\boxed{A^{-1}=\begin{bmatrix}1&-1\\-1&2\end{bmatrix}}
\]

Verificando:

\[
AA^{-1}=\begin{bmatrix}1&0\\0&1\end{bmatrix}=I
\]

## 10. Ligação com sistemas lineares

Considere:

\[
Ax=b
\]

Se \(A^{-1}\) existe:

\[
A^{-1}Ax=A^{-1}b
\]

Logo:

\[
\boxed{x=A^{-1}b}
\]

Essa é uma forma conceitual de resolver sistemas lineares.

## 11. Exemplo de sistema linear

Considere:

\[
\begin{cases}
2x+y=5\\
x+y=3
\end{cases}
\]

Podemos escrever:

\[
\begin{bmatrix}2&1\\1&1\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}
=
\begin{bmatrix}5\\3\end{bmatrix}
\]

Usando a inversa:

\[
\begin{bmatrix}x\\y\end{bmatrix}
=
\begin{bmatrix}1&-1\\-1&2\end{bmatrix}
\begin{bmatrix}5\\3\end{bmatrix}
=
\begin{bmatrix}2\\1\end{bmatrix}
\]

Portanto:

\[
\boxed{x=2,\quad y=1}
\]

## 12. Interpretação geométrica de Ax=b

A equação:

\[
Ax=b
\]

pode ser lida como:

> Qual vetor \(x\), quando transformado por \(A\), produz \(b\)?

A inversa percorre a transformação no sentido contrário:

```text
b
↓ A⁻¹
x
```

## 13. Pré-imagem e transformação um-para-um

Quando \(A\) é invertível, cada saída possui exatamente uma entrada correspondente. Se:

\[
Ax_1=Ax_2
\]

então, aplicando \(A^{-1}\):

\[
x_1=x_2
\]

## 14. Teorema da Matriz Invertível — prévia

Para uma matriz quadrada \(A\), várias afirmações são equivalentes:

\[
A^{-1}\text{ existe}
\]

\[
\Updownarrow
\]

\[
\det(A)\neq0
\]

\[
\Updownarrow
\]

colunas linearmente independentes

\[
\Updownarrow
\]

\[
Ax=0
\]

possui apenas a solução \(x=0\)

\[
\Updownarrow
\]

\[
Ax=b
\]

possui solução única para todo \(b\).

## 15. Inversa de transformações conhecidas

### Rotação

\[
\boxed{R(\theta)^{-1}=R(-\theta)}
\]

Para matrizes de rotação:

\[
\boxed{R^{-1}=R^T}
\]

Essas matrizes pertencem à classe das matrizes ortogonais.

### Reflexão

Para:

\[
F=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
\]

vale:

\[
\boxed{F^{-1}=F}
\]

porque aplicar a reflexão duas vezes devolve o vetor original.

### Escala

Se:

\[
S=\begin{bmatrix}2&0\\0&5\end{bmatrix}
\]

então:

\[
S^{-1}=\begin{bmatrix}1/2&0\\0&1/5\end{bmatrix}
\]

## 16. Composição e inversa

Se:

\[
y=BAx
\]

primeiro aplicamos \(A\) e depois \(B\). Para desfazer, precisamos inverter a ordem:

\[
\boxed{(BA)^{-1}=A^{-1}B^{-1}}
\]

Uma analogia útil: vestir camiseta e depois casaco; para desfazer, tiramos primeiro o casaco e depois a camiseta.

Também:

\[
\boxed{(A^{-1})^{-1}=A}
\]

## 17. Determinante da inversa

Como:

\[
AA^{-1}=I
\]

então:

\[
\det(A)\det(A^{-1})=1
\]

logo:

\[
\boxed{\det(A^{-1})=\frac{1}{\det(A)}}
\]

Se \(A\) multiplica áreas por 6, a inversa as multiplica por \(1/6\).

## 18. NumPy: calculando a inversa

```python
import numpy as np

A = np.array([
    [2.0, 1.0],
    [1.0, 1.0]
])

A_inv = np.linalg.inv(A)

print(A_inv)
print(A @ A_inv)
```

Por causa de ponto flutuante, o resultado pode apresentar valores extremamente pequenos no lugar de zeros exatos.

## 19. Regra importante em programação científica

Embora matematicamente possamos escrever:

\[
x=A^{-1}b
\]

normalmente não calculamos explicitamente a inversa apenas para resolver um sistema.

Evite, quando o objetivo for somente resolver \(Ax=b\):

```python
x = np.linalg.inv(A) @ b
```

Prefira:

```python
x = np.linalg.solve(A, b)
```

Algoritmos de solução direta costumam ser mais eficientes e numericamente mais estáveis.

## 20. Exemplo com solve

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

## 21. Matriz singular

Considere:

```python
A = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])
```

Temos:

\[
\det(A)=0
\]

`np.linalg.inv(A)` falha porque a matriz é singular.

## 22. Sistemas sem solução única

Para matrizes singulares, dependendo de \(b\), podemos ter infinitas soluções ou nenhuma solução.

### Infinitas soluções

\[
\begin{cases}
x+2y=4\\
2x+4y=8
\end{cases}
\]

A segunda equação repete a primeira.

### Nenhuma solução

\[
\begin{cases}
x+2y=4\\
2x+4y=10
\end{cases}
\]

As duas equações são incompatíveis.

## 23. Relação com independência linear

Para uma matriz quadrada:

\[
\boxed{\text{colunas independentes}\iff A\text{ invertível}}
\]

Se as colunas são dependentes:

\[
\det(A)=0
\]

## 24. Próxima grande ideia: posto da matriz

Nem toda matriz é quadrada. Para uma matriz \(100\times50\), uma inversa comum não existe. Precisamos de um conceito mais geral:

\[
\boxed{\text{rank (posto)}}
\]

O rank mede quantas direções independentes realmente existem na transformação. Ele será importante para compressão, PCA, SVD, embeddings e redundância.

## 25. Matrizes quase singulares e condicionamento

Uma matriz pode ter \(\det(A)\neq0\) e ainda estar muito próxima de ser singular. Se duas colunas são quase paralelas, a transformação comprime fortemente alguma direção. Ao inverter, pequenos erros podem ser amplificados.

Isso nos leva ao:

\[
\boxed{\text{condition number}}
\]

ou número de condição, que mede a sensibilidade da solução a pequenos erros nos dados.

## 26. Por que isso importa para IA?

Modelos de IA trabalham com grandes matrizes, otimização, covariâncias, transformações, sistemas lineares e decomposições. Entender inversa e invertibilidade ajuda a compreender:

- quando informação pode ser recuperada;
- quando dimensões foram colapsadas;
- quando um sistema possui solução única;
- por que certas transformações são instáveis;
- como matrizes representam relações entre variáveis.

## 27. Inversas em estatística e regressão

Em estatística aparece, por exemplo:

\[
\Sigma^{-1}
\]

na distância de Mahalanobis.

Em regressão linear aparece a forma conceitual:

\[
\hat\beta=(X^TX)^{-1}X^Ty
\]

Mais adiante veremos por que, computacionalmente, decomposições como QR e SVD podem ser preferíveis a calcular explicitamente uma inversa.

## 28. Pseudoinversa

Quando uma matriz é retangular ou singular, pode ser útil uma generalização:

\[
\boxed{A^+}
\]

chamada pseudoinversa de Moore-Penrose.

Ela aparece em mínimos quadrados, sistemas superdeterminados, sistemas subdeterminados e SVD.

## 29. Inversão e representações em IA

Se:

\[
z=Wx
\]

com \(W\) invertível:

```text
x
↓ W
z
↓ W⁻¹
x
```

Em princípio a representação original pode ser recuperada.

Se \(W\) reduz dimensionalidade e destrói informação, a reconstrução perfeita deixa de ser possível sem estrutura adicional.

Um autoencoder, por exemplo, aprende uma reconstrução aproximada:

```text
entrada
↓ encoder
representação comprimida
↓ decoder
reconstrução
```

O decoder não precisa ser a inversa matemática exata do encoder.

## 30. Mapa mental da Aula 08

```text
A
↓
transformação

se det(A) ≠ 0
      ↓
A é invertível
      ↓
existe A⁻¹
      ↓
A⁻¹A = I
      ↓
podemos desfazer a transformação
      ↓
Ax = b
      ↓
x = A⁻¹b

se det(A) = 0
      ↓
dimensão colapsada
      ↓
informação perdida
      ↓
sem inversa
```

## Exercícios da Aula 08

### Questão 1

Considere:

\[
A=\begin{bmatrix}2&0\\0&4\end{bmatrix}
\]

Qual deve ser \(A^{-1}\)? Explique usando a interpretação de escala.

### Questão 2

Para:

\[
A=\begin{bmatrix}2&1\\1&1\end{bmatrix}
\]

calcule \(\det(A)\) e diga se a matriz possui inversa.

### Questão 3

Para:

\[
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\]

explique por que ela não possui inversa usando: determinante, geometria e perda de informação.

### Questão 4

Se \(y=Ax\) e conhecemos \(y\), qual operação permite recuperar \(x\) quando \(A\) é invertível?

### Questão 5

Resolva:

\[
\begin{cases}
2x+y=5\\
x+y=3
\end{cases}
\]

### Questão 6

Se \(C=BA\), qual é \(C^{-1}\)? Explique por que a ordem é invertida.

### Questão 7

Por que, em programação, `np.linalg.solve(A, b)` costuma ser melhor do que `np.linalg.inv(A) @ b` quando nosso único objetivo é resolver \(Ax=b\)?

### Questão 8 — a mais importante

Explique com suas palavras a diferença entre transformar um vetor e desfazer uma transformação. Relacione:

\[
x\xrightarrow{A}y\xrightarrow{A^{-1}}x
\]

com a perda de informação quando a transformação é singular.

## Mini laboratório em NumPy

```python
import numpy as np

A = np.array([
    [2.0, 1.0],
    [1.0, 1.0]
])

x = np.array([2.0, 1.0])

y = A @ x
A_inv = np.linalg.inv(A)
x_recuperado = A_inv @ y

print("x original:", x)
print("y transformado:", y)
print("A inversa:")
print(A_inv)
print("x recuperado:", x_recuperado)
```

Depois experimente uma matriz singular:

```python
A_singular = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])
```

Calcule o determinante e tente inverter a matriz.

## O que você precisa sair sabendo

Quando olhar para:

\[
A^{-1}
\]

pense:

\[
\boxed{A^{-1}=\text{a transformação que desfaz }A}
\]

Guarde:

\[
A^{-1}A=AA^{-1}=I
\]

Para matrizes quadradas:

\[
\boxed{A^{-1}\text{ existe}\iff\det(A)\neq0}
\]

E principalmente:

> **Uma transformação só pode ser desfeita perfeitamente quando não destruiu a informação necessária para reconstruir a entrada.**

## Próxima aula

### Aula 09 — Sistemas lineares, eliminação de Gauss e posto da matriz

Na próxima aula estudaremos \(Ax=b\) sem depender de calcular a inversa: sistemas determinados, superdeterminados e subdeterminados, eliminação de Gauss, escalonamento, pivôs, posto, dependência linear e condições de existência e unicidade de solução.
