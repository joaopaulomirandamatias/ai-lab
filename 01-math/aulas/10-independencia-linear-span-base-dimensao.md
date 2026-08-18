# Aula 10 — Independência linear, span, base e dimensão

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 09 — Sistemas lineares e rank  
**Objetivo:** entender quando vetores carregam informação nova, como conjuntos de vetores geram espaços e por que base e dimensão são conceitos centrais para representações em IA.

## 1. A pergunta central

Considere três vetores:

\[
v_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
v_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
v_3=\begin{bmatrix}1\\1\end{bmatrix}
\]

O terceiro vetor parece novo, mas:

\[
v_3=v_1+v_2
\]

Então ele não adiciona uma nova direção independente. Essa ideia de **redundância** é o ponto de partida desta aula.

> Em Álgebra Linear, uma pergunta fundamental é: quantas direções realmente novas existem em um conjunto de vetores?

## 2. Combinação linear

Uma combinação linear de vetores \(v_1,\ldots,v_k\) é qualquer vetor da forma:

\[
a_1v_1+a_2v_2+\cdots+a_kv_k
\]

onde os \(a_i\) são escalares.

Exemplo:

\[
2\begin{bmatrix}1\\0\end{bmatrix}+3\begin{bmatrix}0\\1\end{bmatrix}
=\begin{bmatrix}2\\3\end{bmatrix}
\]

Isso significa que os vetores canônicos \([1,0]^T\) e \([0,1]^T\) conseguem construir qualquer vetor do plano.

## 3. Span

O **span** de um conjunto de vetores é o conjunto de todas as combinações lineares possíveis deles:

\[
\operatorname{span}(v_1,\ldots,v_k)
\]

Se temos apenas:

\[
v=\begin{bmatrix}1\\2\end{bmatrix}
\]

seu span é uma reta que passa pela origem:

\[
\{av:a\in\mathbb{R}\}
\]

Com dois vetores não paralelos no plano, o span pode ser todo \(\mathbb{R}^2\).

## 4. Dependência linear

Os vetores são linearmente dependentes se existe uma combinação não trivial que produz zero:

\[
a_1v_1+\cdots+a_kv_k=0
\]

com pelo menos um coeficiente diferente de zero.

Exemplo:

\[
v_1=\begin{bmatrix}1\\2\end{bmatrix},\quad
v_2=\begin{bmatrix}2\\4\end{bmatrix}
\]

Como:

\[
v_2=2v_1
\]

podemos escrever:

\[
2v_1-v_2=0
\]

Logo os vetores são dependentes.

## 5. Independência linear

Um conjunto é linearmente independente quando a única maneira de obter o vetor zero é:

\[
a_1=a_2=\cdots=a_k=0
\]

Intuição:

- independente → cada vetor acrescenta uma direção nova;
- dependente → pelo menos um vetor é redundante.

Essa interpretação se conecta diretamente ao **rank** da aula anterior.

## 6. Testando independência com uma matriz

Coloque os vetores como colunas:

\[
A=\begin{bmatrix}|&|& &|\\v_1&v_2&\cdots&v_k\\|&|&&|\end{bmatrix}
\]

Resolver:

\[
Ax=0
\]

responde se existe uma combinação não trivial das colunas que produz zero.

Se o null space contém apenas \(x=0\), as colunas são independentes.

## 7. Base

Uma **base** de um espaço precisa satisfazer duas propriedades:

1. gerar todo o espaço;
2. ser linearmente independente.

Para \(\mathbb{R}^2\), a base canônica é:

\[
e_1=\begin{bmatrix}1\\0\end{bmatrix},\quad e_2=\begin{bmatrix}0\\1\end{bmatrix}
\]

Mas não é a única base. Também podemos usar:

\[
b_1=\begin{bmatrix}1\\1\end{bmatrix},\quad
b_2=\begin{bmatrix}1\\-1\end{bmatrix}
\]

Eles são independentes e geram o plano inteiro.

## 8. Coordenadas dependem da base

O vetor geométrico:

\[
x=\begin{bmatrix}4\\2\end{bmatrix}
\]

na base canônica tem coordenadas \((4,2)\). Na base \(b_1=[1,1]^T\), \(b_2=[1,-1]^T\), queremos:

\[
x=\alpha b_1+\beta b_2
\]

Daí:

\[
\alpha+\beta=4,\qquad \alpha-\beta=2
\]

resultando em:

\[
\alpha=3,\qquad \beta=1
\]

O mesmo objeto pode ter representações numéricas diferentes em bases diferentes.

## 9. Dimensão

A dimensão de um espaço é o número de vetores em qualquer base desse espaço.

\[
\dim(\mathbb{R}^2)=2,\qquad \dim(\mathbb{R}^3)=3
\]

Se um conjunto de dados vive em \(\mathbb{R}^{1000}\), mas todos os pontos estão próximos de um plano, sua dimensionalidade **ambiente** é 1000, enquanto sua dimensionalidade **intrínseca** pode ser muito menor.

## 10. Ligação com embeddings

Um embedding pode ter 768 ou 1536 componentes. Isso indica a dimensão do espaço de representação, mas não garante que todos os dados usem cada direção de maneira igualmente relevante.

Perguntas que aparecerão mais adiante:

- existem direções redundantes?
- podemos encontrar uma base melhor?
- podemos representar os dados com menos dimensões?

Essas perguntas conduzem diretamente a PCA e SVD.

## 11. Mudança de base e IA

Uma matriz pode transformar coordenadas de uma base para outra. Em IA, muitas camadas aprendem representações nas quais o problema se torna mais simples.

Uma forma mental útil:

```text
representação original
        ↓
transformação aprendida
        ↓
nova base / novas coordenadas
        ↓
padrões mais fáceis de explorar
```

Não significa que toda camada neural execute literalmente uma mudança de base clássica, mas a intuição de **reexpressar informação em coordenadas mais úteis** é poderosa.

## 12. NumPy

```python
import numpy as np

A = np.array([
    [1., 0., 1.],
    [0., 1., 1.]
])

print("rank:", np.linalg.matrix_rank(A))
```

A matriz possui três colunas, mas rank 2. Logo uma das colunas é redundante.

Para resolver coordenadas em outra base:

```python
B = np.array([
    [1.,  1.],
    [1., -1.]
])

x = np.array([4., 2.])
coords = np.linalg.solve(B, x)
print(coords)  # [3. 1.]
```

## 13. Erros comuns

- Confundir número de vetores com dimensão do espaço gerado.
- Achar que qualquer conjunto que gera o espaço é uma base; ele também precisa ser independente.
- Pensar que coordenadas são o vetor em si; coordenadas dependem da base escolhida.
- Confundir dimensão ambiente com dimensionalidade efetiva dos dados.

## Exercícios

1. Os vetores \([1,2]^T\) e \([3,6]^T\) são independentes? Por quê?
2. Qual é o span de um único vetor não nulo em \(\mathbb{R}^3\)?
3. Verifique se \([1,0]^T\) e \([1,1]^T\) formam uma base de \(\mathbb{R}^2\).
4. Expresse \([5,1]^T\) na base \(b_1=[1,1]^T\), \(b_2=[1,-1]^T\).
5. Uma matriz tem 100 colunas e rank 7. O que isso sugere sobre independência entre as colunas?
6. Explique a diferença entre “o vetor” e “as coordenadas do vetor em uma base”.

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cambridge University Press, 2020. Capítulos 2 e 3. https://mml-book.github.io/
- STRANG, G. *MIT 18.06 Linear Algebra*. MIT OpenCourseWare. Tópicos: bases, espaços vetoriais, rank e null space. https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. Cap. 2 — Linear Algebra. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 11 — Os quatro subespaços fundamentais: column space, row space, null space e left null space.**