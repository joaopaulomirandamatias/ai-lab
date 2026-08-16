# Aula 02 — Vetores como objetos geométricos: direção, magnitude, norma e distância

**Trilha:** 01 · Matemática para IA  
**Etapa:** M1–M2 · Nível 1  
**Data de estudo:** 2026-08-16  
**Status:** em andamento · exercício de fixação pendente  
**Objetivo:** interpretar vetores geometricamente e compreender direção, magnitude, norma, normalização e distância, preparando a base para produto escalar e similaridade de cosseno.

---

## 1. Mudando a forma de enxergar um vetor

Na Aula 01, vimos um vetor principalmente como uma coleção ordenada de números. Por exemplo:

$$
v =
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

Agora precisamos acrescentar uma segunda interpretação:

> **Um vetor não é apenas uma lista de números. Ele também pode ser interpretado como uma seta dentro de um espaço geométrico.**

Essa interpretação será fundamental para entender embeddings, similaridade semântica, redes neurais, otimização e, mais adiante, atenção em Transformers.

---

## 2. Vetores em duas dimensões

Considere:

$$
v =
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

Também podemos escrevê-lo como:

$$
v=(3,4)
$$

Geometricamente, podemos imaginá-lo como uma seta que parte da origem $(0,0)$ e termina no ponto $(3,4)$.

```text
y
↑
|
|            ● (3,4)
|          ↗
|        ↗
|      ↗   v
|    ↗
|  ↗
●──────────────────→ x
(0,0)
```

O vetor está dizendo, intuitivamente:

> mova 3 unidades no eixo horizontal e 4 unidades no eixo vertical.

Um vetor possui duas características importantes:

- **direção**;
- **magnitude**.

---

## 3. Direção

Considere:

$$
a=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
2 \\
4
\end{bmatrix}
$$

O segundo vetor é o dobro do primeiro:

$$
b=2a
$$

Geometricamente, eles possuem comprimentos diferentes, mas apontam para a mesma direção.

```text
        b
       ↗
      /
     /
    ↗ a
   /
●────────────────→
```

Multiplicar um vetor por um escalar pode alterar sua magnitude sem alterar sua direção.

Essa ideia será importante quando estudarmos similaridade de cosseno.

---

## 4. Magnitude e norma

Voltemos ao vetor:

$$
v=
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

Queremos saber o comprimento dessa seta.

Pelo teorema de Pitágoras:

$$
3^2+4^2=5^2
$$

Portanto, seu comprimento é:

$$
5
$$

Em álgebra linear, chamamos o comprimento de um vetor de **norma**.

Escrevemos:

$$
\|v\|=5
$$

---

## 5. Norma Euclidiana — L2

Para um vetor:

$$
v=
\begin{bmatrix}
v_1 \\
v_2 \\
\vdots \\
v_n
\end{bmatrix}
$$

sua norma Euclidiana é:

$$
\boxed{
\|v\|_2=
\sqrt{v_1^2+v_2^2+\cdots+v_n^2}
}
$$

Para $v=(3,4)$:

$$
\|v\|_2
=
\sqrt{3^2+4^2}
=
\sqrt{9+16}
=
\sqrt{25}
=
5
$$

Em NumPy:

```python
import numpy as np

v = np.array([3, 4])
norma = np.linalg.norm(v)

print(norma)
```

Resultado:

```text
5.0
```

---

## 6. Por que a norma importa para IA?

Um embedding pode possuir centenas ou milhares de componentes:

$$
x=
\begin{bmatrix}
0.8 \\
0.2 \\
0.9 \\
-0.4 \\
\vdots
\end{bmatrix}
$$

Mesmo que não consigamos visualizar geometricamente um espaço de centenas ou milhares de dimensões, ainda podemos calcular:

$$
\|x\|
$$

Isto leva a uma ideia importante:

> **A álgebra linear permite fazer geometria em espaços que não conseguimos visualizar.**

Embeddings vivem justamente nesses espaços de alta dimensionalidade.

---

## 7. Normalização

Considere novamente:

$$
v=
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

Sabemos que:

$$
\|v\|=5
$$

Podemos dividir o vetor inteiro pela sua norma:

$$
\hat{v}=
\frac{v}{\|v\|}
$$

Portanto:

$$
\hat{v}
=
\frac{1}{5}
\begin{bmatrix}
3 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.6 \\
0.8
\end{bmatrix}
$$

A norma do novo vetor é:

$$
\sqrt{0.6^2+0.8^2}
=
\sqrt{0.36+0.64}
=
1
$$

Portanto:

$$
\boxed{\|\hat{v}\|=1}
$$

Um vetor com norma igual a 1 é chamado de **vetor unitário**.

---

## 8. O que a normalização faz geometricamente?

Antes:

$$
v=(3,4)
$$

possuía magnitude 5.

Depois:

$$
\hat v=(0.6,0.8)
$$

possui magnitude 1.

Mas ambos apontam para a mesma direção.

```text
               v
              ↗
             /
            /
           /
       v̂ ↗
         /
        /
●──────────────────→
```

Assim:

> **Normalizar um vetor significa remover a influência do tamanho, preservando sua direção.**

---

## 9. Direção e embeddings

Considere:

$$
a=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
10 \\
20
\end{bmatrix}
$$

Temos:

$$
b=10a
$$

O vetor $b$ é dez vezes maior, mas aponta exatamente para a mesma direção.

Ao comparar embeddings, muitas vezes a pergunta mais interessante não é:

> Qual vetor é maior?

mas:

> Os vetores apontam para direções semelhantes?

Essa intuição prepara diretamente para a **similaridade de cosseno**.

---

## 10. Distância entre vetores

Considere:

$$
a=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
4 \\
6
\end{bmatrix}
$$

Queremos calcular a distância entre os dois pontos.

Primeiro:

$$
b-a
=
\begin{bmatrix}
4 \\
6
\end{bmatrix}
-
\begin{bmatrix}
1 \\
2
\end{bmatrix}
=
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

Agora calculamos a norma:

$$
\|b-a\|
=
\sqrt{3^2+4^2}
=
5
$$

A distância Euclidiana pode ser escrita como:

$$
\boxed{d(a,b)=\|a-b\|}
$$

Em NumPy:

```python
import numpy as np

a = np.array([1, 2])
b = np.array([4, 6])

distancia = np.linalg.norm(a - b)
print(distancia)
```

Resultado:

```text
5.0
```

---

## 11. Distância e significado

Imagine embeddings conceituais simplificados:

$$
café=
\begin{bmatrix}
0.7 \\
0.4
\end{bmatrix}
$$

$$
chá=
\begin{bmatrix}
0.72 \\
0.43
\end{bmatrix}
$$

$$
automóvel=
\begin{bmatrix}
-0.8 \\
0.9
\end{bmatrix}
$$

A distância entre `café` e `chá` tende a ser pequena neste exemplo, enquanto a distância entre `café` e `automóvel` é maior.

Assim, uma pergunta semântica pode começar a ser transformada em uma pergunta geométrica:

> **Quão próximos estão seus vetores?**

---

## 12. Distância Euclidiana nem sempre é suficiente

Considere:

$$
a=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
100 \\
200
\end{bmatrix}
$$

Eles apontam exatamente na mesma direção, mas sua distância Euclidiana é grande.

Isso mostra que existem perguntas diferentes:

### Pergunta A

> Os vetores estão geometricamente próximos?

Podemos usar distância.

### Pergunta B

> Os vetores apontam para direções parecidas?

Precisamos comparar seus ângulos.

Essa segunda pergunta nos levará à **similaridade de cosseno**.

---

## 13. Distinção fundamental

Guarde esta estrutura mental:

| Conceito | O que mede |
|---|---|
| Direção | para onde o vetor aponta |
| Magnitude | tamanho do vetor |
| Norma | medida da magnitude |
| Normalização | transforma a magnitude em 1 preservando a direção |
| Distância | separação entre dois vetores |
| Cosseno | semelhança de direção — próximo assunto |

---

## 14. Exemplo em Machine Learning

Suponha um cliente representado por:

$$
x=
\begin{bmatrix}
idade \\
renda \\
compras
\end{bmatrix}
$$

Cliente A:

$$
A=
\begin{bmatrix}
30 \\
5000 \\
10
\end{bmatrix}
$$

Cliente B:

$$
B=
\begin{bmatrix}
31 \\
5100 \\
11
\end{bmatrix}
$$

Podemos calcular:

$$
d(A,B)
$$

para procurar clientes próximos.

Esse princípio aparece em algoritmos como **k-Nearest Neighbors (KNN)**, que procura exemplos próximos segundo alguma métrica de distância.

---

## 15. Ligação com RAG

Em um sistema RAG, uma pergunta pode ser transformada em embedding:

```text
Pergunta do usuário
        ↓
embedding
        ↓
vetor da pergunta
        ↓
comparação com vetores de documentos
        ↓
documentos semanticamente relevantes
        ↓
LLM
```

Por exemplo, uma pergunta pode virar:

$$
q\in\mathbb{R}^{1536}
$$

E os documentos podem ser representados por:

$$
d_1,d_2,d_3,\ldots,d_n
$$

O sistema precisa descobrir quais documentos possuem representações mais próximas ou mais semelhantes a $q$.

Por trás desse processo aparecem conceitos como:

- norma;
- distância;
- ângulo;
- produto escalar;
- similaridade de cosseno.

---

## 16. Ideia central para guardar

Ao ver um vetor como:

$$
[3,4]
$$

devemos conseguir pensar de duas formas simultaneamente:

1. **é uma coleção de números**;
2. **é uma seta em um espaço geométrico**.

A segunda visão é fundamental para compreender a geometria dos embeddings e boa parte da álgebra linear usada em IA.

---

## 17. Exercício de fixação

### Questão 1

Para o vetor:

$$
v=
\begin{bmatrix}
6 \\
8
\end{bmatrix}
$$

qual é sua norma?

**Resposta do aluno:** pendente.

---

### Questão 2

Se:

$$
a=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
3 \\
6
\end{bmatrix}
$$

eles têm a mesma direção? Explique.

**Resposta do aluno:** pendente.

---

### Questão 3

Normalize:

$$
v=
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

**Resposta do aluno:** pendente.

---

### Questão 4

Se:

$$
a=(1,1)
$$

$$
b=(4,5)
$$

qual é a distância Euclidiana entre eles?

**Resposta do aluno:** pendente.

---

### Questão 5

Por que, ao comparar embeddings, a direção pode ser mais importante que o tamanho do vetor?

**Resposta do aluno:** pendente.

---

## Próxima aula

**Aula 03 — Produto escalar: a operação que conecta vetores, projeções e similaridade de cosseno.**

Essa aula prepara diretamente para compreender:

- ângulo entre vetores;
- projeção;
- similaridade de cosseno;
- embeddings;
- atenção em Transformers.
