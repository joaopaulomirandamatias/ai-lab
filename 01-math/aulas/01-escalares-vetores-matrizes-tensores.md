# Aula 01 — Escalares, Vetores, Matrizes e Tensores

**Trilha:** 01 · Matemática para IA  
**Etapa:** M1–M2 · Nível 1  
**Data de estudo:** 2026-08-16  
**Objetivo:** entender como dados são representados matematicamente em sistemas de IA e reconhecer escalares, vetores, matrizes e tensores.

---

## 1. Ideia central

Quase tudo que entra, circula e sai de um modelo de IA pode ser representado por números organizados em estruturas matemáticas.

A cadeia mental desta aula é:

```text
dados
  ↓
números
  ↓
vetores
  ↓
matrizes / tensores
  ↓
transformações matemáticas
  ↓
modelo de IA
```

---

## 2. Escalar

Um **escalar** é um único número.

Exemplos:

```python
x = 7
temperatura = 28.5
learning_rate = 0.001
```

Matematicamente:

$$
x \in \mathbb{R}
$$

Isso significa que `x` pertence ao conjunto dos números reais.

Em IA, escalares aparecem em elementos como:

- taxa de aprendizado;
- valor de uma função de perda;
- probabilidade;
- peso individual;
- temperatura de geração de um LLM.

---

## 3. Vetor

Um **vetor** é uma coleção ordenada de números.

Exemplo: representar uma pessoa por idade, altura e peso.

$$
x =
\begin{bmatrix}
35 \\
1.80 \\
90
\end{bmatrix}
$$

Em NumPy:

```python
import numpy as np

x = np.array([35, 1.80, 90])
```

Como possui três componentes:

$$
x \in \mathbb{R}^{3}
$$

### Interpretação importante

Um vetor não deve ser entendido apenas como uma lista de números. Em Machine Learning, ele frequentemente funciona como uma **representação matemática de um objeto**.

Exemplos:

$$
Pessoa =
\begin{bmatrix}
idade \\
altura \\
peso
\end{bmatrix}
$$

ou:

$$
Carro =
\begin{bmatrix}
velocidade \\
peso \\
potencia \\
consumo
\end{bmatrix}
$$

---

## 4. Ligação com IA: embeddings

Modelos de linguagem precisam transformar palavras, tokens, documentos e outros objetos em representações numéricas.

Uma palavra como `café` pode ser associada a um vetor:

$$
café =
\begin{bmatrix}
0.72 \\
-0.14 \\
0.83 \\
0.41 \\
\vdots
\end{bmatrix}
$$

Esse tipo de representação vetorial é chamado de **embedding**.

A ideia fundamental é que objetos semanticamente relacionados podem ocupar regiões próximas em um espaço vetorial.

Por exemplo, vetores associados a `café` e `chá` podem apontar para regiões semelhantes, enquanto um vetor associado a `automóvel` pode estar mais distante.

Essa noção prepara o caminho para conceitos posteriores como:

- produto escalar;
- norma;
- distância;
- similaridade de cosseno;
- busca semântica;
- RAG.

---

## 5. Matriz

Uma **matriz** é uma organização bidimensional de números, em linhas e colunas.

Exemplo com três pessoas e três atributos:

$$
X =
\begin{bmatrix}
35 & 1.80 & 90 \\
28 & 1.65 & 65 \\
42 & 1.75 & 82
\end{bmatrix}
$$

Em NumPy:

```python
X = np.array([
    [35, 1.80, 90],
    [28, 1.65, 65],
    [42, 1.75, 82]
])

print(X.shape)
```

Resultado:

```text
(3, 3)
```

Uma forma inicial de lembrar:

| Estrutura | Intuição |
|---|---|
| Escalar | um número |
| Vetor | uma sequência de números |
| Matriz | uma tabela de números |

---

## 6. A equação fundamental `y = Wx + b`

Uma das transformações mais importantes em redes neurais é:

$$
\boxed{y = Wx + b}
$$

Onde:

- $x$ é o vetor de entrada;
- $W$ é a matriz de pesos;
- $b$ é o vetor de bias;
- $y$ é o vetor de saída.

Exemplo:

$$
x =
\begin{bmatrix}
2 \\
3
\end{bmatrix}
$$

$$
W =
\begin{bmatrix}
0.5 & 0.2 \\
0.1 & 0.8
\end{bmatrix}
$$

$$
b =
\begin{bmatrix}
0.1 \\
0.2
\end{bmatrix}
$$

Primeiro:

$$
Wx =
\begin{bmatrix}
0.5 & 0.2 \\
0.1 & 0.8
\end{bmatrix}
\begin{bmatrix}
2 \\
3
\end{bmatrix}
=
\begin{bmatrix}
1.6 \\
2.6
\end{bmatrix}
$$

Depois:

$$
y = Wx+b =
\begin{bmatrix}
1.6 \\
2.6
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\
0.2
\end{bmatrix}
=
\begin{bmatrix}
1.7 \\
2.8
\end{bmatrix}
$$

### Interpretação

`W` **não é, por si só, toda a transformação**. Ela é a matriz de pesos que parametriza a transformação linear.

- $W$ contém parâmetros aprendidos pelo modelo;
- $Wx$ é a transformação linear aplicada à entrada;
- $Wx+b$ é a transformação afim completa.

Durante o treinamento de uma rede neural, os valores de $W$ e $b$ são ajustados para reduzir o erro do modelo.

---

## 7. Tensor

Um **tensor** generaliza escalares, vetores e matrizes para estruturas com mais dimensões.

Uma simplificação útil no início é:

| Estrutura | Dimensionalidade típica |
|---|---:|
| Escalar | 0D |
| Vetor | 1D |
| Matriz | 2D |
| Tensor | 3D ou mais |

### Imagem RGB

Uma imagem RGB pode ter formato:

$$
altura \times largura \times canais
$$

Por exemplo:

$$
1080 \times 1920 \times 3
$$

Os três canais correspondem a vermelho, verde e azul.

> Observação: bibliotecas diferentes podem organizar as dimensões em ordens diferentes, como `H × W × C` ou `C × H × W`.

Se processarmos um lote de imagens, aparece uma dimensão adicional de **batch**:

$$
batch \times altura \times largura \times canais
$$

Por exemplo, um batch contendo uma única imagem:

$$
1 \times 1080 \times 1920 \times 3
$$

Portanto, `1 × 1920 × 1080 × 3` pode representar um **batch de uma imagem**, enquanto a imagem isolada não precisa do `1` inicial.

---

## 8. Tensores em LLMs

Considere a frase:

> O café brasileiro é excelente

Depois da tokenização, cada token recebe uma representação vetorial.

Se tivermos 5 tokens e cada embedding tiver 768 dimensões, podemos obter uma estrutura:

$$
5 \times 768
$$

Se processarmos 32 sequências simultaneamente:

$$
32 \times 5 \times 768
$$

Esse tensor pode ser interpretado como:

```text
batch × tokens × embedding_dimensions
```

Ao trabalhar com tensores, uma pergunta essencial é:

> **O que cada dimensão representa?**

Essa pergunta será recorrente em Deep Learning e Transformers.

---

## 9. Resumo mental

### Escalar

$$
7
$$

Uma informação numérica.

### Vetor

$$
[7,5,9]
$$

Um objeto representado por várias componentes.

### Matriz

$$
\begin{bmatrix}
7 & 5 & 9 \\
2 & 4 & 8
\end{bmatrix}
$$

Uma estrutura bidimensional de dados ou parâmetros.

### Tensor

```text
batch × tokens × embeddings
```

Uma estrutura multidimensional usada intensamente em Deep Learning.

---

## 10. Exercício de fixação

### Questão 1

Se temos:

```python
temperatura = 27.5
```

Qual estrutura matemática representa esse valor?

**Resposta do aluno:** escalar.  
**Correção:** ✅ Correto.

---

### Questão 2

Se uma pessoa é representada por:

```text
[idade, peso, altura]
```

Qual estrutura temos?

**Resposta do aluno:** vetor.  
**Correção:** ✅ Correto.

---

### Questão 3

Uma imagem RGB de `1920 × 1080 × 3` pode ser vista como qual estrutura?

**Resposta do aluno:** tensor `1 × 1920 × 1080 × 3`.  
**Correção:** ✅ Conceito correto, com uma observação.

A imagem isolada pode ser representada como tensor 3D:

```text
1920 × 1080 × 3
```

ou, conforme a convenção da biblioteca:

```text
1080 × 1920 × 3
```

O `1` inicial acrescenta uma dimensão de batch, transformando a representação em um tensor 4D contendo **um lote com uma imagem**.

---

### Questão 4

Na equação:

$$
y = Wx+b
$$

o que representa $W$ em uma rede neural?

**Resposta do aluno:** transformação.  
**Correção:** 🟡 Intuição correta; precisão conceitual a melhorar.

$W$ é a **matriz de pesos aprendíveis** que define ou parametriza a transformação. A operação $Wx$ realiza a transformação linear sobre $x$, e $Wx+b$ representa a transformação afim completa.

---

## 11. Resultado da aula

**Status:** concluída ✅

O aluno demonstrou compreender:

- diferença entre escalar, vetor e tensor;
- representação vetorial de objetos;
- existência da dimensão de batch;
- ideia geral de transformação em uma rede neural.

Ponto a consolidar:

- distinguir a **matriz de parâmetros $W$** da **transformação $Wx$** que ela produz.

---

## Próxima aula

**Aula 02 — Vetores como objetos geométricos: direção, magnitude, norma e distância.**

Essa aula prepara diretamente para:

- produto escalar;
- similaridade de cosseno;
- embeddings;
- busca semântica;
- RAG.
