# Aula 03 — Produto escalar: alinhamento entre vetores

**Trilha:** 01 · Matemática para IA  
**Etapa:** M1–M2 · Nível 1  
**Data de estudo:** 2026-08-17  
**Objetivo:** compreender o produto escalar numericamente e geometricamente e reconhecer sua presença em redes neurais, embeddings, busca vetorial e Transformers.

---

## 1. Ideia central

O **produto escalar** (*dot product*) recebe dois vetores de mesma dimensão e devolve um único número.

Para:

$$
a =
\begin{bmatrix}
2 \\
3
\end{bmatrix}
\qquad
b =
\begin{bmatrix}
4 \\
1
\end{bmatrix}
$$

calculamos:

$$
a \cdot b = 2(4) + 3(1) = 8 + 3 = 11
$$

A regra é simples:

1. multiplicar componentes correspondentes;
2. somar os resultados.

Para vetores em $\mathbb{R}^n$:

$$
\boxed{a \cdot b = \sum_{i=1}^{n} a_i b_i}
$$

---

## 2. Implementação em NumPy

```python
import numpy as np

a = np.array([2, 3])
b = np.array([4, 1])

print(np.dot(a, b))  # 11
print(a @ b)         # 11
```

Para vetores, `np.dot(a, b)` e `a @ b` produzem o produto escalar.

---

## 3. Interpretação geométrica

O produto escalar também pode ser escrito como:

$$
\boxed{a \cdot b = \|a\|\,\|b\|\cos(\theta)}
$$

onde:

- $\|a\|$ é a magnitude de $a$;
- $\|b\|$ é a magnitude de $b$;
- $\theta$ é o ângulo entre os vetores.

Essa forma é fundamental porque conecta uma operação algébrica ao alinhamento geométrico entre vetores.

### Regra mental

| Resultado | Relação angular | Interpretação |
|---|---|---|
| $a\cdot b > 0$ | $\theta < 90^\circ$ | direções formam ângulo agudo |
| $a\cdot b = 0$ | $\theta = 90^\circ$ | vetores ortogonais |
| $a\cdot b < 0$ | $\theta > 90^\circ$ | direções formam ângulo obtuso |

Casos extremos:

- mesma direção: $\theta = 0^\circ$ e $\cos(\theta)=1$;
- direções opostas: $\theta = 180^\circ$ e $\cos(\theta)=-1$.

---

## 4. Mesma direção

Considere:

$$
a =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
\qquad
b =
\begin{bmatrix}
2 \\
4
\end{bmatrix}
$$

Como $b=2a$, os vetores apontam exatamente para a mesma direção.

$$
a\cdot b = 1(2)+2(4)=10
$$

O resultado é positivo e, geometricamente, o ângulo entre eles é $0^\circ$.

---

## 5. Vetores perpendiculares

Considere:

$$
a =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
\qquad
b =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

Então:

$$
a\cdot b = 1(0)+0(1)=0
$$

Logo:

$$
\boxed{a\cdot b = 0}
$$

Nesse caso, os vetores são ortogonais e formam um ângulo de $90^\circ$.

---

## 6. Direções opostas

Considere:

$$
a =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
\qquad
b =
\begin{bmatrix}
-1 \\
0
\end{bmatrix}
$$

Então:

$$
a\cdot b = -1
$$

O ângulo é $180^\circ$, portanto os vetores apontam para sentidos exatamente opostos.

---

## 7. A magnitude influencia o produto escalar

Considere:

$$
a=[1,2]
$$

$$
b=[2,4]
$$

$$
c=[100,200]
$$

Os vetores $b$ e $c$ têm exatamente a mesma direção de $a$, mas:

$$
a\cdot b = 10
$$

$$
a\cdot c = 500
$$

A diferença não vem da direção: vem da **magnitude**.

Portanto o produto escalar combina duas informações:

$$
\boxed{\text{alinhamento angular + magnitude}}
$$

Isso explica por que o dot product bruto nem sempre é a melhor medida para comparar embeddings de escalas diferentes.

---

## 8. Daí surge a similaridade de cosseno

Partimos de:

$$
a\cdot b = \|a\|\|b\|\cos(\theta)
$$

Isolando o cosseno:

$$
\boxed{
\cos(\theta)=\frac{a\cdot b}{\|a\|\|b\|}
}
$$

Essa expressão é a base da **similaridade de cosseno**.

Ela normaliza o produto escalar pelas magnitudes e permite concentrar a comparação na direção dos vetores.

### Exemplo

Para:

$$
a=[1,2],\quad b=[2,4]
$$

Temos:

$$
a\cdot b=10
$$

$$
\|a\|=\sqrt{5}
$$

$$
\|b\|=\sqrt{20}
$$

Portanto:

$$
\frac{a\cdot b}{\|a\|\|b\|}
=
\frac{10}{\sqrt5\sqrt{20}}
=
\frac{10}{10}
=1
$$

O valor 1 confirma que os vetores têm a mesma direção.

---

## 9. Embeddings e busca semântica

Uma pergunta e os documentos de uma base podem ser transformados em embeddings:

```text
pergunta do usuário
        ↓
embedding q
        ↓
comparação com d₁, d₂, d₃, ...
        ↓
ranking vetorial
        ↓
documentos mais relevantes
```

O produto escalar ou uma medida derivada dele, como a similaridade de cosseno, pode participar do cálculo de relevância entre o vetor da consulta $q$ e os vetores dos documentos $d_i$.

Esse princípio aparece em:

- busca semântica;
- bancos vetoriais;
- recomendação;
- RAG;
- recuperação de vizinhos mais próximos.

---

## 10. Relação com redes neurais

Da Aula 01:

$$
y=Wx+b
$$

Considere uma linha da matriz de pesos:

$$
w=[w_1,w_2,w_3]
$$

E uma entrada:

$$
x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
$$

Uma componente da multiplicação $Wx$ envolve:

$$
w\cdot x = w_1x_1+w_2x_2+w_3x_3
$$

Portanto, multiplicação matricial pode ser vista como vários produtos escalares organizados.

Cadeia conceitual:

```text
vetores
  ↓
produto escalar
  ↓
multiplicação matricial
  ↓
camadas de redes neurais
```

---

## 11. Relação com Transformers

No mecanismo de atenção aparecem as matrizes:

- $Q$ — Query;
- $K$ — Key;
- $V$ — Value.

A expressão clássica é:

$$
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

A multiplicação $QK^T$ é composta por vários produtos escalares entre queries e keys.

Intuitivamente, esses produtos ajudam a medir quanto uma query está alinhada com cada key.

Não é necessário dominar atenção nesta etapa; o objetivo é reconhecer que o produto escalar estudado aqui reaparecerá dentro dos Transformers.

---

## 12. Resumo mental

### Produto escalar

$$
\boxed{a\cdot b = \sum_i a_i b_i}
$$

Interpretação:

> Quanto os vetores estão alinhados, levando também em conta seus tamanhos?

### Similaridade de cosseno

$$
\boxed{
\frac{a\cdot b}{\|a\|\|b\|}
}
$$

Interpretação:

> Quanto suas direções estão alinhadas depois de remover o efeito da magnitude?

---

## 13. Exercício de fixação

### Questão 1

Calcule:

$$
a=[2,3],\qquad b=[5,4]
$$

Quanto vale $a\cdot b$?

### Questão 2

Se:

$$
a=[1,0],\qquad b=[0,5]
$$

quanto vale $a\cdot b$ e o que isso significa geometricamente?

### Questão 3

Se o produto escalar entre dois vetores for negativo, o que isso indica sobre o ângulo entre suas direções?

### Questão 4

Por que o produto escalar bruto pode ser problemático para comparar embeddings de magnitudes muito diferentes?

### Questão 5

Explique com suas palavras a diferença entre:

$$
a\cdot b
$$

e:

$$
\frac{a\cdot b}{\|a\|\|b\|}
$$

**Status dos exercícios:** aguardando resposta do aluno.

---

## Próxima aula

**Aula 04 — Similaridade de cosseno na prática: embeddings, busca semântica e RAG.**

A próxima aula aplicará diretamente produto escalar, norma e normalização em exemplos de comparação de embeddings.
