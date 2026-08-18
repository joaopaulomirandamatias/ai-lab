# Aula 12 — Ortogonalidade, projeções, Gram-Schmidt e decomposição QR

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** subespaços, produto escalar e projeções  
**Objetivo:** entender ortogonalidade como independência geométrica, construir bases ortonormais e introduzir QR e mínimos quadrados.

## 1. Ortogonalidade

Dois vetores são ortogonais quando:

\[
a^Tb=0
\]

No plano isso corresponde a um ângulo de 90°. Em dimensões maiores a visualização desaparece, mas o produto escalar continua funcionando.

Exemplo:

\[
a=\begin{bmatrix}1\\1\end{bmatrix},\quad
b=\begin{bmatrix}1\\-1\end{bmatrix}
\]

\[
a^Tb=1-1=0
\]

## 2. Ortogonal versus ortonormal

Um conjunto é **ortogonal** quando os vetores são mutuamente perpendiculares. É **ortonormal** quando, além disso, cada vetor possui norma 1:

\[
q_i^Tq_j=\begin{cases}1&i=j\\0&i\neq j\end{cases}
\]

Bases ortonormais são extremamente convenientes porque coordenadas podem ser calculadas por simples produtos escalares.

## 3. Projeção sobre um vetor unitário

Se \(q\) é unitário:

\[
\operatorname{proj}_q(x)=(q^Tx)q
\]

O número \(q^Tx\) mede quanto de \(x\) existe na direção de \(q\).

Se \(Q\) contém várias colunas ortonormais, a projeção sobre o subespaço gerado por elas é:

\[
\boxed{P=QQ^T}
\]

E:

\[
\hat x=QQ^Tx
\]

## 4. O vetor residual

A diferença:

\[
r=x-\hat x
\]

é ortogonal ao subespaço projetado.

Isso será central em mínimos quadrados: quando não conseguimos resolver \(Ax=b\) exatamente, procuramos o ponto \(A\hat x\) mais próximo de \(b\). O erro:

\[
r=b-A\hat x
\]

fica ortogonal ao espaço das colunas de \(A\).

## 5. Por que bases ortonormais são boas numericamente?

Vetores ortonormais têm comprimento 1 e não apontam quase na mesma direção. Isso evita amplificações desnecessárias em várias operações e simplifica inversões:

\[
Q^TQ=I
\]

Se \(Q\) é quadrada ortogonal:

\[
\boxed{Q^{-1}=Q^T}
\]

## 6. Gram-Schmidt

Gram-Schmidt transforma vetores independentes em uma base ortonormal para o mesmo espaço.

Comece com \(a_1,a_2\).

Primeiro:

\[
q_1=\frac{a_1}{\|a_1\|}
\]

Depois remova de \(a_2\) a parte alinhada com \(q_1\):

\[
u_2=a_2-(q_1^Ta_2)q_1
\]

Normalize:

\[
q_2=\frac{u_2}{\|u_2\|}
\]

Agora \(q_1\perp q_2\).

## 7. Exemplo

Considere:

\[
a_1=\begin{bmatrix}1\\1\end{bmatrix},\quad
a_2=\begin{bmatrix}1\\0\end{bmatrix}
\]

Temos:

\[
q_1=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}
\]

A projeção de \(a_2\) sobre \(q_1\) é:

\[
(q_1^Ta_2)q_1=\frac12\begin{bmatrix}1\\1\end{bmatrix}
\]

Então:

\[
u_2=\begin{bmatrix}1\\0\end{bmatrix}-\begin{bmatrix}1/2\\1/2\end{bmatrix}
=\begin{bmatrix}1/2\\-1/2\end{bmatrix}
\]

Normalizando:

\[
q_2=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}
\]

## 8. QR

Se aplicarmos Gram-Schmidt às colunas de \(A\), obtemos uma matriz \(Q\) com colunas ortonormais e uma matriz triangular superior \(R\):

\[
\boxed{A=QR}
\]

Essa é a **decomposição QR**.

Intuição:

- \(Q\): direções ortonormais;
- \(R\): coordenadas necessárias para reconstruir as colunas originais nessas direções.

## 9. QR e mínimos quadrados

Para:

\[
Ax\approx b
\]

se \(A=QR\):

\[
QRx\approx b
\]

Multiplicando por \(Q^T\):

\[
Rx\approx Q^Tb
\]

Como \(R\) é triangular, o problema fica mais simples e numericamente melhor comportado do que formar explicitamente \((A^TA)^{-1}\).

## 10. Conexão com attention e embeddings

Ortogonalidade também fornece uma intuição de “informação não redundante”. Em representações aprendidas, direções quase ortogonais podem capturar aspectos distintos, embora não devamos assumir que redes neurais criem bases perfeitamente ortogonais sem restrições específicas.

## 11. NumPy

```python
import numpy as np

A = np.array([
    [1., 1.],
    [1., 0.]
])

Q, R = np.linalg.qr(A)

print("Q=")
print(Q)
print("Q^T Q=")
print(Q.T @ Q)
print("R=")
print(R)
print("Reconstrução=")
print(Q @ R)
```

Você deve observar:

\[
Q^TQ\approx I
\]

e:

\[
QR\approx A
\]

## 12. Erro numérico

O Gram-Schmidt clássico é excelente didaticamente, mas pode perder ortogonalidade em problemas numericamente difíceis. Implementações robustas usam variantes como modified Gram-Schmidt ou reflexões de Householder para obter QR.

Isso é uma ótima ilustração da diferença entre “fórmula matemática” e “algoritmo numérico estável”.

## Exercícios

1. Verifique se \([1,2]^T\) e \([2,-1]^T\) são ortogonais.
2. Normalize \([3,4]^T\).
3. Projete \([3,4]^T\) sobre \([1,0]^T\).
4. Explique por que \(Q^TQ=I\) para colunas ortonormais.
5. Descreva em palavras o objetivo do Gram-Schmidt.
6. Por que QR é útil para mínimos quadrados?

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Tópicos: projections, orthogonal bases, Gram-Schmidt e QR. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 3–4. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 13 — Autovalores e autovetores: direções especiais de uma transformação.**