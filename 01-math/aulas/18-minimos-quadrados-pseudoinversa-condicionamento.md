# Aula 18 — Mínimos quadrados, pseudoinversa e condicionamento

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear aplicada  
**Pré-requisitos:** projeções, QR, SVD e sistemas lineares  
**Objetivo:** resolver sistemas que não têm solução exata, compreender a pseudoinversa e introduzir estabilidade numérica antes do bloco de Cálculo.

## 1. Quando Ax=b não pode ser satisfeito exatamente

Em Machine Learning é comum termos mais observações que parâmetros:

\[
A\in\mathbb{R}^{m\times n},\qquad m\gg n
\]

Pode não existir nenhum \(x\) tal que:

\[
Ax=b
\]

para todas as equações simultaneamente.

Então procuramos o \(x\) que torna a saída o mais próxima possível de \(b\).

## 2. Problema de mínimos quadrados

Definimos:

\[
\boxed{\hat x=\arg\min_x\|Ax-b\|_2^2}
\]

O vetor de erro é:

\[
r=b-A\hat x
\]

Na solução ótima, \(r\) é ortogonal ao espaço das colunas de \(A\):

\[
A^Tr=0
\]

Logo:

\[
A^T(b-A\hat x)=0
\]

## 3. Equações normais

Reorganizando:

\[
\boxed{A^TA\hat x=A^Tb}
\]

Se \(A^TA\) é invertível:

\[
\hat x=(A^TA)^{-1}A^Tb
\]

Essa fórmula é conceitualmente importante, mas nem sempre é a melhor forma numérica de calcular a solução.

## 4. Interpretação geométrica

\(A\hat x\) é a projeção de \(b\) sobre o espaço das colunas de \(A\).

```text
            b •
              | residual r
              |
Col(A) -------• A x̂ --------
```

O residual é perpendicular ao subespaço.

## 5. Pseudoinversa

A pseudoinversa de Moore–Penrose generaliza a inversa:

\[
\boxed{A^+}
\]

A solução de mínimos quadrados de norma mínima pode ser escrita como:

\[
\boxed{\hat x=A^+b}
\]

Isso funciona também para muitas matrizes retangulares ou singulares.

## 6. Pseudoinversa via SVD

Se:

\[
A=U\Sigma V^T
\]

então:

\[
A^+=V\Sigma^+U^T
\]

Valores singulares suficientemente não nulos são invertidos. Direções associadas a valores zero não podem ser recuperadas.

Essa construção conecta diretamente pseudoinversa, rank e perda de informação.

## 7. Solução de norma mínima

Em sistemas subdeterminados pode haver infinitas soluções. A pseudoinversa seleciona, sob condições padrão, a solução com menor norma euclidiana.

Isso é uma forma de preferência implícita:

\[
\min\|x\|_2\quad\text{sujeito a }Ax=b
\]

quando o sistema é consistente.

## 8. Por que evitar formar AᵀA sem necessidade?

O número de condição pode piorar porque:

\[
\kappa(A^TA)\approx\kappa(A)^2
\]

Assim, problemas já sensíveis podem se tornar numericamente mais difíceis.

Por isso QR ou SVD são frequentemente preferíveis para least squares.

## 9. Condicionamento

O número de condição mede sensibilidade da solução a pequenas perturbações.

Em norma 2, para matriz invertível:

\[
\boxed{\kappa_2(A)=\frac{\sigma_{max}}{\sigma_{min}}}
\]

Se \(\kappa\approx1\), o problema é bem condicionado. Se \(\kappa\) é enorme, pequenos erros podem ser amplificados.

## 10. Mal condicionado não é o mesmo que singular

- singular → inversa não existe;
- mal condicionada → inversa pode existir, mas o problema é muito sensível numericamente.

Uma matriz com duas colunas quase paralelas é um bom exemplo de quase singularidade.

## 11. NumPy

```python
import numpy as np

A = np.array([
    [1., 1.],
    [1., 2.],
    [1., 3.],
    [1., 4.]
])

b = np.array([1.1, 1.9, 3.2, 3.9])

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("solução:", x)
print("rank:", rank)
print("singular values:", s)
print("cond:", np.linalg.cond(A))
```

## 12. Pseudoinversa no NumPy

```python
A_plus = np.linalg.pinv(A)
x2 = A_plus @ b

print(np.allclose(x, x2))
```

`pinv` normalmente usa SVD ou técnicas relacionadas e uma tolerância para tratar valores singulares muito pequenos.

## 13. Regressão linear aparece aqui

Se modelamos:

\[
y\approx X\beta
\]

mínimos quadrados procura:

\[
\hat\beta=\arg\min_\beta\|X\beta-y\|^2
\]

Mais adiante resolveremos o mesmo problema por gradient descent. Isso permitirá comparar uma solução algébrica com uma solução iterativa.

## 14. Por que gradient descent existe se há solução fechada?

Nem todo modelo possui solução analítica simples. E mesmo quando existe, matrizes gigantes podem tornar fatorações diretas inviáveis ou inconvenientes.

Gradient-based optimization generaliza para modelos com milhões ou bilhões de parâmetros e funções de perda não lineares.

A partir da próxima aula começamos a construir essa ferramenta.

## 15. Conexão com regularização

Quando o problema é mal condicionado, regularização pode estabilizar a solução. Ridge regression, por exemplo, substitui aproximadamente:

\[
X^TX
\]

por:

\[
X^TX+\lambda I
\]

Essa ideia será estudada no módulo de Machine Learning, mas a motivação numérica já está visível aqui.

## Exercícios

1. O que minimizamos em mínimos quadrados?
2. Por que o residual ótimo é ortogonal a Col(A)?
3. Derive as equações normais a partir de \(A^T(b-A\hat x)=0\).
4. Diferencie inversa e pseudoinversa.
5. O que um número de condição muito alto indica?
6. Por que QR/SVD podem ser preferíveis a formar \(A^TA\)?

## Referências

- STRANG, G. *MIT 18.06 Linear Algebra*. Least squares, projections e pseudoinverse. https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 2–4 e 9. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4 — Numerical Computation. https://www.deeplearningbook.org/contents/numerical.html

## Próxima aula

**Aula 19 — Funções para IA: composição, exponencial, logaritmo, sigmoid e softmax.**