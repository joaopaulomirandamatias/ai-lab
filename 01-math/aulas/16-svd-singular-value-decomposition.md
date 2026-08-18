# Aula 16 — SVD: a decomposição que revela rank, energia e compressão

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisitos:** rank, ortogonalidade, autovalores e decomposições  
**Objetivo:** compreender geometricamente a Singular Value Decomposition e por que ela é uma das ferramentas mais importantes para Machine Learning.

## 1. A fórmula

Para qualquer matriz real:

\[
A\in\mathbb{R}^{m\times n}
\]

podemos escrever:

\[
\boxed{A=U\Sigma V^T}
\]

onde:

- \(U\): colunas ortonormais no espaço de saída;
- \(V\): colunas ortonormais no espaço de entrada;
- \(\Sigma\): matriz diagonal/retangular com valores singulares não negativos.

## 2. Intuição geométrica

Uma transformação linear geral pode ser entendida, essencialmente, como:

```text
entrada
  ↓ Vᵀ
reorientação do espaço
  ↓ Σ
escalas independentes
  ↓ U
nova orientação
  ↓
saída
```

Essa visão é uma das grandes recompensas de estudar Álgebra Linear geometricamente.

## 3. Valores singulares

Os elementos:

\[
\sigma_1\ge\sigma_2\ge\cdots\ge0
\]

na diagonal de \(\Sigma\) são os **valores singulares**.

Eles medem quanto a transformação estica determinadas direções especiais.

Se \(\sigma_i=0\), aquela direção é colapsada.

## 4. Rank pela SVD

O rank da matriz é o número de valores singulares não nulos:

\[
\boxed{rank(A)=\#\{\sigma_i>0\}}
\]

Numericamente usamos uma tolerância porque valores muito pequenos podem ser consequência de arredondamento.

## 5. Vetores singulares direitos e esquerdos

As colunas de \(V\) são direções especiais no espaço de entrada. Se \(v_i\) é uma dessas direções:

\[
Av_i=\sigma_i u_i
\]

A matriz envia \(v_i\) para a direção \(u_i\), escalada por \(\sigma_i\).

Essa equação resume a geometria da SVD.

## 6. Relação com AᵀA

Os vetores singulares direitos são autovetores de:

\[
A^TA
\]

E:

\[
A^TAv_i=\sigma_i^2v_i
\]

Logo os autovalores de \(A^TA\) são os quadrados dos valores singulares.

Da mesma forma, \(u_i\) relaciona-se a \(AA^T\).

## 7. SVD reduzida

Se \(A\) tem rank \(r\), podemos manter apenas os componentes associados aos valores singulares relevantes:

\[
A=U_r\Sigma_rV_r^T
\]

Essa forma evidencia que uma matriz aparentemente enorme pode possuir estrutura efetiva de dimensão muito menor.

## 8. Aproximação de baixo rank

Podemos truncar a SVD:

\[
A_k=U_k\Sigma_kV_k^T
\]

mantendo apenas os \(k\) maiores valores singulares.

O teorema de Eckart–Young mostra que essa é a melhor aproximação de rank \(k\) para \(A\) em normas matriciais importantes como a norma de Frobenius.

Intuição:

> mantenha as direções em que a matriz concentra mais “energia” e descarte as menos importantes.

## 9. Compressão

Uma imagem em tons de cinza pode ser representada por uma matriz. Em vez de armazenar todos os pixels diretamente, uma aproximação de baixo rank pode usar \(U_k\), \(\Sigma_k\) e \(V_k\).

Quanto menor \(k\):

- maior compressão;
- maior perda de detalhe.

Isso permite visualizar concretamente o trade-off entre informação e dimensionalidade.

## 10. Pseudoinversa

A SVD também oferece uma forma estável de construir a pseudoinversa:

\[
\boxed{A^+=V\Sigma^+U^T}
\]

Em \(\Sigma^+\), valores singulares não nulos são invertidos:

\[
\sigma_i\rightarrow\frac{1}{\sigma_i}
\]

Valores zero permanecem zero.

A pseudoinversa funciona inclusive para matrizes retangulares e singulares.

## 11. Condicionamento

Para uma matriz invertível, o número de condição em norma 2 relaciona-se a:

\[
\kappa_2(A)=\frac{\sigma_{max}}{\sigma_{min}}
\]

Se \(\sigma_{min}\) é muito pequeno, a matriz está perto de perder uma direção e o problema pode ser numericamente sensível.

SVD torna essa fragilidade visível.

## 12. Conexão com PCA

Para dados centralizados em uma matriz \(X\):

\[
X=U\Sigma V^T
\]

as colunas principais de \(V\) fornecem direções principais. Assim, PCA pode ser calculado diretamente por SVD, sem formar explicitamente a matriz de covariância.

Essa será a próxima aula.

## 13. Conexão com LoRA

LoRA trabalha com atualizações de baixo rank:

\[
\Delta W=BA
\]

A SVD mostra por que representações low-rank são plausíveis: muitas matrizes podem ter grande parte de sua ação concentrada em poucas direções singulares dominantes.

Isso não prova que toda atualização de LLM seja intrinsicamente de baixo rank, mas fornece a linguagem matemática usada para estudar essa hipótese.

## 14. NumPy

```python
import numpy as np

A = np.array([
    [3., 1.],
    [1., 3.],
    [1., 1.]
])

U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("U=", U)
print("s=", s)
print("Vt=", Vt)

A_rec = U @ np.diag(s) @ Vt
print(np.allclose(A, A_rec))
```

## 15. Aproximação rank-1

```python
k = 1
A1 = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print(A1)
print("erro:", np.linalg.norm(A - A1, ord='fro'))
```

Depois aumente \(k\) e observe o erro diminuir.

## 16. Erros comuns

- Confundir valores singulares com autovalores; valores singulares são sempre não negativos.
- Achar que SVD exige matriz quadrada; ela funciona para matrizes retangulares.
- Pensar que rank numérico é uma decisão absolutamente exata; depende de tolerâncias.
- Formar \(A^TA\) desnecessariamente pode piorar aspectos numéricos em algumas aplicações.

## Exercícios

1. O que representa cada fator em \(A=U\Sigma V^T\)?
2. O que significa \(\sigma_i=0\)?
3. Como o rank aparece na SVD?
4. Explique a ideia de truncar a SVD.
5. Por que uma matriz com \(\sigma_{min}\) muito pequeno pode ser mal condicionada?
6. Qual a ligação entre SVD, PCA e LoRA?

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 4 e 10. https://mml-book.github.io/
- STRANG, G. *MIT 18.06 Linear Algebra*. Singular Value Decomposition e least squares. https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 2. https://www.deeplearningbook.org/contents/linear_algebra.html

## Próxima aula

**Aula 17 — PCA: redução de dimensionalidade preservando as direções de maior variância.**