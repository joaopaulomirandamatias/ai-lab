# Aula 07 — Determinante: área, volume, orientação e perda de informação

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 06 — Transformações geométricas  
**Objetivo:** compreender o determinante geometricamente e usá-lo para identificar mudança de área/volume, inversão de orientação e perda de dimensão.

## 1. O determinante não é apenas uma fórmula

Na aula anterior vimos que uma matriz pode transformar o espaço:

\[
x \rightarrow Wx
\]

Ela pode esticar, comprimir, rotacionar, refletir, projetar e deformar. Agora queremos responder:

> **Quanto essa transformação alterou o espaço?**

Para matrizes quadradas existe um número capaz de nos dar informações importantes sobre isso:

\[
\boxed{\det(W)}
\]

A ideia central desta aula será:

\[
\boxed{|\det(W)|=\text{fator de mudança de área ou volume}}
\]

Além disso, o determinante ajuda a descobrir se a transformação preservou dimensão, inverteu a orientação ou colapsou alguma direção.

## 2. Matriz identidade

Considere:

\[
I=\begin{bmatrix}1&0\\0&1\end{bmatrix}
\]

A matriz identidade não altera o espaço. Um quadrado de área 1 continua com área 1.

\[
\boxed{\det(I)=1}
\]

Geometricamente, a transformação preservou a área.

## 3. Determinante de uma matriz 2×2

Para:

\[
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
\]

seu determinante é:

\[
\boxed{\det(A)=ad-bc}
\]

Exemplo:

\[
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\]

\[
\det(A)=2\cdot3-0\cdot0=6
\]

Mas o valor 6 tem uma interpretação geométrica importante.

## 4. Determinante como mudança de área

A matriz:

\[
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\]

multiplica a componente horizontal por 2 e a vertical por 3.

Um quadrado unitário com área 1 se transforma em um retângulo de largura 2 e altura 3:

\[
\text{área nova}=2\times3=6
\]

Portanto:

\[
\boxed{|\det(A)|=6}
\]

Se uma região possui área \(S\), depois da transformação:

\[
\boxed{S'=|\det(A)|S}
\]

## 5. Interpretando a magnitude do determinante

Se:

\[
|\det(A)|>1
\]

a transformação expande área ou volume.

Se:

\[
0<|\det(A)|<1
\]

a transformação comprime área ou volume.

Por exemplo:

\[
A=\begin{bmatrix}0.5&0\\0&0.5\end{bmatrix}
\]

possui:

\[
\det(A)=0.25
\]

Cada eixo foi reduzido pela metade, mas a área foi reduzida a um quarto.

> **Escalas em dimensões diferentes se multiplicam para determinar a mudança de volume.**

## 6. Rotação e determinante

Considere uma rotação de 90°:

\[
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\]

Seu determinante é:

\[
\det(R)=1
\]

Uma rotação muda a direção dos vetores, mas preserva comprimentos, ângulos e áreas.

A matriz de rotação geral em 2D é:

\[
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\]

Seu determinante é:

\[
\cos^2\theta+\sin^2\theta=1
\]

Logo:

\[
\boxed{\det(R)=1}
\]

para qualquer rotação 2D.

## 7. Reflexão e determinante negativo

Considere:

\[
A=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
\]

Essa matriz reflete o espaço em relação ao eixo horizontal.

\[
\det(A)=-1
\]

A magnitude:

\[
|\det(A)|=1
\]

mostra que a área foi preservada. O sinal negativo indica que a orientação do espaço foi invertida.

Assim:

\[
\boxed{|\det(A)|=\text{mudança de área/volume}}
\]

\[
\boxed{\operatorname{sign}(\det(A))=\text{orientação}}
\]

## 8. O caso mais importante: determinante zero

Considere:

\[
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\]

\[
\det(A)=1\cdot4-2\cdot2=0
\]

As colunas são:

\[
\begin{bmatrix}1\\2\end{bmatrix}
\quad\text{e}\quad
\begin{bmatrix}2\\4\end{bmatrix}
\]

mas:

\[
\begin{bmatrix}2\\4\end{bmatrix}=2\begin{bmatrix}1\\2\end{bmatrix}
\]

As duas colunas apontam para a mesma direção. Em vez de formar uma região 2D, formam apenas uma linha.

> **Quando o determinante é zero, pelo menos uma dimensão foi colapsada.**

## 9. Projeção como exemplo de perda de dimensão

Na Aula 06 vimos:

\[
P=\begin{bmatrix}1&0\\0&0\end{bmatrix}
\]

que projeta tudo sobre o eixo horizontal.

\[
\det(P)=0
\]

A transformação:

\[
(x,y)\rightarrow(x,0)
\]

elimina a componente vertical.

Considere:

\[
x_1=\begin{bmatrix}3\\5\end{bmatrix}
\quad\text{e}\quad
x_2=\begin{bmatrix}3\\100\end{bmatrix}
\]

Então:

\[
Px_1=Px_2=\begin{bmatrix}3\\0\end{bmatrix}
\]

Duas entradas diferentes produziram a mesma saída. A informação sobre a segunda componente foi perdida.

## 10. Determinante e matriz inversa

Se:

\[
y=Ax
\]

queremos saber se podemos recuperar \(x\) a partir de \(y\).

Quando a transformação não perde informação, pode existir uma matriz inversa:

\[
A^{-1}
\]

com:

\[
A^{-1}Ax=x
\]

Mas quando:

\[
\det(A)=0
\]

a transformação destruiu informação e não pode ser completamente desfeita.

Para matrizes quadradas:

\[
\boxed{A\text{ é invertível}\iff\det(A)\neq0}
\]

## 11. Exemplo invertível

Para:

\[
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\]

\[
\det(A)=6\neq0
\]

A inversa existe:

\[
A^{-1}=\begin{bmatrix}1/2&0\\0&1/3\end{bmatrix}
\]

A transformação pode ser desfeita.

## 12. Determinante como área de um paralelogramo

Considere as colunas \(a\) e \(b\) de:

\[
A=\begin{bmatrix}|&|\\a&b\\|&|\end{bmatrix}
\]

Elas formam um paralelogramo. Sua área é:

\[
\boxed{|\det(A)|}
\]

Na Aula 06 vimos que as colunas de uma matriz mostram onde os vetores da base foram parar. Agora podemos acrescentar:

> **O determinante mede a área formada pelos vetores da base transformada.**

## 13. Determinante e independência linear

Se:

\[
\det(A)\neq0
\]

as colunas de uma matriz quadrada são linearmente independentes.

Se:

\[
\det(A)=0
\]

elas são linearmente dependentes e o volume formado por elas é zero.

## 14. Determinante em três dimensões

Para:

\[
A\in\mathbb{R}^{3\times3}
\]

\(|\det(A)|\) mede o fator de alteração do volume.

Exemplo:

\[
A=\begin{bmatrix}2&0&0\\0&2&0\\0&0&2\end{bmatrix}
\]

Um cubo 1×1×1 vira um cubo 2×2×2.

\[
\det(A)=2^3=8
\]

O volume foi multiplicado por oito.

Em \(n\) dimensões, o determinante mede o fator de escala do hipervolume.

## 15. Composição de transformações

Uma propriedade fundamental é:

\[
\boxed{\det(AB)=\det(A)\det(B)}
\]

Se uma transformação multiplica áreas por 2 e outra por 3, a composição multiplica por 6.

Exemplo:

\[
\det(A)=2,\quad\det(B)=5
\]

então:

\[
\det(BA)=10
\]

## 16. Determinante da inversa

Como:

\[
AA^{-1}=I
\]

então:

\[
\det(A)\det(A^{-1})=1
\]

Logo:

\[
\boxed{\det(A^{-1})=\frac{1}{\det(A)}}
\]

Se uma transformação multiplica áreas por 4, sua inversa precisa dividi-las por 4.

## 17. Matrizes diagonais e shear

Para uma matriz diagonal:

\[
D=\begin{bmatrix}d_1&0&0\\0&d_2&0\\0&0&d_3\end{bmatrix}
\]

\[
\boxed{\det(D)=d_1d_2d_3}
\]

Já um shear como:

\[
S=\begin{bmatrix}1&1\\0&1\end{bmatrix}
\]

possui:

\[
\det(S)=1
\]

O quadrado pode virar um paralelogramo sem alterar a área.

Isso mostra que o determinante não descreve toda a transformação. Ele resume especificamente mudança de volume e orientação.

## 18. NumPy

```python
import numpy as np

A = np.array([
    [2.0, 0.0],
    [0.0, 3.0]
])

print(np.linalg.det(A))
# 6.0
```

Para uma matriz singular:

```python
A = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])

print(np.linalg.det(A))
```

Matematicamente o resultado é zero. Em computação numérica, por arredondamentos, pode surgir um valor extremamente pequeno em vez de zero exato.

Por isso, em muitos contextos, prefira:

```python
if np.isclose(np.linalg.det(A), 0):
    print("Matriz singular")
```

## 19. Onde isso aparece em IA?

Você provavelmente não calculará determinantes manualmente todos os dias ao treinar redes neurais. Entretanto, o conceito fornece uma base profunda para:

- dimensionalidade;
- perda de informação;
- invertibilidade;
- independência linear;
- mudança de volume;
- transformações.

Aplicações mais avançadas incluem:

- distribuições Gaussianas multivariadas;
- matrizes de covariância;
- normalizing flows;
- Jacobianos;
- estatística multivariada.

Em normalizing flows, por exemplo, aparece:

\[
\log|\det(J)|
\]

onde \(J\) é uma matriz Jacobiana.

## 20. Preview: Jacobiano

Mais adiante em cálculo veremos funções:

\[
f:\mathbb{R}^n\rightarrow\mathbb{R}^n
\]

cuja derivada pode ser representada por uma matriz Jacobiana \(J\).

Então:

\[
|\det(J)|
\]

indica como pequenos volumes locais são expandidos ou comprimidos pela função.

Essa é uma ponte importante entre Álgebra Linear e Cálculo.

## 21. Perda de informação em IA

Se uma transformação quadrada \(W\) possui:

\[
\det(W)=0
\]

alguma informação foi colapsada. Entradas diferentes podem gerar a mesma saída.

Isso não significa que toda perda de dimensionalidade seja ruim. Em Machine Learning, às vezes queremos reduzir dimensionalidade de forma controlada para remover redundância ou ruído.

Mais adiante estudaremos:

- PCA;
- SVD.

A diferença importante é entre **compressão planejada** e **perda acidental de informação**.

## 22. Mapa mental

```text
matriz A
   ↓
transformação do espaço
   ↓
det(A)
   │
   ├── |det| > 1 → expansão de volume
   ├── 0 < |det| < 1 → compressão
   ├── det < 0 → orientação invertida
   └── det = 0
          ↓
     dimensão colapsada
          ↓
     perda de informação
          ↓
     sem inversa
```

## 23. Conectando as aulas

- Aula 01 — vetores, matrizes e tensores.
- Aula 02 — norma e geometria vetorial.
- Aula 03 — produto escalar.
- Aula 04 — similaridade de cosseno.
- Aula 05 — \(Wx+b\).
- Aula 06 — escala, rotação, reflexão e projeção.
- Aula 07 — \(\det(W)\), mudança de volume e perda de informação.

## Exercícios da Aula 07

### Questão 1

Calcule:

\[
A=\begin{bmatrix}2&0\\0&4\end{bmatrix}
\]

\[
\det(A)=?
\]

Depois explique geometricamente o resultado.

### Questão 2

Considere:

\[
A=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
\]

Calcule o determinante e explique por que ele é negativo.

### Questão 3

Calcule:

\[
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\]

Explique o resultado em termos de:

1. área;
2. dimensionalidade;
3. invertibilidade.

### Questão 4

Uma transformação possui:

\[
\det(A)=0.2
\]

Se uma região original possui área 50, qual será sua área depois da transformação?

### Questão 5

Se:

\[
\det(A)=3
\]

\[
\det(B)=4
\]

quanto vale:

\[
\det(BA)?
\]

### Questão 6

Uma matriz de rotação possui:

\[
\det(R)=1
\]

Explique geometricamente por que isso faz sentido.

### Questão 7 — a mais importante

Explique com suas palavras por que:

\[
\boxed{\det(A)=0}
\]

significa perda de informação. Relacione sua resposta com a ideia de transformar um plano inteiro em uma linha.

## Mini laboratório em NumPy

```python
import numpy as np

matrizes = {
    "identidade": np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),
    "escala": np.array([
        [2.0, 0.0],
        [0.0, 3.0]
    ]),
    "rotacao_90": np.array([
        [0.0, -1.0],
        [1.0,  0.0]
    ]),
    "reflexao": np.array([
        [1.0,  0.0],
        [0.0, -1.0]
    ]),
    "projecao": np.array([
        [1.0, 0.0],
        [0.0, 0.0]
    ])
}

for nome, matriz in matrizes.items():
    print(nome, np.linalg.det(matriz))
```

Antes de executar, tente prever os resultados para identidade, escala, rotação, reflexão e projeção.

## O que você precisa sair sabendo

Quando olhar para:

\[
\det(A)
\]

não pense apenas em \(ad-bc\). Pense:

\[
\boxed{\det(A)=\text{como a transformação altera volume e orientação}}
\]

Guarde principalmente:

\[
|\det(A)|>1
\]

→ expansão de volume.

\[
0<|\det(A)|<1
\]

→ compressão.

\[
\det(A)<0
\]

→ orientação invertida.

E principalmente:

\[
\boxed{\det(A)=0}
\]

→ alguma dimensão foi colapsada;  
→ informação foi perdida;  
→ a transformação não pode ser completamente desfeita;  
→ a matriz não possui inversa.

## Próxima aula

### Aula 08 — Matriz inversa: desfazendo transformações

Na próxima aula estudaremos:

\[
\boxed{A^{-1}}
\]

Vamos entender:

- o que significa inverter uma transformação;
- por que \(A^{-1}A=I\);
- como resolver sistemas lineares;
- quando uma matriz não possui inversa;
- a relação entre inversa, determinante e independência linear;
- por que, em computação numérica, normalmente evitamos calcular explicitamente a inversa quando queremos resolver \(Ax=b\).
