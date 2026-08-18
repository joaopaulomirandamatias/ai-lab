# Aula 24 — Jacobiano, regra da cadeia multivariada e autodiff

**Trilha:** Matemática para IA  
**Etapa:** Cálculo multivariável  
**Pré-requisito:** derivadas parciais e gradiente  
**Objetivo:** entender como derivar funções vetoriais e conectar Jacobianos à regra da cadeia usada por autodiff e backpropagation.

## 1. Quando a saída também é um vetor

Até agora estudamos principalmente:

\[
f:\mathbb{R}^n\rightarrow\mathbb{R}
\]

Mas camadas de redes neurais normalmente fazem:

\[
f:\mathbb{R}^n\rightarrow\mathbb{R}^m
\]

Exemplo:

\[
f(x,y)=\begin{bmatrix}x^2+y\\xy\end{bmatrix}
\]

Precisamos registrar como cada saída muda com cada entrada.

## 2. Jacobiano

O Jacobiano é a matriz de derivadas parciais:

\[
\boxed{J_f(x)=
\begin{bmatrix}
\partial f_1/\partial x_1&\cdots&\partial f_1/\partial x_n\\
\vdots&\ddots&\vdots\\
\partial f_m/\partial x_1&\cdots&\partial f_m/\partial x_n
\end{bmatrix}}
\]

Se \(f:\mathbb{R}^n\to\mathbb{R}^m\), então:

\[
J_f\in\mathbb{R}^{m\times n}
\]

## 3. Exemplo

Para:

\[
f_1=x^2+y,\qquad f_2=xy
\]

obtemos:

\[
J_f(x,y)=
\begin{bmatrix}
2x&1\\
y&x
\end{bmatrix}
\]

No ponto \((2,3)\):

\[
J_f(2,3)=\begin{bmatrix}4&1\\3&2\end{bmatrix}
\]

## 4. Jacobiano como melhor transformação linear local

Para uma pequena perturbação \(\Delta x\):

\[
\boxed{f(x+\Delta x)\approx f(x)+J_f(x)\Delta x}
\]

Essa é a generalização da reta tangente para funções vetoriais.

O Jacobiano descreve como pequenas mudanças na entrada são transformadas em pequenas mudanças na saída.

## 5. Regra da cadeia multivariada

Se:

\[
y=g(x),\qquad z=f(y)
\]

então:

\[
\boxed{J_{f\circ g}(x)=J_f(g(x))J_g(x)}
\]

A ordem das matrizes importa. Shapes ajudam a validar a derivação.

## 6. Exemplo de shapes

Se:

\[
g:\mathbb{R}^3\to\mathbb{R}^5
\]

então:

\[
J_g\in\mathbb{R}^{5\times3}
\]

Se:

\[
f:\mathbb{R}^5\to\mathbb{R}^2
\]

então:

\[
J_f\in\mathbb{R}^{2\times5}
\]

Logo:

\[
J_fJ_g\in\mathbb{R}^{2\times3}
\]

que corresponde corretamente à composição \(\mathbb{R}^3\to\mathbb{R}^2\).

## 7. Jacobiano de uma camada linear

Para:

\[
y=Wx+b
\]

em relação a \(x\):

\[
\boxed{J_y=W}
\]

Isso mostra que a matriz de pesos é literalmente a transformação linear local da camada.

## 8. Vector-Jacobian products

Em redes neurais enormes, materializar um Jacobiano completo seria muito caro. Backpropagation normalmente calcula produtos do tipo:

\[
v^TJ
\]

sem construir explicitamente toda a matriz.

Isso é uma razão prática pela qual reverse-mode autodiff escala bem para uma loss escalar e muitos parâmetros.

## 9. Forward mode versus reverse mode

- **Forward mode:** propaga sensibilidades da entrada para a saída; eficiente quando há poucas entradas.
- **Reverse mode:** propaga sensitividades da saída para trás; eficiente quando há poucas saídas, especialmente uma loss escalar.

Treinamento de redes neurais é o caso clássico do reverse mode.

## 10. Autodiff

Automatic differentiation registra operações elementares e aplica regra da cadeia de maneira algorítmica.

Ele não precisa manipular uma expressão simbólica gigante e não depende de aproximações por diferenças finitas.

Frameworks modernos constroem ou rastreiam grafos computacionais para calcular gradientes eficientemente.

## 11. PyTorch — uma prévia

```python
import torch

x = torch.tensor([2., 3.], requires_grad=True)
y = x[0]**2 + 3*x[0]*x[1] + x[1]**2

y.backward()
print(x.grad)
```

O resultado é o gradiente de \(y\) em relação a \(x\).

No M6 usaremos PyTorch profundamente. Aqui o objetivo é entender o que `backward()` está calculando matematicamente.

## 12. Jacobianos e redes profundas

Uma composição de muitas camadas produz produtos de Jacobianos. Isso ajuda a explicar:

- vanishing gradients;
- exploding gradients;
- sensibilidade das representações;
- importância da inicialização e normalização.

Se sucessivos Jacobianos contraem fortemente os vetores, o gradiente pode desaparecer. Se expandem demais, pode explodir.

## Exercícios

1. Calcule o Jacobiano de \(f(x,y)=[x+y,x-y]^T\).
2. Qual a shape do Jacobiano para \(f:\mathbb{R}^{10}\to\mathbb{R}^4\)?
3. Explique a aproximação \(f(x+\Delta x)\approx f(x)+J\Delta x\).
4. Por que a ordem na regra da cadeia matricial importa?
5. Por que reverse mode é adequado ao treinamento de redes?
6. Qual é o Jacobiano de \(y=Wx+b\) em relação a \(x\)?

## Referências

- BAYDIN, A. G. et al. *Automatic Differentiation in Machine Learning: a Survey*. JMLR, 2018. arXiv:1502.05767.
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 5. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Caps. 4 e 6. https://www.deeplearningbook.org/

## Próxima aula

**Aula 25 — Hessiana, curvatura e aproximação de Taylor.**