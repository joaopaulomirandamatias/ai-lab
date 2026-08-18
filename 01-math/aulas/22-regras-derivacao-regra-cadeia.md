# Aula 22 — Regras de derivação e regra da cadeia: o coração do backpropagation

**Trilha:** Matemática para IA  
**Etapa:** Cálculo  
**Pré-requisito:** derivadas  
**Objetivo:** dominar regras de derivação e entender a regra da cadeia como mecanismo de propagação de sensibilidade em funções compostas.

## 1. Redes neurais são composições

Uma rede simples pode ser escrita como:

\[
y=f(g(h(x)))
\]

Para saber como \(y\) muda quando \(x\) muda, precisamos atravessar todas as transformações intermediárias.

A ferramenta é a **regra da cadeia**.

## 2. Regra da soma

Se:

\[
f(x)=u(x)+v(x)
\]

então:

\[
f'(x)=u'(x)+v'(x)
\]

## 3. Regra do produto

\[
(uv)'=u'v+uv'
\]

## 4. Regra do quociente

\[
\left(\frac uv\right)'=\frac{u'v-uv'}{v^2}
\]

quando \(v\neq0\).

## 5. Regra da cadeia

Se:

\[
y=f(u),\qquad u=g(x)
\]

então:

\[
\boxed{\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}}
\]

Ou:

\[
(f\circ g)'(x)=f'(g(x))g'(x)
\]

## 6. Exemplo

\[
y=(3x+1)^2
\]

Defina:

\[
u=3x+1
\]

Então:

\[
y=u^2
\]

Temos:

\[
\frac{dy}{du}=2u
\]

\[
\frac{du}{dx}=3
\]

Logo:

\[
\frac{dy}{dx}=2(3x+1)\cdot3=6(3x+1)
\]

## 7. Grafo computacional

Podemos representar:

```text
x
↓
u = 3x + 1
↓
y = u²
```

Forward pass:

```text
x → u → y
```

Backward pass:

```text
dy/dy → dy/du → dy/dx
```

É exatamente essa estrutura que o backpropagation explora.

## 8. Derivada local × gradiente acumulado

Cada nó sabe apenas sua derivada local. O gradiente total é obtido multiplicando sensibilidades ao longo dos caminhos.

Se:

\[
z=f(y),\quad y=g(x)
\]

então:

\[
\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}
\]

Essa modularidade permite diferenciar programas complexos.

## 9. Exemplo com neurônio

Considere:

\[
z=wx+b
\]

\[
y=\sigma(z)
\]

Queremos:

\[
\frac{dy}{dw}
\]

Pela cadeia:

\[
\frac{dy}{dw}=\frac{dy}{dz}\frac{dz}{dw}
\]

Sabemos:

\[
\frac{dy}{dz}=\sigma(z)(1-\sigma(z))
\]

\[
\frac{dz}{dw}=x
\]

Logo:

\[
\boxed{\frac{dy}{dw}=\sigma(z)(1-\sigma(z))x}
\]

## 10. Ramificações

Se uma variável influencia a saída por vários caminhos, as contribuições são somadas.

Isso vem da regra multivariada da cadeia e explica por que frameworks acumulam gradientes quando um tensor é reutilizado em diferentes partes do grafo.

## 11. Backpropagation

Backpropagation é uma aplicação eficiente do **reverse-mode automatic differentiation** a grafos computacionais, especialmente adequada quando temos uma saída escalar (loss) e muitos parâmetros.

O artigo clássico de Rumelhart, Hinton e Williams (1986) popularizou o treinamento de redes multicamadas por propagação de erros, embora as ideias de diferenciação reversa sejam mais amplas e anteriores em computação científica.

## 12. Autodiff não é diferenciação simbólica

- simbólica → manipula expressões algébricas;
- diferença finita → aproxima derivadas numericamente;
- automatic differentiation → aplica sistematicamente regras da cadeia às operações executadas.

Autodiff produz derivadas precisas até erro de ponto flutuante e é a base de PyTorch, JAX e outros frameworks.

## 13. NumPy: backprop manual de um escalar

```python
import numpy as np

x = 2.0
w = 0.5
b = 0.1

z = w*x + b
y = 1/(1 + np.exp(-z))

dy_dz = y*(1-y)
dz_dw = x
dy_dw = dy_dz * dz_dw

print(y, dy_dw)
```

Compare com uma diferença finita para validar.

## Exercícios

1. Derive \((x^2+1)^3\) usando cadeia.
2. Derive \(e^{2x}\).
3. Para \(z=wx+b\), calcule \(dz/dw\), \(dz/dx\) e \(dz/db\).
4. Explique forward e backward pass.
5. Diferencie autodiff de diferença finita.
6. Por que reverse-mode é adequado quando uma loss escalar depende de milhões de parâmetros?

## Referências

- RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. *Learning representations by back-propagating errors*. Nature, 323, 533–536, 1986. DOI: 10.1038/323533a0.
- BAYDIN, A. G. et al. *Automatic Differentiation in Machine Learning: a Survey*. Journal of Machine Learning Research, 2018. arXiv:1502.05767.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Caps. 4 e 6. https://www.deeplearningbook.org/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 5. https://mml-book.github.io/

## Próxima aula

**Aula 23 — Derivadas parciais e gradiente: cálculo em muitas dimensões.**