# Aula 23 — Derivadas parciais e gradiente: cálculo em muitas dimensões

**Trilha:** Matemática para IA  
**Etapa:** Cálculo multivariável  
**Pré-requisito:** regra da cadeia  
**Objetivo:** generalizar derivadas para funções de muitos parâmetros e entender o gradiente como direção de maior crescimento local.

## 1. Losses dependem de muitos parâmetros

Em vez de:

\[
L(w)
\]

uma rede neural possui:

\[
L(w_1,w_2,\ldots,w_n)
\]

com \(n\) podendo chegar a bilhões.

Precisamos medir a sensibilidade da loss a cada parâmetro.

## 2. Derivada parcial

A derivada parcial em relação a \(x_i\) varia apenas aquela coordenada, mantendo as outras fixas:

\[
\frac{\partial f}{\partial x_i}
\]

Exemplo:

\[
f(x,y)=x^2+3xy+y^2
\]

Então:

\[
\frac{\partial f}{\partial x}=2x+3y
\]

\[
\frac{\partial f}{\partial y}=3x+2y
\]

## 3. Gradiente

Reunimos todas as derivadas parciais:

\[
\boxed{\nabla f(x)=
\begin{bmatrix}
\partial f/\partial x_1\\
\vdots\\
\partial f/\partial x_n
\end{bmatrix}}
\]

Para o exemplo:

\[
\nabla f(x,y)=\begin{bmatrix}2x+3y\\3x+2y\end{bmatrix}
\]

## 4. Interpretação geométrica

O gradiente aponta na direção de maior crescimento local da função.

Por isso:

\[
-\nabla f
\]

aponta na direção de maior descida local.

Essa propriedade é a base do gradient descent.

## 5. Derivada direcional

Se \(u\) é um vetor unitário, a taxa de mudança na direção \(u\) é:

\[
D_uf=\nabla f^Tu
\]

Pelo produto escalar:

\[
D_uf=\|\nabla f\|\cos\theta
\]

O valor é máximo quando \(u\) aponta na mesma direção do gradiente.

Isso conecta diretamente cálculo multivariável à geometria vetorial das primeiras aulas.

## 6. Exemplo quadrático

\[
f(x,y)=x^2+y^2
\]

Então:

\[
\nabla f=\begin{bmatrix}2x\\2y\end{bmatrix}
\]

No ponto \((3,4)\):

\[
\nabla f(3,4)=\begin{bmatrix}6\\8\end{bmatrix}
\]

Ele aponta para fora do centro. A direção negativa aponta de volta para o mínimo em \((0,0)\).

## 7. Aproximação linear multivariada

Para uma pequena mudança \(\Delta x\):

\[
\boxed{f(x+\Delta x)\approx f(x)+\nabla f(x)^T\Delta x}
\]

O gradiente é, portanto, a versão multivariada da inclinação local.

## 8. Gradiente de uma função linear

Se:

\[
f(x)=w^Tx+b
\]

então:

\[
\nabla_xf=w
\]

A própria matriz/vetor de pesos descreve a sensibilidade da saída às componentes da entrada.

## 9. Gradiente de norma quadrática

\[
f(x)=\|x\|^2=x^Tx
\]

Então:

\[
\boxed{\nabla_xf=2x}
\]

Essa identidade aparece repetidamente em least squares e regularização L2.

## 10. Gradiente de least squares

Considere:

\[
L(w)=\frac{1}{n}\|Xw-y\|^2
\]

Seu gradiente é:

\[
\boxed{\nabla_wL=\frac{2}{n}X^T(Xw-y)}
\]

Essa expressão será implementada do zero na Aula 29.

## 11. Shapes importam

Se:

\[
X\in\mathbb{R}^{n\times d},\quad w\in\mathbb{R}^d
\]

então:

\[
Xw-y\in\mathbb{R}^n
\]

E:

\[
X^T(Xw-y)\in\mathbb{R}^d
\]

O gradiente precisa ter a mesma shape dos parâmetros que ele atualiza.

Essa checagem dimensional detecta muitos erros de derivação.

## 12. NumPy

```python
import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return 2*x

x = np.array([3., 4.])
print(f(x))
print(grad_f(x))
```

## 13. Gradient check multivariado

```python
def numerical_grad(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2*h)
    return g

print(numerical_grad(f, x))
```

Compare com `grad_f(x)`.

## 14. Conexão com treinamento

Treinar significa ajustar parâmetros para diminuir uma loss. O gradiente fornece uma aproximação local de como cada parâmetro influencia essa loss.

Para milhões de parâmetros, autodiff calcula esse vetor de sensibilidades sem precisarmos derivar manualmente cada termo.

## Exercícios

1. Calcule as derivadas parciais de \(f(x,y)=x^2+xy+3y^2\).
2. Monte o gradiente dessa função.
3. Qual a direção de maior descida local?
4. Por que o gradiente tem a mesma dimensão do vetor de parâmetros?
5. Calcule \(\nabla\|x\|^2\).
6. Explique a ligação entre derivada direcional e produto escalar.

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 5. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4. https://www.deeplearningbook.org/contents/numerical.html
- BAYDIN, A. G. et al. *Automatic Differentiation in Machine Learning: a Survey*. JMLR, 2018. arXiv:1502.05767.

## Próxima aula

**Aula 24 — Jacobiano, regra da cadeia multivariada e autodiff.**