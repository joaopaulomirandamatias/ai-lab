# Aula 25 — Hessiana, curvatura e aproximação de Taylor

**Trilha:** Matemática para IA  
**Etapa:** Cálculo multivariável  
**Pré-requisito:** gradiente e Jacobiano  
**Objetivo:** entender curvatura, Hessiana e como aproximações de segunda ordem ajudam a analisar otimização e estabilidade.

## 1. Gradiente diz para onde; Hessiana diz como a inclinação muda

Para:

\[
f:\mathbb{R}^n\to\mathbb{R}
\]

o gradiente é:

\[
\nabla f
\]

A Hessiana reúne as segundas derivadas:

\[
\boxed{H_f(x)=\nabla^2f(x)}
\]

## 2. Forma da Hessiana

\[
H=
\begin{bmatrix}
\partial^2 f/\partial x_1^2&\cdots&\partial^2 f/\partial x_1\partial x_n\\
\vdots&\ddots&\vdots\\
\partial^2 f/\partial x_n\partial x_1&\cdots&\partial^2 f/\partial x_n^2
\end{bmatrix}
\]

Sob condições regulares, as derivadas cruzadas coincidem e a Hessiana é simétrica.

## 3. Exemplo quadrático

\[
f(x,y)=x^2+3y^2
\]

Gradiente:

\[
\nabla f=\begin{bmatrix}2x\\6y\end{bmatrix}
\]

Hessiana:

\[
\boxed{H=\begin{bmatrix}2&0\\0&6\end{bmatrix}}
\]

A função possui curvatura maior na direção \(y\).

## 4. Autovalores da Hessiana

Como a Hessiana é simétrica em muitos problemas suaves, seus autovalores descrevem curvatura em direções ortogonais especiais.

- todos positivos → mínimo local estrito em um ponto crítico;
- todos negativos → máximo local estrito;
- sinais mistos → ponto de sela.

Isso conecta Cálculo diretamente aos autovalores das Aulas 13–14.

## 5. Ponto de sela

Considere:

\[
f(x,y)=x^2-y^2
\]

No ponto \((0,0)\):

\[
\nabla f=0
\]

mas:

\[
H=\begin{bmatrix}2&0\\0&-2\end{bmatrix}
\]

Há uma direção de subida e outra de descida. Logo o gradiente zero não representa um mínimo.

## 6. Taylor de primeira ordem

Já vimos:

\[
f(x+\Delta)\approx f(x)+\nabla f(x)^T\Delta
\]

## 7. Taylor de segunda ordem

Incluindo curvatura:

\[
\boxed{f(x+\Delta)\approx f(x)+\nabla f^T\Delta+\frac12\Delta^TH\Delta}
\]

Esse modelo quadrático local ajuda a motivar métodos de segunda ordem como Newton.

## 8. Método de Newton — prévia

Em várias dimensões, uma atualização idealizada de Newton é:

\[
\theta_{t+1}=\theta_t-H^{-1}\nabla f
\]

Diferente do gradient descent, ele ajusta o passo levando em conta curvatura.

Para redes enormes, formar/inverter a Hessiana completa é geralmente inviável, mas a ideia inspira muitos métodos de otimização e aproximações.

## 9. Condicionamento da superfície

Se a curvatura é muito diferente entre direções, as curvas de nível ficam alongadas.

Gradient descent pode “ziguezaguear” e exigir learning rate menor. O número de condição da Hessiana em problemas quadráticos ajuda a quantificar essa dificuldade.

## 10. Saddle points em Deep Learning

Superfícies de loss de redes profundas são não convexas e de altíssima dimensão. Pontos de sela e regiões de curvatura complexa são comuns na teoria de otimização não convexa.

Por isso a frase “gradiente zero = encontrei o mínimo” não é segura.

## 11. Hessian-vector products

Assim como não precisamos formar Jacobianos completos, é possível calcular produtos:

\[
Hv
\]

sem materializar toda a Hessiana, usando técnicas de autodiff. Isso permite estudar curvatura em modelos maiores.

## 12. NumPy

Para uma função quadrática:

```python
import numpy as np

H = np.array([
    [2., 0.],
    [0., 6.]
])

values, vectors = np.linalg.eigh(H)
print(values)
```

Os autovalores 2 e 6 revelam curvatura distinta nos dois eixos.

## 13. Conexão com IA

- análise de estabilidade e learning rate;
- métodos de segunda ordem;
- curvatura de losses;
- diagnóstico de saddle points;
- compreensão de condicionamento;
- aproximações locais usadas em otimização.

## Exercícios

1. Calcule a Hessiana de \(f(x,y)=x^2+xy+y^2\).
2. O que autovalores positivos da Hessiana indicam em um ponto crítico?
3. Por que \(f=x^2-y^2\) tem um ponto de sela na origem?
4. Escreva Taylor de segunda ordem.
5. Por que Newton pode convergir rápido perto de um mínimo bem comportado?
6. Por que não calculamos a Hessiana completa de um LLM com bilhões de parâmetros?

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Caps. 5 e 7. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4. https://www.deeplearningbook.org/contents/numerical.html
- BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/

## Próxima aula

**Aula 26 — Otimização, funções de perda e convexidade.**