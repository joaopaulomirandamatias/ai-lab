# Aula 26 — Otimização, funções de perda e convexidade

**Trilha:** Matemática para IA  
**Etapa:** Otimização  
**Pré-requisitos:** gradiente e Hessiana  
**Objetivo:** entender treinamento como um problema de otimização e distinguir problemas convexos de superfícies não convexas de redes neurais.

## 1. Treinar é minimizar uma função

Um modelo possui parâmetros:

\[
\theta
\]

Uma função de perda mede a qualidade:

\[
L(\theta)
\]

Treinar significa procurar:

\[
\boxed{\theta^*=\arg\min_\theta L(\theta)}
\]

Essa formulação é uma das ideias unificadoras de Machine Learning.

## 2. Loss de uma amostra versus objetivo total

Para dados \((x_i,y_i)\):

\[
L(\theta)=\frac1n\sum_{i=1}^n\ell(f_\theta(x_i),y_i)
\]

- \(\ell\): perda de uma observação;
- \(L\): objetivo empírico agregado.

Regularização pode adicionar termos extras.

## 3. Mean Squared Error

Para regressão:

\[
\boxed{MSE=\frac1n\sum_i(\hat y_i-y_i)^2}
\]

O quadrado penaliza erros grandes e produz uma função suave conveniente para derivação.

## 4. Log-loss / cross-entropy binária

Para \(y\in\{0,1\}\) e probabilidade prevista \(p\):

\[
\boxed{\ell=-[y\log p+(1-y)\log(1-p)]}
\]

Ela é a negative log-likelihood do modelo Bernoulli e será usada na regressão logística da Aula 30.

## 5. Convexidade

Uma função é convexa quando, intuitivamente, o segmento entre dois pontos do gráfico nunca fica abaixo da função.

Formalmente:

\[
f(tx+(1-t)y)\le tf(x)+(1-t)f(y)
\]

para \(t\in[0,1]\).

## 6. Por que convexidade é especial?

Em um problema convexo bem formulado, qualquer mínimo local é também global.

Isso elimina uma grande fonte de ambiguidade da otimização.

Regressão linear com MSE é um exemplo clássico de objetivo convexo nos parâmetros.

## 7. Convexidade e Hessiana

Para funções duas vezes diferenciáveis, uma condição útil é:

\[
H_f(x)\succeq0
\]

ou seja, Hessiana positiva semidefinida.

Assim, autovalores da Hessiana retornam como critério de curvatura.

## 8. Redes neurais são não convexas

Uma rede profunda possui muitas camadas e simetrias entre parâmetros. A loss em relação a todos os pesos não é, em geral, convexa.

Isso significa que podem existir:

- múltiplos mínimos;
- saddle points;
- vales planos;
- geometrias complexas.

Mesmo assim, métodos baseados em gradiente funcionam muito bem na prática.

## 9. Gradiente como bússola local

O gradiente fornece informação local:

\[
\nabla L(\theta)
\]

Mover contra o gradiente tende a reduzir a loss para passos suficientemente pequenos:

\[
\theta\leftarrow\theta-\eta\nabla L
\]

Mas o gradiente não “vê” toda a superfície. É uma bússola local, não um mapa global.

## 10. Learning rate

\[
\eta
\]

controla o tamanho do passo.

- muito pequeno → aprendizado lento;
- muito grande → oscilações/divergência;
- adequado → progresso estável.

A próxima aula analisará isso experimentalmente.

## 11. Restrições

Alguns problemas exigem:

\[
\min f(x)\quad\text{sujeito a }g_i(x)\le0
\]

Deep Learning frequentemente usa otimização não restrita ou restrições incorporadas à parametrização/loss, mas otimização convexa geral vai muito além do escopo deste Gate I.

## 12. NumPy: uma loss simples

```python
import numpy as np

def loss(w):
    return (w - 3)**2

def grad(w):
    return 2*(w - 3)

for w in [-2., 0., 2., 3., 5.]:
    print(w, loss(w), grad(w))
```

Observe que o sinal do gradiente aponta para longe do mínimo e o negativo do gradiente aponta de volta.

## 13. Modelo, loss e otimizador são peças diferentes

É importante separar:

```text
modelo → produz previsão
loss → mede erro
otimizador → escolhe como atualizar parâmetros
```

Mudar a loss muda o objetivo. Mudar o otimizador muda a trajetória usada para procurar bons parâmetros.

## Exercícios

1. Escreva o problema de treinamento como `argmin`.
2. Qual a diferença entre loss por amostra e loss média?
3. Por que convexidade facilita otimização?
4. O que a Hessiana positiva semidefinida sugere?
5. Por que redes profundas não são, em geral, problemas convexos?
6. Explique o papel do learning rate.

## Referências

- BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 7 — Continuous Optimization. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4. https://www.deeplearningbook.org/contents/numerical.html

## Próxima aula

**Aula 27 — Gradient Descent do zero: derivação, learning rate e convergência.**