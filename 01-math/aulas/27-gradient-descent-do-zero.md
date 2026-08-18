# Aula 27 — Gradient Descent do zero: derivação, learning rate e convergência

**Trilha:** Matemática para IA  
**Etapa:** Otimização  
**Pré-requisitos:** gradiente e função de perda  
**Objetivo:** derivar e implementar gradient descent, entendendo por que a atualização funciona e como o learning rate altera a trajetória.

## 1. A regra central

\[
\boxed{\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)}
\]

Essa será uma das equações mais importantes de toda a formação.

## 2. Por que subtrair o gradiente?

Para uma pequena mudança \(\Delta\theta\):

\[
L(\theta+\Delta)\approx L(\theta)+\nabla L^T\Delta
\]

Se escolhermos:

\[
\Delta=-\eta\nabla L
\]

então:

\[
\nabla L^T\Delta=-\eta\|\nabla L\|^2\le0
\]

Para passos suficientemente pequenos, a loss tende a diminuir.

Essa é a justificativa local do gradient descent.

## 3. Exemplo unidimensional

\[
L(w)=(w-3)^2
\]

Gradiente:

\[
L'(w)=2(w-3)
\]

Atualização:

\[
w_{t+1}=w_t-2\eta(w_t-3)
\]

Comece em \(w_0=0\) e \(\eta=0.1\):

\[
w_1=0-0.1(-6)=0.6
\]

O parâmetro caminhou em direção a 3.

## 4. Implementação mínima

```python
def loss(w):
    return (w - 3)**2

def grad(w):
    return 2*(w - 3)

w = 0.0
lr = 0.1

for step in range(30):
    w = w - lr * grad(w)
    print(step, w, loss(w))
```

Não use otimizador pronto nesta aula.

## 5. Learning rate pequeno

Com \(\eta\) muito pequeno:

```text
passos minúsculos
→ progresso estável
→ muitas iterações
```

O algoritmo pode convergir, mas lentamente.

## 6. Learning rate grande

Com \(\eta\) excessivo:

```text
pula de um lado para o outro
→ pode oscilar
→ pode divergir
```

Para uma quadrática, podemos analisar matematicamente a faixa de estabilidade.

## 7. Experimento obrigatório: três learning rates

Implemente e compare pelo menos:

\[
\eta\in\{0.01,0.1,1.1\}
\]

para uma função quadrática apropriada.

Registre:

- loss por iteração;
- valor do parâmetro;
- convergência ou divergência.

Produza um gráfico com as três curvas. Esse tipo de experimento fará parte do Gate I.

## 8. Muitas dimensões

Para:

\[
\theta\in\mathbb{R}^d
\]

nada muda conceitualmente:

\[
\theta\leftarrow\theta-\eta\nabla L
\]

O gradiente possui uma componente para cada parâmetro.

## 9. Curvas de nível

Em duas dimensões, gradient descent pode ser visualizado sobre curvas de nível. Em uma quadrática mal condicionada, a trajetória pode ziguezaguear entre paredes estreitas do vale.

Isso conecta learning rate ao condicionamento e à Hessiana.

## 10. Critérios de parada

Possibilidades:

- número máximo de iterações;
- \(\|\nabla L\|\) pequeno;
- mudança de loss pequena;
- mudança dos parâmetros pequena;
- desempenho em conjunto de validação.

Em ML real, early stopping pode depender de generalização, não apenas da training loss.

## 11. Convergência não significa generalização

Otimizar muito bem a loss de treino não garante bom desempenho em dados novos.

Essa distinção será central no módulo de Machine Learning:

```text
otimização → ajustar treino
generalização → funcionar fora do treino
```

## 12. Gradient checking

Antes de confiar em um gradiente derivado manualmente, compare com diferenças finitas em problemas pequenos.

```python
def numerical_grad(f, w, h=1e-5):
    return (f(w+h)-f(w-h))/(2*h)

print(grad(2.0), numerical_grad(loss, 2.0))
```

## 13. NumPy vetorial

```python
import numpy as np

def loss(theta):
    return np.sum((theta - np.array([2., -1.]))**2)

def grad(theta):
    return 2*(theta - np.array([2., -1.]))

theta = np.array([0., 0.])
for _ in range(100):
    theta -= 0.1 * grad(theta)

print(theta)
```

## Exercícios

1. Derive a atualização para \(L(w)=(w-5)^2\).
2. Faça manualmente duas iterações com \(w_0=0\) e \(\eta=0.1\).
3. Explique por que o negativo do gradiente é uma direção de descida local.
4. Compare learning rates pequeno, adequado e excessivo.
5. Por que training loss baixa não garante generalização?
6. Implemente o experimento com três learning rates e gere um gráfico.

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 7. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4 — Gradient-Based Optimization. https://www.deeplearningbook.org/contents/numerical.html
- BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Capítulos de métodos de descida. https://web.stanford.edu/~boyd/cvxbook/

## Próxima aula

**Aula 28 — SGD, mini-batch, momentum e dinâmica de treinamento.**