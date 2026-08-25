# Aula 01 — Do modelo linear ao neurônio artificial

- **Trilha:** Especialista em IA
- **Módulo:** M5 · Redes Neurais do Zero
- **Pré-requisito:** álgebra linear, derivadas, regra da cadeia, gradient descent e ML clássico (M1–M4)
- **Objetivo central:** construir um neurônio artificial como uma composição de operações, executar o forward e derivar $\partial L/\partial W$, $\partial L/\partial b$ e $\partial L/\partial x$ sem autograd.

## Objetivos de aprendizagem

Ao final, você deve conseguir:

1. separar pré-ativação, ativação, previsão e loss;
2. interpretar peso e viés sem recorrer a analogias biológicas imprecisas;
3. calcular o forward de um neurônio escalar e em lote;
4. derivar os gradientes locais da transformação afim;
5. aplicar a regra da cadeia até os parâmetros e a entrada;
6. implementar e treinar um classificador de um neurônio em NumPy;
7. conferir o gradiente analítico com diferenças finitas.

## Intuição

Um neurônio artificial é uma unidade de computação parametrizada. Ele recebe números,
combina-os com pesos, adiciona um deslocamento e aplica uma função. A forma central é:

$$
z = w^Tx+b, \qquad a=\phi(z).
$$

$z$ é a **pré-ativação**; $a$ é a saída depois da ativação. Os pesos não representam
“importância” de forma universal. Seu sinal e magnitude dependem de escala, correlação,
codificação e das outras camadas. O viés permite deslocar a fronteira de decisão: sem ele,
o hiperplano linear seria forçado a passar pela origem.

A palavra “neurônio” é histórica, mas a analogia biológica deve parar cedo. O objeto que
estudaremos é uma composição diferenciável de álgebra linear e funções não lineares. Essa
precisão importa: backpropagation não exige imaginar dendritos; exige conhecer dependências,
derivadas locais e a regra da cadeia.

## Fundamento matemático

### Transformação afim

Para $x,w\in\mathbb{R}^d$ e $b\in\mathbb{R}$:

$$
z=\sum_{j=1}^{d}w_jx_j+b.
$$

As derivadas locais são:

$$
\frac{\partial z}{\partial w_j}=x_j,\qquad
\frac{\partial z}{\partial x_j}=w_j,\qquad
\frac{\partial z}{\partial b}=1.
$$

Se $a=\phi(z)$ e a loss escalar é $L(a,y)$, a regra da cadeia introduz o gradiente
upstream $\delta=\partial L/\partial z$:

$$
\delta=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}.
$$

Assim:

$$
\frac{\partial L}{\partial w_j}=\delta x_j,\qquad
\frac{\partial L}{\partial b}=\delta,\qquad
\frac{\partial L}{\partial x_j}=\delta w_j.
$$

Essas três equações serão reutilizadas em toda camada densa. O backward não é uma fórmula
mágica separada: é a derivada da operação feita no forward, multiplicada pelo gradiente que
chega das operações posteriores.

### Forma em lote e shapes

Adotaremos exemplos nas linhas. Para $X\in\mathbb{R}^{n\times d}$,
$W\in\mathbb{R}^{d\times 1}$ e $b\in\mathbb{R}^{1}$:

$$
Z=XW+b,\qquad A=\phi(Z).
$$

Se $dZ\in\mathbb{R}^{n\times1}$ contém $\partial L/\partial Z$, então:

$$
dW=X^TdZ,\qquad db=\sum_{i=1}^{n}dZ_i,\qquad dX=dZW^T.
$$

Confira shapes antes de confiar no resultado:

- $X^TdZ$: $(d\times n)(n\times1)=(d\times1)$, igual a $W$;
- $db$: soma sobre exemplos, shape $(1,)$, igual a $b$;
- $dZW^T$: $(n\times1)(1\times d)=(n\times d)$, igual a $X$.

Dividir por $n$ depende de onde a média foi definida. Se a loss já é a média do lote,
$dZ$ deve carregar esse fator. Duplicar ou esquecer $1/n$ muda a escala do gradiente.

## Explicação passo a passo

### 1. Declare o instante e o significado da entrada

Antes do cálculo, cada componente de $x$ deve corresponder a informação disponível no
momento da previsão. Redes neurais não corrigem data leakage; elas podem explorá-lo com
ainda mais eficiência.

### 2. Calcule a pré-ativação

Faça o produto interno e some o viés. Guarde $x$, $w$ e $z$: o backward precisará desses
intermediários. Em redes profundas, esse conjunto de valores guardados é chamado de cache.

### 3. Aplique uma ativação

Sem não linearidade, empilhar camadas afins continua sendo uma única transformação afim:
$W_2(W_1x+b_1)+b_2=\widetilde W x+\widetilde b$. A ativação é o que permite construir
fronteiras não lineares. Nesta aula usaremos sigmoid para probabilidade binária e ReLU para
expor a derivada local; ambas serão estudadas em profundidade na Aula 04.

### 4. Calcule uma loss escalar

A loss conecta previsão e objetivo. Para regressão, uma escolha didática é
$L=\tfrac12(a-y)^2$. Para classificação binária com $p=\sigma(z)$, usamos:

$$
L=-\left[y\log p+(1-y)\log(1-p)\right].
$$

Com sigmoid seguida de binary cross-entropy, a derivada em relação ao logit simplifica para
$\partial L/\partial z=p-y$. A simplificação deve ser derivada, não apenas memorizada.

### 5. Propague o gradiente para trás

Comece em $L$ e percorra as operações na ordem inversa. Multiplique gradiente upstream pela
derivada local. Em cada etapa, anote valor, shape e interpretação. Quando chegar a $w$ e $b$,
você terá a direção de maior crescimento local da loss.

### 6. Atualize os parâmetros

Gradient descent move no sentido oposto:

$$
w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}.
$$

Uma redução imediata da loss é um teste útil, não uma garantia. Learning rate grande pode
ultrapassar a região em que a aproximação local do gradiente é informativa.

## Exemplo numérico

Considere $x=[2,-1]^T$, $w=[0{,}5,-2]^T$, $b=-0{,}5$, ativação ReLU e target $y=1{,}5$.

**Forward:**

$$
z=0{,}5\cdot2+(-2)\cdot(-1)-0{,}5=2{,}5,
$$

$$
a=\max(0,z)=2{,}5,\qquad L=\frac12(2{,}5-1{,}5)^2=0{,}5.
$$

**Backward:** como $z>0$, $\partial a/\partial z=1$ e
$\partial L/\partial a=a-y=1$. Logo, $\delta=1$:

$$
\frac{\partial L}{\partial w}=\delta x=[2,-1]^T,
\quad \frac{\partial L}{\partial b}=1,
\quad \frac{\partial L}{\partial x}=\delta w=[0{,}5,-2]^T.
$$

Com $\eta=0{,}1$, $w'=[0{,}3,-1{,}9]^T$ e $b'=-0{,}6$. No mesmo exemplo:

$$
z'=0{,}3\cdot2+(-1{,}9)(-1)-0{,}6=1{,}9,
\qquad L'=\frac12(1{,}9-1{,}5)^2=0{,}08.
$$

A loss caiu de $0{,}5$ para $0{,}08$. Refaça o caso com $z<0$: a derivada ReLU será zero,
e esse neurônio não receberá sinal por esse exemplo.

## Aplicação em IA

Uma camada de classificação em um sistema real calcula logits antes de convertê-los em
probabilidades. Em detecção de fraude, por exemplo, cada logit combina uma representação da
transação. O threshold operacional não faz parte do neurônio: é uma decisão posterior baseada
em custo e risco. Em LLMs, a mesma operação afim aparece repetidamente nas projeções de
atenção e nos blocos feed-forward; o M5 prepara a álgebra que será reutilizada no M8.

Também é importante distinguir explicação matemática de interpretação de negócio. Um peso
positivo em uma rede profunda não significa diretamente que uma feature “aumenta a fraude”:
ativações, interações e outras camadas mediam o efeito.

## Laboratório Python

O laboratório treina um neurônio sigmoide para a porta OR. Todo gradiente é manual e o
gradient check usa diferenças centrais. Não há autograd.

```python
import numpy as np

X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([[0.], [1.], [1.], [1.]])

def sigmoid(z):
    # Forma estável para valores positivos e negativos.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out

def forward(X, W, b):
    Z = X @ W + b
    P = sigmoid(Z)
    return Z, P

def bce(P, y):
    eps = 1e-12
    P = np.clip(P, eps, 1 - eps)
    return -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))

def loss_and_grads(X, y, W, b):
    _, P = forward(X, W, b)
    loss = bce(P, y)
    # BCE(sigmoid(z)) -> dL/dZ = (P - y) / n
    dZ = (P - y) / len(X)
    dW = X.T @ dZ
    db = dZ.sum(axis=0)
    dX = dZ @ W.T
    return loss, {"W": dW, "b": db, "X": dX}

rng = np.random.default_rng(42)
W = rng.normal(0, 0.1, size=(2, 1))
b = np.zeros(1)

# Gradient check antes do treino.
loss, grads = loss_and_grads(X, y, W, b)
h = 1e-5
numeric = np.zeros_like(W)
for idx in np.ndindex(W.shape):
    old = W[idx]
    W[idx] = old + h
    plus = loss_and_grads(X, y, W, b)[0]
    W[idx] = old - h
    minus = loss_and_grads(X, y, W, b)[0]
    W[idx] = old
    numeric[idx] = (plus - minus) / (2 * h)

relative_error = np.linalg.norm(numeric - grads["W"]) / (
    np.linalg.norm(numeric) + np.linalg.norm(grads["W"]) + 1e-12
)
print("erro relativo do gradiente:", relative_error)
assert relative_error < 1e-7

learning_rate = 0.8
history = []
for step in range(2000):
    loss, grads = loss_and_grads(X, y, W, b)
    W -= learning_rate * grads["W"]
    b -= learning_rate * grads["b"]
    history.append(loss)

_, probabilities = forward(X, W, b)
print("loss inicial/final:", history[0], history[-1])
print("W:", W.ravel(), "b:", b)
print("probabilidades:", probabilities.ravel())
print("classes:", (probabilities >= 0.5).astype(int).ravel())
assert np.array_equal((probabilities >= 0.5).astype(int), y.astype(int))
```

**Entregue:** derivação de $dZ$, $dW$, $db$ e $dX$; tabela de shapes; curva da loss;
resultado do gradient check; repetição com porta AND; tentativa com XOR e explicação de por
que um único neurônio linear não a separa.

## Armadilhas comuns

- chamar $w^Tx$ de transformação linear quando há viés; tecnicamente, $w^Tx+b$ é afim;
- misturar $z$, $a$ e a previsão final como se fossem o mesmo objeto;
- omitir shapes e aceitar broadcasting acidental;
- somar o gradiente do lote quando a loss foi definida como média, ou dividir duas vezes;
- aplicar sigmoid ingênua e gerar overflow em $e^{-z}$;
- calcular BCE com $\log(0)$ sem estabilidade numérica;
- confundir peso com causalidade ou importância global;
- verificar apenas se o código roda, sem gradient checking;
- usar XOR como evidência de falha de treinamento, quando é limitação de representação de um único neurônio linear.

## Exercícios

1. Derive $\partial z/\partial w$, $\partial z/\partial b$ e $\partial z/\partial x$ usando diferenciais.
2. Derive $\partial L/\partial z=p-y$ para sigmoid seguida de binary cross-entropy.
3. Recalcule o exemplo numérico com $x=[-1,2]$, $w=[1,0{,}5]$ e dois valores de viés.
4. Mostre algebricamente que duas camadas afins sem ativação equivalem a uma camada afim.
5. Altere o laboratório para a porta AND e compare os parâmetros aprendidos.
6. Tente treinar XOR. Plote os quatro pontos e explique a impossibilidade geométrica.
7. Introduza de propósito um erro de transpose em $dW$; use shapes para diagnosticá-lo.
8. Faça gradient check também de $b$ e de uma coordenada de $X$.
9. Compare diferenças progressivas e centrais para $h\in\{10^{-2},10^{-4},10^{-6},10^{-8}\}$.
10. Escreva uma seção curta: “o que o sucesso na porta OR não prova sobre redes profundas”.

## Critério de domínio

Você domina a aula quando, sem consultar o material:

- desenha o grafo $x,w,b\rightarrow z\rightarrow a\rightarrow L$;
- explica cada símbolo e shape;
- deriva $dW$, $db$ e $dX$ a partir do gradiente upstream;
- implementa forward, loss e backward em NumPy;
- obtém erro relativo de gradient check menor que $10^{-7}$ neste exemplo;
- explica por que uma atualização pequena tende a reduzir a loss;
- prevê o comportamento para ReLU negativa e para XOR;
- distingue evidência de execução, correção do gradiente e capacidade de generalização.

## Referências principais

- Goodfellow, Bengio e Courville — [Deep Learning](https://www.deeplearningbook.org/), cap. 6.
- Simon J. D. Prince — [Understanding Deep Learning](https://udlbook.github.io/udlbook/), cap. 3.
- Stanford CS231n — [Neural Networks and Backpropagation](https://cs231n.stanford.edu/slides/2020/lecture_4.pdf).
- Dive into Deep Learning — [Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) e [Forward/Backward Propagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html).
- Rumelhart, Hinton e Williams (1986) — [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0).

## Próxima aula

**Aula 02 — Perceptron, separação linear e regra de aprendizagem.**
