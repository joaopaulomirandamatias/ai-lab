# Aula 28 — SGD, mini-batch, momentum e dinâmica de treinamento

**Trilha:** Matemática para IA  
**Etapa:** Otimização  
**Pré-requisito:** Gradient Descent  
**Objetivo:** entender por que redes grandes usam gradientes aproximados por mini-batches e como momentum altera a dinâmica da otimização.

## 1. Batch Gradient Descent

Se:

\[
L(\theta)=\frac1N\sum_{i=1}^N\ell_i(\theta)
\]

o gradient descent exato calcula:

\[
\nabla L=\frac1N\sum_i\nabla\ell_i
\]

Isso exige passar por todo o dataset antes de cada atualização.

Para milhões ou bilhões de exemplos, pode ser caro.

## 2. Stochastic Gradient Descent

SGD clássico usa uma amostra aleatória:

\[
g_t=\nabla\ell_i(\theta_t)
\]

E atualiza:

\[
\theta_{t+1}=\theta_t-\eta g_t
\]

O gradiente é ruidoso, mas barato.

## 3. Mini-batch Gradient Descent

Na prática, usamos um conjunto \(B_t\):

\[
g_t=\frac1{|B_t|}\sum_{i\in B_t}\nabla\ell_i(\theta_t)
\]

A atualização continua:

\[
\theta_{t+1}=\theta_t-\eta g_t
\]

Mini-batches aproveitam hardware vetorial/GPU e equilibram custo e ruído.

## 4. Epoch, batch e step

- **batch:** subconjunto usado em uma atualização;
- **step/iteration:** uma atualização dos parâmetros;
- **epoch:** uma passagem aproximadamente completa pelo dataset.

Se temos 10.000 exemplos e batch size 100, uma epoch tem cerca de 100 steps.

## 5. Ruído do gradiente

Mini-batches diferentes geram gradientes diferentes. Isso faz a trajetória oscilar.

Esse ruído não é apenas “erro”; pode ajudar a explorar a superfície e é parte importante da dinâmica de treinamento.

Mas batch sizes muito pequenos podem produzir estimativas excessivamente instáveis.

## 6. Momentum

Momentum mantém uma média acumulada das direções passadas. Uma convenção comum:

\[
v_t=\beta v_{t-1}+g_t
\]

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

A ideia é ganhar velocidade em direções consistentes e reduzir ziguezague em direções que alternam.

## 7. Analogia física — com cuidado

É comum comparar momentum a uma bola ganhando velocidade em uma descida. A analogia ajuda, mas a atualização é um algoritmo matemático, não uma simulação física literal.

## 8. Exemplo de vale estreito

Em uma superfície com alta curvatura vertical e baixa curvatura horizontal, gradient descent pode oscilar verticalmente enquanto avança lentamente.

Momentum suaviza parte dessas oscilações e acumula movimento na direção persistente.

## 9. Learning rate schedule

O learning rate pode variar ao longo do treinamento:

- decaimento por etapas;
- exponencial;
- cosine decay;
- warmup + decay.

A ideia geral é permitir passos maiores em fases iniciais e refinamento posterior, embora a melhor política dependa do modelo e do regime.

## 10. Adam — prévia

Adam combina ideias de momentum e adaptação da escala por parâmetro. Ele será estudado no módulo de Deep Learning.

Nesta fase, o objetivo é dominar primeiro SGD e momentum para entender o que otimizadores mais sofisticados estão modificando.

## 11. NumPy: mini-batch

```python
import numpy as np

rng = np.random.default_rng(42)

def batches(X, y, batch_size):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]
```

A cada epoch embaralhamos exemplos e produzimos mini-batches.

## 12. Momentum simples

```python
velocity = np.zeros_like(theta)
beta = 0.9

for grad in gradients:
    velocity = beta * velocity + grad
    theta = theta - lr * velocity
```

Compare a trajetória com SGD sem momentum em uma função quadrática anisotrópica.

## 13. Reprodutibilidade

Treinamento estocástico depende de:

- seed;
- ordem dos dados;
- inicialização;
- operações de hardware potencialmente não determinísticas.

Experimentos científicos precisam registrar essas condições.

## 14. Métricas que vale registrar

Em um laboratório de otimização, registre:

- training loss por step/epoch;
- validation loss;
- norma do gradiente;
- learning rate;
- tempo;
- seed;
- batch size.

Essa disciplina será útil mais tarde em experimentos científicos com modelos.

## Exercícios

1. Diferencie batch GD, SGD e mini-batch GD.
2. Quantos steps há por epoch com 50.000 exemplos e batch 250?
3. Explique o benefício e o custo do ruído estocástico.
4. O que momentum tenta corrigir?
5. Por que embaralhar dados entre epochs pode ser importante?
6. Quais informações você registraria para tornar um treino reproduzível?

## Referências

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 8 — Optimization for Training Deep Models. https://www.deeplearningbook.org/
- BENGIO, Y. *Practical Recommendations for Gradient-Based Training of Deep Architectures*. arXiv:1206.5533, 2012.
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 7. https://mml-book.github.io/

## Próxima aula

**Aula 29 — Regressão linear com Gradient Descent do zero.**