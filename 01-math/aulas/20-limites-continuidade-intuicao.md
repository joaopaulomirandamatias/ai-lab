# Aula 20 — Limites e continuidade: a linguagem para falar de mudança

**Trilha:** Matemática para IA  
**Etapa:** Cálculo  
**Pré-requisito:** Aula 19 — Funções  
**Objetivo:** construir a intuição de limite e continuidade necessária para compreender derivadas, otimização e aproximações locais.

## 1. A ideia de aproximação

Quando escrevemos:

\[
\lim_{x\to a}f(x)=L
\]

não estamos perguntando necessariamente quanto vale \(f(a)\). Perguntamos:

> para qual valor a saída se aproxima quando a entrada se aproxima de \(a\)?

## 2. Exemplo simples

Para:

\[
f(x)=x^2
\]

quando:

\[
x\to2
\]

então:

\[
f(x)\to4
\]

Logo:

\[
\lim_{x\to2}x^2=4
\]

## 3. Um buraco na função

Considere:

\[
f(x)=\frac{x^2-1}{x-1}
\]

Em \(x=1\), temos \(0/0\), mas para \(x\neq1\):

\[
f(x)=x+1
\]

Portanto:

\[
\lim_{x\to1}f(x)=2
\]

mesmo que a expressão original não esteja definida exatamente em 1.

Isso mostra que limite fala do comportamento **ao redor** do ponto.

## 4. Continuidade

Uma função é contínua em \(a\) quando, de modo intuitivo:

1. \(f(a)\) existe;
2. o limite existe;
3. o limite coincide com o valor da função.

\[
\boxed{\lim_{x\to a}f(x)=f(a)}
\]

## 5. Por que continuidade importa?

Derivadas descrevem comportamento local. Se uma função possui saltos abruptos, a noção de inclinação pode falhar naquele ponto.

Otimização baseada em gradientes depende de estruturas suficientemente regulares para que pequenas alterações nos parâmetros produzam mudanças interpretáveis na loss.

## 6. Limite da taxa de variação

A derivada será definida como:

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\]

Ou seja, toda a ideia de derivada nasce de um limite.

## 7. ReLU e não diferenciabilidade

A função:

\[
ReLU(x)=\max(0,x)
\]

é contínua, mas possui uma quina em \(x=0\).

A derivada à esquerda é 0 e à direita é 1. Portanto a derivada clássica não existe exatamente no ponto 0.

Ainda assim, redes neurais usam ReLU com sucesso. Frameworks adotam uma convenção de subgradiente/derivada naquele ponto.

Isso é um bom lembrete: matemática aplicada trabalha com detalhes que a intuição inicial simplifica.

## 8. Limites no infinito

Também podemos estudar:

\[
\lim_{x\to\infty}\frac{1}{x}=0
\]

ou:

\[
\lim_{x\to\infty}\sigma(x)=1
\]

Esses comportamentos ajudam a entender saturação de funções de ativação.

## 9. Saturação

Na sigmoid, para \(|x|\) muito grande, a saída se aproxima de 0 ou 1. Mais adiante veremos que sua derivada também se aproxima de zero.

Isso ajuda a explicar **vanishing gradients** em certas arquiteturas e regimes.

## 10. Continuidade em várias dimensões

Para uma função:

\[
f:\mathbb{R}^n\rightarrow\mathbb{R}
\]

a ideia é semelhante: quando o vetor \(x\) se aproxima de \(a\), queremos que \(f(x)\) se aproxime de \(f(a)\).

Esse será o cenário natural de funções de perda com milhões de parâmetros.

## 11. Precisão numérica não é limite matemático

No computador, não podemos representar “\(h\to0\)” literalmente. Escolher um \(h\) muito pequeno em diferenças finitas pode causar cancelamento e erro de ponto flutuante.

Exemplo para aproximar derivada:

```python
import numpy as np

def f(x):
    return x**2

for h in [1e-1, 1e-3, 1e-6, 1e-10]:
    approx = (f(2+h) - f(2)) / h
    print(h, approx)
```

Você verá aproximação de 4, mas a precisão não melhora indefinidamente.

## 12. Conexão com IA

- derivadas são limites de taxas de variação;
- funções de ativação possuem diferentes níveis de suavidade;
- saturação influencia gradientes;
- estabilidade numérica limita a tradução direta de ideias contínuas para computadores.

## Exercícios

1. Calcule intuitivamente \(\lim_{x\to3}x^2\).
2. Simplifique \((x^2-4)/(x-2)\) para \(x\neq2\) e encontre o limite em 2.
3. Defina continuidade com suas palavras.
4. Por que ReLU não é diferenciável em 0?
5. Qual o limite da sigmoid quando \(x\to-\infty\)?
6. Por que um `h` absurdamente pequeno pode piorar uma aproximação numérica de derivada?

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 5 — Vector Calculus. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Caps. 4 e 6. https://www.deeplearningbook.org/

## Próxima aula

**Aula 21 — Derivada: inclinação, taxa de variação e aproximação local.**