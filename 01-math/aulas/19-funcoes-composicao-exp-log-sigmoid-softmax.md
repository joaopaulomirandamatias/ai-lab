# Aula 19 — Funções para IA: composição, exponencial, logaritmo, sigmoid e softmax

**Trilha:** Matemática para IA  
**Etapa:** Cálculo  
**Pré-requisitos:** álgebra básica e vetores  
**Objetivo:** revisar funções como máquinas de transformação e dominar funções que reaparecem em redes neurais, probabilidades e funções de perda.

## 1. Função como transformação

Uma função recebe uma entrada e produz uma saída:

\[
y=f(x)
\]

Isso é a versão escalar da ideia de transformação que já estudamos com matrizes.

Exemplo:

\[
f(x)=2x+3
\]

Para \(x=4\):

\[
f(4)=11
\]

## 2. Domínio e imagem

O **domínio** é o conjunto de entradas permitidas. A **imagem** é o conjunto de saídas efetivamente produzidas.

Para:

\[
f(x)=\log x
\]

no domínio real:

\[
x>0
\]

Entender domínio evita operações inválidas e bugs numéricos.

## 3. Composição

Se:

\[
z=g(x)
\]

e:

\[
y=f(z)
\]

então:

\[
y=f(g(x))
\]

Redes neurais são composições gigantes de funções:

\[
f_L(f_{L-1}(\cdots f_2(f_1(x))))
\]

A regra da cadeia, estudada em breve, existe justamente para diferenciar composições.

## 4. Função exponencial

\[
f(x)=e^x
\]

Propriedades importantes:

\[
e^{a+b}=e^ae^b
\]

\[
\frac{d}{dx}e^x=e^x
\]

Exponenciais aparecem em probabilidades, softmax, modelos de energia e otimização.

## 5. Logaritmo

\[
y=\log x
\]

é a função inversa da exponencial:

\[
e^{\log x}=x
\]

Propriedades:

\[
\log(ab)=\log a+\log b
\]

\[
\log(a/b)=\log a-\log b
\]

Logs transformam produtos em somas, uma propriedade extremamente útil em probabilidade.

## 6. Por que log em Machine Learning?

Probabilidades pequenas multiplicadas podem underflow numericamente. Em log-space:

\[
\log\prod_i p_i=\sum_i\log p_i
\]

Também usamos negative log-likelihood e cross-entropy, que serão estudadas adiante.

## 7. Sigmoid

A função logística é:

\[
\boxed{\sigma(x)=\frac{1}{1+e^{-x}}}
\]

Ela mapeia qualquer número real para:

\[
(0,1)
\]

Por isso é útil quando queremos interpretar uma saída como probabilidade em classificação binária.

## 8. Comportamento da sigmoid

- \(x\gg0\) → \(\sigma(x)\approx1\);
- \(x\ll0\) → \(\sigma(x)\approx0\);
- \(x=0\) → \(\sigma(x)=0.5\).

Mais adiante veremos sua derivada:

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

## 9. Softmax

Para um vetor de logits \(z\):

\[
\boxed{softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}}
\]

As saídas são positivas e somam 1.

Exemplo:

\[
z=[2,1,0]
\]

softmax transforma esses scores em uma distribuição sobre três classes.

## 10. Softmax não é “pegar o maior”

Softmax preserva informação relativa entre logits. Dois vetores com o mesmo argmax podem gerar distribuições de confiança muito diferentes.

Também existe uma noção de temperatura:

\[
softmax(z/T)
\]

- \(T<1\): distribuição mais concentrada;
- \(T>1\): distribuição mais suave.

Esse conceito reaparece em modelos generativos.

## 11. Estabilidade de softmax

Calcular \(e^{1000}\) causa overflow. Como softmax é invariável à soma da mesma constante em todos os logits:

\[
softmax(z)=softmax(z-c)
\]

escolhemos:

\[
c=\max(z)
\]

Assim:

```python
z_shift = z - np.max(z)
p = np.exp(z_shift) / np.exp(z_shift).sum()
```

Essa técnica simples é um exemplo central de estabilidade numérica.

## 12. Log-sum-exp

A expressão:

\[
\log\sum_i e^{z_i}
\]

aparece com frequência. Uma forma estável é:

\[
m+\log\sum_i e^{z_i-m}
\]

onde:

\[
m=\max_i z_i
\]

Esse “truque” será importante em losses probabilísticas.

## 13. NumPy

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

print(sigmoid(np.array([-2., 0., 2.])))
print(softmax(np.array([2., 1., 0.])))
```

## 14. Conexão com LLMs

Em Transformers, softmax aparece na atenção:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

E também na distribuição final sobre tokens.

Então funções elementares como exponencial e normalização participam diretamente de sistemas de linguagem modernos.

## Exercícios

1. Calcule \(f(3)\) para \(f(x)=x^2+1\).
2. Se \(f(x)=x^2\) e \(g(x)=x+1\), escreva \(f(g(x))\).
3. Quanto vale \(\sigma(0)\)?
4. Por que logs ajudam quando multiplicamos muitas probabilidades pequenas?
5. Por que subtrair `max(z)` não altera softmax?
6. Explique a diferença entre logits e probabilidades softmax.

## Referências

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Caps. 3–4. https://www.deeplearningbook.org/
- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Caps. 5 e 7. https://mml-book.github.io/

## Próxima aula

**Aula 20 — Limites, continuidade e o que significa uma função mudar suavemente.**