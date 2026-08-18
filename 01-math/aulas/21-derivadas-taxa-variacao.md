# Aula 21 — Derivada: inclinação, taxa de variação e aproximação local

**Trilha:** Matemática para IA  
**Etapa:** Cálculo  
**Pré-requisito:** limites e funções  
**Objetivo:** entender derivada geometricamente e interpretar derivadas como sensibilidade, não apenas como manipulação simbólica.

## 1. Taxa média versus taxa instantânea

Considere uma função \(y=f(x)\). Entre \(x\) e \(x+h\), a taxa média é:

\[
\frac{f(x+h)-f(x)}{h}
\]

A derivada leva \(h\) ao limite de zero:

\[
\boxed{f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}}
\]

Geometricamente, é a inclinação da reta tangente.

## 2. Exemplo f(x)=x²

\[
\frac{(x+h)^2-x^2}{h}
=\frac{2xh+h^2}{h}
=2x+h
\]

Tomando \(h\to0\):

\[
\boxed{f'(x)=2x}
\]

Em \(x=3\):

\[
f'(3)=6
\]

Isso significa que perto de 3, uma pequena mudança \(\Delta x\) gera aproximadamente:

\[
\Delta f\approx6\Delta x
\]

## 3. Derivada como sensibilidade

Em IA, essa interpretação é mais útil que “inclinação de uma curva”.

Se:

\[
L(w)
\]

é a loss e:

\[
\frac{dL}{dw}=12
\]

então pequenas alterações positivas em \(w\) tendem a aumentar a loss rapidamente.

Se a derivada é negativa, aumentar \(w\) tende localmente a diminuir a loss.

## 4. Aproximação linear

Perto de um ponto \(x_0\):

\[
\boxed{f(x_0+\Delta x)\approx f(x_0)+f'(x_0)\Delta x}
\]

A derivada constrói a melhor aproximação linear local de primeira ordem.

Isso é uma ideia central que reaparecerá com gradientes, Jacobianos e Taylor.

## 5. Pontos críticos

Quando:

\[
f'(x)=0
\]

a função está localmente “horizontal”. Esse ponto pode ser:

- mínimo;
- máximo;
- ponto de sela em dimensões maiores;
- ou outro caso degenerado.

Gradiente zero não é sinônimo automático de mínimo.

## 6. Exemplo de loss quadrática

\[
L(w)=(w-3)^2
\]

Derivada:

\[
L'(w)=2(w-3)
\]

O ponto crítico ocorre em:

\[
w=3
\]

Como a parábola é convexa, esse é o mínimo global.

## 7. Derivada numérica

A diferença central costuma ser melhor que a diferença para frente:

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

```python
def numerical_grad(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

f = lambda x: x**2
print(numerical_grad(f, 3.0))
```

O resultado deve ficar próximo de 6.

## 8. Gradient checking

Quando implementamos backpropagation manualmente, podemos comparar um gradiente analítico com uma aproximação de diferenças finitas.

Esse procedimento é chamado **gradient checking** e é ótimo para detectar erros em derivações e código.

Não é usado para treinar redes grandes porque exige muitas avaliações da função.

## 9. Derivadas de funções importantes

\[
\frac{d}{dx}x^n=nx^{n-1}
\]

\[
\frac{d}{dx}e^x=e^x
\]

\[
\frac{d}{dx}\log x=\frac1x
\]

\[
\frac{d}{dx}\sigma(x)=\sigma(x)(1-\sigma(x))
\]

Essas identidades serão blocos de construção da regra da cadeia.

## 10. Saturação da sigmoid

Quando \(\sigma(x)\) está próxima de 0 ou 1:

\[
\sigma(x)(1-\sigma(x))\approx0
\]

Logo gradientes podem ficar muito pequenos.

Isso fornece uma primeira explicação matemática do problema de vanishing gradients em redes profundas com ativações saturantes.

## 11. Conexão com otimização

Se a derivada positiva aponta crescimento, para reduzir uma função podemos mover na direção oposta:

\[
w_{t+1}=w_t-\eta L'(w_t)
\]

Essa é a versão unidimensional do gradient descent.

Ainda precisamos aprender derivadas de funções compostas e de muitas variáveis antes de implementar o algoritmo completo.

## Exercícios

1. Derive \(f(x)=x^3\).
2. Calcule \(f'(2)\) para \(f(x)=x^2\).
3. Interprete \(dL/dw=-5\) em termos de sensibilidade local.
4. Para \(L(w)=(w-4)^2\), encontre o ponto crítico.
5. Explique aproximação linear local com suas palavras.
6. Por que gradient checking é útil, mas caro?

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 5. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cap. 4 — Numerical Computation. https://www.deeplearningbook.org/contents/numerical.html

## Próxima aula

**Aula 22 — Regras de derivação e regra da cadeia: o coração do backpropagation.**