# Aula 29 — Regressão linear com Gradient Descent do zero

**Trilha:** Matemática para IA  
**Etapa:** Gate I — projeto prático  
**Pré-requisitos:** least squares, gradiente e gradient descent  
**Objetivo:** derivar e implementar regressão linear sem autograd, comparando a solução iterativa com least squares.

## 1. Modelo

Comece com uma característica:

\[
\hat y=wx+b
\]

Temos parâmetros:

\[
\theta=(w,b)
\]

Queremos ajustar a reta aos dados.

## 2. Loss MSE

\[
\boxed{L(w,b)=\frac1n\sum_{i=1}^n(wx_i+b-y_i)^2}
\]

Defina o erro:

\[
e_i=wx_i+b-y_i
\]

Então:

\[
L=\frac1n\sum e_i^2
\]

## 3. Derivando dL/dw

Pela regra da cadeia:

\[
\frac{\partial L}{\partial w}
=\frac1n\sum 2e_i\frac{\partial e_i}{\partial w}
\]

Como:

\[
\frac{\partial e_i}{\partial w}=x_i
\]

obtemos:

\[
\boxed{\frac{\partial L}{\partial w}=\frac2n\sum e_ix_i}
\]

## 4. Derivando dL/db

Como:

\[
\frac{\partial e_i}{\partial b}=1
\]

então:

\[
\boxed{\frac{\partial L}{\partial b}=\frac2n\sum e_i}
\]

## 5. Atualização

\[
w\leftarrow w-\eta\frac{\partial L}{\partial w}
\]

\[
b\leftarrow b-\eta\frac{\partial L}{\partial b}
\]

Isso é treinamento de um modelo — sem framework.

## 6. Dataset sintético

```python
import numpy as np

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y = 2.5 * X - 1.2 + rng.normal(0, 0.5, size=len(X))
```

A relação verdadeira é aproximadamente:

\[
y=2.5x-1.2
\]

com ruído.

## 7. Implementação do zero

```python
w = 0.0
b = 0.0
lr = 0.05
history = []

for epoch in range(500):
    y_hat = w * X + b
    error = y_hat - y

    loss = np.mean(error**2)
    history.append(loss)

    dw = 2 * np.mean(error * X)
    db = 2 * np.mean(error)

    w -= lr * dw
    b -= lr * db

print(w, b, history[-1])
```

## 8. Verificação do gradiente

Antes de confiar em `dw` e `db`, faça gradient checking por diferenças finitas em uma versão pequena.

O objetivo do Gate I não é apenas obter uma curva descendente; é demonstrar que a derivação e a implementação concordam.

## 9. Forma matricial

Com múltiplas características:

\[
\hat y=Xw+b
\]

MSE:

\[
L=\frac1n\|Xw+b\mathbf1-y\|^2
\]

Gradiente:

\[
\boxed{\nabla_wL=\frac2nX^T(Xw+b\mathbf1-y)}
\]

Isso reúne álgebra linear e cálculo em uma única expressão.

## 10. Comparando com least squares

Podemos construir uma coluna de 1 para o bias:

```python
X_design = np.column_stack([X, np.ones_like(X)])
beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
print(beta)
```

Compare os parâmetros obtidos analiticamente/numericamente via least squares com gradient descent.

Eles devem ficar próximos quando o algoritmo converge.

## 11. Experimento com learning rates

Obrigatório comparar pelo menos três:

```text
0.001 → lento
0.05  → adequado para este exemplo
0.5   → teste comportamento; ajuste dataset/escala se necessário
```

Não force uma narrativa. Execute e observe. Dependendo da escala dos dados, um valor pode convergir ou divergir.

A conclusão precisa vir do experimento.

## 12. Normalização e learning rate

Se uma feature varia entre 0 e 1 e outra entre 0 e 1.000.000, a geometria da loss pode ficar muito alongada.

Padronizar features pode melhorar condicionamento e tornar a otimização mais fácil.

Isso conecta pré-processamento a Álgebra Linear e Hessiana.

## 13. Train/validation

Mesmo neste projeto simples, separe dados para medir generalização.

Não escolha learning rate olhando apenas o conjunto de teste. Use validação para decisões e teste para avaliação final.

## 14. Entregáveis da Aula 29

Crie no laboratório:

1. derivação escrita de \(dw\) e \(db\);
2. implementação NumPy sem autograd;
3. gradient check;
4. gráfico de loss;
5. gráfico dos dados + reta aprendida;
6. comparação com `np.linalg.lstsq`;
7. experimento com três learning rates;
8. breve conclusão baseada em evidência.

## Exercícios

1. Derive novamente \(dL/dw\) sem consultar a aula.
2. Explique por que o bias tem gradiente \(2mean(error)\).
3. Qual a shape de \(\nabla_wL\) para \(X\in\mathbb{R}^{1000\times20}\)?
4. Por que `lstsq` e GD devem produzir resultados próximos neste problema convexo?
5. Como escala das features influencia otimização?
6. Implemente todos os entregáveis.

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cap. 9 — Linear Regression e Cap. 7 — Optimization. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Caps. 4–5. https://www.deeplearningbook.org/
- STRANG, G. *MIT 18.06 Linear Algebra*. Least squares. https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/

## Próxima aula

**Aula 30 — Regressão logística, cross-entropy e Gate I de Matemática para IA.**