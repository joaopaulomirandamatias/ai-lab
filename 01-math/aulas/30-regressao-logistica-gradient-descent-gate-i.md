# Aula 30 — Regressão logística, cross-entropy e Gate I de Matemática para IA

**Trilha:** Matemática para IA  
**Etapa:** Gate I  
**Pré-requisitos:** todas as Aulas 01–29  
**Objetivo:** integrar Álgebra Linear, Cálculo e Otimização em um classificador binário implementado do zero e demonstrar domínio suficiente para avançar ao módulo de Estatística.

## 1. O fechamento do ciclo

Começamos com:

\[
vetores\rightarrow matrizes\rightarrow y=Wx+b
\]

Depois aprendemos derivadas e gradient descent.

Agora construiremos:

\[
\boxed{p(y=1|x)=\sigma(w^Tx+b)}
\]

Essa equação contém quase todo o caminho percorrido:

- vetor \(x\);
- produto escalar \(w^Tx\);
- transformação afim \(+b\);
- função não linear sigmoid;
- probabilidade;
- loss;
- gradiente;
- otimização.

## 2. Logit

Defina:

\[
z=w^Tx+b
\]

Esse score real é chamado de **logit** no contexto da regressão logística.

Aplicamos:

\[
\sigma(z)=\frac1{1+e^{-z}}
\]

obtendo uma probabilidade em \((0,1)\).

## 3. Decisão

Uma regra simples é:

\[
\hat y=1\quad\text{se }p\ge0.5
\]

Mas 0.5 não é um limiar universalmente ótimo. O threshold pode ser escolhido conforme custo de falso positivo, falso negativo e objetivo da aplicação.

Essa discussão será aprofundada em Machine Learning.

## 4. Binary cross-entropy

Para uma observação:

\[
\boxed{\ell=-[y\log p+(1-y)\log(1-p)]}
\]

Para \(n\) exemplos:

\[
L=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]
\]

Essa loss vem da negative log-likelihood de um modelo Bernoulli.

## 5. Gradiente — resultado fundamental

Com:

\[
p=\sigma(Xw+b)
\]

o gradiente em relação aos pesos simplifica para:

\[
\boxed{\nabla_wL=\frac1nX^T(p-y)}
\]

E para o bias:

\[
\boxed{\frac{\partial L}{\partial b}=\frac1n\sum_i(p_i-y_i)}
\]

A derivação usa regra da cadeia e a identidade:

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

## 6. Derivação de uma amostra

Para:

\[
\ell=-[y\log p+(1-y)\log(1-p)]
\]

obtemos:

\[
\frac{\partial \ell}{\partial p}=-\frac yp+\frac{1-y}{1-p}
\]

E:

\[
\frac{dp}{dz}=p(1-p)
\]

Multiplicando e simplificando:

\[
\boxed{\frac{\partial \ell}{\partial z}=p-y}
\]

Como:

\[
z=w^Tx+b
\]

segue:

\[
\frac{\partial \ell}{\partial w}=(p-y)x
\]

Essa simplificação é elegante e muito importante.

## 7. Implementação estável da sigmoid

Para valores extremos, `exp` pode overflow. Uma implementação robusta pode tratar sinais separadamente ou usar bibliotecas com funções estáveis.

Para o laboratório didático:

```python
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

O clipping é uma proteção simples; em código de produção, prefira primitives numericamente estáveis da biblioteca usada.

## 8. Dataset sintético

```python
rng = np.random.default_rng(42)

X0 = rng.normal(loc=(-1, -1), scale=0.8, size=(200, 2))
X1 = rng.normal(loc=( 1,  1), scale=0.8, size=(200, 2))

X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(200), np.ones(200)])
```

## 9. Gradient Descent do zero

```python
w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1
history = []

eps = 1e-12

for epoch in range(1000):
    z = X @ w + b
    p = sigmoid(z)

    loss = -np.mean(
        y*np.log(p + eps) +
        (1-y)*np.log(1-p + eps)
    )
    history.append(loss)

    error = p - y
    dw = X.T @ error / len(X)
    db = np.mean(error)

    w -= lr * dw
    b -= lr * db
```

Nenhum autograd é permitido no Gate I.

## 10. Avaliação mínima

```python
p = sigmoid(X @ w + b)
pred = (p >= 0.5).astype(int)
accuracy = np.mean(pred == y)
print(accuracy)
```

Accuracy sozinha não basta em problemas reais, especialmente com classes desbalanceadas. Métricas mais completas serão estudadas no módulo de ML.

## 11. Gradient checking

Valide componentes de \(w\) por diferenças finitas.

```python
def loss_for_w(w_test):
    p = sigmoid(X @ w_test + b)
    return -np.mean(
        y*np.log(p + eps) +
        (1-y)*np.log(1-p + eps)
    )
```

Compare gradiente analítico e numérico em alguns parâmetros.

## 12. Três learning rates

O Gate exige experimento, não apenas código final.

Use três valores, por exemplo:

\[
\eta\in\{0.001,0.1,2.0\}
\]

A faixa exata deve ser ajustada ao dataset para produzir comportamentos informativos.

Registre curvas e explique:

- qual foi lento;
- qual convergiu melhor;
- qual oscilou/divergiu, se isso ocorrer;
- como escala dos dados afetou o resultado.

## 13. Fronteira de decisão

Para duas features:

\[
w_1x_1+w_2x_2+b=0
\]

é a fronteira onde:

\[
p=0.5
\]

Visualize os dados e a reta aprendida. Isso conecta pesos a geometria.

## 14. Gate I — critérios de aprovação

Você só deve considerar o bloco de Matemática concluído quando conseguir, sem copiar uma implementação pronta:

1. explicar vetores, matrizes e \(y=Wx+b\);
2. calcular norma, dot product e cosine similarity;
3. explicar rank, base, projeções, autovalores, SVD e PCA em nível intuitivo;
4. derivar gradientes usando regra da cadeia;
5. explicar Jacobiano e Hessiana conceitualmente;
6. derivar MSE para regressão linear;
7. derivar cross-entropy logística até \(p-y\);
8. implementar linear e logistic regression do zero;
9. implementar gradient descent sem autograd;
10. comparar pelo menos três learning rates com gráficos e conclusão;
11. fazer gradient checking;
12. escrever testes que comparem operações próprias com NumPy/SciPy quando aplicável.

## 15. Evidências no repositório

O Gate I deve deixar evidência reproduzível:

```text
01-math/
├── aulas/
├── notebooks/
├── src/
├── tests/
└── gate-i/
```

Sugestão para `gate-i/`:

- `derivacao-linear.md`;
- `derivacao-logistica.md`;
- `gradient-descent.ipynb`;
- `learning-rates.png`;
- `README.md` com resultados;
- testes automatizados.

## 16. O que vem depois

O próximo módulo é Probabilidade, Estatística e Teoria da Informação. Isso adicionará a linguagem necessária para:

- incerteza;
- estimação;
- intervalos de confiança;
- testes de hipótese;
- likelihood;
- entropia;
- cross-entropy;
- KL divergence;
- avaliação científica de modelos.

## Exercícios finais

1. Derive \(d\ell/dz=p-y\) sem consultar a aula.
2. Explique por que sigmoid + BCE formam uma combinação matematicamente conveniente.
3. Implemente regressão logística vetorizada com NumPy.
4. Faça gradient checking.
5. Compare três learning rates.
6. Plote a fronteira de decisão.
7. Escreva uma conclusão de uma página: “O que a matemática me permite enxergar dentro de um modelo de IA?”.

## Referências

- DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Caps. 5, 7 e 9. https://mml-book.github.io/
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Part I: Linear Algebra, Numerical Computation e Machine Learning Basics. https://www.deeplearningbook.org/
- BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/
- RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. *Learning representations by back-propagating errors*. Nature 323, 533–536, 1986. DOI: 10.1038/323533a0.

# Gate I concluído

Ao completar os critérios acima, avance para **02-statistics/**. A matemática não desaparece; ela passa a ser usada como linguagem de trabalho nos módulos seguintes.