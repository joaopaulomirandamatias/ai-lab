# Aula 06 — Regressão logística e classificação probabilística

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 05 deste módulo  
**Objetivo central:** Entender como um modelo linear produz probabilidades de classe por meio da função sigmoide e log-loss.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar logits e probabilidades.
- Entender sigmoide e log-odds.
- Relacionar regressão logística a cross-entropy.
- Separar probabilidade prevista de decisão por threshold.
- Interpretar coeficientes em termos de odds.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como um modelo linear produz probabilidades de classe por meio da função sigmoide e log-loss.**

## 2. Ideias fundamentais

### 1. Logit

A combinação linear $z=w^Tx+b$ não é probabilidade. A sigmoide converte o logit em valor entre 0 e 1.

### 2. Log-odds

A regressão logística assume que o logaritmo das odds é linear nas features: $\log(p/(1-p))=w^Tx+b$.

### 3. Cross-entropy

O treinamento por máxima verossimilhança para Bernoulli leva à binary cross-entropy.

### 4. Threshold

O modelo estima score/probabilidade; a regra de decisão pode usar 0.5 ou outro limiar definido pelo custo operacional.

## Aprofundamento — máxima verossimilhança, log-loss e gradiente

Para $y_i\in\{0,1\}$ e $p_i=\sigma(w^Tx_i+b)$, a verossimilhança Bernoulli é

$$
\prod_i p_i^{y_i}(1-p_i)^{1-y_i}.
$$

Maximizar seu log equivale a minimizar binary cross-entropy:

$$
J(w,b)=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

Usando a derivada da sigmoide e a regra da cadeia, o gradiente simplifica para

$$
\nabla_w J=\frac1nX^T(p-y),\qquad \frac{\partial J}{\partial b}=\frac1n\sum_i(p_i-y_i).
$$

Essa forma reaparecerá no M5. O modelo aprende log-odds lineares; não significa que as probabilidades estejam automaticamente bem calibradas sob mudança de distribuição. Regularização, prevalência e seleção do dataset importam.

## 3. Equação para guardar

$$
p(y=1\mid x)=\sigma(w^Tx+b)=\frac{1}{1+e^{-(w^Tx+b)}}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever inadimplência. O modelo pode estimar 0.18 para um cliente e 0.73 para outro. O threshold de ação deve ser escolhido conforme custo de risco e falso alarme.

## Exemplo numérico resolvido

Com $w=1{,}2$, $x=2$ e $b=-1$, o logit é $z=1{,}4$ e

$$
p=\sigma(1{,}4)\approx0{,}802.
$$

Se $y=1$, a loss é $-\log(0{,}802)\approx0{,}221$; se $y=0$, é $-\log(0{,}198)\approx1{,}619$. A previsão confiante recebe penalidade grande quando está errada. Com threshold 0,5 o caso é positivo; com threshold 0,85, negativo. Ranking e política de decisão são etapas distintas.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred_30 = (proba >= 0.30).astype(int)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Implemente sigmoide, BCE e os gradientes em NumPy. Valide cada gradiente com diferenças finitas, treine um dataset binário e compare coeficientes com `LogisticRegression(penalty=None)`. Em seguida varie o threshold e construa uma tabela custo × precision × recall.

## Laboratório guiado completo

Implemente a loss e o gradiente sem autograd e valide com diferenças finitas.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=42)
X = StandardScaler().fit_transform(X)
w = np.zeros(X.shape[1]); b = 0.0

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1/(1+np.exp(-z))

def loss_grad(w, b):
    p = sigmoid(X @ w + b)
    eps = 1e-12
    loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    return loss, X.T @ (p-y)/len(y), np.mean(p-y)

for _ in range(3000):
    loss, dw, db = loss_grad(w, b)
    w -= 0.1*dw; b -= 0.1*db

h = 1e-5
numeric = (loss_grad(w + h*np.eye(1, len(w), 0)[0], b)[0]
           - loss_grad(w - h*np.eye(1, len(w), 0)[0], b)[0])/(2*h)
print("loss", loss, "grad analítico", dw[0], "grad numérico", numeric)
```

**Entregue:** demonstração algébrica de $dL/dz=p-y$; erro relativo do gradient check; curva da loss; threshold escolhido por custo.

### Protocolo investigativo obrigatório

O laboratório não termina quando o código executa. Para transformar execução em aprendizagem e evidência:

1. escreva uma hipótese antes de rodar o experimento;
2. mantenha um baseline e altere uma decisão por vez;
3. use o mesmo split ou os mesmos folds nas comparações;
4. reporte a distribuição das métricas, não apenas o melhor número;
5. inspecione pelo menos cinco erros ou casos extremos;
6. registre seed, versões, hiperparâmetros e tempo de execução;
7. conclua com **o que os resultados sustentam** e **o que não sustentam**.

Salve um relatório curto em Markdown, a configuração em JSON e o código executável. Uma execução sem interpretação não satisfaz o critério de domínio.

## 6. Conexão com o AI Systems Laboratory

Para o projeto longitudinal, aplique este conceito a um dataset real e salve:
- configuração do experimento;
- baseline;
- métricas de validação;
- análise de erros;
- limitações;
- evidência de que o teste não contaminou o treinamento.

Ao longo do M4, esses artefatos serão acumulados até formar o **Gate II**.

## 7. Armadilhas comuns

- Tratar predict() como única saída relevante.
- Escolher threshold pelo teste final.
- Chamar score não calibrado de probabilidade sem verificar.
- Interpretar associação do coeficiente como causalidade.

## 8. Exercícios

1. Calcule a sigmoide de z=0 e interprete.
2. Qual a diferença entre score, probabilidade e classe?
3. Por que threshold 0.5 não é universal?
4. Explique a relação entre regressão logística e cross-entropy.

## Exercícios de aprofundamento e rubrica

### Nível A — reconstrução conceitual

Feche o material e explique o problema, as hipóteses, cada símbolo das equações e a diferença entre treinamento, seleção e avaliação. Desenhe o fluxo de dados sem consultar o texto. Se uma definição depender de palavras vagas como “melhor” ou “parecido”, torne-a operacional.

### Nível B — cálculo e implementação

Refaça o exemplo numérico com valores diferentes e confira manualmente o resultado do código. Implemente a operação matemática central com NumPy ou Python básico antes de usar a abstração do scikit-learn. Compare tolerâncias e explique qualquer diferença numérica.

### Nível C — contraprova experimental

Crie deliberadamente um cenário em que o método falha: ruído, outlier, escala incompatível, shift, grupos repetidos, classe rara ou leakage. Formule antes o comportamento esperado, execute a ablação e confronte hipótese e resultado.

### Nível D — transferência para sistema real

Aplique o conceito a um problema do AI Systems Laboratory. Declare unidade, instante de predição, dados disponíveis, baseline, métrica, custo dos erros e threat to validity. Produza um artefato que outra pessoa consiga auditar.

### Rubrica de 0 a 4

- **0 — reconhecimento:** identifica o nome, mas não explica o mecanismo;
- **1 — reprodução:** executa exemplo pronto;
- **2 — compreensão:** deriva/calcula e interpreta o resultado;
- **3 — diagnóstico:** prevê falhas, escolhe protocolo e analisa erros;
- **4 — transferência:** projeta, implementa e defende um experimento novo e reproduzível.

**Carga sugerida:** 45 min de leitura ativa, 45 min de derivação/cálculo, 90 min de laboratório, 30 min de análise de erros e 30 min de relatório. Avance somente ao atingir pelo menos nível 3.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP, cap. 4 — Classification.
- Hastie et al. — Linear Methods for Classification.
- Murphy — Logistic Regression.
- scikit-learn — LogisticRegression.

## Leitura orientada e fontes verificadas

- James et al. — [ISLP](https://www.statlearning.com/), cap. 4.
- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), classificação linear e Bernoulli.
- scikit-learn — [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- scikit-learn — [Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Próxima aula

**K-Nearest Neighbors: distâncias e maldição da dimensionalidade**
