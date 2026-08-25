# Aula 04 — Regressão linear e mínimos quadrados

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 03 deste módulo  
**Objetivo central:** Conectar o que foi aprendido em Álgebra Linear e Estatística ao primeiro modelo supervisionado clássico.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar coeficientes de regressão.
- Entender mínimos quadrados.
- Relacionar matriz X, vetor de parâmetros e previsão.
- Analisar resíduos.
- Reconhecer limitações da interpretação causal.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **conectar o que foi aprendido em álgebra linear e estatística ao primeiro modelo supervisionado clássico.**

## 2. Ideias fundamentais

### 1. Modelo linear

A regressão linear assume uma relação aproximada $\hat y=\beta_0+x^T\beta$. Linear refere-se aos parâmetros; features podem ser transformadas, por exemplo com termos polinomiais.

### 2. Mínimos quadrados

Os coeficientes são escolhidos para minimizar a soma dos resíduos quadráticos. A solução possui ligação direta com projeções ortogonais.

### 3. Resíduos

Resíduo é $e_i=y_i-\hat y_i$. Padrões nos resíduos podem revelar não linearidade, heterocedasticidade, dependência temporal ou outliers.

### 4. Coeficiente não é automaticamente causa

Um coeficiente descreve associação condicional sob o modelo e os dados observados. Causalidade exige desenho e hipóteses adicionais.

## Aprofundamento — da loss às equações normais

Com intercepto incorporado em uma coluna de uns, a loss é

$$
J(\beta)=\|X\beta-y\|_2^2=(X\beta-y)^T(X\beta-y).
$$

Derivando em relação a $\beta$:

$$
\nabla_\beta J=2X^T(X\beta-y).
$$

No mínimo, $X^TX\hat\beta=X^Ty$. Se $X^TX$ for inversível,

$$
\hat\beta=(X^TX)^{-1}X^Ty.
$$

Na prática, não calcule a inversa explicitamente: decomposições QR/SVD ou `lstsq` são mais estáveis. A condição $X^T(y-X\hat\beta)=0$ mostra a geometria: o vetor de resíduos é ortogonal ao espaço gerado pelas colunas de $X$.

Coeficientes exigem contexto. Escala, codificação, multicolinearidade e interações mudam sua interpretação. Um ajuste preditivo não identifica, por si só, efeito causal.

## 3. Equação para guardar

$$
\hat{\beta}=\arg\min_\beta \|X\beta-y\|_2^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever consumo de energia usando temperatura e ocupação. O coeficiente de temperatura expressa a variação prevista no target por unidade da feature, mantendo as demais do modelo constantes.

## Exemplo numérico resolvido

Para os pontos $(0,1)$, $(1,3)$ e $(2,5)$, a reta $\hat y=1+2x$ produz previsões $[1,3,5]$ e SSE zero. Agora altere o último target para 8. A solução OLS passa a $\hat y=0{,}5+3{,}5x$, com previsões $[0{,}5,4,7{,}5]$ e resíduos $[0{,}5,-1,0{,}5]$. Um único ponto mudou substancialmente a inclinação: a loss quadrática dá peso crescente a erros grandes.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)

pred = reg.predict(X_test)
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R²:", r2_score(y_test, pred))
print("coef:", reg.coef_)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Implemente OLS com `np.linalg.lstsq`, derive o gradiente e ajuste a mesma reta por gradient descent. Compare coeficientes com `LinearRegression`. Depois injete outliers, plote resíduos versus previsão e discuta linearidade, heterocedasticidade e extrapolação.

## Laboratório guiado completo

Compare solução numérica, implementação vetorizada por gradient descent e scikit-learn.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
x = np.linspace(-3, 3, 120)
y = 1.5 + 2.2*x + rng.normal(0, 0.7, len(x))
X = np.c_[np.ones(len(x)), x]

beta_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
beta = np.zeros(2)
lr = 0.03
for _ in range(3000):
    residual = X @ beta - y
    grad = (2/len(y)) * X.T @ residual
    beta -= lr * grad

sk = LinearRegression().fit(x[:, None], y)
print("lstsq", beta_lstsq)
print("gradient descent", beta)
print("sklearn", np.r_[sk.intercept_, sk.coef_])
print("ortogonalidade X^T e", X.T @ (y - X @ beta_lstsq))
```

**Entregue:** derivação do gradiente; gráfico da loss; residual plot; repetição com outliers e explicação da mudança nos coeficientes.

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

- Interpretar R² alto como evidência de causalidade.
- Ignorar extrapolação fora do domínio dos dados.
- Avaliar apenas R² sem observar magnitude dos erros.
- Ignorar resíduos e dependências.

## 8. Exercícios

1. Derive a loss MSE de uma regressão simples.
2. Explique o significado de um coeficiente negativo.
3. Por que a regressão linear pode funcionar mesmo com features transformadas?
4. O que um padrão curvo nos resíduos sugere?

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

- ISLP, cap. 3 — Linear Regression.
- Hastie et al., cap. 3 — Linear Methods for Regression.
- Murphy — PML, linear regression.
- scikit-learn — LinearRegression.

## Leitura orientada e fontes verificadas

- James et al. — [ISLP](https://www.statlearning.com/), cap. 3.
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), cap. 3.
- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), modelos lineares.
- scikit-learn — [Linear models](https://scikit-learn.org/stable/modules/linear_model.html).

## Próxima aula

**Regularização: Ridge, Lasso e Elastic Net**
