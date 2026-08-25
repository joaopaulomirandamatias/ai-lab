# Aula 05 — Regularização: Ridge, Lasso e Elastic Net

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 04 deste módulo  
**Objetivo central:** Entender como penalizar complexidade reduz variância, melhora estabilidade e pode realizar seleção de features.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar o papel de lambda/alpha.
- Distinguir penalizações L1 e L2.
- Entender shrinkage.
- Relacionar regularização a viés-variância.
- Aplicar Ridge, Lasso e ElasticNet corretamente.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como penalizar complexidade reduz variância, melhora estabilidade e pode realizar seleção de features.**

## 2. Ideias fundamentais

### 1. Ridge / L2

Adiciona $\lambda\|\beta\|_2^2$ à loss. Tende a reduzir coeficientes de forma suave, sendo útil com multicolinearidade e muitas features correlacionadas.

### 2. Lasso / L1

Adiciona $\lambda\|\beta\|_1$. A geometria L1 favorece soluções esparsas e pode zerar coeficientes.

### 3. Elastic Net

Combina L1 e L2, oferecendo esparsidade e estabilidade quando há grupos de features correlacionadas.

### 4. Regularização e scaling

Como a penalidade depende da magnitude dos coeficientes, features devem estar em escalas comparáveis quando a penalização é aplicada.

## Aprofundamento — regularização altera o problema, não apenas o código

Ridge resolve

$$
\hat\beta_{ridge}=\arg\min_\beta \|X\beta-y\|_2^2+\lambda\|\beta\|_2^2,
$$

e, quando aplicável, possui solução $(X^TX+\lambda I)^{-1}X^Ty$. O termo $\lambda I$ melhora condicionamento e reduz variância, ao custo de viés. Normalmente o intercepto não é penalizado.

Lasso substitui L2 por $\|\beta\|_1$. Como $|\beta_j|$ tem quina em zero, a solução pode zerar coeficientes. Em features muito correlacionadas, Lasso pode escolher uma arbitrariamente; Elastic Net combina esparsidade e estabilidade de grupos.

Regularização não torna um modelo automaticamente simples ou causal. O valor de $\lambda$ é hiperparâmetro e deve ser escolhido dentro da validação. Sem scaling, uma mesma penalidade representa custos diferentes para features em unidades diferentes.

## 3. Equação para guardar

$$
\min_\beta \|X\beta-y\|_2^2+\lambda\|\beta\|_2^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Com centenas de features correlacionadas, regressão sem penalização pode produzir coeficientes instáveis. Ridge reduz sua magnitude e geralmente melhora estabilidade fora da amostra.

## Exemplo numérico resolvido

Se duas colunas são idênticas, $x_1=x_2$, qualquer par com $\beta_1+\beta_2=4$ produz a mesma previsão: $(4,0)$, $(2,2)$ ou $(0,4)$. A penalidade L2 prefere $(2,2)$ porque

$$
2^2+2^2=8 < 4^2+0^2=16.
$$

Isso ilustra por que Ridge distribui peso entre variáveis correlacionadas. A penalidade L1 vale 4 em todos esses pares e pode escolher soluções esparsas dependendo do restante do problema e do algoritmo.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(X_train, y_train)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Padronize os dados e trace os coeficientes para 50 valores logarítmicos de `alpha`. Compare OLS, Ridge, Lasso e Elastic Net por CV, número de coeficientes não nulos e estabilidade entre folds. Repita sem scaling e explique a diferença.

## Laboratório guiado completo

Trace o caminho de regularização e meça esparsidade e validação sob a mesma escala.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y, true_coef = make_regression(
    n_samples=300, n_features=40, n_informative=8, noise=20,
    coef=True, random_state=42
)
cv = KFold(5, shuffle=True, random_state=42)
for alpha in np.logspace(-3, 2, 10):
    for cls in (Ridge, Lasso):
        model = make_pipeline(StandardScaler(), cls(alpha=alpha, max_iter=20000))
        rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()
        model.fit(X, y)
        coef = model[-1].coef_
        print(cls.__name__, alpha, round(rmse, 2), "não nulos", np.count_nonzero(coef))
```

**Entregue:** curva `alpha × RMSE`; curva `alpha × ||beta||`; número de coeficientes não nulos; comparação com e sem scaling.

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

- Comparar alphas sem padronizar features.
- Escolher regularização pelo teste final.
- Interpretar coeficiente zero do Lasso como prova de irrelevância causal.
- Esquecer que regularização adiciona viés deliberadamente.

## 8. Exercícios

1. Explique por que Ridge ajuda com multicolinearidade.
2. Qual método tende a gerar solução esparsa?
3. Por que scaling importa?
4. Descreva o trade-off de aumentar lambda.

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

- ISLP, cap. 6 — Linear Model Selection and Regularization.
- Tibshirani (1996) — Regression Shrinkage and Selection via the Lasso.
- Zou & Hastie (2005) — Regularization and Variable Selection via the Elastic Net.
- scikit-learn — Linear models.

## Leitura orientada e fontes verificadas

- Tibshirani (1996) — [Regression Shrinkage and Selection via the Lasso](https://academic.oup.com/jrsssb/article/58/1/267/7027929).
- Hastie, Tibshirani e Wainwright — [Statistical Learning with Sparsity](https://hastie.su.domains/StatLearnSparsity/).
- James et al. — [ISLP](https://www.statlearning.com/), cap. 6.
- scikit-learn — [Linear model regularization](https://scikit-learn.org/stable/modules/linear_model.html).

## Próxima aula

**Regressão logística e classificação probabilística**
