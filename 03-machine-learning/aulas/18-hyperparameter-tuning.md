# Aula 18 — Hyperparameter tuning: Grid Search, Random Search e validação aninhada

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 17 deste módulo  
**Objetivo central:** Otimizar hiperparâmetros sem contaminar a estimativa final de desempenho.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir parâmetros e hiperparâmetros.
- Usar GridSearchCV e RandomizedSearchCV.
- Entender espaços logarítmicos.
- Evitar overfitting à validação.
- Conhecer nested cross-validation.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **otimizar hiperparâmetros sem contaminar a estimativa final de desempenho.**

## 2. Ideias fundamentais

### 1. Parâmetro vs hiperparâmetro

Parâmetros são aprendidos pelo fit; hiperparâmetros controlam o processo/modelo e são escolhidos externamente.

### 2. Random Search

Quando poucos hiperparâmetros realmente importam, random search explora valores úteis de forma mais eficiente que uma grade cartesiana rígida.

### 3. Nested CV

Loop interno escolhe hiperparâmetros; loop externo estima generalização. É importante em comparações científicas com datasets pequenos.

### 4. Orçamento

Tuning é um problema de busca sob orçamento. Mais trials não corrigem um protocolo metodológico ruim.

## Aprofundamento — seleção também pode overfitar

Cada configuração obtém uma estimativa ruidosa. Ao escolher o máximo entre muitas configurações, favorecemos também quem teve ruído positivo. Esse “winner's curse” cresce com espaço de busca, variância do CV e intervenção manual.

Random search costuma ser mais eficiente quando apenas alguns hiperparâmetros importam, pois uma grade desperdiça trials repetindo valores dos eixos irrelevantes. Distribuições devem refletir escala: `loguniform` para parâmetros que variam por ordens de magnitude.

Na nested CV, o loop interno escolhe hiperparâmetros e o loop externo mede todo o procedimento de seleção em dados que o loop interno nunca viu. O resultado externo serve para estimativa/comparação; depois, para produção, o procedimento é reexecutado em todos os dados de desenvolvimento antes do teste externo ou deploy.

## 3. Equação para guardar

$$
\lambda^*=\arg\max_{\lambda\in\Lambda}\widehat{Score}_{CV}(\lambda)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Tunar C e gamma de uma SVM dentro dos folds e usar outro loop externo para estimar desempenho reduz o otimismo de selecionar e avaliar nos mesmos folds.

## Exemplo numérico resolvido

Uma grade com 5 valores de $C$, 4 de $\gamma$ e CV de 5 folds exige $5\cdot4\cdot5=100$ ajustes, além do refit. Random search com 20 trials também exige 100 ajustes, mas explora 20 combinações distintas sem ficar preso aos mesmos eixos.

Nested CV com 5 folds externos, 4 internos e 20 trials exige aproximadamente $5\cdot20\cdot4=400$ fits, mais refits. Defina orçamento antes; aumentar busca sem desenho correto apenas automatiza overfitting.

## 5. Laboratório em Python / scikit-learn

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

search = RandomizedSearchCV(
    SVC(),
    param_distributions={
        "C": loguniform(1e-3, 1e3),
        "gamma": loguniform(1e-4, 1e1)
    },
    n_iter=50,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Compare GridSearchCV e RandomizedSearchCV sob o mesmo orçamento de fits. Em seguida passe `RandomizedSearchCV` como estimador para `cross_validate` em folds externos. Reporte scores internos escolhidos e externos; explique o otimismo entre ambos.

## Laboratório guiado completo

Nested CV mede o procedimento de busca inteiro, não apenas a configuração vencedora.

```python
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
inner = StratifiedKFold(4, shuffle=True, random_state=42)
outer = StratifiedKFold(5, shuffle=True, random_state=123)
search = RandomizedSearchCV(
    pipe, {"logisticregression__C": loguniform(1e-4, 1e4)},
    n_iter=30, cv=inner, scoring="roc_auc", random_state=42, n_jobs=-1
)
res = cross_validate(search, X, y, cv=outer, scoring="roc_auc", return_estimator=True)
for outer_score, fitted_search in zip(res["test_score"], res["estimator"]):
    print("inner best", fitted_search.best_score_, "outer", outer_score,
          "params", fitted_search.best_params_)
```

**Entregue:** orçamento total de fits; gap interno–externo; comparação Grid/Random sob igual orçamento; distribuição dos hiperparâmetros escolhidos.

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

- Tunar no teste.
- Usar distribuição linear para hiperparâmetros que variam em ordens de magnitude.
- Reportar melhor score interno como desempenho final.
- Comparar modelos com budgets de tuning muito diferentes sem transparência.

## 8. Exercícios

1. Diferencie Grid e Random Search.
2. O que nested CV resolve?
3. Por que C costuma ser pesquisado em escala log?
4. Explique overfitting à validação.

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

- Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization.
- Cawley & Talbot (2010) — On Over-fitting in Model Selection.
- scikit-learn — Tuning hyper-parameters.
- ISLP — model selection.

## Leitura orientada e fontes verificadas

- Bergstra e Bengio (2012) — [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html).
- Cawley e Talbot (2010) — [Selection bias in performance evaluation](https://jmlr.org/papers/v11/cawley10a.html).
- scikit-learn — [Tuning hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html).
- scikit-learn — [Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Próxima aula

**Feature engineering e seleção de variáveis**
