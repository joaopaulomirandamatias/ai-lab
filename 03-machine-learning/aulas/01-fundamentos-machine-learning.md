# Aula 01 — Fundamentos de Machine Learning: problemas, paradigmas e generalização

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Módulos 01 Matemática e 02 Probabilidade/Estatística concluídos  
**Objetivo central:** Entender o que torna um problema de Machine Learning diferente de uma regra programada manualmente e distinguir treinamento, inferência e generalização.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir aprendizado supervisionado, não supervisionado e semi/self-supervisionado.
- Definir amostra, feature, target, modelo, parâmetro e hiperparâmetro.
- Explicar treinamento, inferência e generalização.
- Reconhecer underfitting, overfitting e erro irredutível.
- Formular um problema de ML antes de escolher um algoritmo.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender o que torna um problema de machine learning diferente de uma regra programada manualmente e distinguir treinamento, inferência e generalização.**

## 2. Ideias fundamentais

### 1. Do programa explícito ao modelo aprendido

Em programação tradicional, regras + dados produzem saídas. Em ML supervisionado, exemplos de entrada e saída são usados para aprender uma função aproximada $\hat f(x;\theta)$ capaz de generalizar para exemplos não vistos.

### 2. Treinamento e inferência

Treinamento é o processo de ajustar parâmetros $\theta$ para reduzir uma função de perda. Inferência é usar os parâmetros já ajustados para produzir previsões. Misturar os dois conceitos leva a erros de arquitetura e avaliação.

### 3. Generalização

O objetivo real não é minimizar o erro nos dados de treinamento, mas o risco esperado em dados provenientes da distribuição de interesse. Um modelo que memoriza o treino pode ter baixo erro empírico e alto erro fora da amostra.

### 4. Viés e variância

Modelos simples podem ter alto viés; modelos muito flexíveis podem ter alta variância. O ponto útil depende do tamanho dos dados, ruído, regularização e desenho experimental.

## Aprofundamento — risco empírico, risco populacional e generalização

O conjunto de treino é uma amostra finita. O objeto que realmente queremos minimizar é o **risco populacional**

$$
R(\theta)=\mathbb{E}_{(X,Y)\sim P_{\text{alvo}}}[L(f_\theta(X),Y)],
$$

mas a distribuição-alvo $P_{\text{alvo}}$ é desconhecida. Substituímos a esperança pelo risco empírico

$$
\hat R_n(\theta)=\frac{1}{n}\sum_{i=1}^{n}L(f_\theta(x_i),y_i).
$$

Minimizar $\hat R_n$ é necessário, porém não suficiente. Se a família de modelos for flexível demais para o tamanho e a qualidade da amostra, o algoritmo pode aprender ruído, identificadores ou particularidades do período observado. A diferença $R(\hat\theta)-\hat R_n(\hat\theta)$ é o **gap de generalização**.

Toda afirmação de generalização depende de hipóteses: a amostra deve representar o contexto de uso; exemplos não podem atravessar indevidamente os splits; e a distribuição de produção não pode mudar de forma relevante. Em dados temporais, médicos ou corporativos, a hipótese iid é frequentemente apenas uma aproximação e precisa ser discutida.

## 3. Equação para guardar

$$
\hat{\theta}=\arg\min_{\theta}\frac{1}{n}\sum_{i=1}^{n}L(f_\theta(x_i),y_i)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Classificar e-mails como spam/não spam. Features podem incluir representações do texto; target é a classe. O modelo é treinado em exemplos rotulados e avaliado em mensagens que não participaram do ajuste.

## Exemplo numérico resolvido

Considere erro de classificação em 1.000 exemplos de treino e 200 de validação:

- árvore profunda: erro de treino $2\%$ e validação $11\%$;
- regressão logística: erro de treino $7\%$ e validação $8\%$;
- baseline majoritário: erro de validação $9\%$.

A árvore tem gap de $9$ pontos percentuais e perde para o baseline fora da amostra: forte sinal de overfitting. A regressão logística tem gap de apenas $1$ ponto e melhora o baseline em $1$ ponto. Não basta olhar o treino nem escolher o modelo mais complexo; a decisão deve considerar incerteza, custo dos erros e estabilidade em outros splits.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Compare `DummyClassifier`, regressão logística e árvore com profundidades 1, 2, 4, 8 e sem limite. Construa curvas de treino e validação para 10%, 25%, 50%, 75% e 100% dos dados. Explique onde aparece alto viés, onde aparece alta variância e se mais dados parecem ajudar.

## Laboratório guiado completo

O experimento abaixo mede treino e teste para um baseline, um modelo linear e uma árvore flexível. Execute antes de alterar parâmetros.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
models = {
    "dummy": DummyClassifier(strategy="prior"),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "tree": DecisionTreeClassifier(random_state=42),
}
for name, model in models.items():
    model.fit(Xtr, ytr)
    train = balanced_accuracy_score(ytr, model.predict(Xtr))
    test = balanced_accuracy_score(yte, model.predict(Xte))
    print(name, {"train": round(train, 3), "test": round(test, 3), "gap": round(train-test, 3)})
```

**Entregue:** tabela com os gaps; árvore com `max_depth` variando; curva de aprendizagem; explicação de por que score de treino não decide o melhor modelo.

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

- Escolher o algoritmo antes de definir claramente target, unidade de análise e métrica.
- Avaliar o modelo nos mesmos dados usados para treiná-lo.
- Confundir desempenho médio histórico com garantia de comportamento futuro.
- Ignorar mudança de distribuição entre treino e produção.

## 8. Exercícios

1. Dê um exemplo de problema que não precisa de ML e explique por quê.
2. Explique com suas palavras a diferença entre parâmetro e hiperparâmetro.
3. Por que erro de treino baixo não garante generalização?
4. Classifique três problemas seus como regressão, classificação ou não supervisionado.

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

- James et al. — An Introduction to Statistical Learning with Applications in Python (ISLP), caps. 1–2.
- Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, caps. 1–2.
- Murphy — Probabilistic Machine Learning: An Introduction, cap. 1.
- scikit-learn User Guide — Supervised learning.

## Leitura orientada e fontes verificadas

- James et al. — [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/), caps. 1–2.
- Murphy — [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html), decisões, risco e empirical risk minimization.
- Hastie, Tibshirani e Friedman — [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), caps. 2 e 7.
- scikit-learn — [Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html) e [model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).

## Próxima aula

**Do problema ao experimento: features, target, splits e baseline**
