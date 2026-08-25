# Aula 02 — Do problema ao experimento: features, target, splits e baseline

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 01 deste módulo  
**Objetivo central:** Aprender a transformar uma pergunta de negócio ou pesquisa em um experimento de ML mensurável e sem vazamento.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Definir unidade de análise, features e target.
- Separar treino, validação e teste com papéis distintos.
- Construir baselines úteis.
- Escolher métricas compatíveis com o custo do erro.
- Entender por que o conjunto de teste deve permanecer intocado.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **aprender a transformar uma pergunta de negócio ou pesquisa em um experimento de ml mensurável e sem vazamento.**

## 2. Ideias fundamentais

### 1. Unidade de análise

Antes de falar em features, defina o que cada linha representa: pessoa, transação, processo, documento, janela temporal, equipamento etc. Uma unidade mal definida cria duplicação, dependência indevida e leakage.

### 2. Train / validation / test

Treino ajusta parâmetros; validação orienta escolhas; teste estima desempenho final. Quando cross-validation é usada, o papel da validação é incorporado aos folds, mas o teste externo continua separado.

### 3. Baseline

Um modelo só é útil se supera uma referência plausível. Em regressão, prever a média é um baseline; em classificação, classe majoritária ou uma regra operacional existente podem ser referências.

### 4. Métrica orientada ao objetivo

Accuracy pode ser inadequada quando falsos negativos e falsos positivos têm custos diferentes. A métrica deve refletir o uso real do sistema.

## Aprofundamento — desenhe o evento de predição antes do dataset

Defina um **instante índice** $t_0$: o momento em que o sistema produzirá a previsão. Separe então quatro janelas:

1. população elegível, definida com informação disponível até $t_0$;
2. janela de features, sempre anterior ou igual a $t_0$;
3. janela de label, normalmente posterior a $t_0$;
4. horizonte de uso, no qual a decisão será tomada.

Esse desenho impede features retrospectivas, como “dias até o desfecho”, “status final” ou agregações calculadas depois da decisão. Também obriga a declarar a unidade de análise. Se uma pessoa possui dez linhas, um split aleatório por linha permite que sua identidade apareça em treino e teste; nesse caso o split deve ser por grupo. Se o uso é futuro, um split cronológico é mais fiel do que embaralhar períodos.

O baseline deve representar uma alternativa real: média, mediana, classe majoritária, regra vigente ou último valor conhecido. “Superar zero” raramente é uma comparação útil.

## 3. Equação para guardar

$$
\text{ganho}=\text{métrica(modelo)}-\text{métrica(baseline)}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever atraso de entrega. Uma linha = uma entrega. O target deve usar apenas informação disponível até o instante em que a previsão será feita. Um baseline pode ser a taxa histórica de atraso por rota.

## Exemplo numérico resolvido

Há 1.000 entregas, 200 atrasadas. Um classificador que sempre prevê “no prazo” obtém

$$
Accuracy=\frac{800}{1000}=0{,}80,
$$

mas recall de atraso igual a zero e balanced accuracy $0{,}50$. Um modelo com 85% de accuracy parece melhor, mas, se 100 entregas do mesmo cliente foram divididas entre treino e teste e o identificador do cliente entrou como feature, a diferença pode ser memorização. Refaça o split por cliente: se a balanced accuracy cair de $0{,}72$ para $0{,}56$, o primeiro resultado não estimava generalização para clientes novos.

## 5. Laboratório em Python / scikit-learn

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# X, y devem estar definidos
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# baseline = DummyClassifier(strategy="prior")
# baseline.fit(X_train, y_train)
# pred = baseline.predict(X_test)
# print(balanced_accuracy_score(y_test, pred))
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Escreva um “contrato de predição” com unidade, $t_0$, horizonte, target, features permitidas, features proibidas, população e métrica. Compare `train_test_split`, `GroupShuffleSplit` e um corte temporal no mesmo dataset. Registre qual pergunta cada split realmente responde.

## Laboratório guiado completo

Aqui cada “entidade” possui várias linhas. Compare split aleatório por linha com split por grupo.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

X, y = make_classification(n_samples=1000, n_features=12, weights=[0.8, 0.2], random_state=42)
groups = np.repeat(np.arange(200), 5)

def evaluate(train, test):
    baseline = DummyClassifier(strategy="prior").fit(X[train], y[train])
    model = LogisticRegression(max_iter=2000).fit(X[train], y[train])
    return {
        "baseline": balanced_accuracy_score(y[test], baseline.predict(X[test])),
        "model": balanced_accuracy_score(y[test], model.predict(X[test])),
    }

idx = np.arange(len(y))
tr1, te1 = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
tr2, te2 = next(GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42).split(X, y, groups))
print("split por linha", evaluate(tr1, te1))
print("split por grupo", evaluate(tr2, te2))
assert set(groups[tr2]).isdisjoint(groups[te2])
```

**Entregue:** contrato de predição; baseline; justificativa do splitter; teste automático de interseção de grupos; lista de features proibidas após $t_0$.

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

- Usar o teste repetidamente durante desenvolvimento.
- Criar features com informação posterior ao momento da previsão.
- Não definir um baseline antes do tuning.
- Fazer split aleatório em dados temporais ou grupos dependentes.

## 8. Exercícios

1. Defina unidade de análise e target para um problema de detecção de fraude.
2. Explique a diferença entre validação e teste.
3. Crie dois baselines para um problema de classificação desbalanceada.
4. Dê um exemplo em que split aleatório seria metodologicamente incorreto.

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

- ISLP, cap. 5 — Resampling Methods.
- Kaufman et al. — Leakage in Data Mining: Formulation, Detection, and Avoidance.
- scikit-learn — Model selection and evaluation.
- Google Rules of ML — orientação prática sobre baselines e métricas.

## Leitura orientada e fontes verificadas

- scikit-learn — [Cross-validation iterators](https://scikit-learn.org/stable/modules/cross_validation.html), incluindo grupos e séries temporais.
- scikit-learn — [Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators).
- James et al. — [ISLP](https://www.statlearning.com/), caps. 2 e 5.
- NeurIPS — [Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist), para explicitar desenho, dados e limitações.

## Próxima aula

**Pré-processamento, pipelines e data leakage**
