# Aula 24 — Gate II — Experimento completo de Machine Learning clássico

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 23 deste módulo  
**Objetivo central:** Integrar todo o módulo em um experimento científico e reproduzível, com baseline, cross-validation e zero leakage.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Definir pergunta e métrica antes do treino.
- Construir baseline e múltiplos modelos.
- Usar pipeline e cross-validation.
- Fazer tuning sem contaminar o teste.
- Produzir relatório com incerteza, erros e limitações.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **integrar todo o módulo em um experimento científico e reproduzível, com baseline, cross-validation e zero leakage.**

## 2. Ideias fundamentais

### 1. Protocolo

A pergunta, unidade de análise, target, split e métricas devem ser definidos antes de examinar o resultado final.

### 2. Comparação

Compare no mínimo baseline, modelo linear e modelo não linear/ensemble sob exatamente o mesmo protocolo.

### 3. Teste final

O conjunto de teste é usado uma vez, após decisões metodológicas. Se o teste orienta tuning, deixa de ser teste.

### 4. Relatório

Inclua distribuição das métricas no CV, intervalo/variabilidade, matriz de confusão ou resíduos, análise de falhas e o que o experimento não prova.

## Aprofundamento — o Gate II avalia método, não a maior métrica

Congele antes do teste: pergunta, unidade, $t_0$, target, população, split, métricas, baseline, modelos candidatos, espaço de tuning e critérios de decisão. O conjunto de teste é um recurso não renovável: depois que um resultado orienta escolhas, ele passou a ser validação.

Compare pelo menos:

1. baseline operacional/dummy;
2. modelo linear com preprocessing adequado;
3. árvore/ensemble sob o mesmo protocolo.

Registre distribuição do CV, custo, latência e complexidade, não apenas média. Faça análise de erros por faixa e grupo; verifique calibração se probabilidades alimentam decisões. O relatório precisa conter threats to validity: sampling bias, label noise, shift, dependência, hiperparâmetros explorados e limites de causalidade.

O artefato aceito deve ligar código, snapshot, configuração e resultados. A conclusão deve caber em três níveis: evidência observada, interpretação plausível e afirmações não sustentadas.

## 3. Equação para guardar

$$
\text{Gate II}=\text{baseline}+\text{CV}+\text{pipeline}+\text{zero leakage}+\text{evidência}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Escolha um dataset real. Compare Dummy baseline, regressão logística e Random Forest/Gradient Boosting. Faça nested ou train+CV para tuning, depois avalie uma vez no teste.

## Exemplo numérico resolvido

CV no treino produz:

| Modelo | ROC-AUC média | AP média | Latência |
|---|---:|---:|---:|
| Dummy | 0,50 | 0,10 | 0,1 ms |
| Logística | 0,78 | 0,42 | 0,3 ms |
| Random Forest | 0,81 | 0,45 | 8 ms |

Se o requisito é AP $\ge0{,}40$ e latência $<1$ ms, a logística vence apesar da métrica menor. Após congelar a decisão, o teste retorna ROC-AUC 0,79 e AP 0,43. Esse é o resultado final; não volte para escolher a floresta olhando o teste.

## 5. Laboratório em Python / scikit-learn

```python
# Estrutura recomendada:
# 1. carregar snapshot/version do dataset
# 2. separar teste externo
# 3. definir preprocessing + estimator em Pipeline
# 4. cross-validation no treino
# 5. tuning apenas dentro do treino
# 6. selecionar modelo
# 7. avaliar uma única vez no teste
# 8. salvar relatório e artefatos

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline = DummyClassifier(strategy="prior")
scores = cross_validate(
    baseline, X_train, y_train,
    cv=cv,
    scoring=["roc_auc", "average_precision"]
)

print(scores["test_roc_auc"].mean())
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Entregue um repositório executável com `data/README`, configuração, pipeline, script de treino, avaliação, relatório e model card. O comando único deve reconstruir CV e teste a partir do snapshot autorizado. Inclua um teste automatizado que falha se preprocessing for ajustado fora do pipeline.

## Laboratório guiado completo — esqueleto do Gate II

O código compara os três níveis sob o mesmo CV e preserva um teste externo.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "dummy": DummyClassifier(strategy="prior"),
    "linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                     random_state=42, n_jobs=-1),
}
for name, model in models.items():
    s = cross_validate(model, Xtr, ytr, cv=cv,
                       scoring=["roc_auc", "average_precision"], n_jobs=-1)
    print(name, {k: (v.mean(), v.std()) for k,v in s.items() if k.startswith("test_")})

chosen = models["linear"].fit(Xtr, ytr)  # decisão congelada após CV + requisitos
p = chosen.predict_proba(Xte)[:,1]
print("TESTE ÚNICO", {"roc_auc": roc_auc_score(yte,p),
                      "average_precision": average_precision_score(yte,p)})
```

**Entregue:** comando único reproduzível; comparação; critério de escolha congelado; teste único; análise de erros; custo/latência; model card e seção “este experimento não prova que...”.

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

- Não ter baseline.
- Tunar no teste.
- Fazer preprocessing fora do pipeline.
- Relatar somente melhor execução.
- Declarar causalidade a partir de associação preditiva.

## 8. Exercícios

1. Escreva a pergunta experimental do seu Gate II.
2. Escolha baseline e dois modelos candidatos.
3. Defina estratégia de split adequada à estrutura dos dados.
4. Liste métricas primária e secundárias.
5. Escreva uma seção 'O que este experimento não prova'.

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

- ISLP — model assessment, linear models, trees and SVM.
- scikit-learn — Model selection, pipelines and common pitfalls.
- Cawley & Talbot (2010) — Over-fitting in model selection.
- Sculley et al. — Hidden Technical Debt in ML Systems.

## Leitura orientada e fontes verificadas

- scikit-learn — [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) e [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- Cawley e Talbot (2010) — [Over-fitting in model selection](https://jmlr.org/papers/v11/cawley10a.html).
- Bergstra e Bengio (2012) — [Random Search](https://jmlr.org/papers/v13/bergstra12a.html).
- NeurIPS — [Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist).
- James et al. — [ISLP](https://www.statlearning.com/) e laboratórios oficiais em Python.

## Próxima aula

**Deep Learning — redes neurais do zero (M5)**
