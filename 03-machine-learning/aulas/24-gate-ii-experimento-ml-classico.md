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

## 3. Equação para guardar

$$
\text{Gate II}=\text{baseline}+\text{CV}+\text{pipeline}+\text{zero leakage}+\text{evidência}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Escolha um dataset real. Compare Dummy baseline, regressão logística e Random Forest/Gradient Boosting. Faça nested ou train+CV para tuning, depois avalie uma vez no teste.

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

## Próxima aula

**Deep Learning — redes neurais do zero (M5)**
