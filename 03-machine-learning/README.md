# 03 · Machine Learning clássico — M4

**Mês M4** · Roadmap: Nível 3 · **🔶 Gate II** no fim do módulo

> Aqui começa a formação propriamente dita em IA.

**Objetivo do módulo:** construir, avaliar e comparar modelos clássicos com protocolo experimental honesto, baseline explícito, cross-validation e **zero data leakage**.

## Gate II

O módulo termina quando você consegue entregar um experimento reproduzível contendo:

- pergunta e target claramente definidos;
- unidade de análise;
- baseline;
- split correto para a estrutura dos dados;
- preprocessing dentro de `Pipeline`;
- cross-validation;
- tuning sem usar o teste;
- comparação de modelos;
- métricas primária/secundárias justificadas;
- análise de erros;
- provenance/lineage;
- limitações;
- avaliação final única no conjunto de teste.

## Sequência de 24 aulas

| Aula | Tema |
|---:|---|
| 01 | [Fundamentos de Machine Learning: problemas, paradigmas e generalização](./aulas/01-fundamentos-machine-learning.md) |
| 02 | [Do problema ao experimento: features, target, splits e baseline](./aulas/02-framing-dataset-split-baseline.md) |
| 03 | [Pré-processamento, pipelines e data leakage](./aulas/03-preprocessamento-pipelines-leakage.md) |
| 04 | [Regressão linear e mínimos quadrados](./aulas/04-regressao-linear-minimos-quadrados.md) |
| 05 | [Regularização: Ridge, Lasso e Elastic Net](./aulas/05-regularizacao-ridge-lasso-elastic-net.md) |
| 06 | [Regressão logística e classificação probabilística](./aulas/06-regressao-logistica-classificacao-probabilistica.md) |
| 07 | [K-Nearest Neighbors: distâncias e maldição da dimensionalidade](./aulas/07-knn-distancias-dimensionalidade.md) |
| 08 | [Naive Bayes: probabilidade condicional aplicada à classificação](./aulas/08-naive-bayes-probabilidade-condicional.md) |
| 09 | [Árvores de decisão: partições, impureza e interpretabilidade](./aulas/09-arvores-decisao.md) |
| 10 | [Bagging e Random Forest: reduzindo variância com ensembles](./aulas/10-random-forest-bagging.md) |
| 11 | [Boosting e Gradient Boosting: aprendendo com os erros anteriores](./aulas/11-gradient-boosting.md) |
| 12 | [Support Vector Machines: margem máxima e kernels](./aulas/12-svm-kernels.md) |
| 13 | [Métricas de regressão: MAE, MSE, RMSE, R² e erro relativo](./aulas/13-metricas-regressao.md) |
| 14 | [Métricas de classificação: matriz de confusão, precision, recall e F1](./aulas/14-metricas-classificacao.md) |
| 15 | [ROC, Precision-Recall, thresholds e calibração](./aulas/15-roc-pr-threshold-calibracao.md) |
| 16 | [Classes desbalanceadas: amostragem, pesos e avaliação correta](./aulas/16-classes-desbalanceadas.md) |
| 17 | [Cross-validation: estimando generalização sem desperdiçar dados](./aulas/17-cross-validation.md) |
| 18 | [Hyperparameter tuning: Grid Search, Random Search e validação aninhada](./aulas/18-hyperparameter-tuning.md) |
| 19 | [Feature engineering e seleção de variáveis](./aulas/19-feature-engineering-selection.md) |
| 20 | [Interpretabilidade: coeficientes, permutation importance e SHAP](./aulas/20-interpretabilidade-modelos.md) |
| 21 | [Clustering: K-Means, hierárquico e DBSCAN](./aulas/21-clustering-kmeans-hierarquico-dbscan.md) |
| 22 | [Redução de dimensionalidade em ML: PCA, t-SNE e UMAP com responsabilidade](./aulas/22-reducao-dimensionalidade-ml.md) |
| 23 | [Reprodutibilidade, provenance, pipelines e zero data leakage](./aulas/23-reprodutibilidade-provenance-leakage.md) |
| 24 | [Gate II — Experimento completo de Machine Learning clássico](./aulas/24-gate-ii-experimento-ml-classico.md) |

## Blocos

### Bloco A — Fundamentos e modelos lineares · 01–06
Problema, desenho experimental, pipelines, regressão, regularização e classificação probabilística.

### Bloco B — Algoritmos clássicos · 07–12
KNN, Naive Bayes, árvores, Random Forest, boosting e SVM.

### Bloco C — Avaliação e seleção · 13–18
Métricas, thresholds, calibração, desbalanceamento, cross-validation e tuning.

### Bloco D — Representação, interpretação e ciência · 19–24
Feature engineering, interpretabilidade, clustering, redução dimensional, reprodutibilidade e Gate II.

## Referências-base

1. James, Witten, Hastie, Tibshirani, Taylor — **An Introduction to Statistical Learning with Applications in Python (ISLP)**.
2. Hastie, Tibshirani, Friedman — **The Elements of Statistical Learning**.
3. Murphy — **Probabilistic Machine Learning: An Introduction**.
4. scikit-learn — **User Guide** e seção **Common pitfalls and recommended practices**.
5. Géron — **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow**.
6. Papers clássicos indicados em cada aula: CART, Random Forests, Gradient Boosting, SVM, SMOTE, SHAP, Random Search etc.

## Fonte principal e apoio

⭐ **Hands-On Machine Learning** — Aurélien Géron, Parte 1, como referência prática.

Use **ISLP** para consolidar teoria e prática, **The Elements of Statistical Learning**
para aprofundamento e **Probabilistic Machine Learning** para a visão probabilística.
StatQuest e materiais técnicos em português podem servir para destravar a intuição, mas
não substituem as referências acadêmicas e a documentação oficial.

## O que realmente se aprende aqui

Não são apenas algoritmos. O núcleo do M4 é **metodologia experimental**, que será
reutilizada ao comparar RAG no M11 e arquiteturas multiagentes no M14:

- **Baseline primeiro.** Um modelo que não supera uma referência simples ainda não
  demonstrou valor.
- **A métrica vem do problema.** Accuracy em dataset desbalanceado pode esconder um
  sistema inútil.
- **Data leakage é silencioso.** Normalizar antes do split, usar informação futura ou
  deixar duplicatas atravessarem os conjuntos produz resultados bons demais e inválidos.
- **Reprodutibilidade é parte do resultado.** Seeds, versões, configuração e lineage
  devem acompanhar cada experimento.

## Projeto do módulo

**P4 — Pipeline de ML sem leakage (M4)** 🔶 **Gate II**

Classificação ou regressão completa: split → baseline → preprocessing → modelo →
validação cruzada → métricas → análise de erros → avaliação final no teste.

*Pronto quando* você consegue apontar pelo menos três riscos de data leakage e mostrar
no código e no protocolo como cada um foi eliminado.

## Regra do módulo

> Um modelo com métrica alta e protocolo contaminado vale menos que um baseline simples medido corretamente.

O objetivo não é colecionar algoritmos; é aprender a produzir **evidência preditiva confiável**.
