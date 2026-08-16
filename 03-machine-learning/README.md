# 03 · Machine Learning clássico

**Mês M4** · Roadmap: Nível 3 · **🔶 Gate II** no fim do mês

> Aqui começa a formação propriamente dita em IA.

---

## O que dominar

**Supervisionado** — regressão linear e logística, k-NN, Naive Bayes, Decision Trees,
Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM.

**Não supervisionado** — K-Means, DBSCAN, clustering hierárquico, PCA, redução de
dimensionalidade, detecção de anomalias.

**Conceitos fundamentais** (dominar profundamente, não reconhecer) — overfitting,
underfitting, bias, variance, regularização L1/L2, hiperparâmetros, cross-validation,
grid search, random search, otimização bayesiana.

**Métricas**
- Classificação: accuracy, precision, recall, F1, ROC, AUC, matriz de confusão
- Regressão: MAE, MSE, RMSE, R²

---

## Fonte principal

⭐ **Hands-On Machine Learning** — Aurélien Géron, Parte 1. O mais prático que existe.

Teoria de apoio: ISLP (que você já usou no M3) · ESL para consulta.
Destravar: StatQuest · Mario Filho (em português, especialmente sobre validação).

---

## Projeto

**P4 — Pipeline de ML sem leakage (M4)** 🔶 **Gate II**

Classificação ou regressão completa: split → baseline → modelo → validação cruzada →
métricas.

*Pronto quando* você consegue apontar três lugares onde havia risco de data leakage e
mostrar no código como cada um foi eliminado.

---

## O que realmente se aprende aqui

Não são os algoritmos — XGBoost você aprende a usar em uma tarde. É **metodologia
experimental**, e ela vale para o resto do roadmap inteiro:

- **Baseline primeiro.** Um modelo que não bate a média ou a classe majoritária não é
  um modelo, é um gerador de números.
- **A métrica é escolhida pelo problema.** Accuracy em dataset desbalanceado é
  desinformação. Escolher métrica por hábito é o erro mais caro do M4.
- **Data leakage é silencioso.** Ele não dá erro — dá resultado bom demais. Normalizar
  antes do split, usar feature que só existe depois do evento, deixar duplicata cruzar
  o split. Você vai cometer os três pelo menos uma vez.

Essa disciplina é literalmente a mesma que o M11 vai cobrar ao comparar RAG, e a mesma
que o M14 vai cobrar ao comparar 1 vs 3 vs 5 agentes. Aprenda direito agora e economize
duas vezes.
