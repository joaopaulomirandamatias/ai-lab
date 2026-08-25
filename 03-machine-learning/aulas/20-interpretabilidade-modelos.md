# Aula 20 — Interpretabilidade: coeficientes, permutation importance e SHAP

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 19 deste módulo  
**Objetivo central:** Interpretar modelos sem confundir explicação preditiva com causalidade.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar coeficientes de modelos lineares.
- Entender permutation importance.
- Conhecer SHAP conceitualmente.
- Distinguir explicação global e local.
- Reconhecer limitações sob features correlacionadas.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **interpretar modelos sem confundir explicação preditiva com causalidade.**

## 2. Ideias fundamentais

### 1. Global vs local

Explicação global tenta resumir comportamento médio; local explica uma previsão específica.

### 2. Permutation importance

Embaralha uma feature e mede a queda de performance. Features correlacionadas podem compartilhar informação e reduzir importância aparente.

### 3. SHAP

Usa ideias de valores de Shapley para atribuir contribuição das features a uma previsão em relação a um baseline.

### 4. Explicação não é causa

Uma feature importante para o modelo pode ser proxy, artefato ou variável correlacionada. Interpretabilidade preditiva não prova efeito causal.

## 3. Equação para guardar

$$
\Delta_j=Score(X)-Score(X_{\pi(j)})
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Um modelo pode usar CEP como proxy socioeconômica. SHAP pode revelar influência, mas isso não estabelece causalidade e ainda levanta questões de fairness.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42
)

print(r.importances_mean)
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

- Tratar SHAP como verdade causal.
- Ignorar correlação entre features.
- Explicar modelo avaliado em dados vazados.
- Mostrar gráficos bonitos sem medir fidelidade/estabilidade.

## 8. Exercícios

1. Diferencie explicação global e local.
2. Por que features correlacionadas complicam importance?
3. O que SHAP não prova?
4. Crie uma hipótese de proxy sensível.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions.
- Molnar — Interpretable Machine Learning.
- Breiman — Random Forest feature importance caveats.
- scikit-learn — Permutation feature importance.

## Próxima aula

**Clustering: K-Means, hierárquico e DBSCAN**
