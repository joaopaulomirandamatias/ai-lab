# Aula 09 — Árvores de decisão: partições, impureza e interpretabilidade

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 08 deste módulo  
**Objetivo central:** Entender como árvores particionam o espaço de features e por que profundidade controla complexidade.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar nós, splits e folhas.
- Entender Gini, entropy e MSE em árvores.
- Relacionar profundidade a overfitting.
- Aplicar pruning/regularização.
- Reconhecer instabilidade de árvores individuais.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como árvores particionam o espaço de features e por que profundidade controla complexidade.**

## 2. Ideias fundamentais

### 1. Partições recursivas

A árvore escolhe regras do tipo $x_j<t$ para dividir dados em regiões progressivamente mais homogêneas.

### 2. Impureza

Em classificação, Gini e entropia medem mistura de classes. O split busca maior redução ponderada de impureza.

### 3. Overfitting

Árvores profundas podem memorizar pequenas irregularidades. max_depth, min_samples_leaf e pruning controlam complexidade.

### 4. Interpretabilidade

Uma árvore pequena é fácil de explicar, mas árvores reais podem crescer e se tornar instáveis. Importância de feature baseada em impureza também possui vieses.

## 3. Equação para guardar

$$
Gini=1-\sum_k p_k^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Uma árvore de risco pode primeiro separar renda < 2000, depois idade < 25 e assim construir regiões de decisão.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    random_state=42
)
tree.fit(X_train, y_train)
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

- Deixar árvore crescer sem controle.
- Confundir regra aprendida com regra causal.
- Confiar cegamente em feature_importances_.
- Ignorar instabilidade sob pequenas perturbações.

## 8. Exercícios

1. Calcule Gini para uma folha 80/20.
2. O que acontece quando max_depth cresce muito?
3. Por que árvore individual pode ter alta variância?
4. Dê um exemplo de split de uma feature contínua.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Breiman et al. — Classification and Regression Trees.
- ISLP — Tree-Based Methods.
- Hastie et al. — Trees.
- scikit-learn — Decision Trees.

## Próxima aula

**Bagging e Random Forest: reduzindo variância com ensembles**
