# Aula 10 — Bagging e Random Forest: reduzindo variância com ensembles

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 09 deste módulo  
**Objetivo central:** Entender como combinar árvores decorrelacionadas pode produzir modelos mais robustos.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar bootstrap aggregation.
- Entender random subspace em Random Forest.
- Relacionar ensemble a redução de variância.
- Usar out-of-bag score.
- Interpretar limites de feature importance.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como combinar árvores decorrelacionadas pode produzir modelos mais robustos.**

## 2. Ideias fundamentais

### 1. Bagging

Treinamos vários modelos em amostras bootstrap e agregamos previsões. Se os erros não forem perfeitamente correlacionados, a média reduz variância.

### 2. Random Forest

Além do bootstrap, cada split considera apenas um subconjunto aleatório de features, decorrelacionando as árvores.

### 3. OOB

Cada árvore deixa de ver aproximadamente uma parte das observações bootstrap; essas amostras podem fornecer uma estimativa out-of-bag.

### 4. Paralelismo

As árvores são treinadas de forma independente, favorecendo paralelização.

## 3. Equação para guardar

$$
\hat f_{\text{ens}}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat f_b(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em risco de crédito, 500 árvores diferentes votam/produzem probabilidades; o ensemble costuma ser mais estável que qualquer árvore isolada.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print("OOB:", rf.oob_score_)
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

- Achar que mais árvores elimina todo overfitting.
- Usar importância por impureza como evidência causal.
- Ignorar custo de memória/inferência.
- Comparar OOB e teste como se fossem exatamente equivalentes.

## 8. Exercícios

1. Explique por que decorrelação entre árvores ajuda.
2. O que é bootstrap?
3. Para que serve OOB?
4. Compare Random Forest e uma árvore única em viés/variância.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Breiman (2001) — Random Forests.
- ISLP — Bagging, Random Forests and Boosting.
- Hastie et al. — Random Forests.
- scikit-learn — Ensemble methods.

## Próxima aula

**Boosting e Gradient Boosting: aprendendo com os erros anteriores**
