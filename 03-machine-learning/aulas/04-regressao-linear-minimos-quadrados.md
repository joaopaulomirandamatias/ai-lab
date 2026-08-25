# Aula 04 — Regressão linear e mínimos quadrados

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 03 deste módulo  
**Objetivo central:** Conectar o que foi aprendido em Álgebra Linear e Estatística ao primeiro modelo supervisionado clássico.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar coeficientes de regressão.
- Entender mínimos quadrados.
- Relacionar matriz X, vetor de parâmetros e previsão.
- Analisar resíduos.
- Reconhecer limitações da interpretação causal.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **conectar o que foi aprendido em álgebra linear e estatística ao primeiro modelo supervisionado clássico.**

## 2. Ideias fundamentais

### 1. Modelo linear

A regressão linear assume uma relação aproximada $\hat y=\beta_0+x^T\beta$. Linear refere-se aos parâmetros; features podem ser transformadas, por exemplo com termos polinomiais.

### 2. Mínimos quadrados

Os coeficientes são escolhidos para minimizar a soma dos resíduos quadráticos. A solução possui ligação direta com projeções ortogonais.

### 3. Resíduos

Resíduo é $e_i=y_i-\hat y_i$. Padrões nos resíduos podem revelar não linearidade, heterocedasticidade, dependência temporal ou outliers.

### 4. Coeficiente não é automaticamente causa

Um coeficiente descreve associação condicional sob o modelo e os dados observados. Causalidade exige desenho e hipóteses adicionais.

## 3. Equação para guardar

$$
\hat{\beta}=\arg\min_\beta \|X\beta-y\|_2^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Prever consumo de energia usando temperatura e ocupação. O coeficiente de temperatura expressa a variação prevista no target por unidade da feature, mantendo as demais do modelo constantes.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)

pred = reg.predict(X_test)
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R²:", r2_score(y_test, pred))
print("coef:", reg.coef_)
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

- Interpretar R² alto como evidência de causalidade.
- Ignorar extrapolação fora do domínio dos dados.
- Avaliar apenas R² sem observar magnitude dos erros.
- Ignorar resíduos e dependências.

## 8. Exercícios

1. Derive a loss MSE de uma regressão simples.
2. Explique o significado de um coeficiente negativo.
3. Por que a regressão linear pode funcionar mesmo com features transformadas?
4. O que um padrão curvo nos resíduos sugere?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP, cap. 3 — Linear Regression.
- Hastie et al., cap. 3 — Linear Methods for Regression.
- Murphy — PML, linear regression.
- scikit-learn — LinearRegression.

## Próxima aula

**Regularização: Ridge, Lasso e Elastic Net**
