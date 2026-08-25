# Aula 07 — K-Nearest Neighbors: distâncias e maldição da dimensionalidade

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 06 deste módulo  
**Objetivo central:** Entender um algoritmo local simples e usá-lo para aprofundar a intuição sobre distância em espaços de alta dimensão.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar a regra do KNN.
- Escolher k e métrica de distância.
- Entender impacto do scaling.
- Reconhecer a maldição da dimensionalidade.
- Distinguir custo de treino e inferência.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender um algoritmo local simples e usá-lo para aprofundar a intuição sobre distância em espaços de alta dimensão.**

## 2. Ideias fundamentais

### 1. Aprendizado local

KNN praticamente armazena o conjunto de treino e decide pela vizinhança no momento da previsão. Isso desloca custo para a inferência.

### 2. Escolha de k

k pequeno pode gerar alta variância; k grande suaviza fronteiras e aumenta viés.

### 3. Scaling

Distâncias ficam dominadas por features de maior escala, portanto padronização costuma ser essencial.

### 4. Dimensionalidade

Em dimensões altas, distâncias tendem a se concentrar e vizinhanças ficam menos informativas. Redução de dimensionalidade ou outras representações podem ajudar.

## 3. Equação para guardar

$$
d(x,z)=\sqrt{\sum_j(x_j-z_j)^2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Classificar uma flor pela classe majoritária entre as k flores mais próximas no espaço de comprimento/largura de pétalas e sépalas.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
knn.fit(X_train, y_train)
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

- Usar KNN sem scaling.
- Escolher k olhando o teste.
- Aplicar KNN ingênuo em milhões de pontos sem considerar custo.
- Assumir que distância Euclidiana é apropriada para qualquer representação.

## 8. Exercícios

1. O que acontece com o viés quando k aumenta?
2. Por que scaling é crítico?
3. Explique a maldição da dimensionalidade.
4. Compare custo de treinamento do KNN com regressão logística.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- ISLP — KNN em classificação.
- Hastie et al. — Nearest-Neighbor Methods.
- Beyer et al. — When Is Nearest Neighbor Meaningful?
- scikit-learn — Nearest Neighbors.

## Próxima aula

**Naive Bayes: probabilidade condicional aplicada à classificação**
