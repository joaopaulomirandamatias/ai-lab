# Aula 19 — Feature engineering e seleção de variáveis

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 18 deste módulo  
**Objetivo central:** Criar representações úteis sem vazar target e selecionar features de forma compatível com validação.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Distinguir transformação e seleção de features.
- Criar interações e features temporais.
- Conhecer filtros, wrappers e métodos embedded.
- Aplicar seleção dentro do pipeline.
- Entender regularização como seleção/controle.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **criar representações úteis sem vazar target e selecionar features de forma compatível com validação.**

## 2. Ideias fundamentais

### 1. Representação

Modelos aprendem sobre as features fornecidas. Uma boa representação pode tornar um problema difícil quase linear.

### 2. Filtros

Métodos univariados avaliam cada feature individualmente e são rápidos, mas ignoram interações.

### 3. Embedded

Lasso e árvores fazem seleção/ponderação durante o próprio treinamento.

### 4. Leakage

Qualquer seleção orientada por y precisa ser ajustada apenas no treino/fold.

## 3. Equação para guardar

$$
x_{\text{novo}}=\phi(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Uma timestamp pode gerar hora, dia da semana, feriado e tempo desde último evento; mas nenhuma feature pode usar informações posteriores ao instante de predição.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    SelectKBest(mutual_info_classif, k=20),
    LogisticRegression(max_iter=1000)
)
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

- Selecionar features antes do CV.
- Criar agregações futuras.
- Confiar em importância univariada para relações complexas.
- Adicionar centenas de features sem avaliar estabilidade.

## 8. Exercícios

1. Dê três features derivadas de timestamp.
2. Por que seleção deve estar dentro do pipeline?
3. Diferencie filtro, wrapper e embedded.
4. Explique uma feature que causaria leakage.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Guyon & Elisseeff (2003) — An Introduction to Variable and Feature Selection.
- ISLP — model selection and feature engineering.
- scikit-learn — Feature selection.
- Murphy — feature selection.

## Próxima aula

**Interpretabilidade: coeficientes, permutation importance e SHAP**
