# Aula 12 — Support Vector Machines: margem máxima e kernels

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 11 deste módulo  
**Objetivo central:** Entender classificação por margem e o papel dos support vectors e kernels.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar hiperplano e margem.
- Identificar support vectors.
- Entender parâmetro C.
- Compreender kernel trick conceitualmente.
- Usar RBF com scaling e tuning correto.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender classificação por margem e o papel dos support vectors e kernels.**

## 2. Ideias fundamentais

### 1. Margem

O classificador busca um hiperplano com grande margem entre classes. Apenas pontos próximos à fronteira influenciam diretamente a solução.

### 2. Soft margin

C controla o custo de violações. C alto penaliza erros com força; C baixo permite margem mais suave.

### 3. Kernel

Kernels permitem calcular produtos internos em espaços transformados sem construir explicitamente todas as novas features.

### 4. RBF

O kernel RBF cria fronteiras flexíveis; gamma controla a escala de influência dos pontos.

## 3. Equação para guardar

$$
K(x,z)=\exp(-\gamma\|x-z\|^2)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Duas classes em círculos concêntricos não são linearmente separáveis no espaço original, mas um kernel pode induzir uma separação adequada.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
)
svm.fit(X_train, y_train)
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

- Usar SVM RBF sem scaling.
- Interpretar probability=True como probabilidade perfeita.
- Tunar C/gamma no teste.
- Usar kernel complexo quando modelo linear já resolve.

## 8. Exercícios

1. O que são support vectors?
2. Qual o efeito de C muito alto?
3. Explique kernel trick sem fórmula.
4. Por que scaling é importante?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Cortes & Vapnik (1995) — Support-Vector Networks.
- Hastie et al. — Support Vector Machines and Flexible Discriminants.
- ISLP — Support Vector Machines.
- scikit-learn — SVM.

## Próxima aula

**Métricas de regressão: MAE, MSE, RMSE, R² e erro relativo**
