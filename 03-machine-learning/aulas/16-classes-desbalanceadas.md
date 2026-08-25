# Aula 16 — Classes desbalanceadas: amostragem, pesos e avaliação correta

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 15 deste módulo  
**Objetivo central:** Tratar desbalanceamento sem transformar a avaliação em uma ilusão.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Entender prevalência e rare events.
- Usar class_weight.
- Conhecer undersampling e oversampling.
- Aplicar resampling somente no treino.
- Escolher métricas robustas ao desbalanceamento.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **tratar desbalanceamento sem transformar a avaliação em uma ilusão.**

## 2. Ideias fundamentais

### 1. Prevalência

Desbalanceamento não é automaticamente um problema; depende da tarefa, custo e informação das features.

### 2. Class weights

Podemos aumentar a penalização de erros na classe minoritária sem duplicar exemplos.

### 3. Resampling

Oversampling e undersampling devem ocorrer dentro de cada fold de treinamento, nunca antes do split/cross-validation.

### 4. Métrica

Balanced accuracy, PR-AUC, recall e precision são frequentemente mais informativas que accuracy.

## 3. Equação para guardar

$$
BalancedAccuracy=\frac{TPR+TNR}{2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Fraude 0.5%: um modelo que sempre prevê 'não fraude' tem 99.5% accuracy e nenhum valor operacional.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(balanced_accuracy_score(y_test, pred))
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

- Oversampling antes do split.
- Usar SMOTE no teste.
- Otimizar accuracy.
- Ignorar prevalência futura na calibração.

## 8. Exercícios

1. Por que 99% accuracy pode ser inútil?
2. Quando class_weight pode ser preferível a oversampling?
3. Por que resampling deve estar dentro do CV?
4. Escolha métricas para fraude.

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- He & Garcia (2009) — Learning from Imbalanced Data.
- Chawla et al. (2002) — SMOTE.
- scikit-learn — Imbalanced datasets / class_weight.
- Saito & Rehmsmeier — PR curves.

## Próxima aula

**Cross-validation: estimando generalização sem desperdiçar dados**
