# Aula 17 — Cross-validation: estimando generalização sem desperdiçar dados

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 16 deste módulo  
**Objetivo central:** Entender resampling como ferramenta de estimativa e seleção, preservando independência do teste final.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar k-fold CV.
- Usar StratifiedKFold e GroupKFold.
- Entender TimeSeriesSplit.
- Distinguir CV de teste final.
- Calcular média e dispersão das métricas.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender resampling como ferramenta de estimativa e seleção, preservando independência do teste final.**

## 2. Ideias fundamentais

### 1. K-fold

Os dados são divididos em k partes; cada fold funciona uma vez como validação e as demais como treino.

### 2. Estratificação e grupos

Em classificação, estratificar preserva proporções. Em dados com múltiplas linhas por sujeito/entidade, grupos devem ficar inteiros em um lado do split.

### 3. Tempo

Para previsão temporal, o treinamento não pode usar o futuro para prever o passado. TimeSeriesSplit respeita ordenação.

### 4. Variância da estimativa

Não reporte apenas a média dos folds; dispersão e intervalos ajudam a avaliar estabilidade.

## 3. Equação para guardar

$$
\bar m=\frac{1}{K}\sum_{k=1}^{K}m_k
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em dados médicos, todas as consultas do mesmo paciente devem pertencer ao mesmo fold para evitar que identidade do paciente vaze entre treino e validação.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["roc_auc", "average_precision"],
    n_jobs=-1
)

print(scores["test_roc_auc"].mean())
print(scores["test_roc_auc"].std())
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

- Fazer preprocessing antes do CV.
- Ignorar grupos.
- Usar KFold aleatório em série temporal.
- Usar o teste como mais um fold.

## 8. Exercícios

1. Quando usar GroupKFold?
2. Por que o teste final continua necessário?
3. Qual problema de usar shuffle em previsão temporal?
4. Por que reportar desvio dos folds?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Kohavi (1995) — A Study of Cross-Validation and Bootstrap.
- ISLP, cap. 5 — Resampling Methods.
- scikit-learn — Cross-validation.
- Varoquaux et al. — Assessing and tuning brain decoders: cross-validation.

## Próxima aula

**Hyperparameter tuning: Grid Search, Random Search e validação aninhada**
