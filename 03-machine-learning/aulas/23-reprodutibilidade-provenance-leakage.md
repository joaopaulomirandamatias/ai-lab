# Aula 23 — Reprodutibilidade, provenance, pipelines e zero data leakage

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 22 deste módulo  
**Objetivo central:** Construir experimentos de ML que possam ser auditados e repetidos.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Controlar seeds e versões.
- Registrar dataset, código e parâmetros.
- Distinguir provenance e lineage.
- Criar pipeline end-to-end.
- Aplicar checklist de leakage.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **construir experimentos de ml que possam ser auditados e repetidos.**

## 2. Ideias fundamentais

### 1. Reprodutibilidade

Uma métrica sem código, dados, split e configuração identificáveis é uma evidência fraca.

### 2. Provenance

Registra de onde o dado veio, quando foi obtido, sob quais regras e transformações.

### 3. Lineage

Rastreia o caminho de transformação do dado até features, modelos e artefatos finais.

### 4. Zero leakage

Qualquer informação da validação/teste ou do futuro que influencie treinamento, seleção ou preprocessing invalida a estimativa.

## 3. Equação para guardar

$$
\text{evidência}=\text{dados}+\text{código}+\text{config}+\text{métrica}+\text{rastreabilidade}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Um experimento deve permitir reconstruir exatamente qual snapshot de dataset e quais hiperparâmetros produziram um modelo e sua métrica.

## 5. Laboratório em Python / scikit-learn

```python
import sklearn, numpy as np

print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)

RANDOM_STATE = 42
# salve também:
# - hash/version do dataset
# - commit do código
# - parâmetros do pipeline
# - folds/split strategy
# - métricas e intervalos
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

- Salvar somente o pickle do modelo.
- Não versionar dataset.
- Alterar preprocessing manualmente fora do pipeline.
- Não registrar seed e estratégia de split.

## 8. Exercícios

1. Crie um checklist mínimo de reprodução.
2. Diferencie provenance de lineage.
3. Liste cinco tipos de leakage.
4. Que artefatos você salvaria após um experimento?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- scikit-learn — Common pitfalls and recommended practices.
- Sculley et al. (2015) — Hidden Technical Debt in Machine Learning Systems.
- W3C PROV — provenance data model.
- MLflow documentation — experiment tracking concepts.

## Próxima aula

**Gate II — Experimento completo de Machine Learning clássico**
