# Aula 22 — Redução de dimensionalidade em ML: PCA, t-SNE e UMAP com responsabilidade

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 21 deste módulo  
**Objetivo central:** Usar redução dimensional para compressão, visualização e pré-processamento sem atribuir significado excessivo a mapas 2D.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Revisar PCA no contexto de pipeline.
- Distinguir redução linear e visualização não linear.
- Entender objetivos de t-SNE e UMAP.
- Evitar leakage ao aplicar PCA.
- Interpretar mapas 2D com cautela.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **usar redução dimensional para compressão, visualização e pré-processamento sem atribuir significado excessivo a mapas 2d.**

## 2. Ideias fundamentais

### 1. PCA

É uma transformação linear orientada por variância e pode ser integrada ao pipeline antes do estimador.

### 2. t-SNE

Preserva vizinhanças locais para visualização; distâncias globais e tamanhos de clusters não devem ser lidos literalmente.

### 3. UMAP

Também busca preservar estrutura de vizinhança, com suposições próprias e sensibilidade a hiperparâmetros.

### 4. Leakage

PCA aprende componentes dos dados e deve ser ajustada apenas dentro do treino/fold quando usada para modelagem supervisionada.

## 3. Equação para guardar

$$
Z=XW_k
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Usar PCA para reduzir 1000 features antes de regressão logística. O PCA precisa estar dentro do Pipeline para que cada fold aprenda componentes sem ver sua validação.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
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

- Fazer PCA antes do split.
- Tratar distâncias em t-SNE como métricas globais.
- Escolher representação só pela aparência visual.
- Confundir componente principal com feature original.

## 8. Exercícios

1. Por que PCA deve estar dentro do pipeline?
2. O que t-SNE tenta preservar?
3. Por que dois clusters afastados no mapa t-SNE não provam classes separáveis?
4. Quando PCA pode ajudar um modelo?

## 9. Critério de domínio

Você domina esta aula quando consegue:
1. explicar o conceito sem consultar a documentação;
2. implementar um experimento mínimo;
3. identificar pelo menos dois modos de leakage ou avaliação enganosa;
4. justificar a métrica e o protocolo de validação.

## 10. Referências principais

- Jolliffe & Cadima — PCA: a review.
- van der Maaten & Hinton (2008) — Visualizing Data using t-SNE.
- McInnes, Healy & Melville — UMAP.
- scikit-learn — Decomposition and manifold learning.

## Próxima aula

**Reprodutibilidade, provenance, pipelines e zero data leakage**
