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

## Aprofundamento — PCA é modelagem linear; t-SNE/UMAP são mapas

Com $X$ centralizada e covariância $S=X^TX/(n-1)$, PCA escolhe direções ortonormais $w_j$ que maximizam $w_j^TSw_j$. São autovetores de $S$ ordenados pelos autovalores. A razão $\lambda_j/\sum_k\lambda_k$ mede variância explicada, não informação sobre o target.

t-SNE transforma vizinhanças em probabilidades e minimiza divergência KL entre relações de alta e baixa dimensão. Ele privilegia estrutura local; distâncias entre ilhas, tamanho e densidade visual não devem ser lidos literalmente. UMAP constrói um grafo fuzzy de vizinhança sob hipóteses de variedade e otimiza uma representação. `n_neighbors`, `min_dist`, métrica e seed mudam o mapa.

Para modelagem supervisionada, PCA deve ficar dentro do pipeline. Para visualização exploratória, ajuste várias seeds/hiperparâmetros, colore por metadados conhecidos e nunca use o desenho 2D como prova isolada de clusters.

## 3. Equação para guardar

$$
Z=XW_k
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Usar PCA para reduzir 1000 features antes de regressão logística. O PCA precisa estar dentro do Pipeline para que cada fold aprenda componentes sem ver sua validação.

## Exemplo numérico resolvido

Dados centralizados $(2,0),(-2,0),(0,1),(0,-1)$. Usando divisor $n$, a covariância é diagonal com variâncias $2$ e $0{,}5$. Os autovetores são os eixos; o primeiro componente é o eixo $x$ e explica

$$
\frac{2}{2+0{,}5}=0{,}80
$$

da variância. Reduzir a uma dimensão preserva 80% da variância total, mas pode descartar sinal preditivo se o target depender do eixo $y$.

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

### Investigação adicional

Compare modelo sem PCA e PCA com 50%, 80%, 95% e 99% de variância dentro de CV. Para o mesmo embedding, gere mapas t-SNE/UMAP em cinco seeds e vários `perplexity`/`n_neighbors`. Registre estruturas estáveis e instáveis.

## Laboratório guiado completo

PCA entra no pipeline; t-SNE é usado apenas como visualização exploratória neste experimento.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for variance in [0.50, 0.80, 0.95, 0.99]:
    model = make_pipeline(StandardScaler(), PCA(n_components=variance),
                          LogisticRegression(max_iter=3000))
    score = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(variance, score.mean(), score.std())

embedding = TSNE(n_components=2, perplexity=30, init="pca",
                 learning_rate="auto", random_state=42).fit_transform(X[:800])
print("embedding", embedding.shape, "não é entrada automática do classificador")
```

**Entregue:** variância × score × dimensões; comparação sem PCA; mapas para cinco seeds/perplexities; lista do que o mapa 2D não permite concluir.

### Protocolo investigativo obrigatório

O laboratório não termina quando o código executa. Para transformar execução em aprendizagem e evidência:

1. escreva uma hipótese antes de rodar o experimento;
2. mantenha um baseline e altere uma decisão por vez;
3. use o mesmo split ou os mesmos folds nas comparações;
4. reporte a distribuição das métricas, não apenas o melhor número;
5. inspecione pelo menos cinco erros ou casos extremos;
6. registre seed, versões, hiperparâmetros e tempo de execução;
7. conclua com **o que os resultados sustentam** e **o que não sustentam**.

Salve um relatório curto em Markdown, a configuração em JSON e o código executável. Uma execução sem interpretação não satisfaz o critério de domínio.

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

## Exercícios de aprofundamento e rubrica

### Nível A — reconstrução conceitual

Feche o material e explique o problema, as hipóteses, cada símbolo das equações e a diferença entre treinamento, seleção e avaliação. Desenhe o fluxo de dados sem consultar o texto. Se uma definição depender de palavras vagas como “melhor” ou “parecido”, torne-a operacional.

### Nível B — cálculo e implementação

Refaça o exemplo numérico com valores diferentes e confira manualmente o resultado do código. Implemente a operação matemática central com NumPy ou Python básico antes de usar a abstração do scikit-learn. Compare tolerâncias e explique qualquer diferença numérica.

### Nível C — contraprova experimental

Crie deliberadamente um cenário em que o método falha: ruído, outlier, escala incompatível, shift, grupos repetidos, classe rara ou leakage. Formule antes o comportamento esperado, execute a ablação e confronte hipótese e resultado.

### Nível D — transferência para sistema real

Aplique o conceito a um problema do AI Systems Laboratory. Declare unidade, instante de predição, dados disponíveis, baseline, métrica, custo dos erros e threat to validity. Produza um artefato que outra pessoa consiga auditar.

### Rubrica de 0 a 4

- **0 — reconhecimento:** identifica o nome, mas não explica o mecanismo;
- **1 — reprodução:** executa exemplo pronto;
- **2 — compreensão:** deriva/calcula e interpreta o resultado;
- **3 — diagnóstico:** prevê falhas, escolhe protocolo e analisa erros;
- **4 — transferência:** projeta, implementa e defende um experimento novo e reproduzível.

**Carga sugerida:** 45 min de leitura ativa, 45 min de derivação/cálculo, 90 min de laboratório, 30 min de análise de erros e 30 min de relatório. Avance somente ao atingir pelo menos nível 3.

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

## Leitura orientada e fontes verificadas

- van der Maaten e Hinton (2008) — [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html).
- McInnes et al. (2018) — [UMAP, Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.00861).
- scikit-learn — [Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html) e [manifold learning](https://scikit-learn.org/stable/modules/manifold.html).
- James et al. — [ISLP](https://www.statlearning.com/), PCA e aprendizagem não supervisionada.

## Próxima aula

**Reprodutibilidade, provenance, pipelines e zero data leakage**
