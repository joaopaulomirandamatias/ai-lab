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

## Aprofundamento — a métrica define a vizinhança

Para classificação, KNN estima localmente

$$
\hat P(Y=c\mid x)=\frac{1}{k}\sum_{i\in N_k(x)}\mathbf{1}(y_i=c),
$$

possivelmente com pesos inversos à distância. A noção de “perto” vem da métrica e da representação. Euclidiana pressupõe comparabilidade de escalas e geometria contínua; distância de cosseno pode ser mais adequada para vetores cuja direção importa; categorias exigem outra codificação.

Com $k=1$, a fronteira é flexível e sensível a ruído. Ao aumentar $k$, reduzimos variância e aumentamos viés. Em alta dimensão, volume e distâncias se concentram: o vizinho mais próximo pode deixar de ser realmente próximo. Acrescentar features irrelevantes pode piorar mesmo sem alterar a informação útil.

## 3. Equação para guardar

$$
d(x,z)=\sqrt{\sum_j(x_j-z_j)^2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Classificar uma flor pela classe majoritária entre as k flores mais próximas no espaço de comprimento/largura de pétalas e sépalas.

## Exemplo numérico resolvido

Consulta $q=(1,1)$ e pontos $A=(1,2)$, $B=(3,1)$, $C=(8,8)$. As distâncias Euclidianas são $1$, $2$ e $\sqrt{98}\approx9{,}90$. Para $k=1$, a classe de $A$ vence.

Agora interprete a segunda feature em milhares: $A=(1,2000)$ e $q=(1,1000)$. Sem scaling, a segunda coordenada domina qualquer diferença na primeira. Padronizar não é detalhe estético: redefine a geometria usada pelo algoritmo.

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

### Investigação adicional

Meça CV para $k\in\{1,3,5,11,21,51\}$ com e sem scaling. Depois acrescente 0, 10, 50 e 200 features de ruído e trace desempenho e razão entre distância do vizinho mais próximo e do mais distante.

## Laboratório guiado completo

Meça simultaneamente o efeito de $k$, scaling e dimensões irrelevantes.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X0, y = make_classification(n_samples=600, n_features=4, n_informative=3,
                            n_redundant=0, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for noise_dim in [0, 10, 50, 200]:
    noise = rng.normal(size=(len(X0), noise_dim))
    X = np.c_[X0, noise]
    for k in [1, 3, 5, 11, 31]:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(k))
        score = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy").mean()
        print(noise_dim, k, round(score, 3))
```

**Entregue:** mapa `ruído × k × score`; comparação sem scaler; tempo de inferência por tamanho do treino; interpretação da concentração de distâncias.

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

- Usar KNN sem scaling.
- Escolher k olhando o teste.
- Aplicar KNN ingênuo em milhões de pontos sem considerar custo.
- Assumir que distância Euclidiana é apropriada para qualquer representação.

## 8. Exercícios

1. O que acontece com o viés quando k aumenta?
2. Por que scaling é crítico?
3. Explique a maldição da dimensionalidade.
4. Compare custo de treinamento do KNN com regressão logística.

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

- ISLP — KNN em classificação.
- Hastie et al. — Nearest-Neighbor Methods.
- Beyer et al. — When Is Nearest Neighbor Meaningful?
- scikit-learn — Nearest Neighbors.

## Leitura orientada e fontes verificadas

- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), métodos de vizinhos mais próximos.
- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), métodos não paramétricos.
- scikit-learn — [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html).
- scikit-learn — [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), para scaling dentro de pipelines.

## Próxima aula

**Naive Bayes: probabilidade condicional aplicada à classificação**
