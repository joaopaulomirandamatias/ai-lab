# Aula 09 — Árvores de decisão: partições, impureza e interpretabilidade

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 08 deste módulo  
**Objetivo central:** Entender como árvores particionam o espaço de features e por que profundidade controla complexidade.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar nós, splits e folhas.
- Entender Gini, entropy e MSE em árvores.
- Relacionar profundidade a overfitting.
- Aplicar pruning/regularização.
- Reconhecer instabilidade de árvores individuais.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como árvores particionam o espaço de features e por que profundidade controla complexidade.**

## 2. Ideias fundamentais

### 1. Partições recursivas

A árvore escolhe regras do tipo $x_j<t$ para dividir dados em regiões progressivamente mais homogêneas.

### 2. Impureza

Em classificação, Gini e entropia medem mistura de classes. O split busca maior redução ponderada de impureza.

### 3. Overfitting

Árvores profundas podem memorizar pequenas irregularidades. max_depth, min_samples_leaf e pruning controlam complexidade.

### 4. Interpretabilidade

Uma árvore pequena é fácil de explicar, mas árvores reais podem crescer e se tornar instáveis. Importância de feature baseada em impureza também possui vieses.

## Aprofundamento — ganho de impureza e controle de complexidade

Para um nó $S$ e divisão em $S_L,S_R$, o ganho é

$$
Gain=I(S)-\frac{|S_L|}{|S|}I(S_L)-\frac{|S_R|}{|S|}I(S_R).
$$

O algoritmo avalia candidatos e escolhe o maior ganho local; isso não garante a árvore globalmente ótima. Features contínuas geram thresholds; categóricas codificadas de forma inadequada podem impor relações artificiais.

Árvores sem restrição tendem a criar folhas pequenas e de alta variância. `max_depth`, `min_samples_leaf`, `max_leaf_nodes` e cost-complexity pruning definem regularização estrutural. Uma árvore pequena pode ser inspecionada, mas a estabilidade também importa: pequenas mudanças na amostra podem mudar os primeiros splits.

## 3. Equação para guardar

$$
Gini=1-\sum_k p_k^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Uma árvore de risco pode primeiro separar renda < 2000, depois idade < 25 e assim construir regiões de decisão.

## Exemplo numérico resolvido

Nó pai com 6 positivos e 4 negativos:

$$
Gini_{pai}=1-0{,}6^2-0{,}4^2=0{,}48.
$$

Um split gera folha esquerda com 4 positivos/0 negativos e direita com 2 positivos/4 negativos. A impureza ponderada é

$$
\frac4{10}(0)+\frac6{10}\left[1-(1/3)^2-(2/3)^2\right]\approx0{,}267.
$$

Logo, o ganho é $0{,}48-0{,}267=0{,}213$. Compare candidatos pelo ganho, mas valide a árvore completa fora da amostra.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    random_state=42
)
tree.fit(X_train, y_train)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Treine profundidades 1–20 e registre score de treino, CV, número de folhas e menor folha. Extraia `cost_complexity_pruning_path`, escolha `ccp_alpha` por CV e teste estabilidade dos primeiros splits em cinco amostras bootstrap.

## Laboratório guiado completo

Observe o caminho de overfitting e depois aplique poda por complexidade.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for depth in [1, 2, 3, 4, 6, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, random_state=42)
    s = cross_validate(tree, X, y, cv=cv, scoring="balanced_accuracy", return_train_score=True)
    print(depth, round(s["train_score"].mean(), 3), round(s["test_score"].mean(), 3))

path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X, y)
print("primeiros alphas", path.ccp_alphas[:10])
```

**Entregue:** curva profundidade × treino/CV; árvore final; cálculo manual de um ganho Gini; estabilidade do primeiro split em bootstraps.

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

- Deixar árvore crescer sem controle.
- Confundir regra aprendida com regra causal.
- Confiar cegamente em feature_importances_.
- Ignorar instabilidade sob pequenas perturbações.

## 8. Exercícios

1. Calcule Gini para uma folha 80/20.
2. O que acontece quando max_depth cresce muito?
3. Por que árvore individual pode ter alta variância?
4. Dê um exemplo de split de uma feature contínua.

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

- Breiman et al. — Classification and Regression Trees.
- ISLP — Tree-Based Methods.
- Hastie et al. — Trees.
- scikit-learn — Decision Trees.

## Leitura orientada e fontes verificadas

- Breiman et al. — *Classification and Regression Trees* (CART), 1984.
- James et al. — [ISLP](https://www.statlearning.com/), cap. 8.
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), cap. 9.
- scikit-learn — [Decision Trees](https://scikit-learn.org/stable/modules/tree.html).

## Próxima aula

**Bagging e Random Forest: reduzindo variância com ensembles**
