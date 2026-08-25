# Aula 11 — Boosting e Gradient Boosting: aprendendo com os erros anteriores

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 10 deste módulo  
**Objetivo central:** Entender boosting como construção sequencial de um modelo aditivo que corrige resíduos/gradientes.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Diferenciar bagging e boosting.
- Explicar modelo aditivo.
- Entender learning rate e número de árvores.
- Interpretar gradient boosting como descida no espaço de funções.
- Conhecer XGBoost/LightGBM/CatBoost conceitualmente.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender boosting como construção sequencial de um modelo aditivo que corrige resíduos/gradientes.**

## 2. Ideias fundamentais

### 1. Sequencialidade

Boosting adiciona novos weak learners para corrigir padrões ainda mal modelados.

### 2. Gradient Boosting

Cada nova árvore é ajustada aos pseudo-resíduos, relacionados ao gradiente negativo da loss.

### 3. Shrinkage

Learning rate pequeno exige mais árvores, mas frequentemente melhora generalização.

### 4. Implementações modernas

XGBoost, LightGBM e CatBoost acrescentam engenharia eficiente, regularização, estratégias para histogramas e tratamento de categorias.

## Aprofundamento — gradient descent no espaço de funções

Gradient Boosting constrói um modelo aditivo $F_m$. Em cada etapa calcula pseudo-resíduos

$$
r_{im}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}},
$$

ajusta um weak learner $h_m$ para prever esses valores e atualiza $F_m=F_{m-1}+\eta h_m$. Para loss quadrática, o gradiente negativo é o resíduo $y_i-F_{m-1}(x_i)$; para log-loss, a expressão muda.

Profundidade das árvores controla ordem de interações; learning rate e número de iterações são acoplados. Valores pequenos de $\eta$ exigem mais árvores. Early stopping deve observar validação interna, nunca o teste final.

XGBoost, LightGBM e CatBoost implementam variações importantes, mas o conceito central continua sendo otimização sequencial de uma loss por modelos aditivos.

## 3. Equação para guardar

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em regressão, a primeira árvore aproxima o target; as próximas modelam sistematicamente os resíduos ainda existentes.

## Exemplo numérico resolvido

Targets $[3,5,9]$. O modelo inicial para MSE é a média $F_0=17/3\approx5{,}667$. Os pseudo-resíduos são

$$
[-2{,}667,-0{,}667,3{,}333].
$$

Suponha que uma árvore preveja $h_1=[-2,-2,3]$ e $\eta=0{,}1$. Então

$$
F_1=[5{,}467,5{,}467,5{,}967].
$$

O terceiro caso se aproxima de 9; os dois primeiros também se movem. A próxima árvore verá novos resíduos. Boosting corrige erros progressivamente, não treina árvores independentes.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.ensemble import HistGradientBoostingClassifier

gb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    random_state=42
)
gb.fit(X_train, y_train)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Implemente três iterações de boosting para regressão usando stumps. Depois compare com `GradientBoostingRegressor`/`HistGradientBoostingClassifier`. Construa uma grade de learning rate × número de árvores e plote train/validation loss por iteração.

## Laboratório guiado completo

Primeiro simule atualizações de resíduos; depois observe learning curves no estimador real.

```python
import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X, y = make_hastie_10_2(n_samples=2500, random_state=42)
y = (y == 1).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
for eta in [0.01, 0.05, 0.2]:
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=eta,
                                       max_depth=2, random_state=42).fit(Xtr, ytr)
    checkpoints = []
    for m, p in enumerate(model.staged_predict_proba(Xte), 1):
        if m in [1, 10, 50, 100, 300]:
            checkpoints.append((m, log_loss(yte, p)))
    print("eta", eta, checkpoints)
```

**Entregue:** derivação dos pseudo-resíduos para MSE; implementação de três stumps; gráfico iteração × loss para três learning rates; ponto de early stopping.

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

- Usar muitas árvores e learning rate alto sem validação.
- Comparar bibliotecas com defaults diferentes como se fossem equivalentes.
- Ignorar calibration em classificação.
- Fazer tuning no teste.

## 8. Exercícios

1. Diferencie bagging de boosting.
2. Qual o papel do learning rate?
3. Por que boosting é sequencial?
4. Explique pseudo-resíduos em linguagem simples.

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

- Friedman (2001) — Greedy Function Approximation: A Gradient Boosting Machine.
- Chen & Guestrin (2016) — XGBoost.
- ISLP — Boosting.
- scikit-learn — Gradient Boosting.

## Leitura orientada e fontes verificadas

- Friedman (2001) — [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.short).
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), cap. 10.
- scikit-learn — [Gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting).
- James et al. — [ISLP](https://www.statlearning.com/), métodos baseados em árvores.

## Próxima aula

**Support Vector Machines: margem máxima e kernels**
