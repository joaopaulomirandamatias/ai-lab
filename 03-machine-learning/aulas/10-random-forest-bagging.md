# Aula 10 — Bagging e Random Forest: reduzindo variância com ensembles

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 09 deste módulo  
**Objetivo central:** Entender como combinar árvores decorrelacionadas pode produzir modelos mais robustos.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar bootstrap aggregation.
- Entender random subspace em Random Forest.
- Relacionar ensemble a redução de variância.
- Usar out-of-bag score.
- Interpretar limites de feature importance.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender como combinar árvores decorrelacionadas pode produzir modelos mais robustos.**

## 2. Ideias fundamentais

### 1. Bagging

Treinamos vários modelos em amostras bootstrap e agregamos previsões. Se os erros não forem perfeitamente correlacionados, a média reduz variância.

### 2. Random Forest

Além do bootstrap, cada split considera apenas um subconjunto aleatório de features, decorrelacionando as árvores.

### 3. OOB

Cada árvore deixa de ver aproximadamente uma parte das observações bootstrap; essas amostras podem fornecer uma estimativa out-of-bag.

### 4. Paralelismo

As árvores são treinadas de forma independente, favorecendo paralelização.

## Aprofundamento — por que a média de árvores funciona

Se cada árvore tem variância $\sigma^2$ e correlação média $\rho$ com as demais, a variância aproximada da média de $B$ árvores é

$$
Var(\bar f)=\sigma^2\left(\rho+\frac{1-\rho}{B}\right).
$$

Mais árvores reduzem a parcela independente, mas não a parcela correlacionada. O sorteio de features em cada split busca justamente reduzir $\rho$ sem enfraquecer demais cada árvore.

Em uma amostra bootstrap de tamanho $n$, a probabilidade de uma observação não ser escolhida é $(1-1/n)^n\to e^{-1}\approx0{,}368$. Essas observações out-of-bag permitem avaliação interna, mas OOB não substitui automaticamente um teste temporal ou por grupos.

Importância por redução de impureza favorece features com muitos thresholds/categorias. Prefira permutation importance no conjunto de validação e interprete com cuidado features correlacionadas.

## 3. Equação para guardar

$$
\hat f_{\text{ens}}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat f_b(x)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em risco de crédito, 500 árvores diferentes votam/produzem probabilidades; o ensemble costuma ser mais estável que qualquer árvore isolada.

## Exemplo numérico resolvido

Com $B=100$ e correlação $\rho=0{,}10$:

$$
Var(\bar f)=\sigma^2(0{,}10+0{,}90/100)=0{,}109\sigma^2.
$$

Se $\rho=0{,}80$, a variância permanece $0{,}802\sigma^2$. O exemplo mostra que “adicionar árvores” não resolve falta de diversidade. `max_features` pode reduzir correlação e melhorar o ensemble.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print("OOB:", rf.oob_score_)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Compare árvore única, bagging sem sorteio de features e Random Forest para 10, 50, 200 e 500 árvores. Registre OOB e CV, variabilidade entre seeds e permutation importance no teste. Verifique quando o ganho satura.

## Laboratório guiado completo

Compare árvore, bagging e floresta e verifique saturação com o número de árvores.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "tree": DecisionTreeClassifier(random_state=42),
    "bagging": BaggingClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "forest": RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                     oob_score=True, random_state=42, n_jobs=-1),
}
for name, model in models.items():
    score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(X, y)
    print(name, score.mean(), score.std(), getattr(model, "oob_score_", None))
```

**Entregue:** curva 10–1.000 árvores; OOB versus CV; variabilidade por seed; permutation importance e crítica às importâncias por impureza.

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

- Achar que mais árvores elimina todo overfitting.
- Usar importância por impureza como evidência causal.
- Ignorar custo de memória/inferência.
- Comparar OOB e teste como se fossem exatamente equivalentes.

## 8. Exercícios

1. Explique por que decorrelação entre árvores ajuda.
2. O que é bootstrap?
3. Para que serve OOB?
4. Compare Random Forest e uma árvore única em viés/variância.

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

- Breiman (2001) — Random Forests.
- ISLP — Bagging, Random Forests and Boosting.
- Hastie et al. — Random Forests.
- scikit-learn — Ensemble methods.

## Leitura orientada e fontes verificadas

- Breiman (2001) — [Random Forests](https://link.springer.com/article/10.1023/A%3A1010933404324).
- Breiman (1996) — *Bagging Predictors*, Machine Learning 24.
- scikit-learn — [Forest ensembles](https://scikit-learn.org/stable/modules/ensemble.html#forest).
- scikit-learn — [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Próxima aula

**Boosting e Gradient Boosting: aprendendo com os erros anteriores**
