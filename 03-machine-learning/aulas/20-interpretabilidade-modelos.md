# Aula 20 — Interpretabilidade: coeficientes, permutation importance e SHAP

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 19 deste módulo  
**Objetivo central:** Interpretar modelos sem confundir explicação preditiva com causalidade.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar coeficientes de modelos lineares.
- Entender permutation importance.
- Conhecer SHAP conceitualmente.
- Distinguir explicação global e local.
- Reconhecer limitações sob features correlacionadas.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **interpretar modelos sem confundir explicação preditiva com causalidade.**

## 2. Ideias fundamentais

### 1. Global vs local

Explicação global tenta resumir comportamento médio; local explica uma previsão específica.

### 2. Permutation importance

Embaralha uma feature e mede a queda de performance. Features correlacionadas podem compartilhar informação e reduzir importância aparente.

### 3. SHAP

Usa ideias de valores de Shapley para atribuir contribuição das features a uma previsão em relação a um baseline.

### 4. Explicação não é causa

Uma feature importante para o modelo pode ser proxy, artefato ou variável correlacionada. Interpretabilidade preditiva não prova efeito causal.

## Aprofundamento — explique o modelo, os dados e a pergunta

Coeficientes lineares são globais apenas na escala e codificação usadas. Com colinearidade, muitos vetores de coeficientes produzem previsões semelhantes. Permutation importance mede quanto o score cai quando a associação de uma feature com o target é quebrada; o resultado depende do conjunto, métrica e correlações.

Para uma explicação aditiva local, SHAP busca

$$
f(x)=E[f(X)]+\sum_j\phi_j,
$$

sob uma definição de coalizões e background. A escolha do background altera o baseline e as contribuições. Dependência entre features torna “feature ausente” ambígua; diferentes variantes fazem hipóteses condicionais/intervencionais.

Uma explicação fiel ao modelo pode revelar que ele usa um proxy indevido, mas não garante que a previsão seja correta, justa ou causal. Valide explicadores com sanity checks, estabilidade e conhecimento de domínio.

## 3. Equação para guardar

$$
\Delta_j=Score(X)-Score(X_{\pi(j)})
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Um modelo pode usar CEP como proxy socioeconômica. SHAP pode revelar influência, mas isso não estabelece causalidade e ainda levanta questões de fairness.

## Exemplo numérico resolvido

$f(x)=2x_1+0{,}5x_2+1$. Para $x=(3,4)$, $f(x)=9$. Se o background é $E[X]=(1,2)$, então $E[f(X)]=4$. Em um modelo linear independente, contribuições locais são

$$
\phi_1=2(3-1)=4,\qquad \phi_2=0{,}5(4-2)=1.
$$

$4+4+1=9$. Mudar o background muda a pergunta “em relação a quê?”.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42
)

print(r.importances_mean)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Gere dados com uma feature causal sintética, uma cópia altamente correlacionada e ruído. Compare coeficientes, impurity importance, permutation importance e SHAP. Embaralhe cada correlata isoladamente e em conjunto; explique a divisão de crédito.

## Laboratório guiado completo

Mostre como features correlacionadas dividem importância sem deixar de ser úteis em conjunto.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
x1 = rng.normal(size=1200)
x2 = x1 + rng.normal(0, 0.03, len(x1))
noise = rng.normal(size=(len(x1), 5))
X = np.c_[x1, x2, noise]
y = 3*x1 + rng.normal(0, 0.4, len(x1))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr,ytr)
r = permutation_importance(model, Xte, yte, scoring="r2", n_repeats=20, random_state=42)
print("R2", model.score(Xte,yte))
print("importance individual", r.importances_mean)
Xboth = Xte.copy(); rng.shuffle(Xboth[:,0]); rng.shuffle(Xboth[:,1])
print("queda conjunta", model.score(Xte,yte) - model.score(Xboth,yte))
```

**Entregue:** importância individual/conjunta; estabilidade por seed; explicação local com dois backgrounds; distinção explícita entre associação e causa.

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

- Tratar SHAP como verdade causal.
- Ignorar correlação entre features.
- Explicar modelo avaliado em dados vazados.
- Mostrar gráficos bonitos sem medir fidelidade/estabilidade.

## 8. Exercícios

1. Diferencie explicação global e local.
2. Por que features correlacionadas complicam importance?
3. O que SHAP não prova?
4. Crie uma hipótese de proxy sensível.

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

- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions.
- Molnar — Interpretable Machine Learning.
- Breiman — Random Forest feature importance caveats.
- scikit-learn — Permutation feature importance.

## Leitura orientada e fontes verificadas

- Lundberg e Lee (2017) — [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).
- scikit-learn — [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- scikit-learn — [Common pitfalls in interpreting coefficients](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html).
- NeurIPS — [Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), limitações e responsabilidade.

## Próxima aula

**Clustering: K-Means, hierárquico e DBSCAN**
