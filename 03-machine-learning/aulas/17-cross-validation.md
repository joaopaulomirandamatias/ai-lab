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

## Aprofundamento — CV estima um procedimento inteiro

Em K-fold, para cada $k$ ajustamos **todo o pipeline** em $D\setminus D_k$ e avaliamos em $D_k$. A média estima o desempenho do procedimento treinado com aproximadamente $(K-1)/K$ dos dados. Os scores dos folds não são observações totalmente independentes porque os conjuntos de treino se sobrepõem; desvio-padrão entre folds é diagnóstico de estabilidade, não um intervalo de confiança clássico automático.

O splitter codifica a hipótese de generalização:

- `KFold`/`StratifiedKFold`: novas amostras da mesma população aproximadamente iid;
- `GroupKFold`: novos grupos, sem identidade compartilhada;
- `TimeSeriesSplit`: futuro previsto apenas com passado;
- splits espaciais ou por site: novos locais.

Escolher o splitter errado não é um detalhe estatístico: muda a pergunta respondida. CV também não protege o teste se o pesquisador usa os resultados para centenas de decisões e reporta apenas a melhor.

## 3. Equação para guardar

$$
\bar m=\frac{1}{K}\sum_{k=1}^{K}m_k
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em dados médicos, todas as consultas do mesmo paciente devem pertencer ao mesmo fold para evitar que identidade do paciente vaze entre treino e validação.

## Exemplo numérico resolvido

Scores de cinco folds: $[0{,}71,0{,}76,0{,}74,0{,}62,0{,}77]$. Média $0{,}72$ e desvio-padrão amostral aproximado $0{,}060$. Reportar apenas 0,72 esconde o fold 0,62. Investigue se ele contém outro período, hospital ou perfil. A variação pode revelar fragilidade estrutural, não apenas “azar”.

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

### Investigação adicional

No mesmo problema com entidades repetidas e timestamp, compare KFold, StratifiedKFold, GroupKFold e TimeSeriesSplit. Visualize índices de treino/validação. Para cada um, escreva em uma frase qual cenário de produção ele simula.

## Laboratório guiado completo

Simule registros repetidos por entidade e compare split ingênuo com `GroupKFold`.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
groups = np.repeat(np.arange(120), 5)
entity_signal = rng.normal(size=(120, 1))
X = np.repeat(entity_signal, 5, axis=0) + rng.normal(0, .2, (600,1))
X = np.c_[X, rng.normal(size=(600,5))]
y = np.repeat((entity_signal[:,0] > 0).astype(int), 5)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
splitters = {
    "kfold": KFold(5, shuffle=True, random_state=42),
    "stratified": StratifiedKFold(5, shuffle=True, random_state=42),
    "group": GroupKFold(5),
}
for name, cv in splitters.items():
    kwargs = {"groups": groups} if name == "group" else {}
    s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", **kwargs)
    print(name, s.mean(), s.std(), s)
```

**Entregue:** visualização dos índices; comparação com corte temporal; análise do fold pior; pergunta de generalização respondida por cada splitter.

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

- Fazer preprocessing antes do CV.
- Ignorar grupos.
- Usar KFold aleatório em série temporal.
- Usar o teste como mais um fold.

## 8. Exercícios

1. Quando usar GroupKFold?
2. Por que o teste final continua necessário?
3. Qual problema de usar shuffle em previsão temporal?
4. Por que reportar desvio dos folds?

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

- Kohavi (1995) — A Study of Cross-Validation and Bootstrap.
- ISLP, cap. 5 — Resampling Methods.
- scikit-learn — Cross-validation.
- Varoquaux et al. — Assessing and tuning brain decoders: cross-validation.

## Leitura orientada e fontes verificadas

- scikit-learn — [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
- James et al. — [ISLP](https://www.statlearning.com/), cap. 5.
- Hastie, Tibshirani e Friedman — [ESL](https://hastie.su.domains/ElemStatLearn/), cap. 7.
- Cawley e Talbot (2010) — [Over-fitting in model selection](https://jmlr.org/papers/v11/cawley10a.html).

## Próxima aula

**Hyperparameter tuning: Grid Search, Random Search e validação aninhada**
