# Aula 12 — Support Vector Machines: margem máxima e kernels

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 11 deste módulo  
**Objetivo central:** Entender classificação por margem e o papel dos support vectors e kernels.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Explicar hiperplano e margem.
- Identificar support vectors.
- Entender parâmetro C.
- Compreender kernel trick conceitualmente.
- Usar RBF com scaling e tuning correto.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender classificação por margem e o papel dos support vectors e kernels.**

## 2. Ideias fundamentais

### 1. Margem

O classificador busca um hiperplano com grande margem entre classes. Apenas pontos próximos à fronteira influenciam diretamente a solução.

### 2. Soft margin

C controla o custo de violações. C alto penaliza erros com força; C baixo permite margem mais suave.

### 3. Kernel

Kernels permitem calcular produtos internos em espaços transformados sem construir explicitamente todas as novas features.

### 4. RBF

O kernel RBF cria fronteiras flexíveis; gamma controla a escala de influência dos pontos.

## Aprofundamento — margem, slack e kernel trick

No caso linear soft-margin, a forma primal é

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i
$$

sujeita a $y_i(w^Tx_i+b)\ge 1-\xi_i$ e $\xi_i\ge0$. Minimizar $\|w\|$ aumenta a margem geométrica; $C$ controla o custo de violações. $C$ alto tenta corrigir mais pontos e pode aumentar variância; $C$ baixo aceita violações para obter margem mais larga.

Na formulação dual, exemplos entram por produtos internos. Um kernel $K(x,z)=\langle\phi(x),\phi(z)\rangle$ permite trabalhar implicitamente em outro espaço. No RBF, `gamma` alto cria influência muito local e fronteira flexível; baixo suaviza. Scaling altera todas as distâncias e é obrigatório na maioria dos usos.

`probability=True` costuma ajustar calibração adicional e aumenta custo; não trate a saída de margem como probabilidade sem método apropriado.

## 3. Equação para guardar

$$
K(x,z)=\exp(-\gamma\|x-z\|^2)
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Duas classes em círculos concêntricos não são linearmente separáveis no espaço original, mas um kernel pode induzir uma separação adequada.

## Exemplo numérico resolvido

Em uma dimensão, negativos em $x=-2,-1$ e positivos em $x=1,2$. O hiperplano $w=1,b=0$ separa em zero. Os pontos $-1$ e $1$ satisfazem $y(wx+b)=1$ e são vetores de suporte. A distância de cada plano de suporte ao centro é $1/\|w\|=1$; a largura total da margem é 2.

O ponto $x=0{,}5$ recebe score $0{,}5$ e classe positiva, mas score não é probabilidade calibrada.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
)
svm.fit(X_train, y_train)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Em `make_moons`, compare kernel linear e RBF. Faça tuning de `C` e `gamma` em escala logarítmica dentro de pipeline com scaler. Plote número de vetores de suporte, fronteira, CV e custo de inferência. Depois faça nested CV.

## Laboratório guiado completo

Faça nested CV para que seleção de $C$ e $\gamma$ não compartilhe dados com a estimativa final.

```python
from scipy.stats import loguniform
from sklearn.datasets import make_moons
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=700, noise=0.25, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
search = RandomizedSearchCV(
    pipe, {"svc__C": loguniform(1e-3, 1e3), "svc__gamma": loguniform(1e-4, 1e2)},
    n_iter=30, cv=4, scoring="roc_auc", random_state=42, n_jobs=-1
)
outer = StratifiedKFold(5, shuffle=True, random_state=123)
result = cross_validate(search, X, y, cv=outer, scoring="roc_auc", return_estimator=True)
print(result["test_score"], result["test_score"].mean())
print([est.best_params_ for est in result["estimator"]])
```

**Entregue:** mapa $C×\gamma$; score interno versus externo; número de vetores de suporte; custo de treino/inferência; comparação kernel linear/RBF.

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

- Usar SVM RBF sem scaling.
- Interpretar probability=True como probabilidade perfeita.
- Tunar C/gamma no teste.
- Usar kernel complexo quando modelo linear já resolve.

## 8. Exercícios

1. O que são support vectors?
2. Qual o efeito de C muito alto?
3. Explique kernel trick sem fórmula.
4. Por que scaling é importante?

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

- Cortes & Vapnik (1995) — Support-Vector Networks.
- Hastie et al. — Support Vector Machines and Flexible Discriminants.
- ISLP — Support Vector Machines.
- scikit-learn — SVM.

## Leitura orientada e fontes verificadas

- Cortes e Vapnik (1995) — [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018).
- Google Research — [registro do artigo Support-Vector Networks](https://research.google/pubs/support-vector-networks/).
- James et al. — [ISLP](https://www.statlearning.com/), cap. 9.
- scikit-learn — [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html).

## Próxima aula

**Métricas de regressão: MAE, MSE, RMSE, R² e erro relativo**
