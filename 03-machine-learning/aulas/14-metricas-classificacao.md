# Aula 14 — Métricas de classificação: matriz de confusão, precision, recall e F1

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 13 deste módulo  
**Objetivo central:** Entender métricas condicionais e escolher a que representa o custo dos erros.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Interpretar TP, FP, TN e FN.
- Calcular precision, recall, specificity e F1.
- Entender por que accuracy falha em classes raras.
- Distinguir macro, micro e weighted averages.
- Relacionar métrica ao custo operacional.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **entender métricas condicionais e escolher a que representa o custo dos erros.**

## 2. Ideias fundamentais

### 1. Matriz de confusão

Toda métrica de classificação binária pode ser entendida a partir de TP, FP, TN e FN.

### 2. Precision

Entre previsões positivas, quantas eram realmente positivas? Importa quando falso positivo é caro.

### 3. Recall

Entre positivos reais, quantos foram detectados? Importa quando falso negativo é caro.

### 4. F1

Média harmônica de precision e recall. É útil quando queremos equilíbrio, mas não incorpora TN.

## Aprofundamento — da matriz de confusão ao custo operacional

Para classe positiva definida explicitamente:

$$
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad Specificity=\frac{TN}{TN+FP}.
$$

Accuracy mistura as quatro células e pode ser dominada pela classe frequente. F1 é a média harmônica de precision e recall, mas ignora verdadeiros negativos e presume que o equilíbrio entre as duas métricas é desejado. $F_\beta$ permite dar mais peso a recall ($\beta>1$) ou precision ($\beta<1$).

Em multiclasse, `macro` calcula a média por classe dando o mesmo peso a cada uma; `weighted` pondera pela frequência; `micro` agrega contagens. Reportar apenas “F1” sem dizer a variante é incompleto.

Uma decisão real deve incluir custos. Se um FN custa R$ 5.000 e um FP custa R$ 50, a função de custo pode ser $5000FN+50FP$, avaliada em thresholds definidos na validação.

## 3. Equação para guardar

$$
F1=2\frac{precision\cdot recall}{precision+recall}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Em detecção de fraude, recall alto captura mais fraudes, mas pode aumentar bloqueios indevidos. A decisão depende do custo e da capacidade de revisão.

## Exemplo numérico resolvido

Em 1.000 casos: $TP=40$, $FP=10$, $FN=20$, $TN=930$.

$$
Precision=40/50=0{,}80,\qquad Recall=40/60\approx0{,}667,
$$

$$
F1\approx0{,}727,\qquad Accuracy=970/1000=0{,}97.
$$

Os 97% de accuracy escondem que um terço dos positivos foi perdido. Se FN custa muito, o threshold precisa ser revisto mesmo com alta accuracy.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Para thresholds de 0,05 a 0,95, gere tabela com TP, FP, FN, TN, precision, recall, specificity, F1 e custo. Compare macro/micro/weighted em um problema multiclasse e explique qual agregação combina com a pergunta.

## Laboratório guiado completo

Transforme scores em decisões em vários thresholds e calcule custo explicitamente.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

y = np.array([1,0,1,0,0,1,0,1,0,0,1,0])
score = np.array([.95,.85,.78,.70,.62,.58,.50,.42,.30,.20,.15,.05])
for threshold in [0.2, 0.4, 0.6, 0.8]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    cost = 50*fp + 5000*fn
    print(threshold, dict(tp=tp, fp=fp, fn=fn, tn=tn,
                          precision=precision, recall=recall, f1=f1, cost=cost))
```

**Entregue:** tabela completa; threshold de menor custo escolhido sem teste; comparação micro/macro/weighted em um dataset multiclasse.

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

- Usar accuracy em dataset 99/1.
- Escolher F1 sem justificar custo.
- Confundir precision com recall.
- Ignorar averaging em multiclasse.

## 8. Exercícios

1. Calcule precision e recall para TP=80, FP=20, FN=40.
2. Qual métrica priorizaria em triagem de doença?
3. Quando accuracy pode ser adequada?
4. Explique macro vs micro F1.

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

- Powers (2011) — Evaluation: From Precision, Recall and F-Measure to ROC.
- scikit-learn — Classification metrics.
- Saito & Rehmsmeier (2015) — Precision-Recall plot.
- ISLP — Classification evaluation.

## Leitura orientada e fontes verificadas

- scikit-learn — [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
- scikit-learn — [Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics).
- James et al. — [ISLP](https://www.statlearning.com/), classificação e avaliação.
- Murphy — [PML: An Introduction](https://probml.github.io/pml-book/book1.html), teoria da decisão.

## Próxima aula

**ROC, Precision-Recall, thresholds e calibração**
