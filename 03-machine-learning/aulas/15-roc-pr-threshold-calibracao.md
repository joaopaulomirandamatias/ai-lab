# Aula 15 — ROC, Precision-Recall, thresholds e calibração

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 14 deste módulo  
**Objetivo central:** Separar ranking, threshold e qualidade probabilística.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Construir curvas ROC e PR.
- Entender AUC como métrica de ranking.
- Escolher threshold por custo.
- Avaliar calibração.
- Entender Brier score e calibration curves.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **separar ranking, threshold e qualidade probabilística.**

## 2. Ideias fundamentais

### 1. ROC

Varia threshold e plota TPR contra FPR. ROC-AUC mede capacidade de ranking sob uma interpretação probabilística por pares.

### 2. Precision-Recall

Mais informativa quando classe positiva é rara e nos interessa a qualidade das detecções positivas.

### 3. Threshold

Threshold é uma política de decisão, não parte intrínseca do ranker. Pode ser adaptado ao contexto.

### 4. Calibração

Se previsões 0.8 ocorrem, aproximadamente 80% desses casos deveriam ser positivos em um modelo bem calibrado.

## Aprofundamento — ranking, política e probabilidade

ROC-AUC pode ser interpretada como a probabilidade de um positivo aleatório receber score maior que um negativo aleatório. Ela é invariante a transformações monotônicas dos scores e, portanto, não mede calibração. Em forte desbalanceamento, muitos TN podem tornar FPR pequeno mesmo com número operacionalmente grande de falsos alarmes; a curva Precision-Recall evidencia a qualidade das detecções positivas. A referência de average precision depende da prevalência.

Threshold é política: selecione-o na validação a partir de custo, capacidade de revisão ou restrições, nunca no teste. Calibração pergunta se eventos previstos com probabilidade $p$ ocorrem aproximadamente com frequência $p$. Brier score e log-loss são proper scoring rules; reliability diagrams complementam o número.

Platt scaling e isotonic regression precisam de dados separados ou cross-validation. Calibrar e avaliar no mesmo conjunto também gera otimismo.

## 3. Equação para guardar

$$
Brier=\frac{1}{n}\sum_i(p_i-y_i)^2
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Dois modelos podem ter ROC-AUC semelhante, mas um produzir probabilidades muito mal calibradas, prejudicando decisões baseadas em risco.

## Exemplo numérico resolvido

Probabilidades $[0{,}9,0{,}8,0{,}7,0{,}1]$ e labels $[1,0,1,0]$.

Com threshold 0,75: $TP=1,FP=1,FN=1,TN=1$. Com 0,65: $TP=2,FP=1,FN=0,TN=1$. O ranking não mudou; mudou a política.

O Brier score é

$$
\frac{(0{,}9-1)^2+(0{,}8-0)^2+(0{,}7-1)^2+(0{,}1-0)^2}{4}=0{,}1875.
$$

O falso positivo confiante de 0,8 domina a penalidade.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

roc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)

print(roc, ap)
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Compare dois modelos em ROC-AUC, average precision, Brier, log-loss e curva de calibração. Aplique `CalibratedClassifierCV` e escolha threshold por custo na validação. Confirme que o teste é usado uma vez.

## Laboratório guiado completo

Separe ranking, qualidade probabilística e decisão no mesmo conjunto de scores.

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)

rng = np.random.default_rng(42)
y = rng.binomial(1, 0.12, 2000)
raw = np.clip(0.05 + 0.65*y + rng.normal(0, 0.18, len(y)), 0.001, 0.999)
distorted = np.clip(raw**0.35, 0.001, 0.999)  # ranking semelhante, calibração diferente
for name, p in {"raw": raw, "distorted": distorted}.items():
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    precision, recall, thresholds = precision_recall_curve(y, p)
    print(name, {"roc": roc_auc_score(y,p), "ap": average_precision_score(y,p),
                 "brier": brier_score_loss(y,p), "logloss": log_loss(y,p)})
    print("calibração", list(zip(mean, frac)))
```

**Entregue:** ROC e PR; reliability diagram; threshold por custo/capacidade; explicação de por que AUC pode ficar parecida enquanto Brier piora.

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

- Chamar AUC de accuracy.
- Assumir que bom ranking implica boa calibração.
- Escolher threshold no conjunto de teste.
- Preferir ROC em classe extremamente rara sem olhar PR.

## 8. Exercícios

1. Explique ranking vs decisão.
2. Quando PR-AUC é mais informativa?
3. O que significa uma probabilidade calibrada?
4. Como escolher threshold por custo?

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

- Fawcett (2006) — An Introduction to ROC Analysis.
- Saito & Rehmsmeier (2015) — PR vs ROC for imbalanced data.
- Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities.
- scikit-learn — Probability calibration.

## Leitura orientada e fontes verificadas

- Fawcett (2006) — [An Introduction to ROC Analysis](https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X).
- Saito e Rehmsmeier (2015) — [Precision-Recall vs ROC em dados desbalanceados](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).
- Niculescu-Mizil e Caruana (2005) — [Predicting Good Probabilities](https://dl.acm.org/doi/10.1145/1102351.1102430).
- scikit-learn — [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) e [threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Próxima aula

**Classes desbalanceadas: amostragem, pesos e avaliação correta**
