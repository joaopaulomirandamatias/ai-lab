# Aula 16 — Classes desbalanceadas: amostragem, pesos e avaliação correta

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 15 deste módulo  
**Objetivo central:** Tratar desbalanceamento sem transformar a avaliação em uma ilusão.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Entender prevalência e rare events.
- Usar class_weight.
- Conhecer undersampling e oversampling.
- Aplicar resampling somente no treino.
- Escolher métricas robustas ao desbalanceamento.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **tratar desbalanceamento sem transformar a avaliação em uma ilusão.**

## 2. Ideias fundamentais

### 1. Prevalência

Desbalanceamento não é automaticamente um problema; depende da tarefa, custo e informação das features.

### 2. Class weights

Podemos aumentar a penalização de erros na classe minoritária sem duplicar exemplos.

### 3. Resampling

Oversampling e undersampling devem ocorrer dentro de cada fold de treinamento, nunca antes do split/cross-validation.

### 4. Métrica

Balanced accuracy, PR-AUC, recall e precision são frequentemente mais informativas que accuracy.

## Aprofundamento — desbalanceamento, custo e mudança de prevalência

Desbalanceamento é uma propriedade da distribuição, não um defeito automaticamente corrigido. Primeiro pergunte: o modelo separa as classes? qual custo dos erros? a prevalência de produção será igual à do treino? são necessárias probabilidades calibradas ou apenas ranking?

`class_weight` altera a contribuição de exemplos na loss e frequentemente desloca a fronteira. Oversampling repete ou sintetiza casos; undersampling descarta informação da maioria. Todos devem ocorrer **dentro do fold de treino**. Fazer SMOTE antes do split cria pontos sintéticos relacionados a amostras que depois podem cair na validação.

SMOTE interpola $x_{new}=x_i+\lambda(x_{nn}-x_i)$ com $\lambda\in[0,1]$. Isso pressupõe que segmentos entre vizinhos minoritários permanecem plausíveis; pode falhar com categorias, overlap ou geometria complexa. Pesos/reamostragem também podem distorcer probabilidades e exigir correção/calibração.

## 3. Equação para guardar

$$
BalancedAccuracy=\frac{TPR+TNR}{2}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Fraude 0.5%: um modelo que sempre prevê 'não fraude' tem 99.5% accuracy e nenhum valor operacional.

## Exemplo numérico resolvido

Cinco fraudes em 1.000 transações. Prever tudo como normal gera 99,5% de accuracy e balanced accuracy 50%. Um segundo sistema sinaliza 20 transações, contendo 4 das 5 fraudes:

$$
Recall=4/5=0{,}80,\qquad Precision=4/20=0{,}20.
$$

Ele produz 16 falsos alarmes. Seu valor depende do custo das quatro fraudes capturadas e da capacidade de revisar 20 alertas — não de maximizar accuracy.

## 5. Laboratório em Python / scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(balanced_accuracy_score(y_test, pred))
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Compare baseline, `class_weight`, undersampling e SMOTE dentro de `imblearn.Pipeline`. Meça PR-AUC, recall em uma capacidade fixa de alertas, calibração e custo. Repita com prevalência de teste diferente e observe a queda de precision.

## Laboratório guiado completo

Compare baseline e loss ponderada sem alterar o conjunto de teste.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=6000, n_features=20, n_informative=7,
                           weights=[0.99, 0.01], flip_y=0.002, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
for weight in [None, "balanced"]:
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight=weight, max_iter=3000
    )).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:,1]
    pred = (p >= 0.5).astype(int)
    print(weight, "AP", average_precision_score(yte,p),
          "balanced", balanced_accuracy_score(yte,pred), "alertas", pred.sum())
```

**Entregue:** PR-AUC, recall@capacidade, precision, calibração e custo; SMOTE apenas dentro de `imblearn.Pipeline`; comparação sob nova prevalência.

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

- Oversampling antes do split.
- Usar SMOTE no teste.
- Otimizar accuracy.
- Ignorar prevalência futura na calibração.

## 8. Exercícios

1. Por que 99% accuracy pode ser inútil?
2. Quando class_weight pode ser preferível a oversampling?
3. Por que resampling deve estar dentro do CV?
4. Escolha métricas para fraude.

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

- He & Garcia (2009) — Learning from Imbalanced Data.
- Chawla et al. (2002) — SMOTE.
- scikit-learn — Imbalanced datasets / class_weight.
- Saito & Rehmsmeier — PR curves.

## Leitura orientada e fontes verificadas

- Chawla et al. (2002) — [SMOTE](https://www.jair.org/index.php/jair/article/view/10302).
- Saito e Rehmsmeier (2015) — [Precision-Recall em dados desbalanceados](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).
- scikit-learn — [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).
- imbalanced-learn — [User Guide](https://imbalanced-learn.org/stable/user_guide.html), pipelines e over-sampling.

## Próxima aula

**Cross-validation: estimando generalização sem desperdiçar dados**
