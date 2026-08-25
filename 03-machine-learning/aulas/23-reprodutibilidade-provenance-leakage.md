# Aula 23 — Reprodutibilidade, provenance, pipelines e zero data leakage

**Trilha:** Especialista em IA  
**Módulo:** 03 · Machine Learning clássico (M4)  
**Pré-requisito:** Aula 22 deste módulo  
**Objetivo central:** Construir experimentos de ML que possam ser auditados e repetidos.

> Nesta fase, o objetivo deixa de ser apenas conhecer algoritmos. Você precisa saber construir um experimento em que o desempenho medido seja uma estimativa honesta de generalização.

## Objetivos de aprendizagem

- Controlar seeds e versões.
- Registrar dataset, código e parâmetros.
- Distinguir provenance e lineage.
- Criar pipeline end-to-end.
- Aplicar checklist de leakage.

## 1. Por que este tema importa para IA?

Machine Learning clássico continua sendo uma ferramenta essencial em sistemas reais. Dados tabulares, risco, fraude, previsão operacional, ranking, manutenção preditiva e inúmeros problemas corporativos frequentemente são resolvidos com modelos lineares, árvores e ensembles de forma mais simples, rápida e auditável do que com redes neurais.

O foco desta aula é **construir experimentos de ml que possam ser auditados e repetidos.**

## 2. Ideias fundamentais

### 1. Reprodutibilidade

Uma métrica sem código, dados, split e configuração identificáveis é uma evidência fraca.

### 2. Provenance

Registra de onde o dado veio, quando foi obtido, sob quais regras e transformações.

### 3. Lineage

Rastreia o caminho de transformação do dado até features, modelos e artefatos finais.

### 4. Zero leakage

Qualquer informação da validação/teste ou do futuro que influencie treinamento, seleção ou preprocessing invalida a estimativa.

## Aprofundamento — do experimento repetível à evidência auditável

Distinga:

- **repeatability**: mesma equipe, ambiente e artefatos repete o resultado;
- **reproducibility**: outra equipe reconstrói com artefatos/metodologia fornecidos;
- **replication**: nova implementação ou novos dados testam a mesma conclusão.

Seed não basta. Paralelismo, bibliotecas numéricas, hardware, ordem dos dados e algoritmos não determinísticos podem alterar resultados. Registre versões, dispositivo, número de threads e distribuição entre seeds quando relevante.

Provenance responde “de onde veio”; lineage responde “por quais transformações passou”. Use hash de conteúdo para snapshots, commit para código, configuração declarativa para hiperparâmetros e identificador único ligando execução, métricas e modelo. Não versionar dados sensíveis diretamente no Git; versionar metadados, hashes e ponteiros controlados.

Um modelo pode ser reexecutável e ainda metodologicamente inválido. Reprodutibilidade preserva o erro; o protocolo de zero leakage e as threats to validity avaliam sua qualidade.

## 3. Equação para guardar

$$
\text{evidência}=\text{dados}+\text{código}+\text{config}+\text{métrica}+\text{rastreabilidade}
$$

Não memorize a fórmula isoladamente. Pergunte sempre: **o que entra, o que é aprendido, qual hipótese está sendo feita e como isso será avaliado fora da amostra?**

## 4. Exemplo mental

Um experimento deve permitir reconstruir exatamente qual snapshot de dataset e quais hiperparâmetros produziram um modelo e sua métrica.

## Exemplo numérico resolvido

Defina

$$
experiment\_id=SHA256(data\_hash\;||\;commit\;||\;config\_hash).
$$

Se dataset `a91...`, commit `ba038...` e config `c72...` geram ID `e4f...`, qualquer mudança cria outro experimento. Uma tabela de resultados deve registrar esse ID, horário, ambiente, seed, folds e métricas. “modelo_final_v7.pkl” não é lineage.

## 5. Laboratório em Python / scikit-learn

```python
import sklearn, numpy as np

print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)

RANDOM_STATE = 42
# salve também:
# - hash/version do dataset
# - commit do código
# - parâmetros do pipeline
# - folds/split strategy
# - métricas e intervalos
```

O código é apenas o início. No laboratório, registre **split, seed, preprocessing, hiperparâmetros, métrica e versão do dataset**. A meta é que outra pessoa consiga reproduzir o experimento.

### Investigação adicional

Implemente um runner que salve `metadata.json`: hash SHA-256 do dataset, commit, `pip freeze`, seed, splitter, parâmetros via `get_params`, métricas por fold e duração. Reexecute duas vezes e compare hashes/resultados. Faça um terceiro run com seed diferente.

## Laboratório guiado completo

Crie um registro de execução que vincule dados, ambiente, código, parâmetros e métricas.

```python
import hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import sklearn

dataset = Path("dataset.csv")
data_hash = hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.exists() else "DEMO"
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "UNKNOWN"
config = {"seed": 42, "splitter": "StratifiedKFold(5)", "metric": "roc_auc"}
config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
metadata = {
    "experiment_id": hashlib.sha256(f"{data_hash}{commit}{config_hash}".encode()).hexdigest(),
    "data_hash": data_hash, "commit": commit, "config": config,
    "python": platform.python_version(), "numpy": np.__version__,
    "sklearn": sklearn.__version__,
}
print(json.dumps(metadata, indent=2))
```

**Entregue:** `metadata.json`; hashes estáveis; dois runs iguais e um com seed diferente; lista de fontes de não determinismo; threats to validity.

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

- Salvar somente o pickle do modelo.
- Não versionar dataset.
- Alterar preprocessing manualmente fora do pipeline.
- Não registrar seed e estratégia de split.

## 8. Exercícios

1. Crie um checklist mínimo de reprodução.
2. Diferencie provenance de lineage.
3. Liste cinco tipos de leakage.
4. Que artefatos você salvaria após um experimento?

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

- scikit-learn — Common pitfalls and recommended practices.
- Sculley et al. (2015) — Hidden Technical Debt in Machine Learning Systems.
- W3C PROV — provenance data model.
- MLflow documentation — experiment tracking concepts.

## Leitura orientada e fontes verificadas

- Pineau et al. — [Improving Reproducibility in Machine Learning Research](https://arxiv.org/abs/2003.12206).
- NeurIPS — [Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist).
- scikit-learn — [Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness).
- NeurIPS — [programa de reprodutibilidade e checklist](https://blog.neurips.cc/2021/03/26/introducing-the-neurips-2021-paper-checklist/).

## Próxima aula

**Gate II — Experimento completo de Machine Learning clássico**
