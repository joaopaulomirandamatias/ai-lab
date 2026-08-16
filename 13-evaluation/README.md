# 13 · Avaliação, observabilidade e operação

**Transversal, do M11 em diante** · Roadmap: Níveis 24, 25, 26, 27, 28, 29 · Pasta nova

Avaliação é o que separa "funciona na minha máquina" de evidência. É a habilidade que
todos os cinco gates cobram, de formas diferentes.

---

## Evaluation de LLMs (Nível 24)

**Benchmarks** — MMLU, HellaSwag, GSM8K, HumanEval, BIG-bench, HELM, Chatbot Arena.
E a limitação de todos eles: contaminação de dados de treino.

**Avaliação de sistema**, que importa mais que benchmark de modelo:
- **LLM-as-a-judge** — e seus vieses conhecidos (posição, verbosidade, auto-preferência)
- Métricas de RAG — faithfulness, groundedness, answer relevance, retrieval precision/recall
- Métricas de agente — taxa de conclusão, número de passos, uso correto de ferramenta, custo por tarefa
- Avaliação humana, e quando ela é insubstituível
- Conjuntos de regressão — o que impede que a melhoria de hoje quebre o de ontem

**Ferramentas** — Ragas · DeepEval · promptfoo · `lm-evaluation-harness` · HELM.

---

## Observabilidade (Nível 25)

Tracing, spans, custo por chamada, latência (p50/p95/p99), taxa de erro, drift,
logging de prompt e resposta, replay.

**Ferramentas** — Langfuse · Arize Phoenix · OpenLLMetry · W&B Weave · LangSmith.

---

## MLOps e LLMOps (Níveis 26–27)

Versionamento de modelo, dado e prompt · registry · CI/CD para modelos · deploy ·
rollback · monitoramento · retraining · gestão de custo.
Ferramentas: MLflow · DVC · Weights & Biases · BentoML.

---

## Arquitetura e roteamento (Níveis 28–29)

Model routing, AI Gateway, fallback, cache semântico, rate limiting, multi-tenant,
escolha modelo local vs API. É a camada **Model Router** do capstone do M15.

---

## Fonte principal

⭐ **AI Engineering** — Chip Huyen. Os capítulos de avaliação são o melhor material
prático publicado sobre o assunto.

Papers: **Judging LLM-as-a-Judge** (Zheng et al., MT-Bench/Chatbot Arena) · HELM ·
Ragas — ver [`recursos/papers-essenciais.md`](../recursos/papers-essenciais.md).
Blog: Eugene Yan · Chip Huyen.

---

## Por que esta pasta é transversal

Cada gate é, no fundo, um problema de medição:

| Gate | O que precisa ser medido |
|---|---|
| **II** | baseline, validação cruzada, métrica adequada ao problema |
| **III** | RAG vs prompt vs fine-tuning — qualidade, custo, latência |
| **IV** | avaliação de sistema agentic, cobertura de audit trail |
| **V** | análise estatística, tamanho de efeito, *threats to validity* |

Uma regra que resolve a maior parte dos erros: **crie o conjunto de avaliação antes de
construir o sistema.** Conjunto criado depois é construído, inconscientemente, para o
sistema passar. É a versão em avaliação do mesmo problema que
[`_templates/experimento/`](../_templates/experimento/) evita ao exigir a hipótese antes
do resultado.
