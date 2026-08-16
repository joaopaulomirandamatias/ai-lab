# Plano de 18 meses

Extraído de `IA ESPECIALISTA.pdf` (conversa original), que contém este cronograma
além do [roadmap de 49 níveis](roadmap-49-niveis.md).

---

## O projeto longitudinal

Em vez de 30 projetos desconectados, **um único projeto** que evolui:

**AI Systems Laboratory** → [`projetos/ai-systems-lab/`](../projetos/)

```
M1  matemática
 ↓
M2  gradient descent
 ↓
M3  estatística
 ↓
M4  ML
 ↓
M5  Neural Network
 ↓
M6  PyTorch
 ↓
M7  Deep Learning
 ↓
M8  Transformer
 ↓
M9  LLM
 ↓
M10 Fine-tuning
 ↓
M11 RAG
 ↓
M12 Agent
 ↓
M13 MCP
 ↓
M14 Multi-Agent
 ↓
M15 Governed AI
 ↓
M16 Evaluation
 ↓
M17 Paper reproduction
 ↓
M18 Original experiment
```

Ao terminar, o resultado é **evidência pública da evolução** — não uma lista de cursos concluídos.

---

## As duas trilhas paralelas

A pesquisa **não começa no mês 16**. Desde o mês 1, duas trilhas correm juntas:

```
TRILHA DE ENGENHARIA
teoria → código → sistema

TRILHA CIENTÍFICA
paper → hipótese → experimento
```

O aprendizado inteiro fica:

```
CONHECIMENTO → CÓDIGO → EXPERIMENTO → MÉTRICA → EVIDÊNCIA → PESQUISA
```

---

## Fase 1 — Fundamentos científicos (M1–M4)

| Mês | Foco | Pasta | Entregável |
|---|---|---|---|
| M1 | Álgebra linear, cálculo | [`01-math/`](../01-math/) | Notebook com operações vetoriais/matriciais implementadas em NumPy |
| M2 | Otimização, gradient descent | [`01-math/`](../01-math/) | Gradient descent **derivado no papel e implementado do zero** |
| M3 | Probabilidade, estatística, teoria da informação | [`02-statistics/`](../02-statistics/) | Teste de hipótese real com intervalo de confiança e tamanho de efeito |
| M4 | ML clássico | [`03-machine-learning/`](../03-machine-learning/) | Experimento com baseline, validação cruzada e **zero data leakage** |

→ **Gate I** no fim do M2 · **Gate II** no fim do M4

---

## Fase 2 — Deep Learning (M5–M8)

| Mês | Foco | Pasta | Entregável |
|---|---|---|---|
| M5 | Redes neurais do zero | [`04-deep-learning/`](../04-deep-learning/) | MLP com backprop escrito à mão (sem autograd) |
| M6 | PyTorch | [`04-deep-learning/`](../04-deep-learning/) | Mesmo MLP reescrito em PyTorch + treino em GPU |
| M7 | Arquiteturas (CNN, RNN/LSTM, seq2seq) | [`04-deep-learning/`](../04-deep-learning/) | Transfer learning funcionando em dataset próprio |
| M8 | **Transformers** | [`05-transformers/`](../05-transformers/) | Transformer do zero + explicar por que a equação de atenção funciona |

O M8 é o mês mais importante da formação. Não passe adiante sem conseguir
desenhar a arquitetura de memória e explicar `softmax(QKᵀ/√d_k)V` sem consultar nada.

---

## Fase 3 — LLM Engineer (M9–M11)

| Mês | Foco | Pasta | Entregável |
|---|---|---|---|
| M9 | Famílias de LLM, pretraining, alinhamento | [`06-llm/`](../06-llm/) | Comparação arquitetural escrita: GPT vs BERT vs T5 vs Llama vs Mistral |
| M10 | Fine-tuning, LoRA/QLoRA, quantização | [`06-llm/`](../06-llm/) | Modelo fine-tunado publicado no Hugging Face com model card |
| M11 | **RAG** | [`07-rag/`](../07-rag/) | Três versões comparadas cientificamente |

O projeto do M11 é explícito:

```
LLM puro   vs   LLM + RAG   vs   LLM + RAG + Reranker
```

Medindo: Retrieval Precision, Retrieval Recall, Faithfulness, Answer Relevance, Groundedness.

→ **Gate III** no fim do M11

---

## Fase 4 — Agentic AI (M12–M15)

É aqui que a formação deixa de ser genérica e vira **sua** especialização.

### M12 — Agentes
**Primeiro: não use framework.** Construa o agente você mesmo.

```
        LLM
         │
  ┌──────┼──────┐
  ↓      ↓      ↓
Memory  Tools  State
  └──────┼──────┘
         ↓
       Action
```

Estudar: tool calling, function calling, ReAct, planning, reflection, state, memory,
routing, retries, termination, guardrails.

E principalmente: **quando NÃO usar um agente.** Comparar `workflow determinístico vs agent`.

### M13 — MCP e protocolos
Dominar: JSON-RPC, MCP (Host / Client / Server), Tools, Resources, Prompts,
capability discovery, authentication, authorization. Depois: A2A, comunicação
agente-agente, discovery, delegation.

**Projeto obrigatório**, do zero e sem framework de agentes:

```
MCP Host → MCP Client → MCP Server → Tool → Banco/API
```

### M14 — Multi-Agent Systems
MAS clássico **+** LLM agents. Não estudar só CrewAI.

```
        Supervisor
       /    |     \
      ↓     ↓      ↓
 Research Planner Executor
      \     |     /
         Critic
           ↓
        Result
```

Projeto: **Research Multi-Agent System** (planner, researcher, critic, evaluator, executor).
Depois medir `1 agente vs 3 agentes vs 5 agentes` comparando qualidade, custo,
latência e número de erros. Essa comparação já é um experimento científico publicável.

### M15 — Governed Agentic AI
A especialização profunda.

**Segurança:** prompt injection, indirect prompt injection, tool abuse, confused deputy,
privilege escalation, data exfiltration, context poisoning.

**Governança:** `Identity + Purpose + Policy + Confidence + Human Review + Evidence`

**HIL formalizado por confiança:**
```
σ alto          → ALLOW
σ intermediário → HIL
σ baixo         → DENY
```

**Evidence — dossiê por decisão:** quem? qual modelo? qual versão? qual entrada?
qual saída? qual confiança? qual política? qual evidência? houve humano?

**Capstone — Mini Governed AI Runtime:**
```
Usuário → Mission Compiler → Planner → Model Router → Agent
        → MCP Gateway → Policy → Tool → Evidence
```
Incluindo: Model Registry, Policy Engine, HIL, Audit Trail, Evaluation, Observability.

→ **Gate IV** no fim do M15

---

## Fase 5 — Pesquisador em IA (M16–M18)

A mudança de `"eu sei construir isso"` para
`"eu consigo produzir evidência científica de que minha proposta funciona"`.

### M16 — Leitura científica
Rotina de **2 papers por semana**. Classificar cada um em:

```
Problema · Pergunta · Hipótese · Contribuição · Método · Dataset
Baseline · Métricas · Experimento · Resultado · Limitações · Threats to validity
```

Meta inicial: **50–80 papers realmente lidos**, não apenas armazenados.
Template em [`_templates/fichamento-paper.md`](../_templates/fichamento-paper.md),
fichamentos em [`14-papers/`](../14-papers/).

### M17 — Reproduzir pesquisa
```
Paper → Reimplementação → Dataset → Baseline → Resultados → Comparação
```
Se o paper diz `Accuracy = 89.7%` e você obtém `87.9%`, **investigue por quê**.
Isso ensina mais que dez cursos.

### M18 — Pesquisa original
```
Estado da arte → Gap → Pergunta → Hipótese → Proposta → Experimento → Evidência → Paper
```

A linha de pesquisa sugerida, que amarra tudo:

> **Como adicionar determinismo, governança semântica e auditabilidade à execução
> de sistemas multiagentes baseados em LLM?**

Liga: LLM · Agentic AI · MAS · MCP · semântica · governança · segurança · avaliação.

O experimento controlado que dá validade científica ao roadmap inteiro:
**comparar workflow, agente e multiagente** sob qualidade, custo, latência e segurança.

→ **Gate V** no fim do M18

---

## Formação em T esperada ao final

```
─────────────────────────────────────────────
ML · DL · CV · NLP · RL · MLOps · Segurança
─────────────────────────────────────────────
                  │ LLMs
                  │ RAG
                  │ Agents
                  │ Multi-Agent Systems
                  │ MCP / A2A
                  │ Semantic Interoperability
                  │ AI Governance
                  │ Governed AI Systems
                  ▼
```

Amplitude em IA, profundidade muito maior em **LLM Systems + Agentic AI + Semantic/Governed AI**.
