# AI Lab — Do básico a especialista em IA

Laboratório de estudo e experimentação de João Paulo.
Objetivo final: **AI Systems + Agentic AI + Semantic Interoperability**, com foco em
sistemas multiagentes governados, auditáveis e comprovados experimentalmente.

> A meta não é "sei usar GPT, Claude, Hugging Face e LangChain".
> A meta é: *entendo matematicamente o modelo, treino, avalio, projeto a arquitetura,
> integro por protocolos, controlo permissões, meço custo e qualidade, registro decisões
> e produzo evidência científica sobre a arquitetura.*

---

## Como este repositório funciona

Três coisas convivem aqui, e é importante não misturá-las:

| Camada | Onde vive | O que é |
|---|---|---|
| **Plano** | [`00-plano/`](00-plano/) | O roadmap, o cronograma de 18 meses, os gates |
| **Conhecimento** | `01-` … `16-` | Notas, derivações, resumos, código de estudo por tema |
| **Evidência** | [`projetos/`](projetos/), [`15-reproductions/`](15-reproductions/), [`16-research/`](16-research/) | Coisas que rodam e produzem números |

Regra que sustenta o resto: **nada é dado como aprendido sem artefato**.
Um tema só sai da fila quando existe código rodando, métrica medida ou nota que
explica o conceito sem consultar a fonte.

---

## Mapa das pastas

### Plano e apoio

| Pasta | Conteúdo |
|---|---|
| [`00-plano/`](00-plano/) | Roadmap de 49 níveis, plano de 18 meses, os 5 gates, rotina semanal |
| [`recursos/`](recursos/) | Livros, sites, cursos, canais de YouTube, papers essenciais, projetos |
| [`projetos/`](projetos/) | O projeto longitudinal (**AI Systems Laboratory**) que evolui M1→M18 |
| [`diario/`](diario/) | Diário científico — 30 min/domingo, o registro do que foi aprendido |
| [`_templates/`](_templates/) | Template de experimento e de fichamento de paper |

### Trilha de estudo (a ordem importa)

| Pasta | Mês | Tema | Gate |
|---|---|---|---|
| [`01-math/`](01-math/) | M1–M2 | Álgebra linear, cálculo, gradient descent | **I** |
| [`02-statistics/`](02-statistics/) | M3 | Probabilidade, inferência, teoria da informação | |
| [`03-machine-learning/`](03-machine-learning/) | M4 | ML clássico, métricas, validação, data leakage | **II** |
| [`04-deep-learning/`](04-deep-learning/) | M5–M7 | Redes neurais, PyTorch, CNN, RNN | |
| [`05-transformers/`](05-transformers/) | M8 | Attention, self-attention, arquitetura completa | |
| [`06-llm/`](06-llm/) | M9–M10 | Famílias, pretraining, fine-tuning, LoRA, quantização | |
| [`07-rag/`](07-rag/) | M11 | Embeddings, vector DB, retrieval, reranking, GraphRAG | **III** |
| [`08-agents/`](08-agents/) | M12 | Tool calling, ReAct, planning, memória, guardrails | |
| [`09-mcp-protocolos/`](09-mcp-protocolos/) | M13 | JSON-RPC, MCP, A2A, capability discovery | |
| [`10-multiagent/`](10-multiagent/) | M14 | Supervisor, planner/executor, negociação, consenso | |
| [`11-interop-semantica/`](11-interop-semantica/) | ∞ | Ontologias, RDF/OWL/SHACL, knowledge graphs | |
| [`12-ai-governance/`](12-ai-governance/) | M15 | Policy, HIL, evidence, segurança de IA, AI Act | **IV** |
| [`13-evaluation/`](13-evaluation/) | ∞ | Benchmarks, LLM-as-judge, Ragas, observabilidade | |
| [`14-papers/`](14-papers/) | M16 | Fichamentos — meta: 50–80 papers *lidos* |  |
| [`15-reproductions/`](15-reproductions/) | M17 | Reprodução de papers |  |
| [`16-research/`](16-research/) | M18 | Pesquisa original, experimento controlado | **V** |

`11-interop-semantica` e `13-evaluation` estão marcadas com ∞ porque não são um mês:
são as duas trilhas que atravessam tudo depois do M11. Interoperabilidade semântica é a
sua especialização declarada; avaliação é o que separa "funciona na minha máquina" de
evidência.

---

## Rotina semanal (~12h)

| Dia | Atividade | Tempo |
|---|---|---|
| Segunda | Matemática / teoria | 2h |
| Terça | Implementação | 2h |
| Quarta | Paper científico | 1h30 |
| Quinta | Laboratório | 2h |
| Sexta | Revisão | 1h |
| Sábado | Projeto principal | 3h |
| Domingo | Revisão + diário científico | 30min |

A proporção muda ao longo do percurso: começa em **40% teoria / 40% implementação /
20% pesquisa** e termina em **20% / 30% / 50%**.

---

## Os 5 gates

Progresso não se mede por curso concluído nem por certificado. Mede-se por gate.

- **Gate I — Scientific Foundations** · você deriva *e* implementa gradient descent.
- **Gate II — ML Engineer** · você monta um experimento de ML sem data leakage, com baseline e métricas apropriadas.
- **Gate III — LLM Engineer** · você decide entre prompt, RAG e fine-tuning — e *prova* qual funciona melhor.
- **Gate IV — Agentic AI Engineer** · você constrói LLM + Tools + State + MCP + Policy + HIL + Evaluation + Audit sem depender intelectualmente de CrewAI/LangChain.
- **Gate V — AI Researcher** · você identifica gap → formula hipótese → implementa baseline → executa experimento → analisa estatisticamente → defende conclusão → publica.

Detalhe de cada um em [`00-plano/gates.md`](00-plano/gates.md).

---

## Por onde começar agora

1. Ler [`00-plano/plano-18-meses.md`](00-plano/plano-18-meses.md) inteiro — uma vez só, para ter o mapa.
2. Abrir [`01-math/README.md`](01-math/README.md) e escolher **uma** fonte principal (não três).
3. Criar `projetos/ai-systems-lab/` no primeiro dia de código — ele cresce com você por 18 meses.
4. Abrir o diário: [`diario/README.md`](diario/README.md).

E a regra que mais economiza tempo: **uma fonte principal por tema, o resto é consulta.**
