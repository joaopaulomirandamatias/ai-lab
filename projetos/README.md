# Projetos

O projeto longitudinal vive em **repositório próprio**:

> **`ai-systems-lab`** — local: `~/dev/ai-systems-lab`

Este repositório (`ai-lab`) guarda o **estudo**: plano, notas, fichamentos, material.
O `ai-systems-lab` guarda o **sistema**: o que roda. A divisão é deliberada — misturar
histórico de anotação com histórico de código torna os dois ilegíveis.

---

## O AI Systems Laboratory

Um projeto, não trinta desconectados. Ele evolui M1 → M18, ganhando uma camada por mês:

```
matemática → gradient descent → estatística → ML → Neural Network → PyTorch
→ Deep Learning → Transformer → LLM → Fine-tuning → RAG → Agent → MCP
→ Multi-Agent → Governed AI → Evaluation → Paper reproduction → Original experiment
```

Ao terminar, o repositório é **evidência pública da evolução** — não uma lista de cursos
concluídos.

### Camadas

| Camada | Nasce em | Papel |
|---|---|---|
| `core/` | M1–M8 | Matemática, treino, modelo |
| `retrieval/` | M11 | Embeddings, busca híbrida, reranking, grafo |
| `agents/` | M12 | Loop, tools, state, memória, guardrails |
| `protocols/` | M13 | MCP Host/Client/Server, JSON-RPC, A2A |
| `governance/` | M15 | Policy engine, HIL, evidence, audit trail |
| `evaluation/` | M11+ | Conjuntos de avaliação, métricas, harness |
| `experiments/` | contínuo | Um diretório por experimento |

---

## Os 18 projetos

Lista completa com **critério de pronto** de cada um:
[`recursos/projetos.md`](../recursos/projetos.md).

Os cinco que mais importam:

| # | Projeto | Mês | Por quê |
|---|---|---|---|
| **P8** | Transformer do zero | M8 | Tudo depois é construído sobre isso |
| **P11** | RAG em três versões | M11 | 🔶 Gate III — prova em vez de opinião |
| **P13** | MCP Host+Client+Server do zero | M13 | O eixo da sua especialização |
| **P14** | Research Multi-Agent System | M14 | A medição já é publicável |
| **P15** | Mini Governed AI Runtime | M15 | 🔶 Gate IV — o capstone |

---

## O que fica aqui

Esta pasta guarda o que é **sobre** os projetos, não os projetos:

- notas de decisão que ainda não viraram código;
- rascunhos de arquitetura;
- ligações entre um projeto e o que você leu em [`14-papers/`](../14-papers/).

O `CHANGELOG.md` do `ai-systems-lab` é o registro oficial de o que foi construído e
por quê — uma entrada por mês. É o que se lê numa entrevista ou numa banca.
