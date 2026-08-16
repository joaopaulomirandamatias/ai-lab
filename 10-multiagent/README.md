# 10 · Sistemas Multiagentes

**Mês M14** · Roadmap: Nível 19

> Agora estudar **MAS clássico + LLM agents**. Não estudar somente CrewAI.

---

## O que dominar

Estudar cientificamente: cooperative agents, competitive agents, communication,
negotiation, coordination, distributed decision making, blackboard, supervisor,
planner/executor, hierarchical agents, swarm, consensus.

**Arquitetura de referência:**

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

---

## Fonte principal

⭐ **An Introduction to MultiAgent Systems** — Michael Wooldridge.
Anterior aos LLMs, e por isso mesmo essencial: coordenação, negociação e protocolos de
interação já tinham teoria antes de virarem prompt.

Papers LLM: AutoGen · CAMEL · MetaGPT · ChatDev · Multiagent Debate (Du et al.) —
ver [`recursos/papers-essenciais.md`](../recursos/papers-essenciais.md).

Formal, se a pesquisa exigir: Shoham & Leyton-Brown, *Multiagent Systems* (grátis,
`masfoundations.org`) — consenso, teoria dos jogos, mecanismo.

Acompanhar: **AAMAS**, a conferência da área, e a categoria **`cs.MA`** no arXiv.
Quase ninguém em "agentic AI" acompanha `cs.MA` — é uma vantagem informacional barata.

---

## Projeto

**P14 — Research Multi-Agent System (M14)** ⭐

Planner, researcher, critic, evaluator, executor sob um supervisor.

Depois **medir**:

```
1 agente   vs   3 agentes   vs   5 agentes
```

Comparando: qualidade · custo · latência · número de erros.

*Pronto quando* existe a tabela comparativa.

---

## Essa medição já é ciência

O plano diz isso explicitamente, e vale sublinhar: a comparação 1 vs 3 vs 5 agentes é um
experimento publicável em workshop. Não porque seja difícil, mas porque **quase ninguém
faz** — o campo produz demonstrações, não medições.

A hipótese que a maioria assume sem testar é "mais agentes, melhor resultado". Na prática
o custo cresce de forma superlinear (cada agente conversa com os outros), a latência
soma, e a qualidade frequentemente satura em 3 — ou piora, porque erro de um agente
contamina os demais.

Se você medir isso direito, com baseline honesto e variância reportada, você tem o
primeiro resultado original do seu M18 — dez meses antes do prazo. Escreva a hipótese
antes de rodar, no formato de [`_templates/experimento/`](../_templates/experimento/).
