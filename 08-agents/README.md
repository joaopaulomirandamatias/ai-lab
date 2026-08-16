# 08 · AI Agents

**Mês M12** · Roadmap: Níveis 18, 21, 22, 23

> **Primeiro: não use framework.** Construa um agente você mesmo.

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

---

## O que dominar

**Núcleo** — tool calling, function calling, ReAct, planning, reflection, state, memory,
routing, retries, termination, guardrails.

**Memória** (Nível 22) — memória de curto prazo, de longo prazo, episódica, semântica,
sumarização de contexto, esquecimento seletivo, persistência.

**Context Engineering** (Nível 23) — janela de contexto, orçamento de tokens, compressão,
seleção do que entra, ordem, cache. É onde está a maior parte do ganho real de qualidade
em sistemas agentic — e a menor parte da atenção do mercado.

**Orquestração** (Nível 21) — DAG, workflow, máquina de estados, event-driven,
human-in-the-loop.

E, principalmente:

> **Quando NÃO usar um agente.**
>
> Comparar `workflow determinístico vs agent`.

---

## Fonte principal

⭐ **Lil'Log (Lilian Weng)** — o artigo sobre LLM-powered autonomous agents é a melhor
síntese existente. Leia antes de escrever a primeira linha.

⭐ **Anthropic — "Building effective agents"** — engenharia honesta, escrita por quem
construiu o MCP: a maior parte dos casos não precisa de agente.

Papers na ordem: CoT → Self-Consistency → **ReAct** → Toolformer → Reflexion.
Curso: HF Agents Course.
Livro clássico: Russell & Norvig, capítulos de agentes racionais — o vocabulário formal
que o campo está redescobrindo.

---

## Projeto

**P12 — Agente do zero, sem framework (M12)** ⭐

Loop próprio: tool calling, ReAct, planning, reflection, retries, condição de parada,
guardrails.

*Pronto quando* você compara o agente com um **workflow determinístico** na mesma tarefa
e consegue dizer em quais casos o workflow ganha.

---

## O objetivo real deste mês

Não é construir um agente — é adquirir o **julgamento de quando não construir**.

Um agente é um loop com um LLM decidindo o próximo passo. Isso o torna flexível e,
pelo mesmo motivo, não determinístico, mais caro, mais lento e mais difícil de auditar.
Para a maioria das tarefas, um workflow determinístico com uma chamada de LLM no meio é
melhor em todas as dimensões que importam.

Essa comparação — workflow vs agente vs multiagente — é literalmente o experimento do
**M18**. Comece a montar o instrumento de medida agora: mesmo conjunto de tarefas, mesmas
métricas de qualidade, custo, latência e erro. O que você medir aqui vira baseline lá.
