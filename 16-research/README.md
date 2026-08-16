# 16 · Pesquisa original

**Mês M18** · Roadmap: Níveis 35, 48, 49 · **🔶 Gate V**

```
Estado da arte → Gap → Pergunta → Hipótese → Proposta
→ Experimento → Evidência → Paper
```

A mudança de `"eu sei construir isso"` para
`"eu consigo produzir evidência científica de que minha proposta funciona"`.

---

## A linha de pesquisa

> **Como adicionar determinismo, governança semântica e auditabilidade à execução de
> sistemas multiagentes baseados em LLM?**

Ela liga tudo que você construiu: LLM · Agentic AI · MAS · MCP · semântica · governança ·
segurança · avaliação.

## O experimento que dá validade ao roadmap inteiro

O roadmap culmina em um sistema de IA governado — mas a validade científica disso depende
de demonstrar que a arquitetura **melhora resultados sob critérios explícitos**. Isso exige:

> comparar **workflow · agente · multiagente** em um experimento controlado,
> medindo qualidade, custo, latência e segurança.

Os instrumentos já existem: o baseline de workflow vs agente saiu do
[M12](../08-agents/), a medição 1 vs 3 vs 5 agentes saiu do [M14](../10-multiagent/), e as
métricas de governança saíram do [M15](../12-ai-governance/). O M18 é onde isso vira
desenho experimental com hipótese e análise estatística.

---

## Estrutura

```
16-research/
├── estado-da-arte.md      # o que já existe, com citações — não impressão
├── gap.md                 # o que NÃO existe, e por que isso importa
├── pergunta.md            # uma frase, com ponto de interrogação
├── hipotese.md            # falseável, escrita ANTES do experimento
├── proposta/              # a arquitetura ou método proposto
├── experimento/           # a partir de _templates/experimento/
├── analise/               # estatística, tamanho de efeito, variância
└── paper/                 # o texto
```

---

## Critérios do Gate V

- [ ] Gap identificado a partir de **leitura real** da literatura, não de intuição
- [ ] Hipótese falseável, escrita e commitada **antes** do experimento
- [ ] Baseline implementado por você
- [ ] Análise estatística com tamanho de efeito, não só p-value
- [ ] Limitações e *threats to validity* declarados honestamente
- [ ] Texto submetido a algum lugar

---

## Onde submeter

Comece por **workshop**, não por conferência principal. Taxa de aceitação maior, revisão
mais construtiva, e é a porta de entrada normal — não um plano B.

- **AAMAS** — a conferência de sistemas multiagentes. Workshops muito alinhados
- **ISWC / ESWC** — web semântica e knowledge graphs
- **NeurIPS / ICLR / ACL** — workshops sobre agentes, avaliação e alinhamento
- **FAccT / AIES** — governança, auditabilidade, responsabilidade
- **arXiv** — preprint desde o primeiro rascunho decente. Estabelece data e permite feedback

Em paralelo, há a via nacional: SBC (SBIA, ENIAC, WebMedia) e periódicos brasileiros,
com revisão em português e comunidade acessível.

---

## O resultado esperado

Depois de 18 meses, a meta não é dizer:

> "Sei usar GPT, Claude, Hugging Face e LangChain."

A meta é conseguir dizer:

> "Consigo entender matematicamente o modelo, treiná-lo e avaliá-lo; construir RAG e
> sistemas agentic; projetar sua arquitetura distribuída; integrar ferramentas por
> protocolos; controlar suas permissões; medir custo e qualidade; registrar suas decisões;
> avaliar experimentalmente seu comportamento e produzir pesquisa científica sobre a
> arquitetura."

Esse é o patamar de **especialista em sistemas de IA** — não o de desenvolvedor que
integra modelos.
