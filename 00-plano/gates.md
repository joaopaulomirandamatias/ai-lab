# Os 5 gates

Progresso não se mede por certificado nem por curso concluído. Mede-se por **gate**:
uma demonstração prática de que você consegue fazer algo, não de que assistiu algo.

Regra: um gate só é considerado passado quando existe **artefato no repositório** —
código que roda, número medido, texto que explica. Não vale "eu acho que entendi".

---

## Gate I — Scientific Foundations
**Quando:** fim do M2 · **Pasta:** [`01-math/`](../01-math/)

> Você consegue **derivar e implementar** gradient descent.

Critério de passagem:
- [ ] Derivar $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$ no papel, partindo da definição de derivada
- [ ] Implementar do zero em NumPy, sem framework, para regressão linear e logística
- [ ] Explicar o efeito de learning rate alto/baixo mostrando as curvas
- [ ] Explicar por que praticamente toda rede neural se constrói sobre $y = Wx + b$

---

## Gate II — ML Engineer
**Quando:** fim do M4 · **Pasta:** [`03-machine-learning/`](../03-machine-learning/)

> Você consegue criar um experimento de ML corretamente, **sem data leakage**,
> usando baseline e métricas apropriadas.

Critério de passagem:
- [ ] Split treino/validação/teste feito antes de qualquer transformação
- [ ] Baseline trivial declarado (média, classe majoritária) e batido
- [ ] Métrica escolhida e **justificada** pelo problema, não por hábito
- [ ] Validação cruzada com intervalo de confiança, não um número solto
- [ ] Você consegue apontar onde havia risco de leakage e como o eliminou

---

## Gate III — LLM Engineer
**Quando:** fim do M11 · **Pasta:** [`07-rag/`](../07-rag/)

> Você consegue decidir e implementar `prompt vs RAG vs fine-tuning` — e **provar**
> qual solução funciona melhor.

Critério de passagem:
- [ ] Três versões implementadas do mesmo sistema
- [ ] Mesmo conjunto de avaliação para as três
- [ ] Métricas de retrieval **e** de geração (precision, recall, faithfulness, groundedness)
- [ ] Custo e latência medidos, não estimados
- [ ] Conclusão escrita que sobreviveria a alguém perguntando "por quê?"

---

## Gate IV — Agentic AI Engineer
**Quando:** fim do M15 · **Pastas:** [`08-agents/`](../08-agents/) · [`09-mcp-protocolos/`](../09-mcp-protocolos/) · [`12-ai-governance/`](../12-ai-governance/)

> Você consegue construir `LLM + Tools + State + MCP + Policy + HIL + Evaluation + Audit`
> **sem depender intelectualmente de CrewAI/LangChain.**

Você deve conseguir explicar tecnicamente:
- [ ] Qual a diferença entre workflow, agente e multiagente?
- [ ] Quando usar MCP?
- [ ] Quando usar A2A?
- [ ] Como impedir um agente de executar algo que não deveria?
- [ ] Como provar posteriormente por que uma ação foi executada?
- [ ] Como avaliar um sistema agentic?

Esse gate é bem mais importante que simplesmente saber LangChain.

---

## Gate V — AI Researcher
**Quando:** fim do M18 · **Pasta:** [`16-research/`](../16-research/)

> Você consegue percorrer o ciclo científico inteiro.

```
identificar gap → formular hipótese → implementar baseline → executar experimento
→ analisar estatisticamente → defender conclusão → publicar
```

Critério de passagem:
- [ ] Gap identificado a partir de leitura real da literatura, não de intuição
- [ ] Hipótese falseável, escrita antes do experimento
- [ ] Baseline implementado por você
- [ ] Análise estatística com tamanho de efeito, não só p-value
- [ ] Limitações e *threats to validity* declarados honestamente
- [ ] Texto submetido a algum lugar (workshop, preprint, periódico)

---

## O teste final

Você deve conseguir olhar para um problema e decidir, em ordem:

1. Preciso realmente de IA?
2. ML tradicional, Deep Learning ou LLM?
3. Modelo local ou API?
4. Prompt, RAG ou fine-tuning?
5. Agente ou workflow determinístico?
6. Um agente ou sistema multiagente?
7. Como medir se funciona?
8. Como tornar seguro?
9. Como garantir rastreabilidade?
10. Quanto custa?
11. Como colocar em produção?
12. Quais evidências científicas comprovam que essa arquitetura é melhor?

Quando você responde tecnicamente, implementa a solução e **demonstra experimentalmente
que ela funciona**, você deixou de ser usuário de IA e está atuando como especialista.
