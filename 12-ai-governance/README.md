# 12 · Governança e segurança de IA

**Mês M15** · Roadmap: Níveis 30, 31, 32 · **🔶 Gate IV** no fim do mês

> Aqui entra sua especialização profunda: **Governed Agentic AI**.

---

## Segurança específica de IA (Nível 30)

prompt injection · **indirect prompt injection** · tool abuse · confused deputy ·
privilege escalation · data exfiltration · context poisoning · jailbreak ·
model extraction · membership inference.

Mais a segurança tradicional que sustenta tudo: authn/authz, least privilege, sandboxing,
secrets, auditoria.

**O ponto que muda o projeto:** em um agente com ferramentas, o ataque não vem do usuário —
vem do **conteúdo que o agente lê**. Uma página web, um e-mail, um documento no RAG podem
carregar instruções. É por isso que política precisa ser aplicada fora do LLM, e não
pedida a ele no prompt.

---

## Governança (Nível 31)

```
Identity + Purpose + Policy + Confidence + Human Review + Evidence
```

**HIL formalizado por confiança:**

```
σ alto          → ALLOW
σ intermediário → HIL
σ baixo         → DENY
```

**Evidence — dossiê por decisão:**
quem? qual modelo? qual versão? qual entrada? qual saída? qual confiança?
qual política? qual evidência? houve humano?

**Explainable AI** (Nível 32) — SHAP, LIME, atribuição, interpretabilidade,
por que o modelo decidiu o que decidiu.

---

## Fonte principal

⭐ **OWASP Top 10 for LLM Applications** + **OWASP Agentic Security Initiative** —
a taxonomia de ataque de referência, gratuita e viva.

⭐ **NIST AI Risk Management Framework (AI RMF 1.0)** + *Generative AI Profile* —
o vocabulário de risco que reguladores e auditores usam.

Papers: **Greshake et al. — *Not what you've signed up for*** (indirect prompt injection,
o paper central para quem constrói agentes) · Zou et al. (ataques adversariais universais).

Normas: EU AI Act · ISO/IEC 42001 e 23894 · LGPD · MITRE ATLAS.
Reporting: Model Cards (Mitchell et al.) · Datasheets for Datasets (Gebru et al.).
XAI: *Interpretable Machine Learning* — Molnar (grátis).

---

## Projeto capstone

**P15 — Mini Governed AI Runtime (M15)** 🔶 **Gate IV** ⭐⭐

```
Usuário → Mission Compiler → Planner → Model Router → Agent
        → MCP Gateway → Policy → Tool → Evidence
```

Incluindo: Model Registry · Policy Engine · HIL · Audit Trail · Evaluation · Observability.

*Pronto quando* você consegue pegar uma ação executada há uma semana e reconstruir
**por que** ela foi executada, a partir do audit trail — sem olhar o código.

---

## O que este mês torna possível

Governança costuma ser tratada como camada de conformidade colada no fim. O capstone
propõe o oposto: governança como **arquitetura** — policy engine no caminho da chamada,
não em um documento.

É essa inversão que dá densidade à pergunta de pesquisa do M18. "Adicionar determinismo,
governança semântica e auditabilidade a sistemas multiagentes" só é uma afirmação
científica se houver como medir: taxa de ações bloqueadas corretamente, custo da camada
de política em latência, cobertura do audit trail, reconstrutibilidade de decisão.

Vale começar a listar essas métricas agora, junto com [`13-evaluation/`](../13-evaluation/).
Sem elas, o M18 vira ensaio de opinião.
