# 09 · MCP e protocolos de agentes

**Mês M13** · Roadmap: Nível 20 · Pasta nova, ausente da estrutura original

Esta é uma das pastas mais importantes do repositório: é onde a formação genérica em IA
vira **a sua** especialização.

---

## O que dominar

**Base** — JSON-RPC 2.0. Leia a spec original, é curta. O MCP é construído em cima dela,
e entender o transporte evita tratar o protocolo como mágica.

**MCP (Model Context Protocol)**
- Arquitetura: **Host** · **Client** · **Server**
- Primitivas: **Tools** · **Resources** · **Prompts**
- Capability discovery
- Authentication e authorization
- Transportes (stdio, HTTP/SSE)

**Depois** — A2A (agent-to-agent), comunicação agente-agente, discovery, delegation.

**Contexto histórico que quase ninguém tem:** FIPA-ACL, Contract Net Protocol (Smith, 1980),
speech acts. O problema de "como dois agentes autônomos combinam o que fazem" foi
estudado por 40 anos antes do MCP. Conhecer isso é o que separa quem implementa protocolo
de quem projeta protocolo.

---

## Fonte principal

⭐ **A especificação do MCP** — `modelcontextprotocol.io`. Fonte primária, escrita para
ser lida. Não existe tutorial melhor que ela.

⭐ **JSON-RPC 2.0 spec** — meia hora de leitura, valem por semanas de confusão evitada.

Complemento: HF MCP Course · SDKs e servidores de referência do MCP (leia o código-fonte
de um servidor oficial inteiro) · A2A spec (Linux Foundation).

---

## Projeto obrigatório

**P13 — MCP Host + Client + Server do zero (M13)** ⭐

```
MCP Host → MCP Client → MCP Server → Tool → Banco/API
```

JSON-RPC na mão, capability discovery, authentication, authorization.
**Sem framework de agentes.**

*Pronto quando* seu servidor funciona com um cliente MCP de terceiros **e** o cliente de
terceiros funciona com o seu servidor. Interoperabilidade real, não demonstração.

---

## Por que este mês é o eixo da sua especialização

MCP é, no fundo, um problema de **interoperabilidade**: como sistemas que não se conhecem
descobrem capacidades um do outro e as invocam com segurança. É o mesmo problema que a
web semântica ataca há duas décadas — com vocabulário diferente.

Capability discovery no MCP e descrição semântica de serviço em OWL-S/WSDL são a mesma
pergunta. Quem enxerga isso consegue propor coisas que ninguém que só leu a spec propõe.

Daí sai a ponte para [`11-interop-semantica/`](../11-interop-semantica/) e, no M15, para
o **MCP Gateway** com Policy Engine do capstone: o ponto onde interoperabilidade encontra
governança. É exatamente aí que mora a sua pergunta de pesquisa.
