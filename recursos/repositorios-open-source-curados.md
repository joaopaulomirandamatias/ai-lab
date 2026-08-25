# Repositórios open source curados para experimentação

**Última verificação:** 2026-08-25

Esta página registra repositórios externos úteis para a formação **Especialista em IA** e
para o AI Systems Laboratory. Eles são fontes de código, padrões e ideias de experimento;
não substituem especificações oficiais, livros ou artigos científicos.

## Regra de uso

Antes de reutilizar qualquer componente:

1. fixe tag ou commit;
2. confirme a licença do repositório e das dependências;
3. execute em ambiente isolado, sem credenciais ou dados reais;
4. escreva a hipótese e as métricas antes do experimento;
5. compare com um baseline próprio;
6. registre versão, configuração, custo, latência, falhas e limitações;
7. atribua a origem no README e nos artefatos derivados.

Popularidade, presença em uma lista “awesome” ou uma demonstração visual não comprovam
segurança, correção, interoperabilidade ou prontidão para produção.

## Mapa de uso

| Repositório | Tipo | Trilhas | Decisão didática |
|---|---|---|---|
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | coleção executável de agentes, RAG e interfaces agentic | M11–M15 | selecionar exemplos como baselines; não adotar como framework |
| [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | catálogo comunitário de servidores MCP | M13 | usar como fonte de descoberta; tratar cada servidor como não confiável até avaliação |
| [punkpeye/awesome-mcp-clients](https://github.com/punkpeye/awesome-mcp-clients) | catálogo comunitário de clientes/hosts MCP | M13 | selecionar clientes independentes para testes de interoperabilidade |
| [ripienaar/free-for-dev](https://github.com/ripienaar/free-for-dev) | catálogo de serviços com free tier | transversal | apoiar protótipos e FinOps; reconfirmar limites no fornecedor |
| [D4Vinci/Scrapling](https://github.com/D4Vinci/Scrapling) | scraping/crawling, Markdown para RAG e servidor MCP | M11, M13 e M15 | experimentar com governança de aquisição web |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | control plane self-hosted para agentes de software | M12–M15 | estudar isolamento, automações e arquitetura de Agent Server |
| [OpenHands/software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) | SDK Python e API para agentes que trabalham com código | M12–M15 | usar como implementação de comparação, não como arquitetura canônica |
| [All-Hands-AI/openhands-resolver](https://github.com/All-Hands-AI/openhands-resolver) | resolvedor histórico de issues | — | não usar: repositório arquivado |
| [sindresorhus/awesome](https://github.com/sindresorhus/awesome) | metaíndice de listas curadas | transversal | descobrir fontes; validar cada lista e item separadamente |

## Experimentos candidatos

### E1 — Web para RAG com proveniência

**Fonte:** Scrapling.

Comparar:

```text
HTML bruto
  vs
extração de conteúdo principal
  vs
extração + Markdown sanitizado + seleção CSS
```

Medir tokens, tempo de ingestão, precisão de recuperação, groundedness e incidência de
conteúdo oculto. O crawler deve usar allowlist de domínios, respeito a `robots.txt`,
limite por domínio, AutoThrottle, retenção de URL/data/hash e dados exclusivamente
públicos ou autorizados.

### E2 — Matriz de interoperabilidade MCP

**Fontes:** Awesome MCP Servers e Awesome MCP Clients.

Selecionar implementações de mantenedores diferentes e registrar:

- versão da especificação;
- transporte;
- autenticação;
- tools, resources e prompts;
- discovery e atualização de capabilities;
- timeout, cancelamento e reconexão;
- comportamento diante de erro;
- resultado do teste cruzado host/client/server.

As listas são apenas inventário. A licença MIT da lista não concede direitos sobre o
código dos projetos apontados.

### E3 — Roteamento multi-MCP e mínimo privilégio

**Fonte:** exemplo Multi-MCP Agent Router do Awesome LLM Apps.

Comparar um agente com todas as ferramentas contra especialistas com subconjuntos de
tools. Medir taxa de conclusão, chamadas indevidas, custo, latência e violações de
política. O roteador do exemplo é baseline didático; a implementação do laboratório
deve adicionar policy fail-closed e testes adversariais.

### E4 — Trust gate e trilha de auditoria

**Fonte:** exemplo Trust-Gated Multi-Agent Research Team do Awesome LLM Apps.

Reproduzir a cadeia de hashes e testar adulteração, mas distinguir:

- integridade encadeada;
- identidade do agente;
- assinatura;
- armazenamento imutável;
- proveniência;
- correção semântica da decisão.

Uma cadeia SHA-256 detecta alteração posterior, mas não prova identidade, veracidade da
entrada ou correção da ação. O experimento deve explicitar essa ameaça à validade.

### E5 — Diagnóstico de falhas em RAG

**Fonte:** RAG Failure Diagnostics Clinic do Awesome LLM Apps.

Usar a taxonomia P01–P12 como ponto de partida para incidentes de retrieval, chunking,
staleness, roteamento, tool misuse, memória, avaliação e interferência multiagente.
Validar cada classe com caso sintético reproduzível e contraprova.

### E6 — Agente de software isolado

**Fontes:** OpenHands e Software Agent SDK.

Comparar execução local direta com workspace efêmero. Medir contenção de filesystem,
rede, secrets, reprodutibilidade, custo, tempo, artefatos e capacidade de reconstruir a
execução. Nunca executar o modo sem sandbox em máquina com credenciais de produção.

## Encaixe nos projetos obrigatórios

| Projeto do roadmap | Uso |
|---|---|
| P11 — RAG em três versões | E1 e E5 |
| P12 — agente do zero | comparar com o SDK somente depois da implementação própria |
| P13 — MCP Host + Client + Server | E2 |
| P14 — Research Multi-Agent System | E3 e E4 |
| P15 — Mini Governed AI Runtime | E1–E4 e E6 como baselines externos |
| P18 — experimento original | resultados controlados acumulados dos experimentos anteriores |

## Restrições de segurança e pesquisa

- Scraping deve respeitar autorização, termos aplicáveis, `robots.txt`, limites e LGPD.
- Conteúdo coletado é entrada não confiável e pode conter prompt injection.
- Servidor MCP comunitário executa com os privilégios concedidos ao processo.
- Agente de software pode ler arquivos, executar comandos e alcançar a rede.
- API keys devem entrar por secret manager ou ambiente isolado, nunca pela interface de
  uma demonstração commitada.
- Resultados de exemplos não são evidência científica independente.
- Mudanças de versão devem invalidar resultados que dependam do comportamento alterado.

## Licenças observadas em 2026-08-25

| Repositório | Licença do repositório |
|---|---|
| awesome-llm-apps | Apache-2.0 |
| awesome-mcp-servers / awesome-mcp-clients | MIT |
| Scrapling | BSD-3-Clause |
| OpenHands / Software Agent SDK / OpenHands Resolver | MIT |
| awesome | CC0-1.0 |
| free-for-dev | nenhuma licença detectada; não reproduzir a base integralmente |

A licença de um catálogo não se estende aos projetos externos que ele referencia.
