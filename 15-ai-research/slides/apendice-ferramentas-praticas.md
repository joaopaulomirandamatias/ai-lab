# Apêndice de Slides — Ferramentas práticas de pesquisa

Use após os 28 slides principais ou em oficina separada.

| # | Título | Visual | Mensagem-chave |
|---:|---|---|---|
| P1 | A stack de pesquisa | ChatGPT + Claude + Zotero + bases | nenhuma ferramenta faz tudo |
| P2 | ChatGPT | pergunta → plugins → fontes | orquestrador, não fonte |
| P3 | Deep Research | plano → múltiplas fontes → relatório | levantamento amplo ≠ revisão sistemática |
| P4 | Plugins acadêmicos | matriz por função | escolha pela tarefa |
| P5 | Claude Research | buscas múltiplas com citações | pesquisa agêntica ainda exige verificação |
| P6 | Claude in Chrome | navegador + painel lateral | automação de interface com supervisão |
| P7 | Prompt injection | página maliciosa → agente | navegador aumenta superfície de risco |
| P8 | Zotero | fonte verificada → biblioteca | memória bibliográfica deve ser controlada |
| P9 | Zotero Connector | browser → metadados → coleção | capture da fonte, não do texto gerado |
| P10 | Skills científicas | skill.md + quality gates | transforme método repetível em artefato versionado |
| P11 | Humano × IA | matriz de discordância | medir erro é melhor que confiar |
| P12 | Workflow final | pergunta → Zotero → evidência → paper | IA acelera; método governa |

## P1 — A stack de pesquisa

```text
ChatGPT/Claude      → raciocínio, busca, síntese, crítica
Plugins acadêmicos  → descoberta/extração especializada
Bases científicas   → corpus e registros
Zotero              → biblioteca verificada
Skill               → workflow e quality gates
Git/OSF             → versão e reprodutibilidade
```

## P4 — Plugins por função

Mostrar tabela simplificada:

| Descoberta | Extração | Biblioteca | Auditoria |
|---|---|---|---|
| Consensus | Elicit | Zotero | Academic Writing Toolkit |
| SciSpace | SciSpace | Zotero | Scholar Sidekick |
| Sider Scholar | Elicit | | |

## P6 — Claude in Chrome

Demonstração recomendada:
1. abrir Semantic Scholar ou PubMed;
2. pedir apenas para identificar filtros;
3. revisar;
4. aplicar filtros aprovados;
5. capturar 10 títulos/abstracts;
6. comparar decisões humano × IA.

## P7 — Segurança

Use o contraste:

```text
Agente lê página
     ↓
Página também pode conter instruções hostis
     ↓
Aprovação + perfil separado + allowlist
```

## P10 — Skill científica

Mostrar a estrutura real do repositório:

```text
pesquisa-cientifica-auditavel/
├── skill.md
└── resources/
    ├── checklist.md
    └── matriz-evidencias.md
```

Mensagem:

> Prompt é uma instrução pontual. Skill é um workflow versionável com critérios e gates.

## P11 — Métrica didática

Não perguntar apenas "funcionou?".

Meça:
- falsos negativos;
- discordâncias;
- campos inventados;
- referência não verificável;
- tempo;
- rastreabilidade.
