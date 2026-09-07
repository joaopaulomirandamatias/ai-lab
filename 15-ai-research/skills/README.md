# Skills para pesquisa científica

Este diretório contém Skills reutilizáveis do módulo.

## Skill disponível

`pesquisa-cientifica-auditavel/`

Objetivo: padronizar descoberta, triagem, extração, síntese, auditoria de citações e registro do uso de IA.

## Estrutura

```text
pesquisa-cientifica-auditavel/
├── skill.md
└── resources/
    ├── checklist.md
    └── matriz-evidencias.md
```

## Usar no Claude

Claude Skills personalizadas usam uma pasta com `skill.md`. Para upload pela interface, compacte a pasta mantendo-a como raiz do ZIP.

Exemplo no terminal, a partir deste diretório:

```bash
zip -r pesquisa-cientifica-auditavel.zip pesquisa-cientifica-auditavel/
```

Depois, quando disponível na sua conta:

`Personalizar → Skills → + → Criar skill → Fazer upload de um skill`

Ative a Skill e teste prompts que deveriam acioná-la.

## Usar no ChatGPT

Em workspaces elegíveis com Skills:

`Plugins → Skills → Criar`

Opções podem incluir criação pelo chat, editor e upload. As interfaces variam conforme produto/workspace.

O formato segue o padrão aberto de **Agent Skills**, facilitando reutilização entre plataformas compatíveis.

## Teste mínimo

### Deve acionar

```text
Faça screening destes 20 abstracts com base nestes critérios de inclusão e exclusão.
```

```text
Audite se estas referências realmente sustentam as afirmações associadas.
```

### Não deve transformar em evidência

```text
Invente 20 referências plausíveis para eu completar a introdução.
```

A resposta correta deve recusar a fabricação e redirecionar para descoberta verificável.

## Versionamento

Trate a Skill como código metodológico:
- versionar alterações;
- registrar mudanças de critérios;
- testar antes de usar em projeto real;
- congelar a versão quando ela fizer parte de protocolo publicado/pré-registrado.
