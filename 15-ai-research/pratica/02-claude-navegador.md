# Prática 2 — Usando Claude, Research e Claude in Chrome

> Meta: usar Claude para **pesquisa orientada por fontes e navegação assistida**, sem delegar cegamente decisões de inclusão/exclusão.

## 1. Três formas de usar Claude na pesquisa

### Chat + busca na web
Com busca na web ativa, Claude pode localizar e analisar páginas e URLs específicas.

### Research
Nos planos pagos compatíveis, o modo **Research** executa buscas múltiplas de forma agêntica, explorando ângulos complementares e retornando respostas com citações.

Ativação típica:
1. clique em `+`;
2. selecione `Research`/`Pesquisa`;
3. mantenha busca na web ativa;
4. forneça pergunta, escopo, período e critérios.

### Claude in Chrome
A extensão oficial **Claude in Chrome** permite que Claude:
- leia a página atual;
- alterne abas;
- clique;
- digite;
- navegue;
- preencha campos;
- compare conteúdo entre páginas.

Em 2026, está disponível em planos pagos do Claude, com diferenças de rollout entre superfícies.

## 2. Quando o navegador é útil para pesquisa

Use o navegador quando o valor está na própria interface da base:
- construir/fazer ajustes numa busca;
- percorrer páginas de resultados;
- abrir títulos e abstracts;
- capturar metadados visíveis;
- comparar filtros;
- organizar uma fila preliminar para triagem humana.

Não use a automação para:
- violar termos de uso da base;
- contornar paywalls ou controles de acesso;
- baixar em massa conteúdo sem autorização;
- decidir sozinha a inclusão final em revisão de alto impacto;
- manipular páginas com dados pessoais/sensíveis sem controle adequado.

## 3. Exemplo — filtrando uma base pelo navegador

### Cenário
Pergunta: **Quais técnicas reduzem alucinações em RAG para aplicações críticas?**

Critérios preliminares:
- 2022–2026;
- inglês/português;
- artigo científico ou preprint claramente identificado;
- avaliação empírica;
- precisa medir factualidade, grounding, hallucination ou equivalente;
- excluir textos apenas opinativos.

### Passo A — abra a base

Use uma base compatível com seu acesso, por exemplo:
- PubMed;
- Semantic Scholar;
- IEEE Xplore;
- ACM Digital Library;
- Scopus/Web of Science, se sua instituição fornecer acesso.

### Passo B — peça navegação supervisionada

```text
Nesta base, ajude-me a executar uma busca para:
[PERGUNTA]

Critérios:
[CRITÉRIOS]

Primeiro:
1. leia a interface;
2. mostre quais filtros estão disponíveis;
3. proponha a string de busca;
4. NÃO clique em filtros ou exclua resultados até eu aprovar.
```

### Passo C — aplique filtros controlados

Depois da revisão humana:

```text
Aplique somente estes filtros aprovados:
- ano: 2022–2026;
- tipo: artigo/conference paper;
- idioma: inglês ou português, se a base permitir.

Não aplique filtro temático adicional sem me mostrar antes.
```

### Passo D — triagem de título/abstract

```text
Analise os 20 resultados visíveis nesta página.
Para cada item, capture somente o que estiver visível:
- título;
- autores;
- ano;
- venue;
- DOI/URL, se visível;
- decisão preliminar: incluir / excluir / incerto;
- justificativa de uma frase baseada nos critérios.

Nunca inferir abstract, método ou resultado que não esteja visível.
Marque como "não disponível" quando necessário.
```

### Passo E — revisão dos excluídos

O pesquisador deve revisar:
- todos os `incertos`;
- uma amostra dos `excluídos`;
- todos os casos em que o abstract não estava visível;
- todo estudo central à conclusão.

## 4. Laboratório de concordância humano × Claude

Selecione 30 títulos/resumos.

1. Humano classifica 10 sem ver a resposta do Claude.
2. Claude classifica os mesmos 10 usando critérios idênticos.
3. Compare discordâncias.
4. Ajuste a instrução, **não os critérios retroativamente**.
5. Classifique os 20 restantes.
6. Revise falsos negativos potenciais.

Tabela:

| ID | Humano | Claude | Concorda? | Motivo da divergência | Decisão final |
|---|---|---|---|---|---|

### Métrica opcional
Calcule concordância simples e, para turmas avançadas, Cohen's kappa.

## 5. Claude Skills para pesquisa

Claude permite Skills personalizadas compostas por uma pasta que contém no mínimo `skill.md` com frontmatter YAML de `name` e `description`.

Estrutura mínima:

```text
pesquisa-cientifica-auditavel/
├── skill.md
└── resources/
    ├── checklist.md
    └── matriz-evidencias.md
```

Fluxo atual de upload:
1. crie a pasta da Skill;
2. compacte em ZIP mantendo a pasta como raiz;
3. abra `Personalizar → Skills`;
4. `+ → Criar skill → Fazer upload de um skill`;
5. ative e teste com prompts que deveriam acioná-la.

A Skill pronta deste curso está em `../skills/pesquisa-cientifica-auditavel/`.

## 6. Um mesmo padrão para ChatGPT e Claude

O formato de Agent Skills é útil porque transforma regras metodológicas em infraestrutura reutilizável:

```text
pergunta
  ↓
Skill científica
  ↓
critérios + workflow + quality gates
  ↓
ferramentas/fontes
  ↓
verificação
  ↓
artefatos auditáveis
```

Não coloque senhas, tokens, dados sensíveis ou material confidencial dentro da Skill.

## 7. Segurança do Claude in Chrome

Ferramentas de IA que controlam navegador introduzem um risco adicional: **prompt injection em páginas web**.

Conteúdo malicioso ou oculto numa página pode tentar induzir o agente a executar ações indevidas.

Boas práticas:
- use modo de aprovação manual durante exercícios;
- prefira um perfil de navegador separado para pesquisa;
- não mantenha contas financeiras, administrativas ou dados sensíveis abertas no mesmo perfil;
- limite sites permitidos quando a organização oferecer allowlist;
- revise ações antes de upload, envio ou exclusão;
- não use em páginas com dados regulados/sensíveis sem política específica.

## 8. Exercício prático — 30 minutos

Objetivo: produzir 10 candidatos verificáveis para leitura.

1. Defina pergunta e critérios.
2. Abra Semantic Scholar, PubMed ou outra base autorizada.
3. Use Claude in Chrome para identificar filtros disponíveis.
4. Aprove manualmente a string/filtros.
5. Capture 20 candidatos.
6. Faça triagem preliminar.
7. Revise todos os incertos e excluídos duvidosos.
8. Salve os 10 melhores no Zotero.
9. Preencha o AI Research Log.

## Fontes oficiais — verificadas em 2026-09-07

- Claude — Research: https://support.claude.com/pt/articles/11088861-usar-pesquisa-no-claude
- Claude — busca na web: https://support.claude.com/pt/articles/10684626-ativar-e-usar-busca-na-web
- Claude in Chrome: https://support.claude.com/pt/articles/12012173-comece-com-claude-no-chrome
- Segurança no Claude in Chrome: https://support.claude.com/pt/articles/12902428-use-claude-in-chrome-com-seguranca
- Criar Skills: https://support.claude.com/pt/articles/12512198-como-criar-habilidades-personalizadas
