# Prática 1 — Usando ChatGPT na pesquisa científica

> Meta: usar ChatGPT como **orquestrador de pesquisa**, não como fonte científica.

## 1. Quatro modos úteis

### Chat comum
Use para:
- decompor uma pergunta;
- melhorar termos de busca;
- revisar clareza;
- criticar argumentos;
- analisar arquivos fornecidos.

### Busca na web
Use quando a tarefa exige informação atual ou fontes externas. Exija links/citações e depois confira a fonte original.

### Deep Research
É adequado para levantamento mais amplo e comparativo. Em 2026, o Deep Research permite controlar melhor o plano e focar a pesquisa em **sites específicos e apps conectados**.

Use para:
- mapear estado da arte;
- comparar normas e políticas;
- localizar grupos de trabalhos;
- levantar lacunas preliminares;
- preparar uma lista de fontes para revisão humana.

Não trate o relatório final como revisão sistemática por si só.

### Plugins
Desde julho de 2026, o diretório de apps do ChatGPT foi migrado para o **Diretório de Plugins**. Um plugin pode combinar Skills e apps conectados.

Quando disponível na interface:
1. abra `Plugins`;
2. instale/conecte o plugin desejado;
3. no chat, use `@NomeDoPlugin` ou `+ → Mais`, conforme a interface;
4. dê uma tarefa bem delimitada;
5. exporte/registre os resultados relevantes;
6. valide no artigo original.

## 2. Plugins acadêmicos úteis

A disponibilidade depende da conta/workspace. Exemplos relevantes em 2026:

- **Consensus** — descoberta e síntese de literatura revisada por pares;
- **Elicit** — busca e extração estruturada de estudos;
- **SciSpace** — descoberta, triagem e tabelas de leitura;
- **Sider Scholar** — busca acadêmica, coleções e RAG sobre documentos;
- **Academic Writing Toolkit** — consistência de citações, lógica de parágrafos e BibTeX;
- **Zotero** — fluxo de biblioteca/referências no Codex quando o plugin está disponível;
- **Scholar Sidekick** — opção para verificação de identificadores, metadados e alertas de correção/retração.

Veja a matriz completa em [`04-plugins-academicos.md`](04-plugins-academicos.md).

## 3. Workflow recomendado: pergunta → papers → evidência

### Passo A — Estruture a pergunta

Prompt:

```text
Minha pergunta científica é:
[PERGUNTA]

Não procure artigos ainda.
1. Identifique conceitos principais.
2. Gere sinônimos e termos técnicos.
3. Separe-os em blocos conceituais.
4. Sugira uma string booleana inicial.
5. Liste ambiguidades que eu preciso resolver antes da busca.
```

### Passo B — Descoberta em plugin acadêmico

Exemplo:

```text
@Consensus
Busque estudos revisados por pares relacionados a [PERGUNTA].
Período: 2021–2026.
Priorize revisões sistemáticas e estudos empíricos.
Retorne título, ano, desenho, população/contexto, achado principal e DOI/URL.
Não faça afirmações que não estejam apoiadas nos registros encontrados.
```

Repita em outro mecanismo apenas quando isso ampliar cobertura — não para "votar" sobre a verdade.

### Passo C — Matriz de evidências

```text
Usando somente os estudos que eu forneci/verifiquei, crie uma tabela com:
- referência;
- pergunta;
- desenho;
- amostra;
- método;
- resultados;
- limitações;
- trecho/fonte que sustenta cada resultado.
Se um campo não estiver disponível, escreva "não localizado".
```

### Passo D — Auditor adversarial

```text
Atue como revisor metodológico.
Para cada conclusão da minha síntese:
1. identifique quais estudos realmente a sustentam;
2. procure generalização indevida;
3. diferencie associação de causalidade;
4. indique limitações e evidência contraditória;
5. marque qualquer afirmação sem fonte verificável.
Não invente referências.
```

## 4. Criando uma Skill de pesquisa no ChatGPT

Em contas/workspaces elegíveis, **Skills** são workflows reutilizáveis que podem incluir instruções, exemplos e código. A disponibilidade atual é específica de determinados planos/workspaces.

Caminhos de criação, quando disponíveis:
- `Plugins → Skills → Criar → Criar com o chat`;
- editor de Skills;
- upload de uma Skill criada externamente.

Também é possível pedir:

```text
Crie uma Skill para pesquisa científica auditável.
Ela deve ser usada em descoberta de literatura, triagem, extração e síntese.
Regras obrigatórias:
- nunca tratar saída de IA como evidência;
- nunca inventar referências;
- exigir fonte original para afirmações críticas;
- registrar ferramenta, data, consulta e validação;
- classificar dados antes de enviá-los a serviços externos;
- diferenciar revisão narrativa de revisão sistemática.
```

O módulo fornece uma implementação pronta em:

`15-ai-research/skills/pesquisa-cientifica-auditavel/skill.md`

## 5. Laboratório rápido — 20 minutos

1. Escolha uma pergunta científica.
2. Gere blocos conceituais e string booleana no ChatGPT.
3. Pesquise com um plugin acadêmico.
4. Selecione 5 referências candidatas.
5. Verifique DOI e fonte original.
6. Salve as referências confirmadas no Zotero.
7. Crie matriz de evidências somente com as fontes verificadas.
8. Registre o uso no `AI Research Log`.

### Critério de aprovação

O estudante precisa demonstrar a cadeia:

```text
afirmação → estudo → trecho/dado → decisão humana
```

## 6. Cuidados

- Não envie dados pessoais/sensíveis sem avaliação de base legal, protocolo e política do serviço.
- Não coloque manuscritos confidenciais em ferramentas externas sem autorização.
- Plugins podem ter permissões próprias; instalação não elimina a necessidade de avaliar privacidade e fornecedor.
- Uma Skill padroniza o processo, mas não torna automaticamente o processo cientificamente válido.

## Fontes oficiais — verificadas em 2026-09-07

- OpenAI — Skills no ChatGPT: https://help.openai.com/pt-br/articles/20001066-skills-no-chatgpt
- OpenAI — Plugins no ChatGPT e Codex: https://help.openai.com/pt-br/articles/20001256-plugins-no-chatgpt-e-no-codex
- OpenAI — Apps/Plugins no ChatGPT: https://help.openai.com/pt-br/articles/11487775
- OpenAI — Release notes / Deep Research: https://help.openai.com/pt-br/articles/6825453-chatgpt-release-notes
