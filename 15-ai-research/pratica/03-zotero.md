# Prática 3 — Zotero como memória bibliográfica da pesquisa

> Meta: impedir que a bibliografia viva apenas no histórico de um chatbot.

## 1. Papel do Zotero no workflow

Use o Zotero como **fonte de verdade bibliográfica** para referências que você decidiu manter.

```text
Descoberta em bases/IA
        ↓
verificação da fonte
        ↓
      Zotero
        ↓
coleções + tags + notas + PDFs
        ↓
BibTeX / Word / Google Docs / LaTeX
```

A IA pode descobrir candidatos; o Zotero organiza os itens que passaram pelo controle humano.

## 2. Instalação recomendada

Instale:
1. Zotero Desktop;
2. **Zotero Connector** no navegador.

O Connector está disponível para Chrome, Firefox e Edge; no Safari ele é distribuído com o Zotero.

O Connector detecta metadados bibliográficos na página e permite salvar um artigo com um clique — em geral é preferível a copiar manualmente uma referência produzida por IA.

## 3. Capturando artigos corretamente

### Evite

```text
LLM gera referência → copiar para Word
```

### Prefira

```text
LLM/base encontra candidato
→ abrir página oficial/DOI
→ verificar artigo
→ Zotero Connector
→ conferir metadados
→ salvar em coleção
```

Ao salvar, confira pelo menos:
- título;
- autores;
- ano;
- periódico/conferência;
- DOI;
- tipo de item;
- PDF/anexo, quando permitido.

## 4. Estrutura de coleções para uma pesquisa

Exemplo:

```text
Projeto — IA e Integridade Científica
├── 00-sementes
├── 01-incluidos
├── 02-excluidos-relevantes
├── 03-revisoes
├── 04-metodos
├── 05-evidencia-contraditoria
└── 99-citar-no-paper
```

Tags sugeridas:
- `screening-incluir`;
- `screening-incerto`;
- `metodo-chave`;
- `risco-vies-alto`;
- `replicar`;
- `citar-introducao`;
- `citar-metodo`;
- `citar-discussao`.

## 5. Notas como camada de evidência

Para cada estudo importante, mantenha uma nota estruturada:

```text
Pergunta:
Método:
Amostra:
Resultado principal:
Limitações:
Trecho que sustenta minha afirmação:
Página/seção:
Minha interpretação:
Contradiz qual outro estudo?
```

Isso permite que a IA trabalhe depois sobre **notas verificadas**, em vez de tentar reconstruir toda a evidência de memória.

## 6. Citações em Word, LibreOffice e Google Docs

Os plugins de processador de texto do Zotero fazem parte do ecossistema oficial do Zotero.

Workflow:
1. escreva a afirmação;
2. insira a citação via Zotero;
3. escolha o item da biblioteca;
4. deixe o Zotero formatar a referência;
5. use a IA apenas para revisar clareza, sem substituir a checagem da fonte.

## 7. Zotero + LaTeX/BibTeX

Para projetos em LaTeX:
- exporte BibTeX/BibLaTeX;
- versionar `references.bib` pode ser útil;
- não altere metadados incorretos apenas no `.bib`: corrija também na biblioteca principal.

## 8. Zotero + Codex

Quando o plugin/Skill Zotero está disponível no Codex, o fluxo local pode pesquisar a biblioteca do Zotero Desktop, exportar BibTeX e inserir chaves de citação em rascunhos.

Exemplos de operações que uma integração local pode realizar:
- verificar status da API local;
- listar coleções/tags;
- buscar itens por título/termo;
- exportar `references.bib`;
- inserir uma citação num arquivo Markdown/LaTeX;
- recuperar texto completo quando solicitado e permitido;
- importar BibTeX/RIS com ação explícita.

### Exemplo de uso conceitual

```text
Procure no meu Zotero artigos sobre "semantic interoperability" publicados desde 2023.
Retorne título, autores, ano e chave de citação.
Não busque na web; quero somente itens que já estão na minha biblioteca.
```

Depois:

```text
Exporte os itens selecionados para references.bib e use as chaves para montar uma seção Related Work.
Não crie referências fora da biblioteca.
```

## 9. Separando duas chaves

Em integrações programáticas, não confunda:
- **item key do Zotero** — identificador interno do item;
- **citation/BibTeX key** — chave usada em documentos como `@autor2026titulo`.

São identificadores diferentes.

## 10. Laboratório — Da base ao Zotero

### Objetivo
Criar uma coleção de 10 referências auditadas.

1. Crie coleção `LAB-IA-PESQUISA`.
2. Encontre 15 candidatos usando base/plugin acadêmico.
3. Abra a fonte original de cada candidato relevante.
4. Salve via Zotero Connector.
5. Corrija metadados quando necessário.
6. Aplique tags de triagem.
7. Escolha 5 artigos.
8. Crie uma nota estruturada para cada um.
9. Exporte uma bibliografia ou `references.bib`.
10. Compare com a lista inicial da IA e documente erros encontrados.

### Entregáveis
- coleção Zotero;
- 5 notas estruturadas;
- bibliografia exportada;
- AI Research Log.

## 11. Plugins do Zotero

Zotero possui comunidade de plugins, mas plugins têm acesso amplo à biblioteca e ao computador. Instale somente extensões de origem confiável e compatíveis com sua versão.

Não confunda:
- **Zotero Connector** — extensão oficial de navegador;
- **plugins do Zotero Desktop** — extensões instaladas no aplicativo;
- **plugin Zotero em IA/Codex** — integração entre o assistente e sua biblioteca local.

## Fontes oficiais — verificadas em 2026-09-07

- Zotero — Adding Items: https://www.zotero.org/support/adding_items_to_zotero
- Zotero Connector: https://www.zotero.org/support/connector
- Download/Connectors: https://www.zotero.org/downloads
- Plugins do Zotero: https://www.zotero.org/support/plugins
