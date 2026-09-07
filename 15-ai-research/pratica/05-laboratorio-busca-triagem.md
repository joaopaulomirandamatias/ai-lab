# Laboratório 5 — Busca, filtragem e triagem assistida por IA

**Duração:** 60–90 min  
**Formato:** individual ou dupla  
**Ferramentas:** ChatGPT ou Claude + 1 plugin acadêmico + navegador + Zotero

## Objetivo

Construir uma mini-base de evidências sem delegar o controle metodológico à IA.

## Pergunta exemplo

> Quais métodos têm sido utilizados para avaliar factualidade e alucinação em sistemas RAG entre 2022 e 2026?

O aluno pode trocar o tema, mantendo o mesmo protocolo.

---

# Fase 1 — Protocolo antes da busca

Preencha:

### Pergunta
[...]

### Inclusão
- período;
- tipo de publicação;
- contexto;
- avaliação empírica;
- idioma, se justificado.

### Exclusão
- opinião sem experimento;
- duplicado;
- fora do escopo;
- sem informação mínima para triagem.

### Campos a extrair
- título;
- autores;
- ano;
- DOI/URL;
- desenho;
- dataset;
- métrica;
- resultado;
- limitações.

**Gate 1:** os critérios devem ser registrados antes da classificação automatizada.

---

# Fase 2 — Construção da estratégia

Use ChatGPT ou Claude somente para expansão inicial:

```text
Minha pergunta científica é:
[PERGUNTA]

Não pesquise artigos ainda.
Gere:
1. conceitos;
2. sinônimos;
3. termos técnicos;
4. acrônimos;
5. string booleana inicial.

Explique quais termos aumentam recall e quais podem reduzir precisão.
```

Revise manualmente a string.

---

# Fase 3 — Busca em plugin acadêmico

Use Consensus, Elicit, SciSpace ou equivalente.

Objetivo: obter **20–30 candidatos**, não uma conclusão.

Exporte/registre:
- consulta;
- data;
- ferramenta;
- filtros;
- quantidade de resultados;
- identificadores retornados.

---

# Fase 4 — Busca em base/navegador

Abra Semantic Scholar, PubMed, IEEE Xplore ou outra base autorizada.

Se usar Claude in Chrome:

```text
Leia a página de resultados.
Mostre os filtros disponíveis.
Não execute nenhuma ação ainda.
Compare a interface com estes critérios de inclusão/exclusão:
[CRITÉRIOS]
```

Depois aprove explicitamente os filtros.

Capture 20 resultados adicionais ou os primeiros 20 de uma busca equivalente.

**Gate 2:** Claude não deve inventar metadados não visíveis.

---

# Fase 5 — Deduplicação

Combine os candidatos numa tabela:

| ID | Título | Ano | DOI | Origem | Duplicado? |
|---|---|---:|---|---|---|

Deduplicate preferencialmente por:
1. DOI;
2. PMID/arXiv ID/outro identificador;
3. título normalizado + ano.

Não use apenas similaridade de título sem revisão.

---

# Fase 6 — Triagem humano × IA

Selecione 20 registros.

### Rodada A — humano
Classifique 10 sem ver a resposta da IA.

### Rodada B — IA

```text
Use somente os critérios fornecidos.
Classifique cada registro em:
- incluir;
- excluir;
- incerto.

Para cada decisão, cite exatamente qual critério foi acionado.
Se o abstract não fornecer informação suficiente, use "incerto".
```

### Comparação

| ID | Humano | IA | Concorda? | Erro provável | Decisão final |
|---|---|---|---|---|---|

**Gate 3:** um falso negativo relevante deve ser discutido explicitamente.

---

# Fase 7 — Verificação bibliográfica

Para cada estudo incluído:
- resolva DOI/identificador;
- abra a página oficial;
- confirme título/autores/ano;
- confira se a publicação realmente existe;
- confirme que o abstract ou texto sustenta a relevância.

Salve os confirmados no Zotero.

---

# Fase 8 — Matriz de evidências

Escolha 5 estudos e preencha:

| Estudo | Método | Dataset/amostra | Métrica | Resultado | Limitação | Fonte verificada? |
|---|---|---|---|---|---|---|

Depois permita que a IA organize/sintetize **somente essa matriz**.

Prompt:

```text
Use exclusivamente a matriz fornecida.
Identifique convergências, divergências e lacunas.
Para cada conclusão, liste os estudos que a sustentam.
Não acrescente referências externas.
```

---

# Fase 9 — Zotero

Crie coleção:

`LAB-BUSCA-TRIAGEM-[DATA]`

Salve:
- estudos incluídos;
- tags de status;
- nota curta sobre motivo de inclusão;
- PDF quando licenciado/autorizado.

---

# Fase 10 — AI Research Log

Registre pelo menos três entradas:
1. expansão da busca;
2. triagem;
3. síntese.

Use `../templates/ai-research-log.csv`.

---

# Entregáveis

1. protocolo preenchido;
2. string de busca final;
3. tabela de candidatos/deduplicação;
4. comparação humano × IA;
5. coleção Zotero;
6. matriz de 5 estudos;
7. síntese de no máximo 500 palavras;
8. AI Research Log.

# Rubrica — 100 pontos

| Critério | Pontos |
|---|---:|
| critérios definidos antes da IA | 15 |
| busca rastreável | 15 |
| deduplicação correta | 10 |
| triagem auditável | 15 |
| verificação das fontes | 20 |
| Zotero organizado | 10 |
| síntese fiel às evidências | 10 |
| AI Research Log | 5 |

# Falha automática do laboratório

O trabalho deve ser refeito se:
- contiver referência inexistente apresentada como real;
- apresentar dado sensível/confidencial enviado sem autorização;
- não for possível reconstruir como os estudos foram selecionados.
