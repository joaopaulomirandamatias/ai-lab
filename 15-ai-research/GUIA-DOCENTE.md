# Guia do Docente

## Propósito pedagógico

A aula não deve ser um treinamento de uma ferramenta específica. O objetivo é ensinar uma competência durável: **integrar IA à pesquisa preservando método, evidência, responsabilidade e auditabilidade**.

## Metodologia didática

Combine:
- Backward Design;
- Problem-Based Learning;
- Case-Based Learning;
- aprendizagem ativa;
- laboratório reproduzível;
- avaliação por evidências do processo.

## Contrato didático

No início, estabeleça quatro regras:

1. Nenhuma saída de IA é aceita como evidência por si só.
2. Referências devem ser verificadas na fonte original.
3. Dados não podem ser enviados a ferramentas sem análise de autorização e risco.
4. Toda decisão metodológica importante precisa ter responsável humano identificável.

## Plano minuto a minuto — 180 min

### 0–10 min — Provocação
Mostre uma resposta de IA com aparência científica e 5 referências. Pergunte:
- Qual parte vocês confiariam?
- Como verificariam?
- Se uma referência não existir, quem responde?

### 10–15 min — Objetivos
Apresente o princípio: IA é uma camada de apoio dentro do método, não substituto do método.

### 15–35 min — Como LLMs funcionam
Ensine:
- probabilidade de tokens;
- contexto;
- grounding;
- alucinação;
- viés;
- desatualização;
- limites de explicabilidade.

Evite transformar a aula em matemática de Transformers; use somente o necessário para explicar por que respostas plausíveis podem estar erradas.

### 35–65 min — Integridade, legislação e governança
Aborde:
- CNPq 2026;
- LGPD e guia da ANPD;
- ética em pesquisa com seres humanos quando aplicável;
- confidencialidade;
- autoria;
- direito autoral;
- política da instituição;
- política do periódico.

Pergunta-chave:
> É tecnicamente possível? É juridicamente permitido? É eticamente aceitável? É metodologicamente defensável? É aceito pelo periódico?

### 65–95 min — Literatura científica com IA
Demonstre:
- busca booleana;
- busca semântica;
- backward/forward citation chaining;
- Elicit/Consensus/Semantic Scholar/Scite;
- Zotero;
- RAG;
- verificação de DOI e fonte.

### 95–105 min — Intervalo

### 105–130 min — Laboratório de referências
Execute `laboratorios/01-cacando-alucinacoes.md`.

### 130–150 min — Dados, código e estatística
Mostre o notebook `02-analise-reprodutivel.ipynb`.
Destaque:
- código que executa não é sinônimo de método correto;
- pressupostos estatísticos;
- versionamento;
- sementes aleatórias;
- rastreabilidade.

### 150–165 min — Escrita e publicação
Ensine:
- IA não é autora;
- humanos respondem pelo conteúdo;
- regras de disclosure;
- confidencialidade em peer review;
- declaração de uso.

### 165–175 min — Casos
Divida a turma em grupos e entregue casos do laboratório 03.

### 175–180 min — Exit ticket
Cada aluno responde:
1. Uma tarefa em que usaria IA.
2. Uma tarefa em que não usaria sem controles adicionais.
3. Como verificaria uma saída crítica.
4. O que registraria no AI Research Log.

---

# Oficina prática estendida — 6–8 horas

Quando houver tempo, acrescente a trilha `pratica/`.

## Bloco A — ChatGPT como orquestrador — 45 min

Use `pratica/01-chatgpt-pesquisa.md`.

Demonstre:
- decomposição de pergunta;
- Deep Research;
- plugin acadêmico;
- matriz de evidências;
- auditor adversarial;
- criação/uso de Skill quando disponível.

### Resultado esperado
O aluno entende a diferença entre:

```text
ChatGPT encontra candidato
```

e

```text
fonte original sustenta afirmação
```

## Bloco B — Claude e navegador — 45–60 min

Use `pratica/02-claude-navegador.md`.

Demonstre uma base real em modo supervisionado:
1. abrir Semantic Scholar/PubMed/outra base;
2. pedir ao Claude para identificar filtros;
3. aprovar filtros manualmente;
4. extrair metadados visíveis;
5. comparar triagem humano × IA.

### Segurança
Antes da demonstração, explique prompt injection em navegador. Use perfil separado e evite páginas sensíveis.

## Bloco C — Zotero — 40 min

Use `pratica/03-zotero.md`.

Demonstre:
- Zotero Connector;
- coleção do projeto;
- tags;
- nota estruturada;
- bibliografia/BibTeX;
- diferença entre referência candidata e referência verificada.

## Bloco D — Ecossistema de plugins — 30 min

Use `pratica/04-plugins-academicos.md`.

Compare a mesma pergunta em pelo menos:
- um plugin acadêmico;
- uma base tradicional.

O objetivo é medir cobertura/triagem, não escolher um “campeão”.

## Bloco E — Laboratório de busca e triagem — 60–90 min

Execute `pratica/05-laboratorio-busca-triagem.md`.

Obrigatório:
- protocolo antes da automação;
- deduplicação;
- humano × IA;
- verificação bibliográfica;
- Zotero;
- AI Research Log.

## Bloco F — Construção de Skill — 45–60 min

Execute `pratica/06-laboratorio-skill.md`.

Mostre `skills/pesquisa-cientifica-auditavel/skill.md` como implementação de referência.

Explique que uma Skill científica deve ser tratada como **artefato metodológico versionado**, não apenas como prompt grande.

### Red team obrigatório
Teste pelo menos:
- pedido para inventar referências;
- pedido para mudar critério retroativamente;
- pedido para inferir amostra ausente;
- tentativa de processar dado sensível sem gate.

## Bloco G — Workflow end-to-end — 30 min

Use `pratica/07-workflow-end-to-end.md` para fechar a oficina.

O aluno deve enxergar a cadeia:

```text
Pergunta
→ estratégia
→ busca
→ candidatos
→ screening
→ verificação
→ Zotero
→ matriz
→ síntese
→ auditoria
→ log
→ escrita
```

---

## Erros conceituais comuns

### "Se duas IAs concordam, está validado"
Não. Pode haver correlação de erro, fontes comuns e mesmos vieses.

### "RAG elimina alucinação"
Não. RAG melhora grounding, mas a recuperação pode ser ruim e a interpretação pode falhar.

### "Dado anonimizado é sempre seguro"
Não necessariamente. Avalie possibilidade de reidentificação e contexto.

### "Código funcionou, então está correto"
Não. Execução é só uma condição necessária.

### "IA não pode ser usada em pesquisa"
Generalização incorreta. O uso depende da tarefa, dados, financiador, instituição, periódico e controles.

### "Detector de IA prova fraude"
Evite tratar detectores como prova conclusiva. Eles possuem limitações e falsos positivos.

### "Uma Skill resolve o método"
Não. A Skill reduz variação operacional e torna regras repetíveis; desenho, julgamento e responsabilidade continuam humanos.

### "Automação no navegador é igual a busca reproduzível"
Não. Interfaces mudam, resultados podem variar e ações podem depender de login/personalização. Registre consulta, filtros, data, base e resultados exportados.

## Critérios de sucesso da aula

O aluno deve ser capaz de:
- justificar decisões, não apenas listar ferramentas;
- reconstruir o caminho da afirmação até a fonte;
- explicar onde a IA participou;
- mostrar como a saída foi validada;
- executar um fluxo prático sem perder governança.
