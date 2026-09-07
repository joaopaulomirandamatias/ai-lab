# 15 · IA na Pesquisa Científica
## Do Prompt à Ciência Reproduzível

Pacote didático completo para uma aula de 3 horas, extensível para **oficina prática de 6–8 horas**.

### Princípio central

> IA não substitui o método científico. Ela deve ser inserida em um processo documentado, verificável, reproduzível e com responsabilidade humana.

## Público-alvo
- graduação avançada;
- pós-graduação;
- pesquisadores;
- orientadores;
- equipes de P&D;
- profissionais que produzem relatórios técnico-científicos.

## Resultados de aprendizagem

Ao final, o participante deverá conseguir:
1. distinguir resposta de IA de evidência científica;
2. selecionar ferramentas adequadas para diferentes etapas da pesquisa;
3. verificar referências e afirmações produzidas por IA;
4. reconhecer riscos de privacidade, confidencialidade e propriedade intelectual;
5. documentar usos relevantes de IA;
6. construir um fluxo de pesquisa assistida por IA auditável;
7. utilizar IA em programação e análise sem abrir mão da validação;
8. preparar uma declaração transparente de uso de IA;
9. aplicar práticas de revisão sistemática, ciência aberta e reprodutibilidade;
10. usar ChatGPT, Claude, Zotero e plugins acadêmicos em um workflow integrado;
11. criar e testar uma **Skill para pesquisa científica auditável**;
12. fazer triagem assistida em bases de dados mantendo supervisão humana.

## Estrutura do módulo

```text
15-ai-research/
├── README.md
├── GUIA-DOCENTE.md
├── aula/
│   └── aula-completa.md
├── slides/
│   └── roteiro-slides.md
├── pratica/
│   ├── README.md
│   ├── 01-chatgpt-pesquisa.md
│   ├── 02-claude-navegador.md
│   ├── 03-zotero.md
│   ├── 04-plugins-academicos.md
│   ├── 05-laboratorio-busca-triagem.md
│   ├── 06-laboratorio-skill.md
│   └── 07-workflow-end-to-end.md
├── skills/
│   ├── README.md
│   └── pesquisa-cientifica-auditavel/
│       ├── skill.md
│       └── resources/
│           ├── checklist.md
│           └── matriz-evidencias.md
├── laboratorios/
│   ├── 01-cacando-alucinacoes.md
│   ├── 02-busca-evidencias.md
│   ├── 03-privacidade-lgpd.md
│   ├── 04-codigo-reprodutivel.md
│   └── 05-revisao-sistematica-com-ia.md
├── exercicios/
│   ├── quiz.md
│   └── gabarito.md
├── templates/
│   ├── ai-research-log.csv
│   ├── checklist-conformidade.md
│   ├── declaracao-uso-ia.md
│   └── protocolo-mini-revisao.md
├── notebooks/
│   ├── 01-auditoria-referencias.ipynb
│   └── 02-analise-reprodutivel.ipynb
├── dados/
│   └── dataset-sintetico.csv
├── prompts/
│   └── biblioteca-prompts.md
├── diagramas/
│   └── diagramas-mermaid.md
├── projeto-final/
│   └── mini-dossie-pesquisa.md
└── referencias/
    └── referencias.md
```

## Roteiro essencial — 3 horas

| Bloco | Tempo | Tema |
|---|---:|---|
| Abertura | 15 min | IA ≠ evidência |
| Fundamentos | 20 min | LLMs, alucinação e viés |
| Integridade e legislação | 30 min | CNPq, LGPD, ética, autoria |
| Literatura científica | 30 min | busca, RAG, citações |
| Intervalo | 10 min | |
| Laboratório | 25 min | auditoria de referências |
| Dados e código | 20 min | programação e reprodutibilidade |
| Publicação | 15 min | autoria, peer review, declaração |
| Casos e avaliação | 15 min | decisões de uso e exit ticket |

## Oficina prática estendida — 6–8 horas

Além do roteiro essencial:

| Bloco | Atividade |
|---|---|
| ChatGPT | Deep Research, plugins, matriz de evidências e Skills |
| Claude | Research, Claude in Chrome e filtragem supervisionada |
| Zotero | captura via Connector, coleções, notas e BibTeX |
| Plugins | comparação Consensus/Elicit/SciSpace/Sider Scholar |
| Skill | criação, upload, testes positivos e red team |
| Laboratório | busca + deduplicação + screening humano × IA |
| Workflow final | evidência verificada → Zotero → síntese → log |

Comece por [`pratica/README.md`](pratica/README.md).

## Skill pronta para pesquisa

A Skill [`pesquisa-cientifica-auditavel`](skills/pesquisa-cientifica-auditavel/skill.md) implementa quality gates para:
- descoberta;
- screening;
- extração;
- auditoria de citações;
- síntese;
- revisão metodológica;
- classificação de dados;
- AI Research Log.

Ela foi desenhada no formato de Agent Skills para facilitar reutilização em plataformas compatíveis.

## Como usar o pacote

1. Leia `GUIA-DOCENTE.md`.
2. Use `slides/roteiro-slides.md` para preparar a apresentação.
3. Entregue `aula/aula-completa.md` como texto-base.
4. Para a parte operacional, siga `pratica/README.md`.
5. Escolha 2–3 laboratórios conforme a carga horária.
6. Use os notebooks em Jupyter/Colab.
7. Teste a Skill em `skills/pesquisa-cientifica-auditavel/`.
8. Encerre com `projeto-final/mini-dossie-pesquisa.md`.

## Mensagem da aula

**Não pergunte apenas "a IA acertou?". Pergunte "como posso verificar que ela acertou?".**
