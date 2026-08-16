# Projetos

Um projeto longitudinal, não trinta desconectados.

```
projetos/
└── ai-systems-lab/     # criar no primeiro dia de código do M1
```

O **AI Systems Laboratory** evolui M1 → M18, ganhando uma camada por mês:

```
matemática → gradient descent → estatística → ML → Neural Network → PyTorch
→ Deep Learning → Transformer → LLM → Fine-tuning → RAG → Agent → MCP
→ Multi-Agent → Governed AI → Evaluation → Paper reproduction → Original experiment
```

Ao terminar, o repositório é **evidência pública da evolução** — não uma lista de cursos
concluídos.

---

## Os 18 projetos

A lista completa, com o **critério de pronto** de cada um, está em
[`recursos/projetos.md`](../recursos/projetos.md).

Os cinco que mais importam:

| # | Projeto | Mês | Por quê |
|---|---|---|---|
| **P8** | Transformer do zero | M8 | Tudo depois é construído sobre isso |
| **P11** | RAG em três versões | M11 | 🔶 Gate III — prova em vez de opinião |
| **P13** | MCP Host+Client+Server do zero | M13 | O eixo da sua especialização |
| **P14** | Research Multi-Agent System | M14 | A medição já é publicável |
| **P15** | Mini Governed AI Runtime | M15 | 🔶 Gate IV — o capstone |

---

## Como estruturar

Sugestão para `ai-systems-lab/`, crescendo por camada e não por mês:

```
ai-systems-lab/
├── README.md           # o que o sistema faz HOJE — atualizado a cada camada
├── CHANGELOG.md        # uma entrada por mês: o que foi adicionado e por quê
├── core/               # matemática, treino, modelo
├── retrieval/          # embeddings, RAG, grafo
├── agents/             # loop, tools, state, memória
├── protocols/          # MCP client/server, A2A
├── governance/         # policy engine, HIL, evidence, audit
├── evaluation/         # conjuntos de avaliação, métricas, harness
└── experiments/        # cada um a partir de _templates/experimento/
```

O `CHANGELOG.md` é o que transforma o repositório em narrativa. Daqui a um ano, ele é a
diferença entre "tenho um projeto" e "consigo mostrar como cheguei aqui" — e é o que se
lê numa entrevista ou numa banca.

---

## Uma decisão que vale tomar cedo

Este repositório (`ai-lab`) é do **estudo** — notas, fichamentos, plano. O
`ai-systems-lab` é o **sistema**. Vale considerar mantê-lo como repositório Git próprio,
não como subpasta:

- histórico de commits limpo, que mostra a evolução do sistema sem ruído de anotação;
- pode ser público desde cedo sem expor notas pessoais;
- CI, testes e releases fazem sentido nele e não fazem aqui.

Se preferir simplicidade agora, comece como subpasta — separar depois é um `git filter-repo`,
não um problema. Só evite o inverso: começar público e ter que limpar histórico.
