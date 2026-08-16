# 15 · Reproduções

**Mês M17** · Roadmap: Nível 47

```
Paper → Reimplementação → Dataset → Baseline → Resultados → Comparação
```

---

## O exercício

Escolha um paper relevante — de preferência um que você fichou no M16 e que se conecta
com a sua pergunta de pesquisa — e reimplemente.

Se o paper diz:

$$\text{Accuracy} = 89.7\%$$

você tenta reproduzir. Se obtiver:

$$\text{Accuracy} = 87.9\%$$

**investigue por quê.**

> Isso ensina muito mais que dez cursos.

---

## Estrutura

Uma pasta por reprodução, a partir de
[`_templates/experimento/`](../_templates/experimento/):

```
15-reproductions/
└── 2022-yao-react/
    ├── README.md          # qual paper, o que foi reproduzido, o que não foi
    ├── hipotese.md        # o que eu espero obter, escrito ANTES
    ├── dataset/
    ├── codigo/
    ├── config/
    ├── metricas/
    ├── resultados/
    ├── graficos/
    └── conclusao.md       # os dois números lado a lado + investigação da diferença
```

---

## Critério de pronto

*Pronto quando* você tem os dois números lado a lado — o do paper e o seu — e uma
**investigação escrita** da diferença.

Divergir é normal. Não investigar não é.

---

## O que a divergência costuma revelar

Quando o número não bate, a causa quase sempre está em algo que o paper não disse — e
descobrir isso é o aprendizado real:

- hiperparâmetro omitido ou escolhido "por busca" sem detalhar o espaço
- pré-processamento não descrito
- versão diferente do dataset, ou split diferente
- seed única, sem variância reportada (e o número publicado era o melhor de várias execuções)
- diferença de versão de biblioteca, tokenizer ou hardware
- para LLMs: **modelo mudou**. `gpt-4` de 2023 não é `gpt-4` de 2024. Registre sempre o
  identificador exato de versão no `config/`

Cada um desses achados é material legítimo de discussão em um paper seu, e vários deles
são exatamente o tipo de *threat to validity* que o M18 vai exigir que você declare no
seu próprio trabalho.

Reprodutibilidade em IA generativa é um problema aberto e reconhecido — se você
documentar bem as suas dificuldades, isso já é contribuição.
