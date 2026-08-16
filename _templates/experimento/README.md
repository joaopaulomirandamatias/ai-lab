# Template de experimento

Estrutura definida no fim do [roadmap](../../00-plano/roadmap-49-niveis.md).
Copie esta pasta inteira para iniciar qualquer experimento.

```
<experimento>/
├── README.md          # o que é, em uma frase, e como rodar
├── hipotese.md        # ESCRITO ANTES DE RODAR QUALQUER COISA
├── dataset/           # dados ou o script que os baixa + hash de verificação
├── codigo/            # implementação
├── config/            # hiperparâmetros, seeds, versões de modelo
├── metricas/          # definição das métricas e por que essas
├── resultados/        # saídas brutas — nunca editadas à mão
├── graficos/          # visualizações geradas a partir de resultados/
└── conclusao.md       # o que se conclui, e o que NÃO se conclui
```

---

## A regra que sustenta o resto

**`hipotese.md` é escrito antes de `resultados/` existir.**

Escrever a hipótese depois de ver o número é a forma mais comum e mais silenciosa de
fazer ciência ruim — e é indetectável depois, porque o texto final fica idêntico.
O commit do `hipotese.md` com data anterior ao do resultado é a sua única prova, para
você mesmo, de que não houve *HARKing*.

Por isso: **commit separado para a hipótese, antes de rodar.**

---

## Checklist antes de considerar concluído

- [ ] `hipotese.md` commitado antes do primeiro resultado
- [ ] Seed fixada e registrada em `config/`
- [ ] Versão exata de modelo/biblioteca registrada (não "gpt-4", mas o identificador completo)
- [ ] Baseline implementado e declarado — e honesto
- [ ] Mais de uma execução, com variância reportada
- [ ] Custo e latência medidos, não estimados
- [ ] `conclusao.md` tem uma seção **"o que este experimento não prova"**
- [ ] Alguém conseguiria reexecutar isso só com o README
