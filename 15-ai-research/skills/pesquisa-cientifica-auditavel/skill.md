---
name: pesquisa-cientifica-auditavel
description: Apoia busca, triagem, extração, síntese e auditoria de evidências científicas com fontes verificáveis, registro de IA e supervisão humana.
---

# Pesquisa Científica Auditável

## Objetivo

Use esta Skill quando a tarefa envolver:
- descoberta de literatura;
- estratégia de busca;
- screening/triagem;
- extração de dados de artigos;
- matriz de evidências;
- verificação de referências;
- síntese de estudos;
- auditoria metodológica;
- documentação do uso de IA.

Não use esta Skill para transformar uma resposta de IA em evidência sem fonte.

## Princípios obrigatórios

1. **IA não é fonte científica.**
2. **Não invente referências, DOI, PMID, autores, resultados ou citações.**
3. Diferencie claramente:
   - informação fornecida pelo usuário;
   - informação recuperada de uma fonte;
   - inferência da IA.
4. Afirmações científicas críticas devem chegar à fonte original ou a um registro bibliográfico confiável.
5. Duas ferramentas concordarem não constitui validação independente.
6. Não altere critérios de inclusão/exclusão retroativamente para acomodar resultados.
7. Não envie dados pessoais, sensíveis, confidenciais ou material de peer review a serviços externos sem autorização apropriada.
8. Toda decisão metodológica importante permanece sob responsabilidade humana.

## Gate 0 — Classifique a tarefa

Antes de executar, determine se a tarefa é:
- `discovery` — encontrar candidatos;
- `screening` — classificar candidatos segundo critérios;
- `extraction` — extrair campos de estudos;
- `synthesis` — sintetizar corpus controlado;
- `citation-audit` — verificar referência/afirmação;
- `method-review` — criticar método;
- `writing-review` — revisar texto sem criar evidência nova.

Se a tarefa misturar etapas, execute-as separadamente e registre a transição.

## Gate 1 — Dados e confidencialidade

Antes de enviar dados para uma ferramenta externa, classifique:
- público;
- interno;
- pessoal;
- pessoal sensível;
- pseudonimizado;
- anonimizado;
- confidencial;
- propriedade intelectual de terceiros.

Se houver risco material, interrompa a automação e solicite/indique revisão de autorização, protocolo, política institucional ou contrato.

## Workflow A — Descoberta de literatura

1. Reformule a pergunta sem alterar o sentido.
2. Extraia blocos conceituais.
3. Gere sinônimos e descritores.
4. Proponha estratégia booleana.
5. Identifique bases adequadas.
6. Execute/solicite busca em fonte apropriada.
7. Retorne **candidatos**, não conclusões.
8. Para cada candidato, preserve identificadores verificáveis quando disponíveis.

### Saída padrão

| ID | Título | Ano | Fonte/base | DOI/ID | Por que é candidato? | Verificado? |
|---|---|---:|---|---|---|---|

## Workflow B — Screening

Pré-condição: critérios explícitos fornecidos/registrados.

Classifique cada item como:
- `incluir`;
- `excluir`;
- `incerto`.

Para cada decisão:
- cite o critério acionado;
- use somente título/abstract/texto disponível;
- não preencha lacunas por inferência;
- prefira `incerto` quando falta informação.

Nunca altere o critério para aumentar concordância com o modelo.

## Workflow C — Extração

Extraia somente campos presentes na fonte.

Campos padrão:
- pergunta/objetivo;
- desenho;
- população/contexto;
- amostra/dataset;
- intervenção/exposição;
- comparador;
- métricas/desfechos;
- resultado;
- limitações;
- seção/trecho de origem.

Se não localizar: `não localizado`.

## Workflow D — Auditoria de citação

Para uma afirmação + referência:

1. A referência existe?
2. Os metadados conferem?
3. O DOI/identificador corresponde ao título/autores?
4. A fonte original foi acessada?
5. A fonte sustenta a afirmação?
6. A população/método/contexto são compatíveis?
7. A afirmação excede o que o estudo permite concluir?

Classifique:
- `confirmada`;
- `parcialmente sustentada`;
- `não sustentada`;
- `não verificável com o material disponível`.

## Workflow E — Síntese

Pré-condição: corpus controlado e preferencialmente verificado.

1. Organize convergências.
2. Organize divergências.
3. Separe resultado de interpretação.
4. Identifique limitações comuns.
5. Aponte lacunas.
6. Para cada conclusão, liste os estudos que a sustentam.
7. Não introduza referências externas sem iniciar novo workflow de discovery.

## Workflow F — Revisão metodológica adversarial

Procure:
- confundidores;
- leakage;
- seleção inadequada de baseline;
- métricas incompatíveis;
- causalidade indevida;
- múltiplas comparações;
- viés de seleção/publicação;
- problemas de unidade experimental;
- ausência de análise de sensibilidade;
- ameaças à validade externa;
- falta de reprodutibilidade.

Não trate uma crítica hipotética como erro comprovado.

## Registro mínimo de IA

Ao final de uso metodologicamente relevante, produza:

```text
Data:
Etapa:
Ferramenta/modelo (quando disponível):
Finalidade:
Consulta/prompt:
Dados/fontes enviados:
Saída usada:
Verificação realizada:
Decisão humana:
Limitações:
```

## Quality gates finais

Antes de concluir, confirme:
- [ ] nenhuma referência foi inventada;
- [ ] afirmações críticas têm fonte rastreável;
- [ ] campos ausentes foram marcados, não inferidos;
- [ ] critérios não foram alterados retroativamente;
- [ ] limitações estão explícitas;
- [ ] decisões humanas estão separadas da sugestão da IA;
- [ ] uso relevante de IA pode ser reconstruído.

## Recursos

Consulte quando necessário:
- `resources/checklist.md` — checklist operacional;
- `resources/matriz-evidencias.md` — template de matriz.
