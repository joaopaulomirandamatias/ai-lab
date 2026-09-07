# Aula Completa — Uso de IA na Pesquisa Científica
## Do Prompt à Ciência Reproduzível

## 1. Pergunta orientadora

**Como utilizar Inteligência Artificial para aumentar a capacidade do pesquisador sem comprometer validade, autoria, ética, privacidade, transparência e reprodutibilidade?**

## 2. Conceito central

Uma saída de IA pode ser útil, bem escrita e plausível sem ser verdadeira.

```text
Resposta fluente ≠ resposta verdadeira ≠ evidência científica
```

IA pode:
- organizar;
- resumir;
- comparar;
- gerar hipóteses;
- sugerir buscas;
- ajudar a programar;
- apoiar análise;
- revisar linguagem.

Mas a evidência continua sendo o artigo, dado, experimento, documento, código, observação ou outra fonte científica verificável.

## 3. O pesquisador aumentado

```text
Problema → pesquisador → ferramentas de IA → verificação → evidência → decisão humana
```

O papel humano muda de executor exclusivo para **orquestrador, verificador e responsável**.

## 4. Onde a IA entra

| Etapa | Apoio possível | Controle |
|---|---|---|
| Pergunta | brainstorming e decomposição | relevância e originalidade humanas |
| Literatura | termos, busca semântica | verificar fonte original |
| Triagem | priorização | critérios explícitos |
| Leitura | explicação e extração | conferir o texto |
| Protocolo | estruturação | decisão metodológica humana |
| Dados | limpeza/código | privacidade e testes |
| Estatística | sugestões | pressupostos e validação |
| Qualitativa | codificação assistida | revisão e interpretação humana |
| Escrita | clareza/estrutura | fatos, autoria e referências |
| Revisão | crítica adversarial | confidencialidade |
| Divulgação | resumo/visualização | fidelidade científica |

## 5. Integridade científica — CNPq

A Portaria CNPq nº 2.664, de 6 de março de 2026, institui a Política de Integridade na Atividade Científica do CNPq. As diretrizes atualizadas incluem declaração do uso de ferramentas de Inteligência Artificial Generativa em qualquer fase relevante da pesquisa apoiada pelo CNPq, com indicação da ferramenta e finalidade.

Pontos didáticos:
- uso de IAG não é simplesmente proibido;
- transparência é central;
- autoria e responsabilidade permanecem humanas;
- conteúdo gerado não deve ser apresentado como se fosse de autoria humana;
- avaliação científica e confidencialidade exigem cuidado especial.

## 6. Cinco camadas de decisão

Antes de usar IA em uma tarefa científica, responda:

1. **Legal:** a legislação permite?
2. **Ética:** o protocolo/comitê permite?
3. **Institucional:** a instituição permite?
4. **Editorial:** o periódico permite?
5. **Metodológica:** o uso preserva validade e reprodutibilidade?

Uma resposta "sim" em uma camada não resolve as demais.

## 7. LGPD e pesquisa

Antes de enviar dados, classifique:
- públicos;
- pessoais;
- pessoais sensíveis;
- pseudonimizados;
- anonimizados;
- confidenciais;
- segredos comerciais;
- propriedade intelectual de terceiros.

Perguntas obrigatórias:
- Qual a finalidade?
- Qual a base legal?
- O titular foi informado quando necessário?
- O protocolo prevê esse processamento?
- Existe transferência para terceiros?
- O provedor retém dados?
- Há risco de reidentificação?
- Preciso anonimizar ou minimizar?

## 8. Confidencialidade

Ter acesso a um material não significa ter autorização para enviá-lo para um serviço externo.

Exemplos:
- manuscrito em peer review;
- projeto de financiamento;
- prontuário;
- entrevista confidencial;
- código proprietário;
- relatório interno.

## 9. Autoria

IA não atende critérios tradicionais de autoria responsável:
- não responde pela integridade do trabalho;
- não aprova versão final;
- não assume conflitos de interesse;
- não pode ser responsabilizada por fraude ou erro.

O humano continua responsável por:
- conteúdo;
- referências;
- permissões;
- plágio;
- análise;
- interpretação.

## 10. Pesquisa bibliográfica assistida

### Busca tradicional
- Scopus
- Web of Science
- PubMed
- IEEE Xplore
- ACM DL
- SciELO
- Google Scholar

### Busca assistida
- Elicit
- Consensus
- Semantic Scholar
- Scite
- ResearchRabbit
- Connected Papers
- OpenAlex

### Estratégia robusta

```text
Pergunta → descritores → sinônimos → string booleana
        ↘ busca semântica
        ↘ citation chaining
        ↘ referências de revisões
→ deduplicação → triagem → fonte original
```

## 11. Referência verdadeira ≠ afirmação correta

Uma referência pode existir e ainda ser usada incorretamente.

Checklist:
- título confere?
- autores?
- ano?
- periódico?
- DOI?
- texto original acessível?
- a afirmação realmente está no estudo?
- população/método são compatíveis?
- o resultado foi retirado do contexto?

## 12. RAG e grounding

RAG combina recuperação de documentos com geração.

```text
Corpus → busca → trechos → modelo → resposta → citação → verificação
```

Benefício: resposta mais ancorada.
Limite: fonte errada ou interpretação errada continuam possíveis.

## 13. Revisão sistemática

IA pode ajudar, mas não substitui:
- protocolo;
- critérios de elegibilidade;
- estratégia de busca;
- deduplicação;
- screening;
- avaliação do texto completo;
- extração;
- avaliação de qualidade/risco de viés;
- síntese;
- relatório PRISMA.

Documente quando algoritmos priorizam ou excluem estudos.

## 14. Programação científica

IA é útil para Python, R, SQL, MATLAB, Julia e notebooks.

Valide:
- lógica;
- método;
- dependências;
- tipos de dados;
- missing values;
- unidades;
- testes;
- aleatoriedade;
- performance;
- segurança.

```text
Código executa ≠ código correto ≠ método adequado
```

## 15. Estatística

Nunca aceite apenas "use teste X".
Pergunte:
- por que esse teste?
- quais pressupostos?
- tipo e distribuição das variáveis?
- independência?
- tamanho de efeito?
- intervalo de confiança?
- análise de sensibilidade?
- alternativas robustas?

## 16. Reprodutibilidade

Ideal:

```text
dados + código + ambiente + dependências + parâmetros + seed + versão = resultado reproduzível
```

Ferramentas úteis:
- Git/GitHub;
- Jupyter;
- R Markdown/Quarto;
- Docker;
- Conda/Poetry/renv;
- OSF;
- Zenodo.

## 17. Escrita científica

Usos mais seguros:
- clareza;
- gramática;
- tradução;
- reorganização;
- crítica;
- identificação de lacunas argumentativas.

Uso de maior risco:
- produzir discussão substantiva sem rastreabilidade;
- gerar citações não verificadas;
- criar resultados;
- formular interpretações sem contato com dados.

### Prompt melhor

Em vez de:
> Escreva minha discussão.

Use:
> Atue como revisor crítico. Liste quais afirmações desta discussão não são diretamente sustentadas pelos resultados apresentados. Não invente referências. Para cada ponto, indique o trecho que motivou a crítica.

## 18. IA como adversário científico

Utilize IA para:
- procurar hipóteses alternativas;
- detectar confundidores;
- sugerir testes de robustez;
- identificar limitações;
- desafiar generalizações.

Depois, valide cada crítica.

## 19. Semáforo

### Verde — menor risco
brainstorming, gramática, tradução, termos de busca, documentação de código.

### Amarelo — controle forte
síntese, triagem, extração, estatística, código, interpretação, texto substancial.

### Vermelho — não aceitável ou alto risco
fabricar dados, referências, manipular resultados, ocultar uso exigido, expor dados confidenciais, mascarar plágio.

## 20. Framework IA-Ciência Auditável

1. **DEFINIR** — o que a IA fará.
2. **CLASSIFICAR** — dados e confidencialidade.
3. **SELECIONAR** — ferramenta adequada.
4. **FUNDAMENTAR** — trabalhar com fontes.
5. **REGISTRAR** — ferramenta, versão, data, finalidade, prompt.
6. **VERIFICAR** — fatos, referências, código e interpretação.
7. **REPRODUZIR** — preservar artefatos metodológicos.
8. **DECLARAR** — informar uso conforme política aplicável.

## 21. Pergunta final

Não pergunte apenas:
> A IA acertou?

Pergunte:
> **Como posso demonstrar que ela acertou?**

## 22. Fórmula

```text
IA + MÉTODO + EVIDÊNCIA + VERIFICAÇÃO + TRANSPARÊNCIA + REPRODUTIBILIDADE
= CIÊNCIA ASSISTIDA POR IA

IA + texto bonito ≠ ciência
```
