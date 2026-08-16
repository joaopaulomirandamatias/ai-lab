# Projetos

O plano é explícito: **um projeto longitudinal**, não trinta desconectados.
Ele se chama **AI Systems Laboratory** e vive em [`projetos/ai-systems-lab/`](../projetos/).

Cada mês adiciona uma camada ao mesmo sistema. Ao final, o repositório é evidência
pública da evolução — muito mais forte que uma lista de cursos concluídos.

Cada projeto abaixo tem um **critério de pronto**. Sem ele, "fiz o projeto" é opinião.

---

## Fase 1 — Fundamentos (M1–M4)

### P1 · Álgebra linear em NumPy — M1
Implementar do zero: produto escalar, produto matricial, transposição, norma,
similaridade de cosseno, autovalores, PCA e SVD.

**Pronto quando:** cada função bate com a versão do NumPy/SciPy em teste automatizado,
e você consegue explicar *geometricamente* o que PCA faz.

### P2 · Gradient descent do zero — M2 🔶 Gate I
Regressão linear e logística treinadas com gradient descent escrito à mão, sem framework.

**Pronto quando:** a derivação está no papel, o código roda, e você tem um gráfico
mostrando o efeito de três learning rates diferentes — e sabe explicar cada curva.

### P3 · Experimento estatístico honesto — M3
Pegue um dataset real e responda uma pergunta com teste de hipótese.

**Pronto quando:** há intervalo de confiança, tamanho de efeito e uma frase declarando
o que o resultado **não** prova.

### P4 · Pipeline de ML sem leakage — M4 🔶 Gate II
Classificação ou regressão completa: split → baseline → modelo → validação cruzada → métricas.

**Pronto quando:** você consegue apontar três lugares onde havia risco de data leakage
e mostrar no código como cada um foi eliminado.

---

## Fase 2 — Deep Learning (M5–M8)

### P5 · MLP com backprop à mão — M5
Rede neural completa em NumPy, com backpropagation derivado e implementado por você.
Referência: `micrograd` do Karpathy.

**Pronto quando:** treina em MNIST e você consegue desenhar o grafo computacional de memória.

### P6 · O mesmo MLP em PyTorch — M6
Reescrever P5 usando `nn.Module`, `Dataset`, `DataLoader`, autograd, GPU, checkpoints,
mixed precision.

**Pronto quando:** os resultados batem com P5 e o treino roda em GPU com AMP.

### P7 · CNN + transfer learning — M7
Classificador de imagem treinado do zero, depois via ResNet pré-treinada.

**Pronto quando:** você mede e explica a diferença de acurácia e de tempo entre as duas abordagens.

### P8 · Transformer do zero — M8 ⭐
O projeto mais importante da formação. Decoder-only, em PyTorch, sem copiar:
tokenização, positional encoding, Q/K/V, multi-head attention, residuais, normalização,
causal masking.

**Pronto quando:** gera texto coerente **e** você explica `softmax(QKᵀ/√d_k)V` sem consultar nada.

### P8b · Tokenizer BPE do zero — M8
Byte-Pair Encoding implementado à mão.

**Pronto quando:** você consegue explicar por que "morango" e "strawberry" quebram de
formas diferentes, e o que isso causa no modelo.

---

## Fase 3 — LLM Engineer (M9–M11)

### P9 · Comparação arquitetural — M9
Texto técnico comparando GPT vs BERT vs T5 vs Llama vs Mistral vs Qwen: o que muda na
arquitetura, não no benchmark.

**Pronto quando:** o texto explica *por que* encoder-only não serve para geração.

### P10 · Fine-tuning com LoRA — M10
Fine-tune de um modelo aberto num dataset próprio, com PEFT/TRL, publicado no Hugging Face
com model card completo.

**Pronto quando:** o model card declara dados, limitações e viés — e o modelo é
comparado com o modelo-base em métrica, não em impressão.

### P11 · RAG em três versões — M11 🔶 Gate III ⭐
```
LLM puro   vs   LLM + RAG   vs   LLM + RAG + Reranker
```
Mesmo conjunto de avaliação para as três. Medir: Retrieval Precision, Retrieval Recall,
Faithfulness, Answer Relevance, Groundedness, custo e latência.

**Pronto quando:** existe uma tabela com as três colunas e uma conclusão que sobrevive
à pergunta "por quê?".

---

## Fase 4 — Agentic AI (M12–M15)

### P12 · Agente do zero, sem framework — M12 ⭐
```
LLM → (Memory, Tools, State) → Action
```
Loop próprio: tool calling, ReAct, planning, reflection, retries, condição de parada, guardrails.

**Pronto quando:** você compara o agente com um **workflow determinístico** na mesma tarefa
e consegue dizer em quais casos o workflow ganha. Saber quando *não* usar agente é o
objetivo real deste projeto.

### P13 · MCP Host + Client + Server do zero — M13 ⭐
```
MCP Host → MCP Client → MCP Server → Tool → Banco/API
```
JSON-RPC na mão, capability discovery, authentication, authorization. **Sem framework de agentes.**

**Pronto quando:** seu servidor funciona com um cliente MCP de terceiros, e o cliente de
terceiros funciona com o seu servidor. Interoperabilidade real, não demonstração.

### P14 · Research Multi-Agent System — M14 ⭐
Planner, researcher, critic, evaluator, executor sob um supervisor.

Depois **medir**: `1 agente vs 3 agentes vs 5 agentes`, comparando qualidade, custo,
latência e número de erros.

**Pronto quando:** existe a tabela comparativa. Essa medição já é, por si só, um
experimento científico publicável em workshop.

### P15 · Mini Governed AI Runtime — M15 🔶 Gate IV ⭐⭐ CAPSTONE
```
Usuário → Mission Compiler → Planner → Model Router → Agent
        → MCP Gateway → Policy → Tool → Evidence
```
Com: Model Registry, Policy Engine, HIL, Audit Trail, Evaluation, Observability.

HIL formalizado por confiança:
```
σ alto → ALLOW    σ intermediário → HIL    σ baixo → DENY
```

Evidence — dossiê por decisão: quem, qual modelo, qual versão, qual entrada, qual saída,
qual confiança, qual política, qual evidência, houve humano.

**Pronto quando:** você consegue pegar uma ação executada há uma semana e reconstruir
**por que** ela foi executada, a partir do audit trail — sem olhar o código.

---

## Fase 5 — Pesquisador (M16–M18)

### P16 · Repositório de fichamentos — M16
50–80 papers lidos e fichados em [`14-papers/`](../14-papers/).

**Pronto quando:** você consegue, de memória, citar três papers que se contradizem
e explicar a divergência metodológica.

### P17 · Reprodução de paper — M17
```
Paper → Reimplementação → Dataset → Baseline → Resultados → Comparação
```
**Pronto quando:** você tem os dois números lado a lado (o do paper e o seu) e uma
investigação escrita da diferença. Divergir é normal; não investigar não é.

### P18 · Experimento original — M18 🔶 Gate V ⭐⭐
O experimento controlado que dá validade científica ao roadmap inteiro:

> **Comparar workflow determinístico vs agente vs sistema multiagente**
> sob qualidade, custo, latência e segurança.

Ligado à pergunta de pesquisa sugerida:

> Como adicionar determinismo, governança semântica e auditabilidade à execução de
> sistemas multiagentes baseados em LLM?

**Pronto quando:** existe hipótese escrita **antes** do experimento, análise estatística
com tamanho de efeito, *threats to validity* declarados, e o texto foi submetido a
algum lugar — workshop, preprint ou periódico.

---

## Como versionar

Cada experimento segue a estrutura definida no fim do roadmap
(template em [`_templates/experimento/`](../_templates/experimento/)):

```
README · hipótese · dataset · código · configuração · métricas · resultados · gráficos · conclusão
```

A parte que as pessoas pulam é **hipótese antes do resultado**. Escrever a hipótese
depois de ver o número é a forma mais comum e mais silenciosa de fazer ciência ruim —
e o hábito de escrever antes é justamente o que o M18 vai cobrar.
