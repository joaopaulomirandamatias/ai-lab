# Papers essenciais

Lista de partida para a meta do M16: **50–80 papers realmente lidos**, não armazenados.
Fichamentos vão em [`14-papers/`](../14-papers/), usando
[`_templates/fichamento-paper.md`](../_templates/fichamento-paper.md).

> ⚠️ Os identificadores arXiv abaixo foram escritos de memória. Confira ao abrir —
> se o número não bater com o título, busque pelo título no Semantic Scholar.
> O título e o primeiro autor são a chave confiável, não o ID.

**Marcados 🔴 os que são leitura obrigatória para a sua especialização** — se o tempo
apertar, esses não podem cair.

---

## Fundamentos de Deep Learning — M5–M7

| Paper | Autor, ano | arXiv |
|---|---|---|
| ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky et al., 2012 | — |
| Dropout: A Simple Way to Prevent Overfitting | Srivastava et al., 2014 | JMLR |
| Batch Normalization | Ioffe & Szegedy, 2015 | 1502.03167 |
| Deep Residual Learning (ResNet) | He et al., 2015 | 1512.03385 |
| Adam: A Method for Stochastic Optimization | Kingma & Ba, 2014 | 1412.6980 |

---

## Embeddings e NLP pré-Transformer — M6 · [`05-transformers/`](../05-transformers/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| Efficient Estimation of Word Representations (Word2Vec) | Mikolov et al., 2013 | 1301.3781 |
| GloVe: Global Vectors for Word Representation | Pennington et al., 2014 | — |
| Sequence to Sequence Learning with Neural Networks | Sutskever et al., 2014 | 1409.3215 |
| Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau et al., 2014 | 1409.0473 |
| Sentence-BERT | Reimers & Gurevych, 2019 | 1908.10084 |

O paper de Bahdanau é onde *attention* nasce, três anos antes do Transformer. Ler os dois
em sequência mostra o que realmente foi a inovação de 2017.

---

## Transformers — M8 · [`05-transformers/`](../05-transformers/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 **Attention Is All You Need** | Vaswani et al., 2017 | 1706.03762 |
| 🔴 BERT | Devlin et al., 2018 | 1810.04805 |
| T5: Exploring the Limits of Transfer Learning | Raffel et al., 2019 | 1910.10683 |
| An Image is Worth 16x16 Words (ViT) | Dosovitskiy et al., 2020 | 2010.11929 |
| RoFormer: Rotary Position Embedding (RoPE) | Su et al., 2021 | 2104.09864 |
| FlashAttention | Dao et al., 2022 | 2205.14135 |

O paper de 2017 deve ser lido **três vezes**: uma para entender, uma para implementar,
uma depois de implementar. É o único desta lista que merece isso.

---

## LLMs — M9 · [`06-llm/`](../06-llm/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Language Models are Few-Shot Learners (GPT-3) | Brown et al., 2020 | 2005.14165 |
| Scaling Laws for Neural Language Models | Kaplan et al., 2020 | 2001.08361 |
| 🔴 Training Compute-Optimal LLMs (Chinchilla) | Hoffmann et al., 2022 | 2203.15556 |
| Switch Transformers (MoE) | Fedus et al., 2021 | 2101.03961 |
| LLaMA / Llama 2 | Touvron et al., 2023 | 2302.13971 / 2307.09288 |
| Mistral 7B · Mixtral of Experts | Jiang et al., 2023/2024 | 2310.06825 / 2401.04088 |
| Emergent Abilities of LLMs | Wei et al., 2022 | 2206.07682 |
| Are Emergent Abilities of LLMs a Mirage? | Schaeffer et al., 2023 | 2304.15004 |
| DeepSeek-R1 | DeepSeek-AI, 2025 | 2501.12948 |

Leia Wei e Schaeffer **em sequência**: é o melhor exemplo disponível de como uma
afirmação forte é contestada metodologicamente. Treino direto para o M16.

---

## Alinhamento e fine-tuning — M10 · [`06-llm/`](../06-llm/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Training LMs to Follow Instructions (InstructGPT/RLHF) | Ouyang et al., 2022 | 2203.02155 |
| 🔴 LoRA: Low-Rank Adaptation | Hu et al., 2021 | 2106.09685 |
| QLoRA | Dettmers et al., 2023 | 2305.14314 |
| Direct Preference Optimization (DPO) | Rafailov et al., 2023 | 2305.18290 |
| Constitutional AI | Bai et al., 2022 | 2212.08073 |
| Self-Instruct | Wang et al., 2022 | 2212.10560 |

---

## RAG e busca semântica — M11 · [`07-rag/`](../07-rag/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Retrieval-Augmented Generation for Knowledge-Intensive NLP | Lewis et al., 2020 | 2005.11401 |
| Dense Passage Retrieval (DPR) | Karpukhin et al., 2020 | 2004.04906 |
| ColBERT: Efficient Late Interaction Retrieval | Khattab & Zaharia, 2020 | 2004.12832 |
| HyDE: Precise Zero-Shot Dense Retrieval | Gao et al., 2022 | 2212.10496 |
| Self-RAG | Asai et al., 2023 | 2310.11511 |
| Corrective RAG (CRAG) | Yan et al., 2024 | 2401.15884 |
| 🔴 GraphRAG: From Local to Global | Edge et al., 2024 | 2404.16130 |
| Lost in the Middle | Liu et al., 2023 | 2307.03172 |
| RAG for LLMs: A Survey | Gao et al., 2023 | 2312.10997 |
| Ragas: Automated Evaluation of RAG | Es et al., 2023 | 2309.15217 |

**GraphRAG é o paper-ponte** entre RAG e a sua especialização em knowledge graphs.
Leia junto com o survey de Pan (seção de interoperabilidade).

---

## Raciocínio e prompting — M12 · [`08-agents/`](../08-agents/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Chain-of-Thought Prompting | Wei et al., 2022 | 2201.11903 |
| Self-Consistency Improves CoT | Wang et al., 2022 | 2203.11171 |
| Tree of Thoughts | Yao et al., 2023 | 2305.10601 |
| Least-to-Most Prompting | Zhou et al., 2022 | 2205.10625 |

---

## Agentes — M12 · [`08-agents/`](../08-agents/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 **ReAct: Synergizing Reasoning and Acting** | Yao et al., 2022 | 2210.03629 |
| 🔴 Toolformer | Schick et al., 2023 | 2302.04761 |
| Reflexion: Verbal Reinforcement Learning | Shinn et al., 2023 | 2303.11366 |
| Generative Agents: Interactive Simulacra | Park et al., 2023 | 2304.03442 |
| Voyager: Open-Ended Embodied Agent | Wang et al., 2023 | 2305.16291 |
| A Survey on LLM-based Autonomous Agents | Wang et al., 2023 | 2308.11432 |
| The Rise and Potential of LLM Based Agents | Xi et al., 2023 | 2309.07864 |

---

## Sistemas multiagentes — M14 · [`10-multiagent/`](../10-multiagent/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 AutoGen: Multi-Agent Conversation Framework | Wu et al., 2023 | 2308.08155 |
| 🔴 CAMEL: Communicative Agents | Li et al., 2023 | 2303.17760 |
| MetaGPT: Meta Programming for Multi-Agent | Hong et al., 2023 | 2308.00352 |
| ChatDev: Communicative Agents for Software Development | Qian et al., 2023 | 2307.07924 |
| Improving Factuality via Multiagent Debate | Du et al., 2023 | 2305.14325 |
| Encouraging Divergent Thinking (Multi-Agent Debate) | Liang et al., 2023 | 2305.19118 |

**Complemento clássico obrigatório:** a literatura de MAS anterior aos LLMs —
Wooldridge, Jennings, protocolos FIPA-ACL, Contract Net Protocol (Smith, 1980),
blackboard systems. É o que quase ninguém em "agentic AI" leu, e é onde estão os
conceitos que o campo está redescobrindo com outro nome.

---

## Interoperabilidade semântica e Knowledge Graphs · [`11-interop-semantica/`](../11-interop-semantica/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Knowledge Graphs (survey completo) | Hogan et al., 2020 | 2003.02320 |
| 🔴 Unifying LLMs and Knowledge Graphs: A Roadmap | Pan et al., 2023 | 2306.08302 |
| Translating Embeddings for Modeling Multi-relational Data (TransE) | Bordes et al., 2013 | — |
| A Review of Relational ML for Knowledge Graphs | Nickel et al., 2015 | 1503.00759 |
| Ontology Matching / alinhamento de ontologias | Euzenat & Shvaiko | livro + surveys |

Somado às specs W3C (RDF, OWL, SHACL, PROV-O) listadas em
[`sites-e-cursos.md`](sites-e-cursos.md). **Esta é a seção mais rala em papers de moda e
mais rica em fundamento** — exatamente onde está o seu diferencial competitivo.

---

## Segurança de IA — M15 · [`12-ai-governance/`](../12-ai-governance/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Not what you've signed up for (indirect prompt injection) | Greshake et al., 2023 | 2302.12173 |
| 🔴 Universal and Transferable Adversarial Attacks on Aligned LMs | Zou et al., 2023 | 2307.15043 |
| Jailbroken: How Does LLM Safety Training Fail? | Wei et al., 2023 | 2307.02483 |
| Extracting Training Data from LLMs | Carlini et al., 2020 | 2012.07805 |
| Poisoning Web-Scale Training Datasets | Carlini et al., 2023 | 2302.10149 |

Greshake é **o** paper para quem constrói agentes com ferramentas: mostra que o ataque
não vem do usuário, vem do conteúdo que o agente lê. Base direta do seu Policy Engine.

---

## Governança, avaliação e ética — M15–M16 · [`13-evaluation/`](../13-evaluation/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| 🔴 Judging LLM-as-a-Judge (MT-Bench, Chatbot Arena) | Zheng et al., 2023 | 2306.05685 |
| HELM: Holistic Evaluation of Language Models | Liang et al., 2022 | 2211.09110 |
| Model Cards for Model Reporting | Mitchell et al., 2018 | 1810.03993 |
| Datasheets for Datasets | Gebru et al., 2018 | 1803.09010 |
| On the Dangers of Stochastic Parrots | Bender et al., 2021 | FAccT |
| Beyond the Imitation Game (BIG-bench) | Srivastava et al., 2022 | 2206.04615 |

---

## Infraestrutura e treino distribuído · [`04-deep-learning/`](../04-deep-learning/)

| Paper | Autor, ano | arXiv |
|---|---|---|
| Megatron-LM | Shoeybi et al., 2019 | 1909.08053 |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al., 2019 | 1910.02054 |
| GPipe | Huang et al., 2018 | 1811.06965 |

---

## Multimodal e generativo além dos LLMs

| Paper | Autor, ano | arXiv |
|---|---|---|
| CLIP: Learning Transferable Visual Models | Radford et al., 2021 | 2103.00020 |
| Denoising Diffusion Probabilistic Models | Ho et al., 2020 | 2006.11239 |
| High-Resolution Image Synthesis with Latent Diffusion | Rombach et al., 2021 | 2112.10752 |
| Whisper: Robust Speech Recognition | Radford et al., 2022 | 2212.04356 |

---

## Ordem sugerida de leitura

Não leia por lista — leia por **necessidade do mês**:

```
M8   Bahdanau → Vaswani → BERT → GPT-3
M9   Kaplan → Chinchilla → Llama → Wei/Schaeffer (o debate)
M10  InstructGPT → LoRA → QLoRA → DPO
M11  Lewis → DPR → Lost in the Middle → Self-RAG → GraphRAG → Ragas
M12  CoT → Self-Consistency → ReAct → Toolformer → Reflexion
M13  spec do MCP + JSON-RPC (não são papers, mas leia com o mesmo rigor)
M14  AutoGen → CAMEL → MetaGPT → Multiagent Debate → Wooldridge (clássico)
M15  Greshake → Zou → OWASP LLM Top 10 → NIST AI RMF
M16+ Hogan → Pan → o que a sua pergunta de pesquisa exigir
```

A partir do M16 a lista para de ser útil: você passa a montar a sua própria,
seguindo citações a partir do gap que identificou. Esse é o ponto onde o roadmap acaba
e a pesquisa começa.
