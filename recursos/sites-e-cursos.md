# Sites, cursos e ferramentas

🆓 = gratuito.

---

## Cursos estruturados

### Fundamentos
| Curso | Onde | Nota |
|---|---|---|
| 🆓 **MIT 18.06 — Linear Algebra** (Gilbert Strang) | MIT OpenCourseWare | Aulas completas. O curso de álgebra linear. |
| 🆓 **Machine Learning Specialization** (Andrew Ng) | Coursera (audit grátis) | A porta de entrada canônica do M4. |
| 🆓 **Statistical Rethinking** (McElreath) | YouTube + repositório | Curso completo em vídeo, acompanha o livro. |
| 🆓 **StatQuest — Statistics Fundamentals** | YouTube | Playlist, não curso formal. Ver [`youtube.md`](youtube.md). |

### Deep Learning
| Curso | Onde | Nota |
|---|---|---|
| 🆓 **Practical Deep Learning for Coders** (fast.ai) | `course.fast.ai` | Abordagem top-down: você treina modelo na aula 1 e entende a matemática depois. Ótimo se teoria pura te trava. |
| 🆓 **Neural Networks: Zero to Hero** (Karpathy) | YouTube + GitHub | Cobre M5–M8 melhor que qualquer curso pago. Ver [`youtube.md`](youtube.md). |
| 🆓 **MIT 6.S191 — Intro to Deep Learning** | `introtodeeplearning.com` | Curso anual, material sempre atualizado. |
| 🆓 **Stanford CS231n** — CNNs for Visual Recognition | `cs231n.stanford.edu` | Notas de aula excelentes, mesmo sem os vídeos. |
| **Deep Learning Specialization** (Andrew Ng) | Coursera | Sólido, um pouco datado em NLP. |

### NLP, Transformers e LLM
| Curso | Onde | Nota |
|---|---|---|
| 🆓 **Stanford CS224n — NLP with Deep Learning** | `web.stanford.edu/class/cs224n` | Slides e trabalhos públicos. Referência. |
| 🆓 **Stanford CS336 — Language Modeling from Scratch** | site do curso | Constrói um LLM inteiro do zero: tokenizer, treino, inferência, alinhamento. Muito alinhado ao M8–M10. |
| 🆓 **Stanford CS25 — Transformers United** | YouTube | Seminários com autores dos papers. |
| 🆓 **Hugging Face LLM Course** | `huggingface.co/learn` | Prático, do ecossistema que você vai usar de verdade. |

### Agentes, MCP e produção
| Curso | Onde | Nota |
|---|---|---|
| 🆓 **Hugging Face Agents Course** | `huggingface.co/learn/agents-course` | Do zero até agente funcional. |
| 🆓 **Hugging Face MCP Course** | `huggingface.co/learn/mcp-course` | Diretamente o M13. |
| 🆓 **Anthropic — cursos e cookbook** | `anthropic.com` / `github.com/anthropics` | Documentação de engenharia de agentes e de contexto, de quem construiu o MCP. |
| 🆓 **DeepLearning.AI Short Courses** | `deeplearning.ai/short-courses` | Dezenas de cursos de 1–2h sobre RAG, agentes, avaliação. Bom para amostragem rápida, raso como base. |
| 🆓 **Full Stack Deep Learning** | `fullstackdeeplearning.com` | Sistemas de ML em produção. |
| 🆓 **Made With ML** | `madewithml.com` | MLOps na prática, com código. |

---

## Prática e infraestrutura

| Recurso | Para quê |
|---|---|
| **Kaggle** | Competições + `Kaggle Learn` (micro-cursos) + GPU grátis semanal |
| **Hugging Face Hub / Spaces** | Modelos, datasets, deploy de demo. Você vai publicar aqui no M10. |
| **Google Colab** | GPU grátis para estudo; Pro quando o modelo crescer |
| **Lightning AI Studios** | Ambiente persistente com GPU, bom para projeto longo |
| **Modal · RunPod · Vast.ai** | GPU sob demanda quando precisar treinar de verdade |
| **Ollama / LM Studio** | Rodar LLM local na sua máquina — essencial para experimentar sem custo de API |

---

## Descoberta e leitura de papers — M16 · [`14-papers/`](../14-papers/)

| Recurso | Para quê |
|---|---|
| **arXiv** — `cs.CL`, `cs.AI`, `cs.LG`, `cs.MA`, `cs.SE` | A fonte. `cs.MA` (multiagent) é a categoria que quase ninguém acompanha e é a sua. |
| **huggingface.co/papers** | Daily papers com discussão — bom filtro do que saiu hoje |
| **Semantic Scholar** | Busca com grafo de citações; API boa para automatizar |
| **Connected Papers** | Mapa visual de um paper e seus vizinhos. Excelente para achar o *gap* |
| **Litmaps** | Acompanhar a evolução de uma linha de pesquisa ao longo do tempo |
| **OpenReview** | Ler as **revisões** de ICLR/NeurIPS. Ver papers sendo criticados ensina a ler criticamente |
| **Elicit · Consensus** | Busca semântica sobre literatura, útil para revisão sistemática |
| **Google Scholar (alertas)** | Alerta por termo: `multi-agent LLM governance`, `semantic interoperability agents` |
| **Portal de Periódicos CAPES** | Acesso institucional a bases pagas |

Onde publicar / acompanhar, na sua área: **AAMAS** (multiagentes), **NeurIPS**, **ICLR**,
**ACL/EMNLP**, **ISWC** e **ESWC** (web semântica), **FAccT** (governança e ética),
**AIES**. Workshops dessas conferências são a porta de entrada realista para o M18.

---

## Blogs que valem acompanhar

| Blog | Por quê |
|---|---|
| **Lil'Log** — Lilian Weng | Os melhores artigos-síntese sobre agentes, alucinação, prompt engineering, RL. Leitura obrigatória do M12. |
| **Ahead of AI** — Sebastian Raschka | LLMs explicados com rigor e código |
| **Jay Alammar** | *The Illustrated Transformer* / *Illustrated Word2Vec* — leia antes do M8 |
| **Chip Huyen** | Arquitetura de sistemas de IA e custos reais |
| **Eugene Yan** | RAG, avaliação, ML aplicado em produto |
| **Simon Willison** | Cobertura diária honesta do ecossistema, incluindo segurança e prompt injection |
| **Anthropic Engineering** | *Building effective agents* e engenharia de contexto |
| **Interconnects** — Nathan Lambert | Post-training, RLHF, DPO com profundidade |
| **OpenAI Cookbook** | Receitas de código verificadas |
| **Distill.pub** | Arquivado, mas os artigos de visualização continuam insuperáveis |
| **The Gradient** | Ensaios longos sobre direção do campo |

---

## Especificações e normas — a espinha da sua especialização

Aqui está o que separa "usa MCP" de "entende interoperabilidade". Leia as fontes primárias.

### Protocolos de agentes — M13 · [`09-mcp-protocolos/`](../09-mcp-protocolos/)
- **MCP** — `modelcontextprotocol.io` · especificação, SDKs e servidores de referência
- **JSON-RPC 2.0** — a base sobre a qual o MCP é construído. Leia a spec original, é curta
- **A2A (Agent-to-Agent)** — protocolo de comunicação entre agentes, hoje sob a Linux Foundation
- **OpenAPI / AsyncAPI** — descrição de capacidade que antecede tudo isso

### Web semântica — [`11-interop-semantica/`](../11-interop-semantica/)
Padrões **W3C**, todos gratuitos e normativos:
- **RDF 1.1** · **RDFS** · **OWL 2** — modelagem e inferência
- **SPARQL 1.1** — consulta
- **SHACL** — validação de grafos (o equivalente a schema/constraint)
- **SKOS** — vocabulários e taxonomias
- **PROV-O** — proveniência. Diretamente relevante para a sua camada de *evidence*
- **JSON-LD** — RDF em JSON, a ponte para APIs
- **schema.org** — vocabulário de fato da web

### Governança e segurança — M15 · [`12-ai-governance/`](../12-ai-governance/)
- **OWASP Top 10 for LLM Applications** — a taxonomia de ataque de referência
- **OWASP Agentic Security Initiative** — ameaças específicas de agentes
- **NIST AI Risk Management Framework (AI RMF 1.0)** + *Generative AI Profile*
- **EU AI Act** — texto oficial e AI Act Explorer
- **ISO/IEC 42001** (sistema de gestão de IA) e **ISO/IEC 23894** (gestão de risco)
- **Model Cards** (Mitchell et al.) e **Datasheets for Datasets** (Gebru et al.)
- **LGPD — Lei 13.709/2018** — aplicada a datasets e decisão automatizada
- **MITRE ATLAS** — táticas e técnicas adversariais contra sistemas de IA

---

## Ferramentas por camada

Aprenda a camada **antes** do framework que a esconde.

| Camada | Ferramentas |
|---|---|
| **Treino** | PyTorch · Lightning · HF `transformers` / `trl` / `peft` / `accelerate` · DeepSpeed · FSDP |
| **Inferência** | vLLM · llama.cpp · Ollama · SGLang · TensorRT-LLM |
| **Vector DB** | pgvector · Qdrant · Milvus · Weaviate · Chroma · FAISS |
| **RAG** | LlamaIndex · Haystack · LangChain *(use depois de entender, não antes)* |
| **Agentes** | SDKs oficiais do MCP · Claude Agent SDK · OpenAI Agents SDK · LangGraph · Pydantic AI · smolagents · CrewAI · AutoGen |
| **Avaliação** | Ragas · DeepEval · promptfoo · `lm-evaluation-harness` · HELM |
| **Observabilidade** | Langfuse · Arize Phoenix · OpenLLMetry · W&B Weave · LangSmith |
| **KG / semântica** | Neo4j · Apache Jena · GraphDB · RDFLib · Protégé · pySHACL · Virtuoso |
| **MLOps** | MLflow · DVC · Weights & Biases · BentoML · KServe |

**Sobre frameworks de agente:** o plano é explícito em construir agente, MCP server e
sistema multiagente **do zero** primeiro (M12–M14). Não é purismo — é que o Gate IV exige
que você explique o mecanismo, e framework é exatamente o que impede isso.
