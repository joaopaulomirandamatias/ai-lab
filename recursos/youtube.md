# YouTube

Vídeo é péssimo como fonte principal e excelente como **destravador**: quando um conceito
não entra pelo texto, um bom vídeo resolve em 20 minutos.

Regra: os canais marcados ⭐ você assiste **implementando junto**, com o editor aberto.
Os outros você assiste e volta para o livro.

---

## Os essenciais

### ⭐ 3Blue1Brown — *Grant Sanderson*
O melhor material visual de matemática que existe. Não substitui o livro; faz o livro
fazer sentido.

- **Essence of Linear Algebra** (15 vídeos) → M1. Assista **antes** de abrir o MML.
- **Essence of Calculus** (12 vídeos) → M1
- **Neural Networks** (série) → M5. O vídeo de backpropagation é a melhor explicação existente.
- **Série sobre Transformers e attention** → M8. Visualiza o que `QKᵀ` faz de verdade.

### ⭐ Andrej Karpathy
Provavelmente o recurso isolado mais valioso desta lista inteira para o M5–M9.

- **Neural Networks: Zero to Hero** — a série completa: `micrograd` (autograd do zero) →
  `makemore` (MLP, batchnorm, WaveNet) → **Let's build GPT from scratch** → **Let's build the GPT Tokenizer**.
- **Intro to Large Language Models** (~1h) — panorama conceitual honesto
- **Deep Dive into LLMs like ChatGPT** (~3h30) — pretraining, SFT, RLHF, alucinação, do começo ao fim

Não assista passivamente. O ganho está em digitar cada linha junto.

### ⭐ Umar Jamil
Implementações do zero com o paper aberto ao lado: Transformer, LLaMA, LoRA, DPO,
Flash Attention, quantização, treinamento distribuído. Talvez o melhor canal para M8–M10 —
é raro alguém explicar paper *e* código na mesma respiração.

### StatQuest — *Josh Starmer*
Estatística e ML clássico explicados com uma clareza quase absurda. M3–M4.
Playlists de *Statistics Fundamentals*, *Machine Learning* e *Neural Networks*.

---

## Papers e profundidade — M16 · [`14-papers/`](../14-papers/)

| Canal | Uso |
|---|---|
| **Yannic Kilcher** | Reviews de paper com opinião. Ver como ele *critica* um paper é treino direto para o M16. |
| **AI Coffee Break with Letitia** | Papers explicados em 10–15 min. Bom filtro antes de decidir ler o original. |
| **Machine Learning Street Talk** | Entrevistas longas e conceitualmente densas. Amplitude, não implementação. |
| **Two Minute Papers** | Panorama do que está saindo. Útil para não ficar por fora, inútil como estudo. |

---

## Cursos completos no YouTube

| Canal | Conteúdo |
|---|---|
| **Stanford Online** | CS229 (ML), CS224n (NLP), **CS25 — Transformers United** com autores dos papers |
| **MIT OpenCourseWare** | 18.06 (Strang, álgebra linear), 6.S191 (deep learning) |
| **DeepMind / Google DeepMind** | Séries de RL e de deep learning |
| **Hugging Face** | Tutoriais do ecossistema — `transformers`, `peft`, agentes, MCP |
| **Weights & Biases** | MLOps, experiment tracking, avaliação na prática |
| **Neo4j** | Knowledge graphs, GraphRAG, GraphAcademy → [`11-interop-semantica/`](../11-interop-semantica/) |
| **Sebastian Raschka** | Acompanha o livro *Build an LLM from Scratch* |
| **fast.ai / Jeremy Howard** | O curso inteiro, abordagem top-down |

---

## Em português

Menos volume, mas útil quando o conceito é novo e o inglês vira atrito adicional.

| Canal | Foco |
|---|---|
| **Mario Filho** | ML aplicado, validação, Kaggle. Grandmaster brasileiro — o conteúdo sobre *validação e leakage* é diretamente o Gate II. |
| **Programação Dinâmica** | Ciência de dados, estatística e Python |
| **Asimov Academy** | Python aplicado a IA, projetos práticos |
| **Sandeco** | Agentes de IA, CrewAI, LLM em português |

Vale conferir o catálogo atual de cada um antes de investir tempo — canais em português
mudam de foco com frequência.

---

## Mapa: o que assistir em cada mês

| Mês | Assistir |
|---|---|
| **M1** | 3Blue1Brown — *Essence of Linear Algebra* + *Essence of Calculus* |
| **M2** | 3Blue1Brown — *Neural Networks* (backprop) · Karpathy — `micrograd` |
| **M3** | StatQuest — *Statistics Fundamentals* · McElreath — *Statistical Rethinking* |
| **M4** | StatQuest — *Machine Learning* · Mario Filho — validação e leakage |
| **M5–M6** | Karpathy — `makemore` (série completa) |
| **M7** | fast.ai — partes de CNN e transfer learning |
| **M8** | 3Blue1Brown (attention) → Karpathy *Let's build GPT* → Umar Jamil (Transformer) → Stanford CS25 |
| **M9** | Karpathy — *Deep Dive into LLMs* · Karpathy — *GPT Tokenizer* |
| **M10** | Umar Jamil — LoRA, quantização, DPO |
| **M11** | Neo4j — GraphRAG · canais de avaliação de RAG (Ragas) |
| **M12–M13** | Hugging Face — Agents Course e MCP Course |
| **M14** | Yannic Kilcher — reviews de papers de multiagentes |
| **M16–M18** | Yannic Kilcher · Machine Learning Street Talk · gravações de AAMAS/NeurIPS |

---

## Um alerta

YouTube dá uma sensação de progresso que o estudo real não dá. Quatro horas de vídeo
passam rápido e rendem quase nada se nenhuma linha de código foi escrita.

Uma heurística que funciona: **para cada hora de vídeo, uma hora de implementação.**
Se você não consegue reimplementar o que assistiu, você não assistiu — você se distraiu
com qualidade de produção.
