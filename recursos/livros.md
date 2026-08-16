# Livros

Marcados com 🆓 os que têm versão oficial gratuita e legal do autor/editora.
Marcados com ⭐ a **fonte principal** sugerida para cada trilha — os outros são consulta.

> Regra que economiza mais tempo que qualquer outra: **uma fonte principal por tema.**
> Ler três livros de álgebra linear pela metade ensina menos que terminar um.

---

## Matemática — M1–M2 · [`01-math/`](../01-math/)

| Livro                                    | Autor                   | Nota                                                                                                                                                                               |
| ---------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⭐🆓 **Mathematics for Machine Learning** | Deisenroth, Faisal, Ong | `mml-book.github.io` · Escrito para quem vai fazer ML, não para matemáticos. Cobre exatamente o necessário: álgebra linear, cálculo vetorial, probabilidade, otimização, PCA, SVD. |
| 🆓 **The Matrix Cookbook**               | Petersen & Pedersen     | Referência de identidades matriciais. Não se lê — se consulta ao derivar backprop.                                                                                                 |
| **Introduction to Linear Algebra**       | Gilbert Strang          | Companheiro do MIT 18.06. Use se o MML ficar seco demais.                                                                                                                          |
| 🆓 **Linear Algebra Done Right**         | Sheldon Axler           | Mais abstrato e elegante. Opcional — só se você gostar de matemática por si.                                                                                                       |

**Atalho honesto:** para o Gate I, MML capítulos 2–5 + a série do 3Blue1Brown já bastam.

---

## Estatística e probabilidade — M3 · [`02-statistics/`](../02-statistics/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐🆓 **An Introduction to Statistical Learning (ISLP)** | James, Witten, Hastie, Tibshirani | `statlearning.com` · A edição **Python** é a certa. Ponte perfeita entre estatística e ML. |
| 🆓 **Statistical Rethinking** | Richard McElreath | Bayesiano, com curso em vídeo gratuito do próprio autor. O melhor livro que existe sobre *pensar* estatisticamente em vez de aplicar receita. |
| **All of Statistics** | Larry Wasserman | Compacto e denso. Inferência inteira em um volume — bom como referência. |
| 🆓 **Think Stats / Think Bayes** | Allen Downey | Leves, com código. Bons para destravar. |
| **Elements of Information Theory** | Cover & Thomas | Para entropia, KL, informação mútua de verdade. Consulta, não leitura linear. |

---

## Machine Learning clássico — M4 · [`03-machine-learning/`](../03-machine-learning/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** | Aurélien Géron | O mais prático que existe. Parte 1 cobre o M4 inteiro. |
| 🆓 **The Elements of Statistical Learning (ESL)** | Hastie, Tibshirani, Friedman | Grátis no site de Stanford. Teoria pesada — consulta. |
| **Deep Learning: Foundations and Concepts** | Bishop & Bishop (2023) | Sucessor moderno do *PRML*. Excelente ponte ML→DL. |
| **Machine Learning Engineering** | Andriy Burkov | Sobre o que dá errado fora do notebook. Curto, prático. |
| **The Hundred-Page Machine Learning Book** | Andriy Burkov | Panorama em um fim de semana. Bom para o começo, insuficiente sozinho. |

---

## Deep Learning — M5–M7 · [`04-deep-learning/`](../04-deep-learning/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐🆓 **Understanding Deep Learning** | Simon J.D. Prince | `udlbook.github.io` · Moderno, visual, com notebooks. Provavelmente a melhor recomendação isolada desta lista inteira. |
| 🆓 **Dive into Deep Learning (D2L)** | Zhang, Lipton, Li, Smola | `d2l.ai` · Cada conceito com código PyTorch executável ao lado. |
| 🆓 **Deep Learning** | Goodfellow, Bengio, Courville | `deeplearningbook.org` · Fundacional. Partes I e II envelheceram bem; parte III está datada. |
| 🆓 **Deep Learning with PyTorch** | Stevens, Antiga, Viehmann | Publicado gratuitamente pela própria PyTorch. Bom para o M6. |

---

## NLP, Transformers e LLM — M8–M10 · [`05-transformers/`](../05-transformers/) · [`06-llm/`](../06-llm/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **Build a Large Language Model (From Scratch)** | Sebastian Raschka | Constrói um GPT do zero, camada por camada. É *exatamente* o M8–M9. |
| 🆓 **Speech and Language Processing (3ª ed.)** | Jurafsky & Martin | Draft gratuito no site de Stanford. A referência canônica de NLP. |
| **Hands-On Large Language Models** | Jay Alammar & Maarten Grootendorst | Visual, do mesmo autor do *Illustrated Transformer*. Ótimo complemento. |
| **Natural Language Processing with Transformers** | Tunstall, von Werra, Wolf | Escrito pelo time da Hugging Face. Cobre o ecossistema do M9. |

---

## Engenharia de sistemas de IA — M9–M15

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **AI Engineering** | Chip Huyen (2025) | O livro mais alinhado com a sua fase M9–M15: RAG, avaliação, fine-tuning, custo, arquitetura. |
| **Designing Machine Learning Systems** | Chip Huyen | O anterior, focado em ML tradicional em produção. Continua válido. |
| **Designing Data-Intensive Applications** | Martin Kleppmann | Não é livro de IA — é o livro de sistemas distribuídos que sustenta a arquitetura. |

---

## Agentes e sistemas multiagentes — M12–M14 · [`08-agents/`](../08-agents/) · [`10-multiagent/`](../10-multiagent/)

Área com poucos livros bons e muitos papers. Aqui os livros dão o **fundamento clássico**
que o hype de LLM agents ignora — e é justamente essa base que diferencia sua pesquisa.

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **An Introduction to MultiAgent Systems** | Michael Wooldridge | O clássico de MAS. Agentes, ambientes, coordenação, negociação, protocolos de interação. Anterior aos LLMs e por isso mesmo essencial. |
| 🆓 **Multiagent Systems: Algorithmic, Game-Theoretic and Logical Foundations** | Shoham & Leyton-Brown | `masfoundations.org` · Formal. Consenso, mecanismo, teoria dos jogos. |
| **Multiagent Systems** | Gerhard Weiss (ed.), MIT Press | Coletânea abrangente. Consulta por capítulo. |
| **Artificial Intelligence: A Modern Approach (4ª ed.)** | Russell & Norvig | Capítulos de agentes racionais, busca e planejamento. A base que quase ninguém em "agentic AI" tem. |

---

## Interoperabilidade semântica e Knowledge Graphs — sua especialização · [`11-interop-semantica/`](../11-interop-semantica/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **Semantic Web for the Working Ontologist (3ª ed.)** | Allemang, Hendler, Gandon | RDF, RDFS, OWL, SHACL na prática, com modelagem real. É *o* livro para essa trilha. |
| 🆓 **Knowledge Graphs** | Hogan et al. | Também disponível como survey no arXiv. Cobre o campo inteiro com rigor. |
| **Designing and Building Enterprise Knowledge Graphs** | Sequeda & Lassila | Foco em KG corporativo — mapeamento relacional→RDF, governança. Muito próximo do seu contexto. |
| 🆓 **Knowledge Graphs: Data in Context for Responsive Businesses** | Barrasa & Webber | Gratuito pela Neo4j. Leve, bom para começar. |
| **Building Ontologies with Basic Formal Ontology** | Arp, Smith, Spear | Se a pesquisa for para ontologia formal séria (BFO, ontologias de alto nível). |

---

## Reinforcement Learning — [`04-deep-learning/`](../04-deep-learning/)

| Livro | Autor | Nota |
|---|---|---|
| 🆓 **Reinforcement Learning: An Introduction (2ª ed.)** | Sutton & Barto | A referência. Você precisa dos capítulos iniciais para entender RLHF/DPO. |
| **Grokking Deep Reinforcement Learning** | Miguel Morales | Mais acessível, com código. |

---

## Governança, segurança e XAI — M15 · [`12-ai-governance/`](../12-ai-governance/)

| Livro | Autor | Nota |
|---|---|---|
| 🆓 **Interpretable Machine Learning** | Christoph Molnar | `christophm.github.io/interpretable-ml-book` · SHAP, LIME, PDP. Referência de XAI. |
| **The Alignment Problem** | Brian Christian | Contexto histórico e conceitual de alinhamento. Leitura, não manual. |
| **Weapons of Math Destruction** | Cathy O'Neil | Contexto ético. Rápido e desconfortável, no bom sentido. |

Para segurança de IA aplicada, o material vivo está em normas e guias, não em livros —
ver [`sites-e-cursos.md`](sites-e-cursos.md), seção *Especificações e normas*.

---

## Escrita científica — M16–M18 · [`16-research/`](../16-research/)

| Livro | Autor | Nota |
|---|---|---|
| ⭐ **Writing for Computer Science** | Justin Zobel | Direto ao ponto, escrito para a nossa área. |
| **The Craft of Research** | Booth, Colomb, Williams | Como formular pergunta, argumento e evidência. |
| **Style: Lessons in Clarity and Grace** | Joseph Williams | Como escrever frases que não escondem o raciocínio. |

---

## Se fosse para comprar só cinco

Considerando o seu objetivo específico (agentic AI + interoperabilidade semântica):

1. **Understanding Deep Learning** — Prince 🆓
2. **Build a Large Language Model (From Scratch)** — Raschka
3. **AI Engineering** — Chip Huyen
4. **Semantic Web for the Working Ontologist** — Allemang et al.
5. **An Introduction to MultiAgent Systems** — Wooldridge

Os dois últimos são o que quase ninguém no mercado de IA leu. É exatamente daí que sai
diferencial de pesquisa.
