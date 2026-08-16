# 07 · Embeddings, busca semântica e RAG

**Mês M11** · Roadmap: Níveis 13 e 14 · **🔶 Gate III** no fim do mês

```
Query → Retrieval → Context → LLM → Answer
```

---

## Embeddings e busca semântica (Nível 13)

Essencial para seus trabalhos de interoperabilidade.

Embeddings, vector spaces, similarity, cosine similarity, dot product, dimensionalidade,
chunking, indexing, reranking.

**Vector databases** — pgvector, Qdrant, Milvus, Weaviate, Pinecone, Chroma, FAISS.

---

## RAG (Nível 14)

Dominar profundamente: ingestion, parsing, chunking, embeddings, retrieval, reranking,
metadata filtering, vector DB, **BM25**, hybrid search, query rewriting,
contextual retrieval, citations, grounded generation, **GraphRAG**.

**Avaliação** — muito importante:

```
Retrieval Precision · Retrieval Recall · Faithfulness
Answer Relevance · Groundedness
```

---

## Fonte principal

⭐ **AI Engineering** — Chip Huyen (capítulos de RAG e avaliação)

Papers na ordem: Lewis 2020 (RAG) → DPR → *Lost in the Middle* → Self-RAG →
**GraphRAG** → Ragas. Ver [`recursos/papers-essenciais.md`](../recursos/papers-essenciais.md).

Blog: Eugene Yan (RAG na prática) · Lil'Log.
Ferramentas: LlamaIndex, Haystack · Ragas para avaliação · pgvector se já usa Postgres.

---

## Projeto

**P11 — RAG em três versões (M11)** 🔶 **Gate III** ⭐

```
LLM puro   vs   LLM + RAG   vs   LLM + RAG + Reranker
```

Mesmo conjunto de avaliação para as três. Medir as cinco métricas acima + custo + latência.

*Pronto quando* existe uma tabela com as três colunas e uma conclusão que sobrevive à
pergunta "por quê?".

---

## Onde este mês conecta com a sua especialização

**GraphRAG é o paper-ponte.** RAG comum recupera trechos por similaridade vetorial e
perde a estrutura — relações, hierarquias, restrições. GraphRAG recupera sobre um grafo,
o que traz de volta exatamente o que ontologia oferece.

Ou seja: este mês é o ponto em que a trilha de LLM encontra a de
[interoperabilidade semântica](../11-interop-semantica/). Não é coincidência que a sua
pergunta de pesquisa fique nessa interseção.

Duas coisas que quase todo mundo erra e que valem a sua atenção:

- **Chunking é decisão de projeto, não default.** O tamanho do chunk determina o teto do
  seu retrieval. Testar três estratégias custa uma tarde e muda o resultado inteiro.
- **Busca híbrida (BM25 + vetorial) quase sempre ganha da vetorial pura.** Nome próprio,
  código, sigla e número são justamente onde o embedding falha — e é o que mais aparece
  em domínio técnico e jurídico.
