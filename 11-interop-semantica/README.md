# 11 · Interoperabilidade semântica e Knowledge Graphs

**Transversal, do M11 em diante** · Roadmap: Níveis 15, 34, 43 · Pasta nova

Esta é a sua **especialização declarada** e o principal diferencial competitivo do plano.
Não é um mês — é uma trilha que corre em paralelo a partir do M11 (RAG/GraphRAG).

---

## O que dominar

**Knowledge Graphs** (Nível 15) — nós, arestas, propriedades, triplas, RDF, RDFS, OWL,
SPARQL, SHACL, SKOS, ontologias, taxonomias, vocabulários controlados, inferência,
raciocínio, entity linking, entity resolution, alinhamento de ontologias,
embeddings de grafo.

**IA simbólica e neuro-simbólica** (Nível 34) — lógica de primeira ordem, regras,
sistemas especialistas, raciocínio dedutivo, e a combinação com modelos neurais.
É a área que dá fundamento formal a "determinismo em sistemas LLM".

**Interoperabilidade semântica** (Nível 43) — mapeamento entre esquemas, mediação,
identidade, proveniência (**PROV-O**), versionamento de ontologia, governança semântica.

---

## Fonte principal

⭐ **Semantic Web for the Working Ontologist** (3ª ed.) — Allemang, Hendler, Gandon.
RDF, OWL e SHACL na prática, com modelagem real. É *o* livro desta trilha.

⭐ **Especificações W3C** — RDF 1.1, OWL 2, SPARQL 1.1, SHACL, SKOS, **PROV-O**, JSON-LD.
Normativas, gratuitas, escritas para leitura. Fonte primária sempre.

Papers-ponte: Hogan et al. (*Knowledge Graphs*, survey) · **Pan et al. — *Unifying LLMs
and Knowledge Graphs: A Roadmap*** · GraphRAG (Edge et al.).

Ferramentas: Neo4j · Apache Jena · RDFLib · Protégé · pySHACL · GraphDB.
Conferências: **ISWC** e **ESWC**.

---

## Onde isso encaixa no plano

```
RAG (M11) ──→ GraphRAG ──→ Knowledge Graph ──→ Ontologia
                                                   │
MCP (M13) ──→ capability discovery ────────────────┤
                                                   ↓
                                    Interoperabilidade semântica
                                                   ↓
                                          AI Governance (M15)
                                                   ↓
                                        Governed AI Systems
```

Três conexões que valem enxergar cedo:

1. **GraphRAG** é onde a trilha de LLM encontra esta. RAG vetorial perde estrutura;
   grafo devolve relações, hierarquia e restrição.
2. **Capability discovery no MCP** e descrição semântica de serviço são o mesmo problema
   com vocabulários diferentes. Poucos enxergam isso — e é onde nasce contribuição original.
3. **PROV-O** é o padrão W3C de proveniência. A camada de *Evidence* do M15 — quem, qual
   modelo, qual entrada, qual política, houve humano — é um problema de proveniência já
   modelado formalmente há mais de uma década. Reusar o padrão em vez de inventar
   estrutura própria é exatamente o tipo de argumento que sustenta um paper.

---

## Por que essa combinação é rara

O mercado de IA generativa quase não tem gente com fundamento em web semântica, e a
comunidade de web semântica quase não tem gente construindo sistemas agentic. A
interseção é pequena — e é onde a sua pergunta de pesquisa vive:

> Como adicionar determinismo, **governança semântica** e auditabilidade à execução de
> sistemas multiagentes baseados em LLM?

Ontologia é justamente o instrumento que permite responder "governança semântica" com
algo verificável, e não com política escrita em prosa.
