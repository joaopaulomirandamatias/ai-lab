# 05 · NLP e Transformers

**Mês M8** · Roadmap: Níveis 6 e 7

> Esse é um dos pontos mais importantes da formação.

---

## Primeiro: NLP (Nível 6)

Antes de estudar LLM profundamente, domine NLP: tokenização, stemming, lemmatization,
bag-of-words, TF-IDF, n-grams, embeddings, Word2Vec, GloVe, FastText, similaridade de
cosseno, classificação de texto, Named Entity Recognition, sentiment analysis,
information extraction, busca semântica.

NER e information extraction voltam com força no
[`11-interop-semantica/`](../11-interop-semantica/) — é assim que texto vira grafo.

---

## Depois: Transformers (Nível 7)

Entender profundamente o paper **Attention Is All You Need**.

Aprender: tokens, embeddings, positional encoding, Query, Key, Value, attention,
self-attention, scaled dot-product attention, multi-head attention, feed-forward network,
residual connections, normalização, encoder, decoder, causal masking.

Compreender:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

E conseguir explicar **por que essa equação permite que um Transformer aprenda relações
contextuais entre tokens** — não apenas recitá-la.

---

## Fonte principal

⭐ **Build a Large Language Model (From Scratch)** — Sebastian Raschka. É exatamente este mês.

Ordem que funciona:
1. Jay Alammar — *The Illustrated Transformer* (intuição visual)
2. 3Blue1Brown — série sobre attention (o que `QKᵀ` faz geometricamente)
3. Karpathy — *Let's build GPT from scratch* (implementar junto)
4. Vaswani et al., 2017 — o paper (`1706.03762`)
5. Umar Jamil — Transformer from scratch (implementação com o paper aberto)
6. Karpathy — *Let's build the GPT Tokenizer*

Referência de NLP: Jurafsky & Martin, *Speech and Language Processing* 3ª ed. (grátis).
Curso: Stanford CS224n · CS25 (Transformers United).

---

## Projetos

**P8 — Transformer do zero (M8)** ⭐ o projeto mais importante da formação.
Decoder-only em PyTorch, sem copiar: positional encoding, Q/K/V, multi-head attention,
residuais, normalização, causal masking. *Pronto quando* gera texto coerente **e** você
explica a equação de atenção sem consultar nada.

**P8b — Tokenizer BPE do zero (M8)** — *Pronto quando* você explica por que "morango" e
"strawberry" quebram de formas diferentes e o que isso causa no modelo.

---

## Leia o paper três vezes

É o único do roadmap que merece isso: uma vez para entender, uma para implementar, uma
**depois** de implementar. A terceira leitura é onde você percebe o que não tinha
entendido nas duas primeiras.

Não avance para o M9 sem o P8 rodando. Todo o resto — LLM, RAG, agentes — é construído
em cima desta arquitetura. Uma lacuna aqui vira dívida por dez meses.
