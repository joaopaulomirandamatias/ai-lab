# 06 · Large Language Models

**Meses M9–M10** · Roadmap: Níveis 8, 9, 10, 11, 12

> Aqui começa sua especialização em IA generativa.

---

## M9 — Entender os modelos

**Famílias** — GPT, BERT, T5, Llama, Qwen, Mistral, Gemma, DeepSeek.
Não apenas *usar* os modelos: entender suas **diferenças arquiteturais**
(encoder-only vs decoder-only vs encoder-decoder, MoE, atenção, contexto).

**Treinamento de LLM** — pretraining, next-token prediction, masked language modeling,
instruction tuning, supervised fine-tuning, preference optimization, RLHF, RLAIF, DPO,
reward models, synthetic data.

**Hugging Face** (Nível 9) — Hub, `transformers`, `datasets`, `tokenizers`, `accelerate`,
`peft`, `trl`, safetensors, Spaces. Baixar, inferir, criar e publicar datasets e modelos,
treinar, fine-tunar, versionar, escrever model cards.

---

## M10 — Adaptar e otimizar

**Fine-tuning** (Nível 10) — full fine-tuning, transfer learning, LoRA, QLoRA, PEFT,
adapters, instruction tuning.

E o julgamento que importa mais que a técnica:
- quando fine-tuning é necessário;
- quando RAG é melhor;
- quando prompt engineering resolve;
- quando treinamento adicional não vale o custo.

**Quantização e otimização** (Nível 11) — FP32, FP16, BF16, INT8, INT4, GPTQ, AWQ, GGUF,
KV Cache, Flash Attention, batching, speculative decoding.
Ferramentas: llama.cpp, vLLM, Ollama, TensorRT-LLM.

**Prompt Engineering** (Nível 12) — zero-shot, few-shot, role prompting, structured
prompting, templates, chain-of-thought, decomposição, self-consistency, structured
outputs, constrained generation, JSON Schema, tool prompting.

> Importante, mas apenas uma pequena parte da especialização. E principalmente:
> **aprender a avaliar prompts empiricamente em vez de escolhê-los por impressão subjetiva.**

---

## Fonte principal

⭐ **AI Engineering** — Chip Huyen (para a parte de decisão e arquitetura)
⭐ **Build a LLM from Scratch** — Raschka (continuação do M8)

Vídeo: Karpathy — *Deep Dive into LLMs like ChatGPT* (3h30) · Umar Jamil (LoRA, DPO, quantização).
Curso: HF LLM Course · Stanford CS336.
Papers: GPT-3 · Chinchilla · InstructGPT · LoRA · QLoRA · DPO —
ver [`recursos/papers-essenciais.md`](../recursos/papers-essenciais.md).

---

## Projetos

**P9 — Comparação arquitetural (M9)** — texto técnico comparando GPT vs BERT vs T5 vs
Llama vs Mistral. *Pronto quando* explica por que encoder-only não serve para geração.

**P10 — Fine-tuning com LoRA (M10)** — modelo publicado no Hugging Face com model card
completo. *Pronto quando* o model card declara dados, limitações e viés, e o modelo é
comparado com o base em métrica.

---

## O erro que define o nível

A pergunta "prompt, RAG ou fine-tuning?" é a que mais separa quem sabe de quem repete.
A resposta padrão do mercado é fine-tuning — quase sempre errada, porque é a opção mais
cara e a que menos resolve o problema real, que costuma ser **falta de contexto**, não
falta de capacidade do modelo.

O Gate III (M11) vai exigir que você **prove** a resposta, não que a defenda.
Comece a coletar as evidências já aqui.
