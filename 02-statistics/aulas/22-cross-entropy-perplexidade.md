# Aula 22 — Cross-entropy, log-loss e perplexidade

## Objetivos
- Derivar cross-entropy H(p,q)=−Σp log q.
- Relacionar classificação one-hot a negative log-likelihood.
- Entender perplexidade como exp(cross-entropy média) na convenção natural.
- Reconhecer limites de perplexidade para comparar tokenizadores/domínios diferentes.

## Prática
Calcular cross-entropy manualmente; comparar previsões confiantes corretas/incorretas; calcular perplexidade de sequência simples.

## Conexão com IA
Cross-entropy é a loss central de classificação e treinamento autoregressivo de LLMs.

## Referências
Goodfellow et al.; Jurafsky & Martin, capítulos de LMs; Cover & Thomas.