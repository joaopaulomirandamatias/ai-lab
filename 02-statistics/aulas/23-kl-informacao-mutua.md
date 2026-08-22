# Aula 23 — KL divergence e informação mútua

## Objetivos
- Calcular D_KL(P||Q)=ΣP log(P/Q).
- Entender que KL não é distância simétrica.
- Interpretar informação mútua I(X;Y) como redução de incerteza/dependência informacional.
- Relacionar MI a entropias.

## Prática
Comparar duas categorical distributions; mostrar D_KL(P||Q)≠D_KL(Q||P); calcular MI em tabela 2×2.

## Conexão com IA
KL aparece em VAEs, distillation, RLHF/regularização de políticas e aproximação variacional; MI aparece em seleção de features e representação.

## Referências
Cover & Thomas; MacKay; Murphy; Goodfellow et al.